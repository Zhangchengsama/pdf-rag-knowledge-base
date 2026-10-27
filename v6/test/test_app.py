"""原生 AppTest 使用合成 PDF/聊天记录，隔离模型下载与外部 API。"""
import ast
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import pymupdf
from streamlit.testing.v1 import AppTest
from test_regressions import TinyModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class OfflineModel(TinyModel):
    def __init__(self, *args, **kwargs):
        pass


class AppTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix='rag_app_'))
        self.pdf = self.directory / 'sample.pdf'
        self.index = self.directory / 'sample.bin'
        self.histories = self.directory / 'data' / 'conversations'
        self.histories.mkdir(parents=True)
        with pymupdf.open() as document:
            for text in ('apple', 'banana'):
                document.new_page().insert_text((72, 72), text)
            document.save(str(self.pdf))
        self.prompts = []
        embeddings = types.ModuleType('sentence_transformers')
        embeddings.SentenceTransformer = OfflineModel
        llm = types.ModuleType('LLM')
        llm.ask_llm = lambda prompt: self.prompts.append(prompt) or 'offline answer'
        replacements = {'sentence_transformers': embeddings, 'LLM': llm}
        previous = {name: sys.modules.get(name) for name in replacements}
        sys.modules.update(replacements)
        def restore_modules():
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        self.addCleanup(restore_modules)

        # 仅替换测试副本的路径和输入配置，页面及服务执行真实代码。
        tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
        values = {'road': f'Path({str(self.pdf)!r})', 'index_path': repr(str(self.index)),
                  'max_pages': '30', 'force_rebuild': 'False'}
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in values):
                node.value = ast.parse(values[node.targets[0].id], mode='eval').body
        tree.body.insert(0, ast.parse(f'__file__ = {str(self.directory / "app.py")!r}').body[0])
        self.app = AppTest.from_string(ast.unparse(ast.fix_missing_locations(tree)), default_timeout=10)

    def run_app(self):
        self.app.run()
        self.assertFalse(self.app.exception, [item.message for item in self.app.exception])
        return self.app

    def history(self, identity, title, answer, modified_ns):
        path = self.histories / f'{identity}.jsonl'
        records = [{'role': 'user', 'content': title},
                   {'role': 'assistant', 'content': answer, 'sources': [
                       {'file_name': 'sample.pdf', 'page_number': 1, 'text': '历史原文'}]}]
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records),
                        encoding='utf-8')
        os.utime(path, ns=(modified_ns, modified_ns))
        return path

    def test_open_only_reports_status_without_building(self):
        app = self.run_app()
        self.assertFalse(self.index.exists())
        self.assertFalse(self.pdf.with_suffix('.json').exists())
        text = '\n'.join(item.value for item in list(app.markdown) + list(app.caption))
        self.assertIn('实际读取：2 页', text)
        self.assertIn('尚未建库', text)
        self.assertIn('文本块数：未统计', text)
        self.assertEqual(self.prompts, [])

    def test_chat_history_sources_titles_and_recent_order_survive_reruns(self):
        older_id, newer_id = 'a' * 32, 'b' * 32
        history = self.history(older_id, '旧问题', '旧回答', 1_000_000_000)
        self.history(newer_id, '较新问题', '较新回答', 2_000_000_000)
        app = self.run_app()
        self.assertEqual(app.selectbox[0].options, ['较新问题', '旧问题'])
        app.selectbox[0].select(older_id)
        self.run_app()
        self.assertEqual(app.session_state['messages'][1]['content'], '旧回答')
        self.assertEqual(len(app.expander), 1)
        app.chat_input[0].set_value('apple followup')
        self.run_app()
        self.assertEqual(len(self.prompts), 1)
        self.assertEqual(app.selectbox[0].value, older_id)
        self.assertEqual(app.selectbox[0].options, ['旧问题', '较新问题'])
        self.assertEqual(len(app.session_state['messages']), 4)
        records = [json.loads(line) for line in history.read_text(encoding='utf-8').splitlines()]
        self.assertEqual([row['role'] for row in records], ['user', 'assistant', 'user', 'assistant'])
        self.assertEqual(records[-1]['content'], 'offline answer')
        self.assertEqual([row['page_number'] for row in records[-1]['sources']], [1, 2])
        self.assertEqual(set(records[-1]), {'role', 'content', 'sources'})
        self.assertEqual(len(app.expander), 2)
        self.assertIn('文本块数：2', '\n'.join(item.value for item in app.markdown))
        self.run_app()
        self.assertEqual(len(self.prompts), 1)
        self.assertEqual(history.read_text(encoding='utf-8').count('apple followup'), 1)
        app.button[0].click()
        self.run_app()
        self.assertEqual(app.session_state['messages'], [])
        self.assertNotEqual(app.session_state['conversation_id'], older_id)
        app.selectbox[0].select(older_id)
        self.run_app()
        self.assertEqual(len(app.session_state['messages']), 4)
        self.assertEqual(len(app.expander), 2)
        self.assertEqual(len(self.prompts), 1)
        service = importlib.import_module('rag_service')
        service.building_data(self.pdf, str(self.index), TinyModel(), max_pages=30, force_rebuild=True)
        self.run_app()
        self.assertIn('文本块数：未统计', '\n'.join(item.value for item in app.markdown))

    def test_same_title_history_selection_keeps_original_ids(self):
        first, second = 'a' * 32, 'b' * 32
        self.history(first, '同一个标题', '第一段回答', 1_000_000_000)
        self.history(second, '同一个标题', '第二段回答', 2_000_000_000)
        app = self.run_app()
        self.assertTrue(all('同一个标题' in label for label in app.selectbox[0].options))
        self.assertEqual(len(set(app.selectbox[0].options)), 2)
        for identity, expected in ((first, '第一段回答'), (second, '第二段回答')):
            app.selectbox[0].select(identity)
            self.run_app()
            self.assertEqual(app.session_state['conversation_id'], identity)
            self.assertEqual(app.session_state['messages'][1]['content'], expected)
        self.assertEqual(self.prompts, [])
