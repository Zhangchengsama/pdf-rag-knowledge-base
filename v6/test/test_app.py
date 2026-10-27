"""原生 AppTest 使用合成 PDF/聊天记录，隔离模型下载与外部 API。"""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pymupdf
from streamlit.testing.v1 import AppTest
from test_regressions import TinyModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def render_test_app(settings):
    from ui import run_web
    run_web(settings)


def render_test_entry(settings):
    import runpy
    import sys
    import streamlit as st
    from unittest.mock import patch
    with patch('config.load_settings', return_value=settings), \
            patch.object(sys, 'argv', ['main.py', '--web']):
        runpy.run_module('main', run_name='__main__')
    st.caption('入口执行完成')


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
        self.title_prompts = []
        embeddings = types.ModuleType('sentence_transformers')
        embeddings.SentenceTransformer = OfflineModel
        llm = types.ModuleType('LLM')
        llm.ask_llm = lambda prompt: self.prompts.append(prompt) or 'offline answer'
        llm.generate_title = lambda question: self.title_prompts.append(question) or '概念与实验讨论'
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

        # 注入配置，不改写页面源码或页面的私有全局变量。
        from config import load_settings
        self.settings = load_settings(base_dir=self.directory, pdf_path=self.pdf, index_path=self.index)
        self.app = AppTest.from_function(render_test_app, args=(self.settings,), default_timeout=10)

    def run_app(self):
        self.app.run()
        self.assertFalse(self.app.exception, [item.message for item in self.app.exception])
        return self.app

    def history(self, identity, title, answer, modified_ns, bind=True):
        path = self.histories / f'{identity}.jsonl'
        records = [{'role': 'user', 'content': title},
                   {'role': 'assistant', 'content': answer, 'sources': [
                       {'file_name': 'sample.pdf', 'page_number': 1, 'text': '历史原文'}]}]
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records),
                        encoding='utf-8')
        os.utime(path, ns=(modified_ns, modified_ns))
        if bind:
            from conversations import save_metadata
            save_metadata(path, {'title': title, 'document': {
                'pdf_path': str(self.pdf), 'index_path': str(self.index), 'max_pages': 30}})
        return path

    def sources(self):
        return [item for item in self.app.expander if item.label == '📚 查看参考资料']

    def test_open_only_reports_status_without_building(self):
        app = self.run_app()
        self.assertFalse(self.index.exists())
        self.assertFalse(self.pdf.with_suffix('.json').exists())
        text = '\n'.join(item.value for item in list(app.markdown) + list(app.caption))
        self.assertIn('实际读取：2 页', text)
        self.assertIn('尚未建库', text)
        self.assertIn('文本块数：未统计', text)
        self.assertEqual(self.prompts, [])

    def test_main_script_returns_normally_in_streamlit_worker(self):
        self.app = AppTest.from_function(render_test_entry, args=(self.settings,), default_timeout=3)
        app = self.run_app()
        self.assertIn('入口执行完成', [item.value for item in app.caption])
        self.assertEqual(len(app.chat_input), 1)

    def test_chat_history_sources_titles_and_recent_order_survive_reruns(self):
        older_id, newer_id = 'a' * 32, 'b' * 32
        history = self.history(older_id, '旧问题', '旧回答', 1_000_000_000)
        self.history(newer_id, '较新问题', '较新回答', 2_000_000_000)
        app = self.run_app()
        self.assertEqual(app.selectbox[0].options, ['较新问题', '旧问题'])
        app.selectbox[0].select(older_id)
        self.run_app()
        self.assertEqual(app.session_state['messages'][1]['content'], '旧回答')
        self.assertEqual(len(self.sources()), 1)
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
        self.assertEqual(len(self.sources()), 2)
        self.assertIn('文本块数：2', '\n'.join(item.value for item in app.markdown))
        self.run_app()
        self.assertEqual(len(self.prompts), 1)
        self.assertEqual(history.read_text(encoding='utf-8').count('apple followup'), 1)
        app.button(key='new_conversation').click()
        self.run_app()
        self.assertEqual(app.session_state['messages'], [])
        self.assertNotEqual(app.session_state['conversation_id'], older_id)
        app.selectbox[0].select(older_id)
        self.run_app()
        self.assertEqual(len(app.session_state['messages']), 4)
        self.assertEqual(len(self.sources()), 2)
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

    def test_failed_answer_retry_and_generated_title_are_not_duplicated(self):
        from conversations import load_metadata, read_messages
        app = self.run_app()
        identity = app.session_state['conversation_id']
        history = self.histories / f'{identity}.jsonl'
        with patch.object(sys.modules['LLM'], 'ask_llm', side_effect=TimeoutError('timeout')):
            app.chat_input[0].set_value('apple question')
            self.run_app()
        self.assertTrue(any('超时' in item.value for item in app.error))
        self.assertEqual(len(read_messages(history)), 1)
        self.assertTrue(app.chat_input[0].disabled)
        app.button(key='retry_request').click()
        self.run_app()
        self.assertEqual(len(read_messages(history)), 2)
        self.assertEqual(len(self.prompts), 1)
        self.assertEqual(self.title_prompts, ['apple question'])
        self.assertEqual(app.selectbox[0].options, ['概念与实验讨论'])
        state = load_metadata(history)
        self.assertEqual(state['request']['attempts'], 2)
        self.assertGreater(state['request']['elapsed_seconds'], 0)
        self.assertEqual(set(state['request']['stages']),
                         {'model', 'knowledge_base', 'retrieval', 'generation', 'title'})
        self.run_app()
        self.assertEqual(len(self.prompts), 1)

    def test_title_failure_only_retries_title(self):
        from conversations import load_metadata, read_messages
        app = self.run_app()
        history = self.histories / f"{app.session_state['conversation_id']}.jsonl"
        with patch.object(sys.modules['LLM'], 'generate_title', side_effect=TimeoutError('timeout')):
            app.chat_input[0].set_value('apple first question')
            self.run_app()
        self.assertEqual(len(read_messages(history)), 2)
        self.assertEqual(load_metadata(history)['title_status'], 'failed')
        self.assertEqual(len(self.prompts), 1)
        app.button(key='retry_title').click()
        self.run_app()
        self.assertEqual(self.title_prompts, ['apple first question'])
        self.assertEqual(len(self.prompts), 1)

    def test_document_and_full_range_are_bound_before_answer(self):
        from conversations import load_metadata
        other = self.directory / 'other.pdf'
        with pymupdf.open() as document:
            for text in ('apple', 'banana', 'pear'):
                document.new_page().insert_text((72, 72), text)
            document.save(str(other))
        app = self.run_app()
        identity = app.session_state['conversation_id']
        app.text_input[0].set_value(str(other))
        app.radio[0].set_value('全文')
        app.button(key='apply_document').click()
        self.run_app()
        app.chat_input[0].set_value('banana')
        self.run_app()
        state = load_metadata(self.histories / f'{identity}.jsonl')
        self.assertEqual(state['document']['pdf_path'], str(other))
        self.assertIsNone(state['document']['max_pages'])
        self.assertNotEqual(state['document']['index_path'], str(self.index))
        self.assertEqual(app.session_state['knowledge_base_details']['actual_pages'], 3)
        self.assertTrue(app.button(key='apply_document').disabled)

    def test_legacy_history_needs_explicit_binding_and_can_generate_title(self):
        from conversations import load_metadata
        identity = 'c' * 32
        history = self.history(identity, '完整旧问题', '旧回答', 1_000_000_000, bind=False)
        before = history.read_bytes()
        app = self.run_app()
        app.selectbox[0].select(identity)
        self.run_app()
        self.assertTrue(app.chat_input[0].disabled)
        self.assertEqual(self.prompts, [])
        self.assertEqual(self.title_prompts, [])
        app.button(key='apply_document').click()
        self.run_app()
        self.assertFalse(app.chat_input[0].disabled)
        self.assertEqual(history.read_bytes(), before)
        self.assertEqual(load_metadata(history)['document']['pdf_path'], str(self.pdf))
        app.button(key='retry_title').click()
        self.run_app()
        self.assertEqual(self.title_prompts, ['完整旧问题'])
        self.assertEqual(self.prompts, [])
