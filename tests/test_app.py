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

ROOT = Path(__file__).resolve().parents[1] / 'src'
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

    def show_manager(self):
        self.app.switch_page(str(ROOT / 'knowledge_ui.py'))
        return self.run_app()

    def show_chat(self):
        self.app.switch_page(str(ROOT / 'chat_ui.py'))
        return self.run_app()

    def test_chat_has_no_document_controls_and_management_uses_its_own_page(self):
        app = self.run_app()
        self.assertEqual([item.proto.label for item in app.main.get('page_link')], ['问答', '知识库管理'])
        self.assertEqual(len(app.sidebar.get('page_link')), 0)
        self.assertEqual(len(app.text_input), 0)
        self.assertEqual(len(app.radio), 0)
        self.assertEqual(len(app.chat_input), 1)
        app.switch_page(str(ROOT / 'knowledge_ui.py'))
        self.run_app()
        self.assertEqual(len(app.chat_input), 0)
        self.assertEqual(len(app.main.text_input), 1)
        self.assertEqual(len(app.sidebar.text_input), 0)
        self.assertEqual(len(app.sidebar.radio), 0)
        self.assertIn('知识库管理', [item.value for item in app.title])
        self.assertEqual(list(self.histories.glob('*.meta.json')), [])
        app.switch_page(str(ROOT / 'chat_ui.py'))
        self.run_app()
        self.assertEqual(len(app.chat_input), 1)

    def test_open_only_reports_status_without_building(self):
        self.run_app()
        app = self.show_manager()
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
        self.show_manager()
        self.assertIn('文本块数：2', '\n'.join(item.value for item in app.markdown))
        self.show_chat()
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
        self.show_manager()
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

    def test_shared_document_and_range_can_change_without_starting_a_new_conversation(self):
        from conversations import load_metadata
        from document_settings import load_active_document
        other = self.directory / 'other.pdf'
        with pymupdf.open() as document:
            for text in ('apple', 'banana', 'pear'):
                document.new_page().insert_text((72, 72), text)
            document.save(str(other))
        app = self.run_app()
        identity = app.session_state['conversation_id']
        self.show_manager()
        app.text_input[0].set_value(str(other))
        app.radio[0].set_value('全文')
        app.button(key='apply_document').click()
        self.run_app()
        self.assertFalse(self.index.exists())
        self.assertEqual(list(self.histories.glob('*.meta.json')), [])
        self.show_chat()
        app.chat_input[0].set_value('banana')
        self.run_app()
        state = load_metadata(self.histories / f'{identity}.jsonl')
        self.assertNotIn('document', state)
        self.assertEqual(state['request']['document']['pdf_path'], str(other))
        self.assertIsNone(state['request']['document']['max_pages'])
        self.assertNotEqual(state['request']['document']['index_path'], str(self.index))
        self.assertEqual(app.session_state['knowledge_base_details']['actual_pages'], 3)
        self.show_manager()
        self.assertFalse(app.button(key='apply_document').disabled)
        app.radio[0].set_value('前 N 页')
        app.number_input[0].set_value(1)
        app.button(key='apply_document').click()
        self.run_app()
        self.show_chat()
        self.assertEqual(app.session_state['conversation_id'], identity)
        app.chat_input[0].set_value('apple again')
        self.run_app()
        state = load_metadata(self.histories / f'{identity}.jsonl')
        self.assertEqual(state['request']['document']['max_pages'], 1)
        self.assertEqual(app.session_state['messages'][-1]['sources'][0]['file_name'], 'other.pdf')
        self.assertEqual({source['page_number'] for source in app.session_state['messages'][-1]['sources']}, {1})
        self.assertEqual(load_active_document(self.settings)['max_pages'], 1)
        self.app = AppTest.from_function(render_test_app, args=(self.settings,), default_timeout=10)
        self.run_app()
        app = self.show_manager()
        self.assertEqual(app.text_input[0].value, str(other))
        self.assertEqual(app.number_input[0].value, 1)

    def test_legacy_history_can_continue_without_document_binding_and_generate_title(self):
        from conversations import load_metadata
        identity = 'c' * 32
        history = self.history(identity, '完整旧问题', '旧回答', 1_000_000_000, bind=False)
        before = history.read_bytes()
        app = self.run_app()
        app.selectbox[0].select(identity)
        self.run_app()
        self.assertFalse(app.chat_input[0].disabled)
        self.assertEqual(self.prompts, [])
        self.assertEqual(self.title_prompts, [])
        self.assertEqual(history.read_bytes(), before)
        app.button(key='retry_title').click()
        self.run_app()
        self.assertEqual(self.title_prompts, ['完整旧问题'])
        self.assertEqual(self.prompts, [])
        self.assertNotIn('document', load_metadata(history))
        app.chat_input[0].set_value('apple followup')
        self.run_app()
        self.assertEqual(len(self.prompts), 1)

    def test_retry_keeps_original_range_after_shared_settings_change(self):
        from conversations import load_metadata
        app = self.run_app()
        identity = app.session_state['conversation_id']
        history = self.histories / f'{identity}.jsonl'
        with patch.object(sys.modules['LLM'], 'ask_llm', side_effect=TimeoutError('timeout')):
            app.chat_input[0].set_value('apple question')
            self.run_app()
        self.show_manager()
        app.number_input[0].set_value(1)
        app.button(key='apply_document').click()
        self.run_app()
        self.show_chat()
        app.button(key='retry_request').click()
        self.run_app()
        self.assertEqual(load_metadata(history)['request']['document']['max_pages'], 30)
        self.assertEqual({source['page_number'] for source in app.session_state['messages'][-1]['sources']}, {1, 2})
        self.assertEqual(len(app.session_state['messages']), 2)
        app.chat_input[0].set_value('apple next question')
        self.run_app()
        self.assertEqual(load_metadata(history)['request']['document']['max_pages'], 1)
        self.assertEqual({source['page_number'] for source in app.session_state['messages'][-1]['sources']}, {1})

    def test_old_binding_does_not_override_shared_settings_for_new_questions(self):
        from document_settings import prepare_document, save_active_document
        from conversations import load_metadata
        identity = 'd' * 32
        history = self.history(identity, '旧标题', '旧回答', 1_000_000_000)
        other = self.directory / 'current.pdf'
        other.write_bytes(self.pdf.read_bytes())
        selected = prepare_document(other, 1, self.pdf, self.index, self.settings.index_dir)
        save_active_document(self.settings, selected)
        app = self.run_app()
        app.selectbox[0].select(identity)
        self.run_app()
        app.chat_input[0].set_value('apple')
        self.run_app()
        self.assertEqual(load_metadata(history)['document']['pdf_path'], str(self.pdf))
        self.assertEqual(app.session_state['messages'][1]['content'], '旧回答')
        self.assertEqual({source['file_name'] for source in app.session_state['messages'][-1]['sources']}, {'current.pdf'})

    def test_invalid_document_does_not_replace_saved_knowledge_settings(self):
        from document_settings import prepare_document, save_active_document, load_active_document
        selected = prepare_document(self.pdf, 1, self.pdf, self.index, self.settings.index_dir)
        save_active_document(self.settings, selected)
        self.run_app()
        app = self.show_manager()
        app.text_input[0].set_value(str(self.directory / 'missing.pdf'))
        app.button(key='apply_document').click()
        self.run_app()
        self.assertTrue(app.error)
        self.assertEqual(load_active_document(self.settings), selected)
        self.assertFalse(self.index.exists())
        self.assertEqual(self.prompts, [])

    def test_corrupt_shared_settings_can_be_repaired_on_management_page(self):
        from document_settings import load_active_document
        self.settings.knowledge_settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.knowledge_settings_path.write_text('{broken', encoding='utf-8')
        app = self.run_app()
        self.assertTrue(app.error)
        self.assertTrue(app.chat_input[0].disabled)
        self.show_manager()
        self.assertEqual(len(app.text_input), 1)
        app.button(key='apply_document').click()
        self.run_app()
        self.assertEqual(load_active_document(self.settings)['pdf_path'], str(self.pdf))
        self.show_chat()
        self.assertFalse(app.chat_input[0].disabled)
        self.assertFalse(self.index.exists())
