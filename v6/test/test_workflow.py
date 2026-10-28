"""失败重试、独立标题、请求资料及旧调用使用真实文件，模型和 API 隔离。"""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import pymupdf
from test_regressions import TinyModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix='rag_workflow_'))
        self.pdf = self.directory / 'sample.pdf'
        self.index = self.directory / 'sample.bin'
        self.history = self.directory / 'conversations' / 'test.jsonl'
        with pymupdf.open() as document:
            for text in ('apple', 'banana'):
                document.new_page().insert_text((72, 72), text)
            document.save(str(self.pdf))
        self.answers, self.titles = [], []
        llm = types.ModuleType('LLM')
        llm.ask_llm = lambda prompt: self.answers.append(prompt) or 'offline answer'
        llm.generate_title = lambda question: self.titles.append(question) or '概念与实验讨论'
        embeddings = types.ModuleType('sentence_transformers')
        embeddings.SentenceTransformer = TinyModel
        previous = {key: sys.modules.get(key) for key in ('LLM', 'sentence_transformers')}
        sys.modules.update(LLM=llm, sentence_transformers=embeddings)
        def restore():
            for key, module in previous.items():
                if module is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = module
        self.addCleanup(restore)

    def bind(self, limit=30):
        from conversations import bind_document
        bind_document(self.history, {'pdf_path': str(self.pdf), 'index_path': str(self.index),
                                     'max_pages': limit})

    def test_failed_answer_retry_does_not_duplicate_user_or_title(self):
        from chat_workflow import start_request, run_request
        from conversations import read_messages, load_metadata
        self.bind()
        start_request(self.history, 'apple?')
        with patch.object(sys.modules['LLM'], 'ask_llm', side_effect=TimeoutError('timeout')):
            state = run_request(self.history, TinyModel)
        self.assertEqual(state['request']['status'], 'failed')
        self.assertIn('超时', state['request']['error'])
        self.assertEqual(len(read_messages(self.history)), 1)
        self.assertEqual(self.titles, [])
        state = run_request(self.history, TinyModel)
        self.assertEqual(state['request']['status'], 'complete')
        self.assertEqual(state['request']['attempts'], 2)
        self.assertEqual([row['role'] for row in read_messages(self.history)], ['user', 'assistant'])
        self.assertEqual(self.titles, ['apple?'])
        self.assertEqual(load_metadata(self.history)['title'], '概念与实验讨论')
        run_request(self.history, TinyModel)
        self.assertEqual(len(self.answers), 1)
        self.assertEqual(len(self.titles), 1)

    def test_title_failure_can_retry_without_regenerating_answer(self):
        from chat_workflow import start_request, run_request, generate_session_title
        from conversations import read_messages
        self.bind()
        start_request(self.history, '完整的首个问题？')
        with patch.object(sys.modules['LLM'], 'generate_title', side_effect=TimeoutError('timeout')):
            state = run_request(self.history, TinyModel)
        self.assertEqual(state['request']['status'], 'complete')
        self.assertEqual(state['title_status'], 'failed')
        self.assertEqual(len(read_messages(self.history)), 2)
        state = generate_session_title(self.history)
        self.assertEqual(state['title'], '概念与实验讨论')
        self.assertEqual(self.titles, ['完整的首个问题？'])
        self.assertEqual(len(self.answers), 1)
        start_request(self.history, '第二个问题')
        run_request(self.history, TinyModel)
        self.assertEqual(len(self.titles), 1)

    def test_model_failure_reports_stage_and_can_retry(self):
        from chat_workflow import start_request, run_request
        self.bind()
        start_request(self.history, 'apple')
        def missing_model():
            raise OSError('local model missing')
        state = run_request(self.history, missing_model)
        self.assertEqual(state['request']['stage'], 'model')
        self.assertIn('模型', state['request']['error'])
        self.assertEqual(self.answers, [])
        self.assertEqual(run_request(self.history, TinyModel)['request']['status'], 'complete')

    def test_recover_old_unanswered_question_and_completed_disk_write(self):
        from chat_workflow import recover_request, run_request
        from conversations import append_message, load_metadata, read_messages
        append_message(self.history, {'role': 'user', 'content': 'apple'})
        self.bind()
        state = recover_request(self.history)
        self.assertEqual(state['request']['user_index'], 0)
        self.assertEqual(state['request']['status'], 'failed')
        # 模拟答案已落盘，但完成标记还未来得及保存的中断。
        append_message(self.history, {'role': 'assistant', 'content': 'already saved', 'sources': []})
        run_request(self.history, TinyModel)
        self.assertEqual(load_metadata(self.history)['request']['status'], 'complete')
        self.assertEqual(len(read_messages(self.history)), 2)
        self.assertEqual(self.answers, [])

    def test_document_binding_prevents_mixed_document_or_range(self):
        from chat_workflow import start_request
        from conversations import bind_document, load_metadata
        self.bind(1)
        start_request(self.history, 'apple')
        with self.assertRaisesRegex(ValueError, '新建对话'):
            bind_document(self.history, {'pdf_path': str(self.pdf), 'index_path': str(self.index),
                                         'max_pages': None})
        self.assertEqual(load_metadata(self.history)['document']['max_pages'], 1)

    def test_new_request_snapshot_overrides_legacy_binding_and_is_independent(self):
        from chat_workflow import start_request, run_request
        from conversations import load_metadata, read_messages
        self.bind(1)
        document = {'pdf_path': str(self.pdf), 'index_path': str(self.index), 'max_pages': None}
        start_request(self.history, 'banana', document=document)
        document['max_pages'] = 1
        state = run_request(self.history, TinyModel)
        self.assertEqual(state['request']['status'], 'complete')
        self.assertIsNone(state['request']['document']['max_pages'])
        self.assertEqual(load_metadata(self.history)['document']['max_pages'], 1)
        self.assertEqual({item['page_number'] for item in read_messages(self.history)[-1]['sources']}, {1, 2})

    def test_old_unbound_unanswered_request_can_retry_using_current_knowledge_base(self):
        from chat_workflow import recover_request, run_request
        from conversations import append_message, read_messages
        append_message(self.history, {'role': 'user', 'content': 'apple'})
        recover_request(self.history)
        document = {'pdf_path': str(self.pdf), 'index_path': str(self.index), 'max_pages': 1}
        state = run_request(self.history, TinyModel, document=document)
        self.assertEqual(state['request']['status'], 'complete')
        self.assertEqual(state['request']['document'], document)
        self.assertNotIn('document', state)
        self.assertEqual(len(read_messages(self.history)), 2)

    def test_shared_settings_round_trip_and_explicit_full_range(self):
        from config import load_settings
        from document_settings import load_active_document, save_active_document
        settings = load_settings(base_dir=self.directory, pdf_path=self.pdf, index_path=self.index)
        self.assertEqual(load_active_document(settings)['max_pages'], 30)
        self.assertFalse(settings.knowledge_settings_path.exists())
        document = {'pdf_path': str(self.pdf), 'index_path': str(self.index), 'max_pages': None}
        save_active_document(settings, document)
        reopened = load_settings(base_dir=self.directory, pdf_path=self.pdf, index_path=self.index)
        self.assertEqual(load_active_document(reopened), document)
        self.assertFalse(self.index.exists())
        self.assertFalse(self.history.exists())

    def test_invalid_or_corrupt_shared_settings_are_reported_without_overwriting(self):
        from config import load_settings
        from document_settings import load_active_document, save_active_document
        settings = load_settings(base_dir=self.directory, pdf_path=self.pdf, index_path=self.index)
        document = {'pdf_path': str(self.pdf), 'index_path': str(self.index), 'max_pages': 1}
        save_active_document(settings, document)
        before = settings.knowledge_settings_path.read_bytes()
        for limit in (0, -1, True, '全文'):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                save_active_document(settings, dict(document, max_pages=limit))
            self.assertEqual(settings.knowledge_settings_path.read_bytes(), before)
        settings.knowledge_settings_path.write_text('{broken', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '知识库设置'):
            load_active_document(settings)
        self.assertEqual(settings.knowledge_settings_path.read_text(encoding='utf-8'), '{broken')

    def test_document_selection_uses_independent_indexes_and_full_pdf(self):
        from document_settings import prepare_document
        other_dir = self.directory / 'other'
        other_dir.mkdir()
        other = other_dir / 'sample.pdf'
        other.write_bytes(self.pdf.read_bytes())
        first = prepare_document(self.pdf, None, self.pdf, self.index, self.directory / 'indexes')
        second = prepare_document(other, 1, self.pdf, self.index, self.directory / 'indexes')
        self.assertEqual(first['index_path'], str(self.index))
        self.assertIsNone(first['max_pages'])
        self.assertNotEqual(first['index_path'], second['index_path'])
        self.assertFalse((self.directory / 'indexes').exists())
        with self.assertRaises(ValueError):
            prepare_document(self.pdf, 0, self.pdf, self.index, self.directory / 'indexes')

    def test_uploaded_pdf_is_kept_and_reused_without_name_collisions(self):
        from document_settings import save_uploaded_pdf
        upload = types.SimpleNamespace(name='sample.pdf', getvalue=self.pdf.read_bytes)
        first = save_uploaded_pdf(upload, self.directory / 'uploads')
        before = first.stat().st_mtime_ns
        self.assertEqual(save_uploaded_pdf(upload, self.directory / 'uploads'), first)
        self.assertEqual(first.stat().st_mtime_ns, before)
        self.assertEqual(first.read_bytes(), self.pdf.read_bytes())

    def test_title_failure_waits_for_explicit_retry_on_later_questions(self):
        from chat_workflow import start_request, run_request
        self.bind()
        attempts = []
        def failed_title(question):
            attempts.append(question)
            raise TimeoutError('timeout')
        with patch.object(sys.modules['LLM'], 'generate_title', side_effect=failed_title):
            start_request(self.history, '首个问题')
            run_request(self.history, TinyModel)
            start_request(self.history, '另一个问题')
            state = run_request(self.history, TinyModel)
        self.assertEqual(state['request']['status'], 'complete')
        self.assertEqual(attempts, ['首个问题'])

    def test_failed_completion_marker_does_not_regenerate_saved_answer(self):
        from chat_workflow import start_request, run_request
        from conversations import read_messages, save_metadata
        self.bind()
        start_request(self.history, 'apple')
        failed = []
        def fail_once(path, metadata):
            if metadata.get('request', {}).get('status') == 'complete' and not failed:
                failed.append(True)
                raise PermissionError('cannot save completion marker')
            save_metadata(path, metadata)
        with patch('chat_workflow.save_metadata', side_effect=fail_once):
            state = run_request(self.history, TinyModel)
        self.assertEqual(state['request']['status'], 'failed')
        self.assertEqual(len(read_messages(self.history)), 2)
        self.assertEqual(run_request(self.history, TinyModel)['request']['status'], 'complete')
        self.assertEqual(len(read_messages(self.history)), 2)
        self.assertEqual(len(self.answers), 1)

    def test_failed_user_append_is_recovered_without_losing_question(self):
        from chat_workflow import recover_request, start_request, run_request
        from conversations import read_messages
        self.bind()
        with patch('chat_workflow.append_message', side_effect=PermissionError('cannot append')):
            with self.assertRaises(PermissionError):
                start_request(self.history, 'apple')
        self.assertEqual(read_messages(self.history), [])
        self.assertEqual(recover_request(self.history)['request']['question'], 'apple')
        self.assertEqual(run_request(self.history, TinyModel)['request']['status'], 'complete')
        self.assertEqual([row['role'] for row in read_messages(self.history)], ['user', 'assistant'])

    def test_interrupted_upload_can_be_uploaded_again_without_reusing_partial_pdf(self):
        from document_settings import save_uploaded_pdf
        content = self.pdf.read_bytes()
        upload = types.SimpleNamespace(name='sample.pdf', getvalue=lambda: content)
        original = Path.write_bytes
        def interrupted(path, data):
            original(path, data[:20])
            raise OSError('interrupted upload write')
        with patch.object(Path, 'write_bytes', interrupted):
            with self.assertRaises(OSError):
                save_uploaded_pdf(upload, self.directory / 'uploads')
        restored = save_uploaded_pdf(upload, self.directory / 'uploads')
        self.assertEqual(restored.read_bytes(), content)

    def test_live_request_in_another_tab_cannot_be_recovered_or_retried_twice(self):
        from chat_workflow import recover_request, start_request, run_request
        from conversations import read_messages
        self.bind()
        start_request(self.history, 'apple')
        entered, release = threading.Event(), threading.Event()
        def slow_answer(prompt):
            self.answers.append(prompt)
            entered.set()
            release.wait(5)
            return 'offline answer'
        results = []
        with patch.object(sys.modules['LLM'], 'ask_llm', side_effect=slow_answer):
            worker = threading.Thread(target=lambda: results.append(run_request(self.history, TinyModel)))
            worker.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(recover_request(self.history)['request']['status'], 'running')
                from conversation_lock import ConversationBusyError
                with self.assertRaises(ConversationBusyError):
                    run_request(self.history, TinyModel)
            finally:
                release.set()
                worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results[0]['request']['status'], 'complete')
        self.assertEqual(len(self.answers), 1)
        self.assertEqual([row['role'] for row in read_messages(self.history)], ['user', 'assistant'])

    def test_interrupted_stage_has_no_invented_elapsed_time(self):
        from chat_workflow import recover_request, start_request
        from conversations import load_metadata, save_metadata
        self.bind()
        start_request(self.history, 'apple')
        state = load_metadata(self.history)
        state['request'].update(status='running', attempts=1, stage='generation', stages={
            'generation': {'stage': 'generation', 'status': 'running', 'elapsed_seconds': 0.001}})
        save_metadata(self.history, state)
        state = recover_request(self.history)
        self.assertEqual(state['request']['status'], 'failed')
        stage = state['request']['stages']['generation']
        self.assertEqual(stage['status'], 'failed')
        self.assertTrue(stage['interrupted'])
        self.assertIsNone(stage['elapsed_seconds'])

    def test_same_pdf_in_two_conversations_cannot_rebuild_or_query_concurrently(self):
        from chat_workflow import start_request, run_request
        from conversations import bind_document, read_messages
        self.bind()
        other = self.history.with_name('other.jsonl')
        bind_document(other, {'pdf_path': str(self.pdf), 'index_path': str(self.index),
                              'max_pages': 1})
        start_request(self.history, 'apple')
        start_request(other, 'banana')
        entered, release = threading.Event(), threading.Event()
        def slow_answer(prompt):
            self.answers.append(prompt)
            entered.set()
            release.wait(5)
            return 'offline answer'
        with patch.object(sys.modules['LLM'], 'ask_llm', side_effect=slow_answer):
            worker = threading.Thread(target=lambda: run_request(self.history, TinyModel))
            worker.start()
            try:
                self.assertTrue(entered.wait(5))
                state = run_request(other, TinyModel)
                self.assertEqual(state['request']['status'], 'failed')
                self.assertIn('文档', state['request']['error'])
                self.assertEqual(len(read_messages(other)), 1)
            finally:
                release.set()
                worker.join(5)
        self.assertEqual(run_request(other, TinyModel)['request']['status'], 'complete')
        self.assertEqual(len(self.answers), 2)
        self.assertEqual(len(read_messages(other)), 2)
