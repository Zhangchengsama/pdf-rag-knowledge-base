"""验证统一入口、纯算法接口及导入时没有运行副作用。"""
import dataclasses
import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StructureTests(unittest.TestCase):
    def test_chunker_keeps_character_overlap_and_tail(self):
        from chunker import split_text
        from get_pdf import split_text as old_split
        self.assertEqual(split_text('abcdefghij', 4, 1), ['abcd', 'defg', 'ghij', 'j'])
        self.assertEqual(split_text('', 4, 1), [])
        self.assertEqual(old_split('abcdefghij', 4, 1), ['abcd', 'defg', 'ghij', 'j'])

    def test_fusion_keeps_weighted_scores_and_stable_ties(self):
        from fusion import fuse_rankings
        indices, scores = fuse_rankings([2, 1], [1, 2], dense_weight=1, sparse_weight=1)
        self.assertEqual(indices, [2, 1])
        self.assertEqual(scores, [1 / 61 + 1 / 62, 1 / 62 + 1 / 61])
        self.assertEqual(fuse_rankings([], []), ([], []))
        self.assertEqual(fuse_rankings([4], [4]), ([4], [0.8 / 61 + 1.2 / 61]))

    def test_settings_are_read_only_and_default_to_thirty_pages(self):
        from config import load_settings
        settings = load_settings(base_dir=ROOT)
        self.assertEqual(settings.max_pages, 30)
        self.assertEqual(settings.conversation_dir, ROOT / 'data' / 'conversations')
        with self.assertRaises(dataclasses.FrozenInstanceError):
            settings.max_pages = 1
        self.assertEqual(load_settings(max_pages=None).max_pages, None)

    def test_launcher_defaults_to_web_and_separates_cli_and_web_worker(self):
        import main
        with patch.object(main, 'launch_web', return_value=0) as launch:
            self.assertEqual(main.main([]), 0)
            launch.assert_called_once()
        with patch('cli.run_cli', return_value={'answer': 'ok'}) as cli, \
                patch.object(main, 'launch_web', side_effect=AssertionError('不应启动网页')):
            self.assertEqual(main.main(['--cli']), 0)
            self.assertEqual(cli.call_args.args[0].max_pages, 30)
        with patch('ui.run_web') as page, \
                patch.object(main, 'launch_web', side_effect=AssertionError('不应递归启动')):
            self.assertEqual(main.main(['--web']), 0)
            page.assert_called_once()

    def test_web_launcher_uses_current_interpreter_and_explicit_worker_flag(self):
        import main
        with patch.object(main.subprocess, 'run') as run:
            run.return_value.returncode = 0
            self.assertEqual(main.launch_web(), 0)
        self.assertEqual(run.call_args.args[0], [
            sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'main.py'), '--', '--web'])

    def test_cli_uses_injected_dependencies_and_original_answer_format(self):
        from cli import run_cli
        from config import load_settings
        output, received = [], []
        model = object()
        def answer(**kwargs):
            received.append(kwargs)
            return {'answer': '命令行回答', 'sources': []}
        result = run_cli(load_settings(), input_fn=lambda label: '用户问题',
                         output_fn=output.append, model_loader=lambda: model, answer_fn=answer)
        self.assertEqual(output, ['最终回答：', '命令行回答'])
        self.assertEqual(result, {'answer': '命令行回答', 'sources': []})
        self.assertEqual(received[0]['max_pages'], 30)
        self.assertEqual(received[0]['text_input'], '用户问题')
        self.assertIs(received[0]['model'], model)

    def test_all_modules_import_without_input_server_model_or_client(self):
        code = '''
import builtins, importlib, sys, types
from unittest.mock import patch
embedding = types.ModuleType('sentence_transformers')
def forbidden(*args, **kwargs):
    raise AssertionError('导入时执行了运行行为')
embedding.SentenceTransformer = forbidden
sys.modules['sentence_transformers'] = embedding
# 第三方库可能查询操作系统版本；这里检查的是项目导入，不拦截库初始化。
import streamlit, openai, pymupdf, hnswlib, jieba, rank_bm25
modules = ['config', 'main', 'ui', 'cli', 'chunker', 'embedding', 'fusion',
           'cache_store', 'knowledge_base', 'get_pdf', 'dense', 'sparse',
           'query', 'prompt', 'rag_service', 'LLM', 'eval', 'conversations',
           'chat_workflow', 'document_settings', 'conversation_lock', 'progress']
with patch('builtins.input', forbidden), patch('subprocess.run', forbidden), patch('openai.OpenAI', forbidden):
    for name in modules:
        importlib.import_module(name)
print('import safe')
'''
        environment = dict(os.environ)
        environment.pop('DEEPSEEK_API_KEY', None)
        process = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                 env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('import safe', process.stdout)

    def test_help_does_not_start_an_application(self):
        process = subprocess.run([sys.executable, '-B', str(ROOT / 'main.py'), '--help'],
                                 capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('--cli', process.stdout)
