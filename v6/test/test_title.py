"""只替换 HTTP 客户端，检查标题确实由独立大模型提示词生成。"""
import importlib.util
import os
from pathlib import Path
import types
import unittest
from unittest.mock import MagicMock, patch


class TitleTests(unittest.TestCase):
    def load_llm(self, content):
        client = MagicMock()
        client.chat.completions.create.return_value = types.SimpleNamespace(choices=[
            types.SimpleNamespace(message=types.SimpleNamespace(content=content))])
        source = Path(__file__).resolve().parents[1] / 'LLM.py'
        spec = importlib.util.spec_from_file_location('title_llm_under_test', source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-test'}), \
                patch('openai.OpenAI', return_value=client):
            spec.loader.exec_module(module)
        return module, client

    def test_title_uses_first_question_as_prompt_and_normalizes_model_output(self):
        module, client = self.load_llm('标题：“氧化还原的判断方法”')
        question = '我不理解为什么钠与水反应属于氧化还原反应，能解释一下吗？'
        self.assertEqual(module.generate_title(question), '氧化还原的判断方法')
        payload = client.chat.completions.create.call_args.kwargs
        self.assertIn(question, payload['messages'][1]['content'])
        self.assertIn('标题', payload['messages'][0]['content'])
        self.assertFalse(payload['stream'])
        self.assertLessEqual(payload['max_tokens'], 100)

    def test_empty_title_is_a_failure_instead_of_copying_question(self):
        module, _ = self.load_llm(None)
        with self.assertRaises(ValueError):
            module.generate_title('完整原问题')
