"""用原有 JSONL 格式验证会话标题、排序与只读兼容性。"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix='rag_conversations_'))

    def write_history(self, name, messages, modified_ns):
        path = self.directory / f'{name}.jsonl'
        path.write_text('\n'.join(json.dumps(m, ensure_ascii=False) for m in messages) + '\n',
                        encoding='utf-8')
        os.utime(path, ns=(modified_ns, modified_ns))
        return path

    def test_old_records_get_titles_and_recent_first_without_changes(self):
        from conversations import list_conversations
        old = self.write_history('older', [
            {'role': 'user', 'content': '  氧化还原\n反应是什么  '},
            {'role': 'assistant', 'content': '回答', 'sources': [{'text': '原文'}]},
        ], 1_000_000_000)
        new = self.write_history('newer', [{'role': 'user', 'content': '钠与水反应'}],
                                 2_000_000_000)
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (old, new)}
        items = list_conversations(self.directory)
        self.assertEqual([item['id'] for item in items], ['newer', 'older'])
        self.assertEqual([item['title'] for item in items], ['钠与水反应', '氧化还原 反应是什么'])
        for path, expected in before.items():
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), expected)

    def test_duplicate_titles_keep_distinct_ids_and_stable_ties(self):
        from conversations import list_conversations
        for name in ('bbb', 'aaa'):
            self.write_history(name, [{'role': 'user', 'content': '同一个问题'}], 1_000_000_000)
        items = list_conversations(self.directory)
        self.assertEqual([item['id'] for item in items], ['aaa', 'bbb'])
        self.assertEqual([item['title'] for item in items], ['同一个问题', '同一个问题'])

    def test_long_title_is_shortened_and_missing_directory_is_empty(self):
        from conversations import list_conversations
        self.assertEqual(list_conversations(self.directory / 'missing'), [])
        self.write_history('long', [{'role': 'user', 'content': '化' * 30}], 1_000_000_000)
        self.assertEqual(list_conversations(self.directory)[0]['title'], '化' * 24 + '…')

    def test_empty_or_malformed_title_does_not_break_history_list(self):
        from conversations import list_conversations
        self.write_history('empty', [], 1_000_000_000)
        broken = self.directory / 'broken.jsonl'
        broken.write_text('{invalid\n[]\n', encoding='utf-8')
        items = {item['id']: item['title'] for item in list_conversations(self.directory)}
        self.assertEqual(items['empty'], '未命名对话')
        self.assertEqual(items['broken'], '未命名对话')
