"""迁移必须保留原件、聊天及可用缓存，并能安全重复执行。"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from test_regressions import TinyModel


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='rag_migration_'))
        self.pdf = self.root / 'v6/data/documents/content-id/book.pdf'
        self.pdf.parent.mkdir(parents=True)
        with pymupdf.open() as document:
            document.new_page().insert_text((72, 72), 'apple')
            document.save(self.pdf)
        digest = hashlib.sha256(str(self.pdf).casefold().encode('utf-8')).hexdigest()[:24]
        self.index = self.root / f'v6/data/indexes/{digest}.bin'
        from knowledge_base import build_knowledge_base
        build_knowledge_base(self.pdf, self.index, TinyModel())
        self.old = {'pdf_path': str(self.pdf), 'index_path': str(self.index), 'max_pages': 30}
        (self.root / 'v6/data/knowledge_base.json').write_text(json.dumps(self.old), encoding='utf-8')
        self.chat = self.root / 'v6/data/conversations/history.jsonl'
        self.chat.parent.mkdir()
        self.chat.write_bytes(b'{"role":"user","content":"apple?"}\n')
        self.meta = self.chat.with_suffix('.meta.json')
        self.meta.write_text(json.dumps({'request': {'document': dict(self.old, max_pages=1)}}), encoding='utf-8')

    def test_migration_keeps_upload_chat_and_cache_and_is_idempotent(self):
        from migrate_layout import migrate
        from config import load_settings
        from document_settings import load_active_document, resolve_document
        from knowledge_base import get_knowledge_base_status
        original = self.pdf.read_bytes()
        messages, metadata = self.chat.read_bytes(), self.meta.read_bytes()
        migrate(self.root)
        settings = load_settings(base_dir=self.root)
        current = load_active_document(settings)
        self.assertEqual(Path(current['pdf_path']).read_bytes(), original)
        self.assertEqual(list((self.root / 'data/uploads').rglob('*.json')), [])
        self.assertEqual((settings.conversation_dir / self.chat.name).read_bytes(), messages)
        self.assertEqual((settings.conversation_dir / self.meta.name).read_bytes(), metadata)
        self.assertEqual(get_knowledge_base_status(
            current['pdf_path'], current['index_path'], output_path=current['output_path'])['status'], 'ready')
        retried = resolve_document(dict(self.old, max_pages=1), settings.path_map_path)
        self.assertEqual(retried['max_pages'], 1)
        self.assertEqual(retried['pdf_path'], current['pdf_path'])
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        migrate(self.root)
        after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(after, before)

    def test_conflict_stops_before_moving_any_original(self):
        from migrate_layout import migrate
        conflict = self.root / 'data/conversations/history.jsonl'
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b'other content')
        with self.assertRaises(FileExistsError):
            migrate(self.root)
        self.assertTrue(self.pdf.exists())
        self.assertTrue(self.chat.exists())
        self.assertEqual(conflict.read_bytes(), b'other content')

    def test_pending_request_uses_migrated_files_and_original_range_with_injected_root(self):
        from migrate_layout import migrate
        from config import load_settings
        from chat_workflow import start_request, run_request
        history = self.root / 'v6/data/conversations/retry.jsonl'
        start_request(history, 'apple?', document=dict(self.old, max_pages=1))
        migrate(self.root)
        settings = load_settings(base_dir=self.root)
        with patch('LLM.ask_llm', return_value='offline'), patch('LLM.generate_title', return_value='test'):
            result = run_request(settings.conversation_dir / history.name, TinyModel,
                                 mapping_path=settings.path_map_path)
        self.assertEqual(result['request']['status'], 'complete', result['request'].get('error'))
        self.assertEqual(result['request']['document']['max_pages'], 1)
        self.assertTrue(Path(result['request']['document']['pdf_path']).is_file())
