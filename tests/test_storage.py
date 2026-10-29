"""使用真实 PDF/索引验证集中存储及旧路径调用兼容。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from test_regressions import TinyModel


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='rag_storage_'))
        self.pdf = self.root / 'originals' / 'sample.pdf'
        self.pdf.parent.mkdir()
        with pymupdf.open() as document:
            for text in ('apple', 'banana'):
                document.new_page().insert_text((72, 72), text)
            document.save(self.pdf)
        self.index = self.root / 'data/cache/example/index.bin'
        self.output = self.index.with_suffix('.json')

    def test_central_cache_reuses_and_returns_same_answer_sources_as_legacy(self):
        import cache_store
        from knowledge_base import build_knowledge_base, get_knowledge_base_status
        from rag_service import answer_question
        original = self.pdf.read_bytes()
        state = get_knowledge_base_status(self.pdf, self.index, output_path=self.output)
        self.assertEqual(state['status'], 'not_built')
        self.assertFalse(self.index.parent.exists())
        build_knowledge_base(self.pdf, self.index, TinyModel(), output_path=self.output)
        with patch('knowledge_base.get_pdf_text', side_effect=AssertionError('不应重新提取')):
            result = build_knowledge_base(self.pdf, self.index, TinyModel(), output_path=self.output)
        self.assertFalse(result['rebuilt'])
        self.assertEqual(get_knowledge_base_status(
            self.pdf, self.index, output_path=self.output)['status'], 'ready')
        self.assertEqual(len(cache_store.load_chunks(self.output)), 2)
        self.assertEqual(self.pdf.read_bytes(), original)
        self.assertEqual(list(self.pdf.parent.iterdir()), [self.pdf])
        with patch('LLM.ask_llm', return_value='same answer'):
            central = answer_question(self.pdf, self.index, 'apple', TinyModel(), output_path=self.output)
            legacy = answer_question(self.pdf, self.root / 'legacy.bin', 'apple', TinyModel())
        self.assertEqual(central, legacy)

    def test_settings_and_new_selection_use_project_data_directories(self):
        from config import load_settings
        from document_settings import prepare_document, save_active_document, load_active_document
        settings = load_settings(base_dir=self.root, pdf_path=self.pdf)
        self.assertEqual(settings.document_dir, self.root / 'data/uploads')
        self.assertEqual(settings.index_dir, self.root / 'data/cache')
        self.assertEqual(settings.knowledge_settings_path, self.root / 'data/settings/knowledge_base.json')
        selected = prepare_document(self.pdf, None, settings.pdf_path,
                                    settings.index_path, settings.index_dir)
        self.assertTrue(Path(selected['output_path']).is_relative_to(settings.index_dir))
        save_active_document(settings, selected)
        self.assertEqual(load_active_document(settings), selected)
        self.assertFalse(Path(selected['output_path']).exists())

    def test_old_request_mapping_preserves_original_page_limit(self):
        from document_settings import resolve_document
        mapping = self.root / 'paths.json'
        old = {'pdf_path': str(self.root / 'v6/data/documents/old.pdf'),
               'index_path': str(self.root / 'v6/data/indexes/old.bin'), 'max_pages': 1}
        target = {'pdf_path': str(self.pdf), 'index_path': str(self.index), 'output_path': str(self.output)}
        mapping.write_text(json.dumps({old['pdf_path']: target}), encoding='utf-8')
        self.assertEqual(resolve_document(old, mapping), dict(target, max_pages=1))
