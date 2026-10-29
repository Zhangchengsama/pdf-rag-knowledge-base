"""知识库浏览只读取原件和已完成的缓存，不触发模型或建库。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from test_regressions import TinyModel


class KnowledgeViewTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='knowledge_view_'))
        self.pdf = self.root / 'book.pdf'
        with pymupdf.open() as document:
            for text in ['apple one', 'banana two']:
                document.new_page().insert_text((72, 72), text)
            document.save(self.pdf)
        self.document = {'pdf_path': str(self.pdf), 'index_path': str(self.root / 'cache/index.bin'),
                         'output_path': str(self.root / 'cache/index.json'), 'max_pages': 30}

    def build(self):
        from knowledge_base import build_knowledge_base
        return build_knowledge_base(self.pdf, self.document['index_path'], TinyModel(),
                                    output_path=self.document['output_path'])

    def test_catalog_lists_only_uploaded_pdfs_with_distinct_paths_and_no_writes(self):
        from knowledge_view import list_uploaded_documents
        directory = self.root / 'uploads'
        for folder in ['a', 'b']:
            destination = directory / folder / 'same.PDF'
            destination.parent.mkdir(parents=True)
            destination.write_bytes(self.pdf.read_bytes())
        (directory / 'ignore.json').write_text('{}')
        before = {str(p): p.stat().st_mtime_ns for p in directory.rglob('*') if p.is_file()}
        rows = list_uploaded_documents(directory)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row['path'] for row in rows}), 2)
        self.assertTrue(all(row['name'] == 'same.PDF' and row['size_bytes'] > 0 for row in rows))
        self.assertEqual(before, {str(p): p.stat().st_mtime_ns for p in directory.rglob('*') if p.is_file()})
        self.assertEqual(list_uploaded_documents(self.root / 'missing'), [])

    def test_overview_reports_total_range_and_missing_files_without_building(self):
        from knowledge_view import document_overview
        before = list(self.root.iterdir())
        overview = document_overview(dict(self.document, max_pages=1))
        self.assertEqual(overview['total_pages'], 2)
        self.assertEqual(overview['state']['actual_pages'], 1)
        self.assertEqual(overview['state']['status'], 'not_built')
        self.assertEqual(overview['cache_bytes'], 0)
        self.assertEqual(len(overview['files']), 6)
        self.assertTrue(all(not item['exists'] for item in overview['files']))
        self.assertEqual(list(self.root.iterdir()), before)

    def test_page_preview_is_bounded_png_in_memory_and_rejects_invalid_page(self):
        from knowledge_view import render_pdf_page
        first = render_pdf_page(self.pdf, 1)
        second = render_pdf_page(self.pdf, 2)
        self.assertTrue(first.startswith(b'\x89PNG'))
        self.assertNotEqual(first, second)
        pixmap = pymupdf.Pixmap(first)
        self.assertLessEqual(pixmap.width, 1000)
        self.assertLessEqual(pixmap.height, 1400)
        for page in [0, 3]:
            with self.subTest(page=page), self.assertRaises(ValueError):
                render_pdf_page(self.pdf, page)
        self.assertEqual(list(self.root.iterdir()), [self.pdf])

    def test_indexed_text_uses_current_cache_and_filters_without_changing_content(self):
        from knowledge_view import load_indexed_chunks, filter_chunks
        self.build()
        path = Path(self.document['output_path'])
        before = path.read_bytes()
        rows, signature = load_indexed_chunks(self.document)
        self.assertEqual(len(rows), 2)
        self.assertTrue(signature)
        self.assertEqual([item['page_number'] for item in filter_chunks(rows, 'APPLE')], [1])
        self.assertEqual(filter_chunks(rows, 'apple', 2), [])
        self.assertEqual([item['page_number'] for item in filter_chunks(rows, page=2)], [2])
        self.assertEqual(path.read_bytes(), before)

    def test_unbuilt_or_stale_cache_is_not_read(self):
        from knowledge_view import load_indexed_chunks
        loader = Mock(side_effect=AssertionError('不应读取过期文本'))
        with self.assertRaises(ValueError):
            load_indexed_chunks(self.document, loader)
        self.build()
        with self.assertRaises(ValueError):
            load_indexed_chunks(dict(self.document, max_pages=1), loader)
        loader.assert_not_called()

    def test_text_browser_honors_document_lock(self):
        from knowledge_view import load_indexed_chunks
        from conversation_lock import document_lock, DocumentBusyError
        self.build()
        with document_lock(self.pdf), self.assertRaises(DocumentBusyError):
            load_indexed_chunks(self.document)

    def test_malformed_text_cache_is_reported_without_rebuilding(self):
        from knowledge_view import load_indexed_chunks
        self.build()
        path = Path(self.document['output_path'])
        path.write_text(json.dumps({'wrong': 'format'}))
        with self.assertRaisesRegex(ValueError, '文本缓存'):
            load_indexed_chunks(self.document)
