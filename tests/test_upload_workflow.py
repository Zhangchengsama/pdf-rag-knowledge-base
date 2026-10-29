"""批量上传逐份校验，复用原保存接口，单个失败不影响其他文件。"""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class UploadWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='batch_upload_'))
        self.directory = self.root / 'data/uploads'

    def upload(self, name, text):
        with pymupdf.open() as document:
            document.new_page().insert_text((72, 72), text)
            content = document.tobytes()
        return SimpleNamespace(name=name, getvalue=lambda: content)

    def test_batch_keeps_valid_files_when_another_file_is_invalid(self):
        from upload_workflow import save_upload_batch
        files = [self.upload('first.pdf', 'apple'),
                 SimpleNamespace(name='broken.pdf', getvalue=lambda: b'not a PDF'),
                 self.upload('last.pdf', 'banana')]
        result = save_upload_batch(files, self.directory)
        self.assertEqual([row['status'] for row in result], ['success', 'failed', 'success'])
        self.assertEqual({path.name for path in self.directory.rglob('*.pdf')}, {'first.pdf', 'last.pdf'})
        for row, original in zip([result[0], result[2]], [files[0], files[2]]):
            self.assertEqual(Path(row['path']).read_bytes(), original.getvalue())

    def test_same_name_different_content_remains_separate_and_reupload_reuses(self):
        from upload_workflow import save_upload_batch
        files = [self.upload('same.pdf', 'apple'), self.upload('same.pdf', 'banana')]
        first = save_upload_batch(files, self.directory)
        paths = [Path(row['path']) for row in first]
        self.assertNotEqual(paths[0], paths[1])
        before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
        again = save_upload_batch(files, self.directory)
        self.assertEqual([row['path'] for row in first], [row['path'] for row in again])
        self.assertEqual(before, [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths])

    def test_all_invalid_or_empty_selection_does_not_create_upload_directory(self):
        from upload_workflow import save_upload_batch
        self.assertEqual(save_upload_batch([], self.directory), [])
        rows = save_upload_batch([SimpleNamespace(name='empty.pdf', getvalue=lambda: b'')], self.directory)
        self.assertEqual(rows[0]['status'], 'failed')
        self.assertFalse(self.directory.exists())
