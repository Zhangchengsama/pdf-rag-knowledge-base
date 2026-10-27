"""验证页面的文档设置，为每个 PDF 分配独立索引；上传文件持久保留。"""
import hashlib
from pathlib import Path

from get_pdf import get_pdf_page_count


def prepare_document(pdf_path, max_pages, default_pdf, default_index, cache_directory):
    pdf = Path(pdf_path).expanduser().resolve()
    if pdf.suffix.lower() != '.pdf':
        raise ValueError('请选择 PDF 文件')
    if not pdf.is_file():
        raise FileNotFoundError(f'找不到 PDF：{pdf}')
    get_pdf_page_count(pdf, max_pages=max_pages)
    if pdf == Path(default_pdf).resolve():
        index = Path(default_index).resolve()
    else:
        identity = hashlib.sha256(str(pdf).casefold().encode('utf-8')).hexdigest()[:24]
        index = Path(cache_directory).resolve() / f'{identity}.bin'
    return {'pdf_path': str(pdf), 'index_path': str(index), 'max_pages': max_pages}


def save_uploaded_pdf(uploaded, directory):
    name = Path(uploaded.name).name
    if Path(name).suffix.lower() != '.pdf':
        raise ValueError('请选择 PDF 文件')
    content = uploaded.getvalue()
    if not content:
        raise ValueError('上传的 PDF 为空')
    identity = hashlib.sha256(content).hexdigest()
    path = Path(directory) / identity / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.uploading')
        temporary.write_bytes(content)
        temporary.replace(path)
    return path
