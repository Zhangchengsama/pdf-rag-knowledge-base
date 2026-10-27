"""验证页面的文档设置，为每个 PDF 分配独立索引；上传文件持久保留。"""
import hashlib
from pathlib import Path

from get_pdf import get_pdf_page_count


def prepare_document(pdf_path, max_pages, default_pdf, default_index, cache_directory):
    # 选择文档
    # 1. 检查 PDF 路径和读取页数。
    # 2. 默认教材沿用原索引，其他 PDF 按路径分配独立索引。
    # 3. 返回绑定设置，不在这里创建知识库。
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
    # 保存上传
    # 1. 检查文件名和内容，按内容摘要确定保存目录。
    # 2. 已有文件直接复用。
    # 3. 新文件先完整写入临时位置，再发布为正式 PDF。
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
