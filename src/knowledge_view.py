"""知识库管理页的只读资料目录、原页预览和文本浏览；不执行建库或检索。"""
from pathlib import Path

import pymupdf
import cache_store
from conversation_lock import document_lock
from get_pdf import get_pdf_page_count
from knowledge_base import get_knowledge_base_status


def list_uploaded_documents(directory):
    # 列出上传
    # 1. 只检查上传目录内 PDF 的路径和文件信息，不解析所有文档。
    # 2. 按最近修改时间排序，同名文件仍使用独立路径作为标识。
    directory = Path(directory).resolve()
    rows = []
    if not directory.exists():
        return rows
    for path in directory.rglob('*'):
        if path.suffix.lower() != '.pdf' or not path.resolve().is_relative_to(directory):
            continue
        if not path.is_file():
            continue
        info = path.stat()
        rows.append({'path': str(path.resolve()), 'name': path.name,
                     'size_bytes': info.st_size, 'modified_ns': info.st_mtime_ns})
    return sorted(rows, key=lambda row: (-row['modified_ns'], row['name'].casefold(), row['path']))


def document_overview(document):
    # 汇总状态
    # 1. 复用原有只读缓存状态，补充源文件总页数与体积。
    # 2. 查看四种缓存和双记录的大小、修改时间及缺失情况。
    pdf, index = Path(document['pdf_path']), Path(document['index_path'])
    output = document.get('output_path')
    state = get_knowledge_base_status(pdf, index, document['max_pages'], output)
    result = {'state': state, 'total_pages': None, 'pdf_bytes': None, 'cache_bytes': 0, 'files': []}
    try:
        result['pdf_bytes'] = pdf.stat().st_size
        result['total_pages'] = get_pdf_page_count(pdf, max_pages=None)
    except (OSError, RuntimeError, ValueError):
        pass  # 状态接口已保留原始错误，页面仍可展示缓存位置。
    paths = (*cache_store.cache_paths(pdf, index, output),
             *cache_store.metadata_paths(pdf, index, output))
    labels = ['文本缓存', '向量缓存', '关键词索引', '向量索引', '文本建库记录', '索引建库记录']
    for number, (label, path) in enumerate(zip(labels, paths)):
        row = {'kind': label, 'path': str(path), 'exists': False,
               'size_bytes': 0, 'modified_ns': None, 'error': None}
        try:
            info = path.stat()
            row.update(exists=path.is_file(), size_bytes=info.st_size, modified_ns=info.st_mtime_ns)
        except FileNotFoundError:
            pass
        except OSError as error:
            row['error'] = str(error)
        result['files'].append(row)
        if number < 4 and row['exists']:
            result['cache_bytes'] += row['size_bytes']
    return result


def render_pdf_page(pdf_path, page_number):
    # 原页预览
    # 1. 验证 PDF 页序，只渲染所选的一页。
    # 2. 限制输出像素尺寸，返回内存 PNG，不保存预览文件。
    with pymupdf.open(pdf_path) as document:
        if type(page_number) is not int or not 1 <= page_number <= document.page_count:
            raise ValueError('PDF 页序超出文档范围。')
        page = document[page_number - 1]
        scale = min(1.5, 1000 / max(page.rect.width, 1), 1400 / max(page.rect.height, 1))
        return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes('png')


def load_indexed_chunks(document, loader=None):
    # 浏览文本
    # 1. 取得与问答相同的文档锁，重新核对当前范围的缓存状态。
    # 2. 仅加载可复用的文本 JSON，不加载向量、模型或 BM25 对象。
    # 3. 检查浏览所需字段，返回独立视图，不改写原文本块。
    with document_lock(document['pdf_path']):
        state = get_knowledge_base_status(document['pdf_path'], document['index_path'],
                                         document['max_pages'], document.get('output_path'))
        if state['status'] != 'ready':
            raise ValueError('当前文档尚无与读取范围匹配的完整缓存，请先提问完成建库。')
        text_path = cache_store.cache_paths(document['pdf_path'], document['index_path'],
                                           document.get('output_path'))[0]
        signature = state['cache_signature']
        rows = loader(text_path, signature) if loader else cache_store.load_chunks(text_path)
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) or not isinstance(row.get('text'), str)
            or type(row.get('page_number')) is not int or row['page_number'] < 1 for row in rows
        ):
            raise ValueError('文本缓存格式无法浏览，请检查文件或重新建库。')
        return [dict(row, chunk_id=row.get('chunk_id', number))
                for number, row in enumerate(rows, start=1)], signature


def filter_chunks(chunks, keyword='', page=None):
    # 筛选文本
    # 1. 按 PDF 页序和原文关键词筛选，保留原块顺序。
    keyword = keyword.strip().casefold()
    return [row for row in chunks if (page is None or row['page_number'] == page)
            and (not keyword or keyword in row['text'].casefold())]
