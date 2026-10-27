"""逐页读取 PDF，保留来源字段和旧调用入口。"""
from pathlib import Path
import pymupdf
import chunker
from cache_store import save_to_json
from chunker import split_text
from config import DEFAULT_MAX_PAGES


def resolve_page_count(total_pages, max_pages=DEFAULT_MAX_PAGES):
    # 计算页数
    # 1. 检查读取上限是否为正整数或全文。
    # 2. 取 PDF 总页数与上限的较小值，空白页也计数。
    if max_pages is not None and (type(max_pages) is not int or max_pages <= 0):
        raise ValueError("max_pages 必须是正整数或 None（读取全文）")
    return total_pages if max_pages is None else min(total_pages, max_pages)


def get_pdf_page_count(road, max_pages=DEFAULT_MAX_PAGES):
    # 查看页数
    # 1. 打开 PDF，只读取总页数。
    # 2. 返回本次配置实际覆盖的页数。
    with pymupdf.open(road) as document:
        return resolve_page_count(document.page_count, max_pages)


def get_pdf_text(road, max_pages=DEFAULT_MAX_PAGES):
    # 提取文档
    # 1. 按页读取 PDF 的文本层。
    # 2. 每页调用独立的 chunker，不跨页拼接。
    # 3. 按原顺序给文本块补上文件名、页码和块编号。
    road = Path(road)
    data = []
    with pymupdf.open(road) as document:
        for page_id in range(resolve_page_count(document.page_count, max_pages)):
            page = document[page_id]
            for text in chunker.split_text(page.get_text()):
                data.append({
                    "file_name": road.name,
                    "page_id": page.number,          # 从 0 开始的页下标。
                    "page_number": page.number + 1,  # 从 1 开始的 PDF 页序。
                    "chunk_id": len(data) + 1,        # 块编号，不是检索下标。
                    "text": text,
                })
    return data

