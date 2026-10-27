import pymupdf
import json
from pathlib import Path

def split_text(text, chunk_size=200, overlap=50):
    """将文本分块
    text: 输入的文本字符串
    chunk_size: 每个文本块的大小
    overlap: 文本块之间的重叠部分大小
    return: 返回一个包含所有文本块的列表
    """
    # 按字符数切分而非 token 数；调用时需保证 chunk_size > overlap >= 0。
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # 保留相邻块的重叠文本，减轻知识点被截断在边界的问题。
        start += chunk_size - overlap
    return chunks

def resolve_page_count(total_pages, max_pages=11):
    """将读取上限转换成实际页数；None 表示全文，空白页也计数。"""
    if max_pages is not None and (type(max_pages) is not int or max_pages <= 0):
        raise ValueError("max_pages 必须是正整数或 None（读取全文）")
    return total_pages if max_pages is None else min(total_pages, max_pages)


def get_pdf_page_count(road, max_pages=11):
    """只读取 PDF 页数信息，用于在提取全文和编码前判断是否需要重建。"""
    with pymupdf.open(road) as doc:
        return resolve_page_count(doc.page_count, max_pages)


def get_pdf_text(road, max_pages=11):
    """读取PDF文件并将其文本内容分块
    road: PDF文件的路径
    max_pages: 最多读取前多少页，默认 11；None 表示读取全文
    return: 返回一个包含所有文本块的列表，每个文本块是一个字典，包含文件名、页码、块编号和文本内容
    """
    road=Path(road)
    data=[]
    # 上下文管理器保证提取异常时也能关闭 PDF 文件。
    with pymupdf.open(road) as doc:
        for page_id in range(resolve_page_count(doc.page_count, max_pages)):
            page = doc[page_id]
            # 提取 PDF 文本层；当前流程不包含扫描图片的 OCR 识别。
            text_chunks = split_text(page.get_text())
            for chunk in text_chunks:
                data.append({
                    "file_name": road.name,
                    # page_id 从 0 开始；page_number 是从 1 开始的 PDF 页序。
                    "page_id": page.number,
                    "page_number": page.number + 1,
                    # chunk_id 不是检索数组下标，而是从 1 开始的文件内块编号。
                    "chunk_id": len(data) + 1,
                    "text": chunk,
                })
    return data

def save_to_json(data, road):
    """保存已读取的PDF文本数据到JSON文件
    data: 包含PDF文本数据的列表
    road: PDF文件的路径
    return: 返回保存的JSON文件路径
    """
    road=Path(road)
    output_path = road.with_suffix(".json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data, 
            f,
            ensure_ascii=False,
            indent=2
            )

    return output_path

