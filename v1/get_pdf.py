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
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_pdf_text(road):
    """读取PDF文件并将其文本内容分块
    road: PDF文件的路径
    return: 返回一个包含所有文本块的列表，每个文本块是一个字典，包含文件名、页码、块编号和文本内容
    """
    road=Path(road)
    doc=pymupdf.open(road)
    data=[]
    for page in doc:
        text=page.get_text()
        text_chunks = split_text(text)
        for chunk in text_chunks:
            data.append(
            {
                "file_name": road.name,
                "page_id": page.number,
                "page_number": page.number+1,
                "chunk_id": len(data)+1,
                "text": chunk
            }
        )
        if page.number>=10:
            break
    doc.close()
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

