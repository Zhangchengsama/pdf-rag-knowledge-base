"""字符切块；更换算法只修改 split_text 的实现，保留输入输出。"""
from config import CHUNK_OVERLAP, CHUNK_SIZE


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    # 文本切块
    # 1. 从当前字符位置取出一个文本块。
    # 2. 向后移动，留下相邻块的重叠部分。
    # 3. 一直读到文本末尾，按顺序返回各块。
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
