"""嵌入模型和文本编码。"""
import numpy as np
from config import EMBEDDING_MODEL


def load_embedding_model(model_name=EMBEDDING_MODEL):
    # 加载模型
    # 1. 需要编码时再导入模型库。
    # 2. 只加载本机已有的模型，不自动下载。
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, local_files_only=True)


def get_embedding(text, model):
    # 文本编码
    # 1. 使用调用方复用的模型生成向量。
    return model.encode(text)


def encode_chunks(chunks, model):
    # 批次编码
    # 1. 沿用逐块编码方式，不改变文本顺序。
    # 2. 转成 float32 矩阵，第 i 行对应第 i 个文本块。
    vectors = [get_embedding(chunk["text"], model).tolist() for chunk in chunks]
    return np.array(vectors, dtype=np.float32)
