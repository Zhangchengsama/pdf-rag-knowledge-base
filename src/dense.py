"""稠密索引和向量召回，编码及文件读写交给对应模块。"""
from pathlib import Path
import hnswlib
import numpy as np
import cache_store
import embedding
from config import FINAL_K, INDEX_M, INDEX_SPACE, SEARCH_EF


def cos_similarity(v1, v2):
    # 余弦相似度
    # 1. 用点积除以两个非零向量的长度乘积。
    return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_embedding(text, model):
    # 兼容编码
    # 1. 保留旧入口，交给编码模块生成向量。
    return embedding.get_embedding(text, model)


def get_similarity(input_embedding, item_embedding):
    # 比较向量
    # 1. 返回两个向量的余弦相似度。
    return cos_similarity(input_embedding, item_embedding)


def save_np_embedding(output_path: Path, model):
    # 编码入库
    # 1. 读取文本块，按原顺序逐块编码。
    # 2. 保存 float32 向量矩阵，并返回给索引构建流程。
    vectors = embedding.encode_chunks(cache_store.load_chunks(output_path), model)
    cache_store.save_vectors(vectors, output_path)
    return vectors


def build_dense_index(vectors, space=INDEX_SPACE):
    # 创建索引
    # 1. 根据向量列数确定维度。
    # 2. 使用原来的 M 参数创建 HNSW。
    # 3. 用从 0 开始的位置作为标签，保持与文本列表一致。
    index = hnswlib.Index(space=space, dim=vectors.shape[1])
    index.init_index(max_elements=vectors.shape[0], M=INDEX_M)
    index.add_items(vectors, np.arange(vectors.shape[0]))
    return index


def save_dense_index(output_path: Path, index_path: Path, model, space=INDEX_SPACE):
    # 保存索引
    # 1. 复用旧入口生成并保存向量。
    # 2. 创建 HNSW，再交给缓存模块保存。
    vectors = save_np_embedding(output_path, model)
    cache_store.save_dense_index(build_dense_index(vectors, space), index_path)


def query_dense_text_top_k(text_input, index_path: Path, output_path: Path, model,
                           k=FINAL_K, ef=SEARCH_EF, space=INDEX_SPACE, data=None):
    # 向量召回
    # 1. 读取或复用文本，并加载对应的向量维度和索引。
    # 2. 按索引容量限制 k，再编码问题并查询。
    # 3. 将距离转成相似度，沿原排名返回文本、分数和下标。
    data = cache_store.load_chunks(output_path, data)
    vectors = cache_store.load_vectors(output_path)
    index = cache_store.load_dense_index(index_path, vectors.shape[1], space)
    k = min(k, index.get_current_count())
    if k <= 0:
        return [], [], []
    index.set_ef(max(ef, k))
    labels, distances = index.knn_query(embedding.get_embedding(text_input, model), k=k)
    results = [(1 - distance, data[position], position)
               for position, distance in zip(labels[0], distances[0])]
    results.sort(key=lambda item: item[0], reverse=True)
    selected = results[:k]
    if selected:
        return ([item["text"] for _, item, _ in selected],
                [score for score, _, _ in selected],
                [position for _, _, position in selected])
    return None, 0, 0
