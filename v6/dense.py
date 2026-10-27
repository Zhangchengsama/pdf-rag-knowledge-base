from sentence_transformers import SentenceTransformer
import json
import hnswlib
from pathlib import Path
import numpy as np


def cos_similarity(v1,v2):
    """计算两个非零向量的余弦相似度，值越大表示方向越接近。"""
    return v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def get_embedding(text, model):
    """用调用方提供的模型编码文本；建库与查询应使用同一模型。"""
    return model.encode(text)

def get_similarity(input_embedding, item_embedding):
    """计算输入文本与已有文本的相似度"""
    sim_A_B = cos_similarity(input_embedding, item_embedding)
    return sim_A_B

def save_np_embedding(output_path:Path, model):
    """获得嵌入向量并保存PDF文本数据的嵌入向量到.npy文件

    output_path: 已保存的PDF文本数据的JSON文件路径

    model: 用于生成嵌入的模型
    """
    data = []
    item_embedding = []

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item_embedding.append(get_embedding(item["text"], model).tolist())

    # 第 i 行向量对应 JSON 的第 i 个文本块，后续通过这个顺序回查原文。
    embeddings = np.array(item_embedding, dtype=np.float32)

    np.save(output_path.with_suffix(".npy"), embeddings) 

    return embeddings

def save_dense_index(output_path: Path, index_path: Path, model, space: str = "cosine"):
    """保存稠密向量索引到指定路径

    output_path: 已保存的PDF文本数据的JSON文件路径

    index_path: 保存索引的路径

    model: 用于生成嵌入的模型

    space: 距离度量空间(默认值为 "cosine")
    """

    embeddings = save_np_embedding(output_path, model)

    # 向量列数就是嵌入维度，加载和创建 HNSW 索引时必须一致。
    dim = embeddings.shape[1]

    index = hnswlib.Index(
        space=space,
        dim=dim
        )

    index.init_index(
        max_elements=embeddings.shape[0],#可以改成可变参数
        M=16,  # HNSW 图的连接数参数，影响内存占用与召回效果。
        )

    # 标签从 0 开始，是 JSON 列表下标，不是从 1 开始的 chunk_id。
    ids = np.arange(embeddings.shape[0])
    
    index.add_items(
        embeddings,
        ids
    )

    index.save_index(str(index_path))

def query_dense_text_top_k(text_input, index_path: Path, output_path:Path, model, k=5, ef=50 ,space: str = "cosine", data=None):
    """查询文本与已保存的PDF文本数据的相似度

    text_input: 用户输入的文本字符串

    index_path: 已保存的索引文件路径

    output_path: 已保存的PDF文本数据的JSON文件路径

    model: 用于生成嵌入的模型

    k: 返回最相似的k个结果

    ef: HNSW索引的ef参数

    space: 距离度量空间

    data: 可选的完整文本块列表；省略时从 output_path 加载
    
    return: 返回按顺序排列的文本(top_k_texts)，相似度得分(top_k_scores)和索引列表(top_k_indices)
    """

    results = []

    # 调用方可传入本次问答已加载的文本；旧调用仍从文件读取。
    if data is None:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    embeddings = np.load(output_path.with_suffix(".npy"))

    # 向量列数就是嵌入维度，加载和创建 HNSW 索引时必须一致。
    dim = embeddings.shape[1]

    index = hnswlib.Index(
        space=space,
        dim=dim,
        )

    index.load_index(index_path)

    # HNSW 不能请求超过索引容量的结果；小型知识库按实际块数返回。
    k = min(k, index.get_current_count())
    if k <= 0:
        return [], [], []

    # ef 控制搜索候选范围；通常至少取 k，越大越偏向召回而非速度。
    index.set_ef(max(ef, k))

    input_embedding = get_embedding(text_input,model)

    labels, distances = index.knn_query(input_embedding, k=k)

    # 单条查询仍返回二维结果，取第 0 行得到这一问题的标签和距离。
    ids = labels[0]
    dists = distances[0]

    for i, dist in zip(ids, dists):
        chunk = data[i]
        # cosine 空间下距离为 1 - 余弦相似度，此处转换为越大越相关的分数。
        similarity = 1 - dist
        results.append((similarity, chunk , i))

    results.sort(key=lambda x: x[0], reverse=True)
    top_k_results = results[:k]

    if top_k_results:
        top_k_texts = [item['text'] for _, item , _ in top_k_results]
        top_k_scores = [similarity for similarity, _ , _ in top_k_results]
        top_k_indices = [index for _, _, index in top_k_results]
        return top_k_texts, top_k_scores, top_k_indices
    else:
        return None, 0 , 0
