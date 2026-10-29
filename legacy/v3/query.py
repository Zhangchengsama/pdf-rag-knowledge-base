import json
import hnswlib
import numpy as np
from pathlib import Path
from embedding import get_embedding

def query_text_top_k(text_input, index_path: Path, output_path:Path, model, k=5, ef=50 ,space: str = "cosine"):
    """查询文本与已保存的PDF文本数据的相似度
    text_input: 用户输入的文本字符串
    index_path: 已保存的索引文件路径
    output_path: 已保存的PDF文本数据的JSON文件路径
    model: 用于生成嵌入的模型
    k: 返回最相似的k个结果
    ef: HNSW索引的ef参数
    space: 距离度量空间
    return: 返回按顺序排列的文本(top_k_texts)，相似度得分(top_k_scores)和索引列表(top_k_indices)
    """

    results = []

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = np.load(output_path.with_suffix(".npy"))

    dim = embeddings.shape[1]

    index = hnswlib.Index(
        space=space,
        dim=dim,
        )

    index.load_index(index_path)

    index.set_ef(ef)

    input_embedding = get_embedding(text_input,model)

    labels, distances = index.knn_query(input_embedding, k=k)

    ids = labels[0]
    dists = distances[0]

    for i, dist in zip(ids, dists):
        chunk = data[i]
        similarity = 1 - dist
        results.append((similarity, chunk , i))

    results.sort(key=lambda x: x[0], reverse=True)
    top_k_results = results[:k]

    if top_k_results:
        top_k_texts = [item['text'] for _, item in top_k_results]
        top_k_scores = [similarity for similarity, _ in top_k_results]
        top_k_indices = [index for _, _, index in top_k_results]
        return top_k_texts, top_k_scores, top_k_indices
    else:
        return None, 0 , 0



