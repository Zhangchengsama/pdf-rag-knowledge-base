import json
import numpy as np
from dense import query_dense_text_top_k
from sparse import query_bm25_top_k


def RRF_top_k_query_indece(text_input, index_path, output_path, model, bm25, c=60 , candidate_k=20, final_k=5, ef=50, space="cosine", dense_weight=1.0, sparse_weight=1.0):
    """使用加权RRF算法进行查询，默认等权时与普通RRF一致

    text_input: 用户输入的文本字符串

    index_path: 已保存的索引文件路径

    output_path: 已保存的PDF文本数据的JSON文件路径

    model: 用于生成嵌入的模型

    bm25: BM25模型对象

    candidate_k: 两种方法的搜索深度

    final_k: 返回最相似的k个结果

    ef: HNSW索引的ef参数

    space: 距离度量空间

    dense_weight: Dense 排名分数的权重，默认 1.0

    sparse_weight: BM25 排名分数的权重，默认 1.0

    return: 返回按顺序排列的列表(top_k_indices)，相似度得分(top_k_scores)
    """
    
    dense_texts , dense_scores, dense_indices = query_dense_text_top_k(text_input, index_path, output_path, model, k=candidate_k, ef=ef, space=space)
    sparse_texts, sparse_scores, sparse_indices = query_bm25_top_k(output_path, bm25, text_input, k=candidate_k)

    ##print(sparse_scores)
    ##print(dense_scores)

    # RRF算法融合结果
    combined_results = {}

    for i , dense_indice in enumerate(dense_indices,start=1):
        combined_results[dense_indice] = dense_weight / (i + c)  # 只使用排名，不混合原始分数

    for j , sparse_indice in enumerate(sparse_indices,start=1):
        if sparse_indice in combined_results:
            combined_results[sparse_indice] += sparse_weight / (j + c)
        else:
            combined_results[sparse_indice] = sparse_weight / (j + c)

    # 根据RRF得分排序
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)

    top_k_indices = [item[0] for item in sorted_results[:final_k]]
    top_k_scores = [item[1] for item in sorted_results[:final_k]]

    return top_k_indices, top_k_scores

def top_k_query_indice2text(top_k_indices, output_path):
    """根据索引获取对应的文本内容

    top_k_indices: RRF算法返回的前k个索引列表

    output_path: 已保存的PDF文本数据的JSON文件路径

    return: 返回按顺序排列的文本(top_k_texts)
    """
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    top_k_texts = []
    for indice in top_k_indices:
        top_k_texts.append(data[indice]["text"])

    return top_k_texts

def top_k_query_indice2chunk(top_k_indices, output_path):
    """根据索引获取对应的文本内容

    top_k_indices: RRF算法返回的前k个索引列表

    output_path: 已保存的PDF文本数据的JSON文件路径

    return: 返回按顺序排列的chunk及其相关信息(top_k_chunk)
    """
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    top_k_chunk = []
    for indice in top_k_indices:
        top_k_chunk.append(data[indice])

    return top_k_chunk
