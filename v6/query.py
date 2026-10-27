import json
import numpy as np
from dense import query_dense_text_top_k
from sparse import query_bm25_top_k


def RRF_top_k_query_indece(text_input, index_path, output_path, model, bm25, c=60 , candidate_k=20, final_k=5, ef=50, space="cosine", dense_weight=0.8, sparse_weight=1.2, data=None):
    """使用加权RRF算法进行查询，权重相等时与普通RRF的排序一致

    text_input: 用户输入的文本字符串

    index_path: 已保存的索引文件路径

    output_path: 已保存的PDF文本数据的JSON文件路径

    model: 用于生成嵌入的模型

    bm25: BM25模型对象

    c: 排名平滑常数，越大则相邻名次的分数差距越小

    candidate_k: 两种方法的搜索深度

    final_k: 返回最相似的k个结果

    ef: HNSW索引的ef参数

    space: 距离度量空间

    dense_weight: Dense 排名分数的权重，默认 0.8

    sparse_weight: BM25 排名分数的权重，默认 1.2

    data: 可选的完整文本块列表，向两路检索透传，不改变融合计算

    return: 返回按顺序排列的列表(top_k_indices)，融合排名得分(top_k_scores)，并非余弦相似度
    """
    
    # 只在显式提供数据时向下传递新参数，兼容旧的独立调用方式。
    data_kwargs = {} if data is None else {"data": data}
    dense_texts , dense_scores, dense_indices = query_dense_text_top_k(text_input, index_path, output_path, model, k=candidate_k, ef=ef, space=space, **data_kwargs)
    sparse_texts, sparse_scores, sparse_indices = query_bm25_top_k(output_path, bm25, text_input, k=candidate_k, **data_kwargs)

    ##print(sparse_scores)
    ##print(dense_scores)

    # RRF算法融合结果
    # 键是 JSON 中从 0 开始的文本块下标；值为两路排名贡献之和。
    combined_results = {}

    # 排名从 1 开始；同一文本块同时命中两路检索时会累加分数。
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

def top_k_query_indice2text(top_k_indices, output_path, data=None):
    """根据索引获取对应的文本内容

    top_k_indices: RRF算法返回的前k个索引列表

    output_path: 已保存的PDF文本数据的JSON文件路径

    return: 返回按顺序排列的文本(top_k_texts)
    """
    if data is None:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # 保留传入排名顺序，不按页码或 chunk_id 重新排序。
    top_k_texts = []
    for indice in top_k_indices:
        top_k_texts.append(data[indice]["text"])

    return top_k_texts

def top_k_query_indice2chunk(top_k_indices, output_path, data=None):
    """根据索引获取对应的文本内容

    top_k_indices: RRF算法返回的前k个索引列表

    output_path: 已保存的PDF文本数据的JSON文件路径

    return: 返回按顺序排列的chunk及其相关信息(top_k_chunk)
    """
    if data is None:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # 返回完整记录，供提示词和前端共同使用来源信息。
    top_k_chunk = []
    for indice in top_k_indices:
        top_k_chunk.append(data[indice])

    return top_k_chunk
