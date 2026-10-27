"""编排两路召回；融合计算在 fusion，来源回查在 cache_store。"""
import cache_store
import fusion
from dense import query_dense_text_top_k
from sparse import query_bm25_top_k
from config import (CANDIDATE_K, DENSE_WEIGHT, FINAL_K, INDEX_SPACE,
                    RRF_C, SEARCH_EF, SPARSE_WEIGHT)


def hybrid_search(text_input, index_path, output_path, model, bm25, c=RRF_C,
                  candidate_k=CANDIDATE_K, final_k=FINAL_K, ef=SEARCH_EF,
                  space=INDEX_SPACE, dense_weight=DENSE_WEIGHT,
                  sparse_weight=SPARSE_WEIGHT, data=None):
    # 混合检索
    # 1. 两路按各自的方法召回候选文本块。
    # 2. 只把排名下标交给独立的融合函数。
    # 3. 返回融合下标和得分，保留旧参数顺序。
    data_kwargs = {} if data is None else {"data": data}
    _, _, dense_indices = query_dense_text_top_k(
        text_input, index_path, output_path, model, k=candidate_k, ef=ef,
        space=space, **data_kwargs)
    _, _, sparse_indices = query_bm25_top_k(
        output_path, bm25, text_input, k=candidate_k, **data_kwargs)
    return fusion.fuse_rankings(dense_indices, sparse_indices, c,
                               dense_weight, sparse_weight, final_k)


# 兼容旧拼写；新代码调用 hybrid_search。
RRF_top_k_query_indece = hybrid_search


def top_k_query_indice2text(top_k_indices, output_path, data=None):
    # 回查文本
    # 1. 沿排名选出来源，再取出文本。
    return [chunk["text"] for chunk in cache_store.select_chunks(top_k_indices, output_path, data)]


def top_k_query_indice2chunk(top_k_indices, output_path, data=None):
    # 回查来源
    # 1. 保留旧入口，按排名返回完整文本块。
    return cache_store.select_chunks(top_k_indices, output_path, data)
