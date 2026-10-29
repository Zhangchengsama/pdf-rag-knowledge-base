"""加权 RRF；只接收排名，不读取文件或调用模型。"""
from config import DENSE_WEIGHT, FINAL_K, RRF_C, SPARSE_WEIGHT


def fuse_rankings(dense_indices, sparse_indices, c=RRF_C,
                  dense_weight=DENSE_WEIGHT, sparse_weight=SPARSE_WEIGHT,
                  final_k=FINAL_K) -> tuple[list[int], list[float]]:
    # 排名融合
    # 1. 按 Dense 名次计算各文本块的贡献，名次从 1 开始。
    # 2. 累加 BM25 名次的贡献，重复命中仍对应同一文本块。
    # 3. 按总分降序取前几名，返回下标和分数。
    combined = {}
    for rank, index in enumerate(dense_indices, start=1):
        combined[index] = dense_weight / (rank + c)
    for rank, index in enumerate(sparse_indices, start=1):
        if index in combined:
            combined[index] += sparse_weight / (rank + c)
        else:
            combined[index] = sparse_weight / (rank + c)
    # 同分时保留插入顺序：Dense 先加入，BM25 新下标随后加入。
    ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    return ([item[0] for item in ranked[:final_k]],
            [item[1] for item in ranked[:final_k]])
