"""知识库缓存读写，不负责切块、编码或排名。"""
import json
import pickle
from pathlib import Path
import numpy as np


def load_chunks(output_path, data=None):
    # 读取文本
    # 1. 已提供数据时直接复用，包括空列表。
    # 2. 否则加载原来的文本 JSON。
    if data is not None:
        return data
    with Path(output_path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_to_json(data, road):
    # 保存文本
    # 1. 使用 PDF 同名的 JSON 路径。
    # 2. 沿用原字段和缩进写入，返回文件位置。
    path = Path(road).with_suffix(".json")
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    return path


def select_chunks(top_k_indices, output_path, data=None):
    # 回查来源
    # 1. 加载或复用完整文本列表。
    # 2. 按检索排名取出来源，不按页码重新排序。
    chunks = load_chunks(output_path, data)
    return [chunks[index] for index in top_k_indices]


def save_vectors(vectors, output_path):
    # 保存向量
    # 1. 将向量矩阵写到同名 NPY 文件。
    np.save(Path(output_path).with_suffix(".npy"), vectors)


def load_vectors(output_path):
    # 读取向量
    # 1. 读取与文本 JSON 同名的 NPY 文件。
    return np.load(Path(output_path).with_suffix(".npy"))


def save_bm25(bm25, output_path):
    # 保存词库
    # 1. 沿用同名 PKL 文件保存 BM25。
    with Path(output_path).with_suffix(".pkl").open("wb") as stream:
        pickle.dump(bm25, stream)


def load_bm25(output_path):
    # 读取词库
    # 1. 从可信的本地缓存恢复 BM25。
    with Path(output_path).with_suffix(".pkl").open("rb") as stream:
        return pickle.load(stream)


def save_dense_index(index, index_path):
    # 保存索引
    # 1. 将已创建的 HNSW 索引写入指定位置。
    index.save_index(str(index_path))


def load_dense_index(index_path, dim, space):
    # 读取索引
    # 1. 按原维度和距离方式创建索引对象。
    # 2. 加载指定的 HNSW 文件。
    import hnswlib
    index = hnswlib.Index(space=space, dim=dim)
    index.load_index(str(index_path))
    return index


def cache_paths(road, index_path):
    # 缓存位置
    # 1. 按原规则列出文本、向量、BM25 和 HNSW 文件。
    road = Path(road)
    return (road.with_suffix(".json"), road.with_suffix(".npy"),
            road.with_suffix(".pkl"), Path(index_path))


def metadata_paths(road, index_path):
    # 记录位置
    # 1. 定位 PDF 和索引各自的建库记录。
    road, index_path = Path(road), Path(index_path)
    return (road.with_suffix(".build.json"),
            index_path.with_suffix(index_path.suffix + ".meta.json"))


def read_build_records(paths):
    # 读取记录
    # 1. 读取两份建库记录。
    # 2. 缺失或损坏时返回空记录，让建库流程决定重建。
    records = []
    for path in paths:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            records.append(None)
    return records


def write_build_records(paths, record):
    # 写入记录
    # 1. 按原格式将同一份状态写到两个位置。
    for path in paths:
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def cache_signature(paths):
    # 缓存标记
    # 1. 读取路径、修改时间和大小。
    # 2. 让页面判断旧的文本块统计是否还能复用。
    signature = []
    for path in paths:
        info = path.stat()
        signature.append((str(path.resolve()), info.st_mtime_ns, info.st_size))
    return tuple(signature)
