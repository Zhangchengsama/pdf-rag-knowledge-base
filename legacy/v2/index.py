import json
import hnswlib
import numpy as np
from pathlib import Path
from embedding import get_embedding

def save_np_embedding(output_path:Path, model):
    """保存PDF文本数据的嵌入向量到.npy文件
    output_path: 已保存的PDF文本数据的JSON文件路径
    model: 用于生成嵌入的模型
    """
    data = []
    item_embedding = []

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item_embedding.append(get_embedding(item["text"], model).tolist())

    embeddings = np.array(item_embedding, dtype=np.float32)

    np.save(output_path.with_suffix(".npy"), embeddings) 

    return embeddings

def save_index(output_path: Path, index_path: Path, model, space: str = "cosine"):
    """保存索引到指定路径
    output_path: 已保存的PDF文本数据的JSON文件路径
    index_path: 保存索引的路径
    model: 用于生成嵌入的模型
    space: 距离度量空间(默认值为 "cosine")
    """

    embeddings = save_np_embedding(output_path, model)

    dim = embeddings.shape[1]

    index = hnswlib.Index(
        space=space,
        dim=dim
        )

    index.init_index(
        max_elements=embeddings.shape[0],#可以改成可变参数
        M=16,
        )

    ids = np.arange(embeddings.shape[0])
    
    index.add_items(
        embeddings,
        ids
    )

    index.save_index(str(index_path))









