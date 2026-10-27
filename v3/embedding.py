from sentence_transformers import SentenceTransformer
import numpy as np


def cos_similarity(v1,v2):
    return v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def get_embedding(text, model):
    return model.encode(text)

def get_similarity(input_embedding, item_embedding):
    """计算输入文本与已有文本的相似度"""
    sim_A_B = cos_similarity(input_embedding, item_embedding)
    return sim_A_B


