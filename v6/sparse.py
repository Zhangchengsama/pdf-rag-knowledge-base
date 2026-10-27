import jieba
import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

def segment_text(text):
    """对文本进行分词处理

    text: 输入的文本字符串

    return: 返回一个包含分词结果的列表
    """
    segments = list(jieba.cut(text))
    result = []

    # 逐词去掉两端空白，再按下方列举的标点进行过滤。
    for word in segments:
        word = word.strip()
        if (word and word != "，" 
            and word != "。" and word != "？" and word != "！" 
            and word != "：" and word != "；" and word != "、" 
            and word != "（" and word != "）" and word != "【" 
            and word != "】" and word != "《" and word != "》"):
            result.append(word)
    return result

def build_corpus(output_path: Path):
    """构建语料库

    output_path: 用于读取 JSON 或派生同名 .pkl 缓存的 Path 路径

    return: 返回一个包含分词结果的列表
    """
    data = []

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 外层列表与 JSON 文本块一一对应；每个内层列表保存该块的分词结果。
    corpus = []
    for item in data:
        corpus.append(segment_text(item["text"]))
    return corpus

def build_bm25(corpus):
    """构建BM25模型

    corpus: 已分词的二维列表（文本块 -> 词语列表），不是文件路径
    
    return: 返回一个BM25模型对象
    """
    bm25 = BM25Okapi(corpus)

    return bm25

def save_bm25(bm25, output_path: Path):
    """保存BM25模型到指定路径

    bm25: BM25模型对象

    output_path: 用于读取 JSON 或派生同名 .pkl 缓存的 Path 路径
    """
    with open(output_path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(bm25, f)

def load_bm25(output_path: Path):
    """加载BM25模型

    output_path: 用于读取 JSON 或派生同名 .pkl 缓存的 Path 路径

    return: 返回一个BM25模型对象
    """
    with open(output_path.with_suffix(".pkl"), "rb") as f:
        bm25 = pickle.load(f)
    return bm25

def query_bm25(bm25: BM25Okapi, query: str):
    """使用BM25模型进行查询

    bm25: BM25模型对象

    query: 查询字符串

    return: 返回一个包含每个文档的相关性得分的列表
    """
    # 查询与建库使用同一分词和过滤流程，scores 的下标对应语料块下标。
    query_tokens = segment_text(query)
    scores = bm25.get_scores(query_tokens)
    return scores

def query_bm25_top_k(output_path, bm25: BM25Okapi, query: str, k: int = 5):
    """使用BM25模型进行查询并返回前k个相关性最高的文档

    bm25: BM25模型对象

    query: 查询字符串

    k: 返回的文档数量

    return: 返回一个包含前k个相关性最高(得分大于零)的文段(top_k_texts)和得分的列表(top_k_scores),以及对应的索引列表(top_k_indices)
    """
    scores = query_bm25(bm25, query)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    # 只保留正分候选，因此返回数量可能少于 k，也可能为空。
    top_k_indices = [i for i in top_k_indices if scores[i] > 0]
    top_k_scores = [scores[i] for i in top_k_indices]

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 按分数排序后的下标回查原文，三个返回列表按位置一一对应。
    text_texts = []
    for indice in top_k_indices:
        text_texts.append(data[indice]["text"])

    return text_texts, top_k_scores , top_k_indices

if __name__ == "__main__":
    road = Path(r"C:\Users\36143\Desktop\教材资料\化学\普通高中教科书·化学必修 第一册.pdf")
    output_path = road.with_suffix(".json")
    query = "氧化还原反应是什么"
    corpus = build_corpus(output_path)
    bm25 = build_bm25(corpus)
    top_k_texts , top_k_scores, top_k_indices= query_bm25_top_k(output_path, bm25, query, k=5)
    print(top_k_texts)
