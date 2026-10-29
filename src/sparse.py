"""中文分词、BM25 创建与召回。"""
from pathlib import Path
import jieba
from rank_bm25 import BM25Okapi
import cache_store
from cache_store import load_bm25, save_bm25
from config import FINAL_K

PUNCTUATION = frozenset("，。？！：；、（）【】《》")


def segment_text(text):
    # 中文分词
    # 1. 使用原来的 jieba 分词。
    # 2. 去掉两端空白，再过滤空词和原有标点。
    result = []
    for word in jieba.cut(text):
        word = word.strip()
        if word and word not in PUNCTUATION:
            result.append(word)
    return result


def build_corpus(output_path: Path):
    # 准备语料
    # 1. 加载文本列表。
    # 2. 逐块分词，第 i 条语料仍对应第 i 个文本块。
    return [segment_text(item["text"]) for item in cache_store.load_chunks(output_path)]


def build_bm25(corpus):
    # 创建词库
    # 1. 将已经分词的二维语料交给原 BM25 实现。
    return BM25Okapi(corpus)


def query_bm25(bm25: BM25Okapi, query: str):
    # 关键词评分
    # 1. 使用与建库相同的分词规则处理问题。
    # 2. 取得每个文本块的 BM25 得分。
    return bm25.get_scores(segment_text(query))


def query_bm25_top_k(output_path, bm25: BM25Okapi, query: str, k=FINAL_K, data=None):
    # 关键词召回
    # 1. 按得分降序取候选，保留原来的同分顺序。
    # 2. 过滤零分和负分，下标与分数一一对应。
    # 3. 读取或复用文本，沿排名回查原文。
    scores = query_bm25(bm25, query)
    positions = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    positions = [i for i in positions if scores[i] > 0]
    selected_scores = [scores[i] for i in positions]
    chunks = cache_store.load_chunks(output_path, data)
    return [chunks[i]["text"] for i in positions], selected_scores, positions
