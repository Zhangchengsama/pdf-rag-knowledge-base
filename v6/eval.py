"""检索评测及结果报告。"""
import time
import cache_store
import embedding
import knowledge_base
from config import load_settings, FINAL_K
from dense import query_dense_text_top_k
from sparse import query_bm25_top_k
from query import hybrid_search
import json

def load_eval_questions(eval_set_path):
    # 读取题集
    # 1. 加载指定评测文件，返回问题列表。
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_questions = json.load(f)
    return eval_questions

def get_page_number(output_path, top_k_indices):
    # 回查页码
    # 1. 按检索下标读取来源。
    # 2. 返回 PDF 页序，保留同页重复和原排名。
    return [chunk["page_number"] for chunk in cache_store.select_chunks(top_k_indices, output_path)]


def evaluate_model(eval_questions, index_path, output_path, model, bm25, c=60, candidate_k=20, final_k=5, ef=50, space="cosine"):
    # 评测检索
    # 1. 每道题分别运行 Dense、BM25 和融合检索。
    # 2. 将文本下标回查成 PDF 页序。
    # 3. 按原规则记录三种方法的 Hit@1 和 Hit@k。
    # 每项记录一道题的三路检索页序与命中标记，最终汇总时再取平均。
    results = []


    
    for i,question in enumerate(eval_questions, start=1):
        text_input = question["question"]
        dense_top_k_texts, dense_top_k_scores, dense_top_k_indices = query_dense_text_top_k(text_input, index_path, output_path, model, k=final_k, ef=ef, space=space)

        sparse_top_k_texts, sparse_top_k_scores, sparse_top_k_indices = query_bm25_top_k(output_path, bm25, text_input, k=final_k)

        rrf_top_k_indices, rrf_top_k_scores = hybrid_search(text_input, index_path, output_path, model, bm25, c=c , candidate_k=candidate_k, final_k=final_k, ef=ef, space=space)

        rrf_top_k_page_number =get_page_number(output_path= output_path, top_k_indices= rrf_top_k_indices)
        dense_top_k_page_number =get_page_number(output_path= output_path, top_k_indices= dense_top_k_indices)
        sparse_top_k_page_number =get_page_number(output_path= output_path, top_k_indices= sparse_top_k_indices)

        # 下方字段沿用 *_indices 名称，但实际存储的是页序而非文本块下标。
        item = {
            "question": f"Q{i}:{text_input}",
            "gold_page": question["gold_page"],
            "dense_top_k_indices": dense_top_k_page_number,
            "sparse_top_k_indices": sparse_top_k_page_number,
            "rrf_top_k_indices": rrf_top_k_page_number,
        }
        # Hit@1 判断第一块；Hit@k 判断指定范围内是否至少有一块来自 gold_page。
        for method, pages in (
            ("dense", dense_top_k_page_number),
            ("sparse", sparse_top_k_page_number),
            ("rrf", rrf_top_k_page_number),
        ):
            item[f"{method}_hit_at_1"] = int(bool(pages) and pages[0] == question["gold_page"])
            item[f"{method}_hit_at_k"] = int(question["gold_page"] in pages[:final_k])

        # 保留原来的 hit 字段，但改为 Hybrid 的 Hit@k。
        item["hit"] = item["rrf_hit_at_k"]
        results.append(item)
    
    return results



def get_eval_set_path(settings=None):
    # 定位题集
    # 1. 通过共享配置返回题集路径，不依赖当前终端目录。
    return (settings or load_settings()).evaluation_path


def print_evaluation_report(results, k=FINAL_K):
    # 展示评测
    # 1. 按问题输出三种方法的页码和命中情况。
    # 2. 汇总 Hit@1 和 Hit@k，空题集单独提示。
    methods = (("dense", "Dense"), ("sparse", "BM25"), ("rrf", "Hybrid RRF"))

    for item in results:
        print(item["question"])
        print(f"正确 PDF 页序: {item['gold_page']}")
        for method, label in methods:
            print(
                f"{label}: {item[f'{method}_top_k_indices']} | "
                f"Hit@1={item[f'{method}_hit_at_1']} | "
                f"Hit@{k}={item[f'{method}_hit_at_k']}"
            )
        print()

    total = len(results)
    if total == 0:
        print("评测集为空，无法计算命中率。")
    else:
        print(f"评测汇总（共 {total} 题；按前 {k} 个文本块的 PDF 页序判定命中）")
        print(f"{'方法':<16} {'Hit@1':<20} Hit@{k}")
        for method, label in methods:
            hit_at_1 = sum(item[f"{method}_hit_at_1"] for item in results)
            hit_at_k = sum(item[f"{method}_hit_at_k"] for item in results)
            hit_at_1_text = f"{hit_at_1}/{total} ({hit_at_1 / total:.1%})"
            hit_at_k_text = f"{hit_at_k}/{total} ({hit_at_k / total:.1%})"
            print(f"{label:<16} {hit_at_1_text:<20} {hit_at_k_text}")

def run_evaluation(settings=None, model_loader=None):
    # 运行评测
    # 1. 加载共享配置和嵌入模型。
    # 2. 按配置准备知识库，再读取题集。
    # 3. 运行评测并输出报告，不调用大模型。
    settings = settings or load_settings()
    started = time.perf_counter()
    model = model_loader() if model_loader else embedding.load_embedding_model(settings.embedding_model)
    print("正在加载 embedding 模型...")
    print(f"模型加载完成，用时 {time.perf_counter() - started:.2f}s")
    knowledge_base.build_knowledge_base(settings.pdf_path, settings.index_path, model,
                                       max_pages=settings.max_pages, force_rebuild=settings.force_rebuild)
    bm25 = cache_store.load_bm25(settings.pdf_path)
    questions = load_eval_questions(get_eval_set_path(settings))
    results = evaluate_model(questions, settings.index_path, settings.pdf_path.with_suffix(".json"),
                             model, bm25, final_k=FINAL_K)
    print_evaluation_report(results)
    return results


if __name__ == "__main__":
    run_evaluation()





