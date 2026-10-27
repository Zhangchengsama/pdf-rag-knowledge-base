from dense import query_dense_text_top_k
from sparse import query_bm25_top_k
from query import RRF_top_k_query_indece, top_k_query_indice2text
import json

def load_eval_questions(eval_set_path):
    """加载评估问题集

    eval_set_path: 评估问题集的JSON文件路径

    return: 返回一个包含评估问题的列表
    """
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_questions = json.load(f)
    return eval_questions

def get_page_number(output_path, top_k_indices):
    """把 JSON 文本块下标映射为 PDF 页序，保留排名及同页重复结果。"""
    top_k_page_number= []

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for indice in top_k_indices:
        top_k_page_number.append(data[indice]["page_number"])

    return top_k_page_number



def evaluate_model(eval_questions, index_path, output_path, model, bm25, c=60, candidate_k=20, final_k=5, ef=50, space="cosine"):
    """评估模型在给定问题集上的表现

    eval_questions: 评估问题集的列表

    index_path: 已保存的索引文件路径

    output_path: 已保存的PDF文本数据的JSON文件路径

    model: 用于生成嵌入的模型

    bm25: BM25模型对象

    c: RRF算法中的常数参数

    candidate_k: 两种方法的搜索深度

    final_k: 返回最相似的k个结果

    ef: HNSW索引的ef参数

    space: 距离度量空间
    
    return: 返回每个问题的检索页码，以及三种方法的 Hit@1 / Hit@k 标记列表。
    每题只标注一个 gold_page；前 k 个文本块中有任意一个来自该页，就记为命中。
    """
    # 每项记录一道题的三路检索页序与命中标记，最终汇总时再取平均。
    results = []


    
    for i,question in enumerate(eval_questions, start=1):
        text_input = question["question"]
        dense_top_k_texts, dense_top_k_scores, dense_top_k_indices = query_dense_text_top_k(text_input, index_path, output_path, model, k=final_k, ef=ef, space=space)

        sparse_top_k_texts, sparse_top_k_scores, sparse_top_k_indices = query_bm25_top_k(output_path, bm25, text_input, k=final_k)

        rrf_top_k_indices, rrf_top_k_scores = RRF_top_k_query_indece(text_input, index_path, output_path, model, bm25, c=c , candidate_k=candidate_k, final_k=final_k, ef=ef, space=space)

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


if __name__ == '__main__':
    import time
    from rag_service import building_data
    from sparse import load_bm25
    from pathlib import Path
    from sentence_transformers import SentenceTransformer

    # 直接运行本文件才执行模型加载和评测；导入时仅定义评测函数。
    t0 = time.perf_counter()

    # 相对于脚本定位评测集，避免依赖旧版本目录或当前启动目录。
    eval_set_path = Path(__file__).resolve().parent / "test" / "eval_questions.json"
    road=Path(r"C:\Users\36143\Desktop\教材资料\化学\普通高中教科书·化学必修 第一册.pdf")
    output_path = road.with_suffix(".json")
    index_path = r"C:\Users\36143\Desktop\Python_work\index\test_index.bin"
    max_pages = 11  # 读取前多少页；None 表示全文，实际页数变化时自动重建。
    force_rebuild = False  # 同页数内容发生变化时可改为 True，重建后再改回 False。
    Model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    local_files_only=True
    )

    print("正在加载 embedding 模型...")

    print(f"模型加载完成，用时 {time.perf_counter() - t0:.2f}s")

    # 与问答共用建库判断，避免评测继续使用旧页数的缓存。
    building_data(road, index_path, Model,
                  max_pages=max_pages, force_rebuild=force_rebuild)
    bm25 = load_bm25(road)

    eval_questions = load_eval_questions(eval_set_path)

    k = 5
    result = evaluate_model(eval_questions= eval_questions , index_path =index_path , output_path=output_path , model= Model , bm25= bm25, final_k=k)

    # 内部字段前缀与终端显示名称的映射，供逐题明细和汇总共用。
    methods = (("dense", "Dense"), ("sparse", "BM25"), ("rrf", "Hybrid RRF"))

    for item in result:
        print(item["question"])
        print(f"正确 PDF 页序: {item['gold_page']}")
        for method, label in methods:
            print(
                f"{label}: {item[f'{method}_top_k_indices']} | "
                f"Hit@1={item[f'{method}_hit_at_1']} | "
                f"Hit@{k}={item[f'{method}_hit_at_k']}"
            )
        print()

    total = len(result)
    if total == 0:
        print("评测集为空，无法计算命中率。")
    else:
        print(f"评测汇总（共 {total} 题；按前 {k} 个文本块的 PDF 页序判定命中）")
        print(f"{'方法':<16} {'Hit@1':<20} Hit@{k}")
        for method, label in methods:
            hit_at_1 = sum(item[f"{method}_hit_at_1"] for item in result)
            hit_at_k = sum(item[f"{method}_hit_at_k"] for item in result)
            hit_at_1_text = f"{hit_at_1}/{total} ({hit_at_1 / total:.1%})"
            hit_at_k_text = f"{hit_at_k}/{total} ({hit_at_k / total:.1%})"
            print(f"{label:<16} {hit_at_1_text:<20} {hit_at_k_text}")





