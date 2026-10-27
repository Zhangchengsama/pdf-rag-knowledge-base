import json
from prompt import get_prompt
from get_pdf import get_pdf_text, get_pdf_page_count, save_to_json
from dense import save_dense_index
from sparse import build_bm25, save_bm25 ,load_bm25 ,build_corpus
from query import RRF_top_k_query_indece ,top_k_query_indice2chunk
from pathlib import Path

def building_data(road, index_path, model, max_pages=11, force_rebuild=False):
    """按实际读取页数复用或整套重建知识库，返回页数与是否重建的状态。

    max_pages 为读取上限，None 表示全文；force_rebuild=True 强制更新。
    页数相同但 PDF 内容、模型或切块方式改变时，需要主动强制重建。
    旧缓存无记录、记录不完整或任一缓存缺失时，也会重建全部四种文件。
    当前按串行调用设计，同一组缓存不应被多个进程同时写入。
    """
    road = Path(road).resolve()
    index_path = Path(index_path).resolve()
    output_path = road.with_suffix(".json")
    actual_pages = get_pdf_page_count(road, max_pages=max_pages)
    record = {
        "version": 1,
        "pdf_path": str(road),
        "index_path": str(index_path),
        "actual_pages": actual_pages,
        "status": "ready",
    }
    # 源文件缓存与独立索引各留一份记录，防止切换 PDF / 索引后误用旧配对。
    metadata_paths = (road.with_suffix(".build.json"),
                      index_path.with_suffix(index_path.suffix + ".meta.json"))
    records_match = True
    for path in metadata_paths:
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            previous = None
        records_match = records_match and previous == record

    cache_paths = (output_path, road.with_suffix(".npy"),
                   road.with_suffix(".pkl"), index_path)
    if not force_rebuild and records_match and all(p.is_file() for p in cache_paths):
        return {"rebuilt": False, "actual_pages": actual_pages}

    print(f"正在重建知识库：读取前 {actual_pages} 页")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    # 必须先标记未完成，再覆盖缓存。任何一步失败，下次都不会复用半成品。
    pending = dict(record, status="building")
    for path in metadata_paths:
        path.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")

    data = get_pdf_text(road, max_pages=max_pages)
    if not data:
        raise ValueError("所选页没有可提取的文本，无法建库；扫描版 PDF 需要先进行 OCR")
    save_to_json(data, road)
    corpus = build_corpus(output_path)
    if not any(corpus):
        raise ValueError("所选页分词后没有有效词语，无法创建 BM25 索引")
    # save_dense_index 内部已生成并保存 .npy，避免同一批文本重复编码。
    save_dense_index(output_path, index_path, model)
    save_bm25(build_bm25(corpus), road)

    # 四种文件全部写入成功后，才发布成功记录。记录损坏也会触发下次重建。
    for path in metadata_paths:
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"rebuilt": True, "actual_pages": actual_pages}


def answer_question(road, index_path, text_input, model, max_pages=11, force_rebuild=False):
    """完成单次问答，返回 {answer: 回答文本, sources: 检索文本块列表}。

    model 由入口复用；其余文件及 BM25 对象在当前调用中准备或加载。
    max_pages / force_rebuild 透传建库设置，不改变回答与来源的返回格式。
    """

    building_data(road, index_path=index_path, model=model,
                  max_pages=max_pages, force_rebuild=force_rebuild)

    road=Path(road)
    output_path = road.with_suffix(".json")

    bm25 = load_bm25(road)

    # 两路各召回 20 块，按加权 RRF 融合后选出 5 块交给大模型。
    top_k_indices, top_k_scores = RRF_top_k_query_indece(text_input, index_path, output_path, model, bm25, c=60, candidate_k=20, final_k=5, ef=50, space="cosine")

    prompt = get_prompt(
    text_input,
    top_k_indices,
    output_path
    )

    # 来源与提示词使用同一组下标，保证页面展示的资料对应本次检索。
    chunks =top_k_query_indice2chunk(top_k_indices= top_k_indices, output_path=output_path)

    # 仅生成回答时才加载 API 客户端；独立建库和评测不需要大模型密钥。
    from LLM import ask_llm
    answer = ask_llm(prompt)
    result = {"answer":answer,
              "sources":chunks
              }

    return result
