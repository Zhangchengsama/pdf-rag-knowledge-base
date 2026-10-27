import json
from prompt import get_prompt, format_prompt
from get_pdf import get_pdf_text, get_pdf_page_count, save_to_json
from dense import save_dense_index
from sparse import build_bm25, save_bm25 ,load_bm25 ,build_corpus
from query import RRF_top_k_query_indece ,top_k_query_indice2chunk
from pathlib import Path
from progress import track_stage


def _build_layout(road, index_path, max_pages):
    """统一生成建库记录和关联文件，供只读状态检查与原建库流程共用。"""
    record = {
        "version": 1,
        "pdf_path": str(road),
        "index_path": str(index_path),
        "actual_pages": get_pdf_page_count(road, max_pages=max_pages),
        "status": "ready",
    }
    metadata_paths = (road.with_suffix(".build.json"),
                      index_path.with_suffix(index_path.suffix + ".meta.json"))
    cache_paths = (road.with_suffix(".json"), road.with_suffix(".npy"),
                   road.with_suffix(".pkl"), index_path)
    return record, metadata_paths, cache_paths


def _read_build_records(metadata_paths):
    """缺失或损坏的记录不能作为复用依据，保持原有判断规则。"""
    records = []
    for path in metadata_paths:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            records.append(None)
    return records


def _can_reuse(record, previous_records, cache_paths):
    """双记录完全一致且四种缓存存在，才符合当前的复用条件。"""
    return (all(previous == record for previous in previous_records)
            and all(path.is_file() for path in cache_paths))


def _cache_signature(cache_paths):
    """轻量记录文件版本，避免状态面板继续显示重建前的块数。"""
    signature = []
    for path in cache_paths:
        info = path.stat()
        signature.append((str(path.resolve()), info.st_mtime_ns, info.st_size))
    return tuple(signature)


def get_knowledge_base_status(road, index_path, max_pages=11):
    """只读检查当前配置，不提取文本、不加载模型、不创建或重建缓存。"""
    road = Path(road).resolve()
    index_path = Path(index_path).resolve()
    state = {"pdf_path": str(road), "index_path": str(index_path),
             "max_pages": max_pages, "actual_pages": None,
             "cache_signature": None, "missing_caches": []}
    try:
        record, metadata_paths, cache_paths = _build_layout(road, index_path, max_pages)
        previous_records = _read_build_records(metadata_paths)
        state["actual_pages"] = record["actual_pages"]
        state["missing_caches"] = [str(path) for path in cache_paths if not path.is_file()]
        if _can_reuse(record, previous_records, cache_paths):
            state["status"] = "ready"
            state["cache_signature"] = _cache_signature(cache_paths)
        elif any(isinstance(item, dict) and item.get("status") == "building"
                 for item in previous_records):
            state["status"] = "incomplete"
        elif not any(path.exists() for path in metadata_paths + cache_paths):
            state["status"] = "not_built"
        else:
            state["status"] = "rebuild_required"
    except (OSError, RuntimeError, ValueError) as error:
        state.update(status="unavailable", error=str(error))
    return state


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
    record, metadata_paths, cache_paths = _build_layout(road, index_path, max_pages)
    actual_pages = record["actual_pages"]
    # 源文件缓存与独立索引各留一份记录，防止切换 PDF / 索引后误用旧配对。
    previous_records = _read_build_records(metadata_paths)
    if not force_rebuild and _can_reuse(record, previous_records, cache_paths):
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


def answer_question(road, index_path, text_input, model, max_pages=11, force_rebuild=False,
                    on_build_status=None, on_progress=None):
    """完成单次问答，返回 {answer: 回答文本, sources: 检索文本块列表}。

    model 由入口复用；其余文件及 BM25 对象在当前调用中准备或加载。
    max_pages / force_rebuild 透传建库设置，不改变回答与来源的返回格式。
    可选 on_build_status 回调接收建库结果、块数和文件版本；旧调用无需传入。
    可选 on_progress 回调接收阶段名称、状态和 elapsed_seconds，不改变结果格式。
    """

    with track_stage('knowledge_base', on_progress):
        build_state = building_data(road, index_path=index_path, model=model,
                                    max_pages=max_pages, force_rebuild=force_rebuild)

    with track_stage('retrieval', on_progress):
        road = Path(road)
        output_path = road.with_suffix(".json")
        # 建库完成后读取一次，本次检索、提示词和来源展示共用同一份文本。
        with output_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if on_build_status is not None:
            cache_paths = (output_path, road.with_suffix(".npy"),
                           road.with_suffix(".pkl"), Path(index_path))
            on_build_status(dict(build_state, chunk_count=len(data),
                                 cache_signature=_cache_signature(cache_paths)))
        bm25 = load_bm25(road)
        # 两路各召回 20 块，按加权 RRF 融合后选出 5 块交给大模型。
        top_k_indices, top_k_scores = RRF_top_k_query_indece(text_input, index_path, output_path, model, bm25, c=60, candidate_k=20, final_k=5, ef=50, space="cosine", data=data)
        # 来源与提示词使用同一组下标，保证页面展示的资料对应本次检索。
        chunks = top_k_query_indice2chunk(top_k_indices, output_path, data=data)
        prompt = format_prompt(text_input, chunks)

    # 仅生成回答时才加载 API 客户端；独立建库和评测不需要大模型密钥。
    with track_stage('generation', on_progress):
        from LLM import ask_llm
        answer = ask_llm(prompt)
    result = {"answer":answer,
              "sources":chunks
              }

    return result
