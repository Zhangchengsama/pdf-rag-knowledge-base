"""知识库准备与状态判断，缓存格式和复用规则保持原样。"""
from pathlib import Path
import cache_store
from config import DEFAULT_MAX_PAGES
from dense import save_dense_index
from get_pdf import get_pdf_page_count, get_pdf_text
from sparse import build_bm25, build_corpus, save_bm25


def build_layout(road, index_path, max_pages, output_path=None):
    # 准备记录
    # 1. 计算本次实际读取页数。
    # 2. 按原字段生成成功记录，并定位缓存与两份状态文件。
    record = {"version": 1, "pdf_path": str(road), "index_path": str(index_path),
              "actual_pages": get_pdf_page_count(road, max_pages=max_pages), "status": "ready"}
    return (record, cache_store.metadata_paths(road, index_path, output_path),
            cache_store.cache_paths(road, index_path, output_path))


def can_reuse(record, previous_records, paths):
    # 判断复用
    # 1. 两份记录都要与当前配置完全相同。
    # 2. 四种缓存都存在时才能复用。
    return (all(previous == record for previous in previous_records)
            and all(path.is_file() for path in paths))


def get_knowledge_base_status(road, index_path, max_pages=DEFAULT_MAX_PAGES, output_path=None):
    # 查看状态
    # 1. 只读页数、建库记录和缓存文件信息。
    # 2. 区分可复用、未建库、需重建、未完成或无法读取。
    # 3. 返回页面需要的状态，不创建模型或重建缓存。
    road, index_path = Path(road).resolve(), Path(index_path).resolve()
    state = {"pdf_path": str(road), "index_path": str(index_path),
             "max_pages": max_pages, "actual_pages": None,
             "cache_signature": None, "missing_caches": []}
    try:
        record, metadata, paths = build_layout(road, index_path, max_pages, output_path)
        previous = cache_store.read_build_records(metadata)
        state["actual_pages"] = record["actual_pages"]
        state["missing_caches"] = [str(path) for path in paths if not path.is_file()]
        if can_reuse(record, previous, paths):
            state["status"] = "ready"
            state["cache_signature"] = cache_store.cache_signature(paths)
        elif any(isinstance(item, dict) and item.get("status") == "building" for item in previous):
            state["status"] = "incomplete"
        elif not any(path.exists() for path in metadata + paths):
            state["status"] = "not_built"
        else:
            state["status"] = "rebuild_required"
    except (OSError, RuntimeError, ValueError) as error:
        state.update(status="unavailable", error=str(error))
    return state


def build_knowledge_base(road, index_path, model, max_pages=DEFAULT_MAX_PAGES, force_rebuild=False,
                         output_path=None):
    # 准备知识库
    # 1. 核对实际页数、双记录和四种缓存，符合条件时直接复用。
    # 2. 重建前先写入未完成状态，再逐页提取和分词。
    # 3. 创建稠密索引与 BM25，全部写入成功后发布成功记录。
    road, index_path = Path(road).resolve(), Path(index_path).resolve()
    output_path = Path(output_path).resolve() if output_path is not None else road.with_suffix(".json")
    record, metadata, paths = build_layout(road, index_path, max_pages, output_path)
    previous = cache_store.read_build_records(metadata)
    if not force_rebuild and can_reuse(record, previous, paths):
        return {"rebuilt": False, "actual_pages": record["actual_pages"]}
    print(f"正在重建知识库：读取前 {record['actual_pages']} 页")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_store.write_build_records(metadata, dict(record, status="building"))
    data = get_pdf_text(road, max_pages=max_pages)
    if not data:
        raise ValueError("所选页没有可提取的文本，无法建库；扫描版 PDF 需要先进行 OCR")
    cache_store.save_to_json(data, road, output_path)
    corpus = build_corpus(output_path)
    if not any(corpus):
        raise ValueError("所选页分词后没有有效词语，无法创建 BM25 索引")
    save_dense_index(output_path, index_path, model)
    save_bm25(build_bm25(corpus), output_path)
    cache_store.write_build_records(metadata, record)
    return {"rebuilt": True, "actual_pages": record["actual_pages"]}
