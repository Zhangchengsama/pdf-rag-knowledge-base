"""单次问答编排；旧建库与状态入口继续转发。"""
from pathlib import Path
import cache_store
import knowledge_base
import query
from config import (CANDIDATE_K, DEFAULT_MAX_PAGES, FINAL_K,
                    INDEX_SPACE, RRF_C, SEARCH_EF)
from progress import track_stage
from prompt import format_prompt


def building_data(road, index_path, model, max_pages=DEFAULT_MAX_PAGES, force_rebuild=False,
                   output_path=None):
    # 兼容建库
    # 1. 保留旧参数顺序，转发到知识库模块。
    options = {} if output_path is None else {'output_path': output_path}
    return knowledge_base.build_knowledge_base(road, index_path, model, max_pages, force_rebuild, **options)


def get_knowledge_base_status(road, index_path, max_pages=DEFAULT_MAX_PAGES, output_path=None):
    # 兼容状态
    # 1. 将旧的状态查询转发到知识库模块。
    return knowledge_base.get_knowledge_base_status(road, index_path, max_pages, output_path)


def answer_question(road, index_path, text_input, model, max_pages=DEFAULT_MAX_PAGES,
                    force_rebuild=False, on_build_status=None, on_progress=None, output_path=None):
    # 回答问题
    # 1. 检查或准备知识库，并报告该阶段的状态。
    # 2. 只加载一次文本，完成混合检索和提示词拼接。
    # 3. 调用大模型，返回原来的回答与来源字典。
    options = {} if output_path is None else {'output_path': output_path}
    with track_stage("knowledge_base", on_progress):
        build_state = knowledge_base.build_knowledge_base(
            road, index_path=index_path, model=model,
            max_pages=max_pages, force_rebuild=force_rebuild, **options)
    with track_stage("retrieval", on_progress):
        road = Path(road)
        output_path = Path(output_path) if output_path is not None else road.with_suffix(".json")
        data = cache_store.load_chunks(output_path)
        if on_build_status is not None:
            signature = cache_store.cache_signature(cache_store.cache_paths(road, index_path, output_path))
            on_build_status(dict(build_state, chunk_count=len(data), cache_signature=signature))
        bm25 = cache_store.load_bm25(output_path)
        positions, _ = query.hybrid_search(
            text_input, index_path, output_path, model, bm25, c=RRF_C,
            candidate_k=CANDIDATE_K, final_k=FINAL_K, ef=SEARCH_EF,
            space=INDEX_SPACE, data=data)
        chunks = cache_store.select_chunks(positions, output_path, data)
        prompt = format_prompt(text_input, chunks)
    with track_stage("generation", on_progress):
        from LLM import ask_llm
        answer = ask_llm(prompt)
    return {"answer": answer, "sources": chunks}
