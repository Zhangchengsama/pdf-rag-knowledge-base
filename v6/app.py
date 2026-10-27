from collections import Counter
from pathlib import Path
from uuid import uuid4

import streamlit as st
from sentence_transformers import SentenceTransformer

from chat_workflow import (failure_message, generate_session_title, recover_request,
                           run_request, start_request)
from conversations import (bind_document, first_question, list_conversations,
                           read_messages)
from document_settings import prepare_document, save_uploaded_pdf
from rag_service import get_knowledge_base_status

# 保留原默认文档、索引和页数；页面设置保存到每个会话的额外元数据。
road = Path(
    r"C:\Users\36143\Desktop\教材资料\化学\普通高中教科书·化学必修 第一册.pdf"
)
index_path = r"C:\Users\36143\Desktop\Python_work\index\test_index.bin"
max_pages = 30
force_rebuild = False
base_dir = Path(__file__).resolve().parent
conversation_dir = base_dir / "data" / "conversations"
STAGE_LABELS = {
    "model": "加载嵌入模型", "knowledge_base": "检查与准备知识库",
    "retrieval": "检索参考资料", "generation": "生成回答", "title": "生成对话标题",
}


def render_sources(sources):
    if not sources:
        return
    with st.expander("📚 查看参考资料"):
        for i, source in enumerate(sources, start=1):
            st.markdown(
                f"**资料 {i}｜{source['file_name']}｜第 {source['page_number']} 页**"
            )
            st.write(source["text"])


@st.cache_resource
def load_model():
    """仅处理问题时加载，让缺少模型时仍能查看历史和修改文档设置。"""
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        local_files_only=True,
    )


def record_build_status(state):
    st.session_state["knowledge_base_details"] = state


def new_conversation():
    """按钮回调先更新状态，再创建本轮页面控件。"""
    st.session_state["conversation_id"] = uuid4().hex
    st.session_state["selected_history"] = None


def stage_text(stages):
    labels = {"running": "进行中", "complete": "完成", "failed": "失败"}
    return "\n\n".join(
        f"{STAGE_LABELS.get(name, name)}：{'中断' if event.get('interrupted') else labels[event['status']]}"
        f" · {elapsed_text(event['elapsed_seconds'])}"
        for name, event in stages.items()
    )


def elapsed_text(seconds):
    return "耗时未完整记录" if seconds is None else f"{seconds:.2f} 秒"


def process_current_request(history_path):
    status = st.status("正在处理问题…", expanded=True)
    output = status.empty()
    stages = {}
    def report(event):
        stages[event["stage"]] = event
        output.markdown(stage_text(stages))
    try:
        metadata = run_request(history_path, load_model, on_progress=report,
                               on_build_status=record_build_status,
                               force_rebuild=force_rebuild)
        request = metadata["request"]
        status.update(
            label=f"处理{'完成' if request['status'] == 'complete' else '失败'}"
                  f" · {elapsed_text(request['elapsed_seconds'])}",
            state="complete" if request["status"] == "complete" else "error",
        )
    except Exception as error:
        status.update(label="无法保存或恢复本次处理状态", state="error")
        st.error(failure_message(error, "storage"))
        return
    st.rerun()


def process_title(history_path):
    status = st.status("正在根据首个问题生成标题…", expanded=True)
    output = status.empty()
    def report(event):
        output.markdown(stage_text({"title": event}))
    try:
        metadata = generate_session_title(history_path, on_progress=report)
        status.update(label="标题已生成" if metadata.get("title") else "标题生成失败",
                      state="complete" if metadata.get("title") else "error")
    except Exception as error:
        status.update(label="标题生成失败", state="error")
        st.error(failure_message(error, "title"))
        return
    st.rerun()


def render_knowledge_base_status(document):
    state = get_knowledge_base_status(document["pdf_path"], document["index_path"],
                                     max_pages=document["max_pages"])
    st.subheader("当前知识库")
    st.caption(Path(document["pdf_path"]).name)
    limit = document["max_pages"]
    st.write(f"读取上限：{'全文' if limit is None else f'前 {limit} 页'}")
    pages = state["actual_pages"]
    st.write(f"实际读取：{pages} 页" if pages is not None else "实际读取：无法确定")
    labels = {
        "ready": "符合当前缓存复用条件",
        "not_built": "尚未建库，首次提问时创建",
        "rebuild_required": "缓存不完整或配置变化，下次提问时重建",
        "incomplete": "上次建库未完成，下次提问时重建",
        "unavailable": "文档或缓存记录无法读取",
    }
    if state["status"] == "unavailable":
        st.warning(labels["unavailable"])
    else:
        st.caption(f"知识库状态：{labels[state['status']]}")
    details = st.session_state.get("knowledge_base_details", {})
    if state["status"] == "ready" and details.get("cache_signature") == state["cache_signature"]:
        st.write(f"文本块数：{details['chunk_count']}")
        st.caption("最近一次提问：" + ("重建知识库" if details["rebuilt"] else "复用已有缓存"))
    else:
        st.write("文本块数：未统计")
    if force_rebuild:
        st.caption("当前已开启强制重建，下一次提问将重新建库。")


def render_document_form(history_path, metadata, messages, document):
    busy = metadata.get('request', {}).get('status') in ('created', 'running') or metadata.get('title_status') == 'running'
    locked = bool(busy or (messages and metadata.get("document")))
    with st.expander("文档与读取范围", expanded=not locked):
        if locked:
            st.caption("文档和范围已与本会话绑定；新建对话后可以更改。")
        elif messages:
            st.warning("旧会话尚未绑定文档，请先确认 PDF 和读取范围。")
        if not locked:
            st.caption("修改后请点击“应用文档和范围”；上传文件时优先使用上传的 PDF。")
        with st.form(f"document_form_{history_path.stem}"):
            pdf_text = st.text_input("本机 PDF 路径", value=document["pdf_path"],
                                    disabled=locked, key=f"pdf_path_{history_path.stem}")
            upload = st.file_uploader("或选择并上传 PDF", type=["pdf"], disabled=locked,
                                      key=f"pdf_upload_{history_path.stem}")
            scope = st.radio("读取范围", ["前 N 页", "全文"],
                             index=1 if document["max_pages"] is None else 0,
                             disabled=locked, key=f"scope_{history_path.stem}")
            pages = st.number_input("读取页数", min_value=1,
                                    value=document["max_pages"] or max_pages or 30, step=1,
                                    disabled=locked, help="选择全文时忽略此项。",
                                    key=f"pages_{history_path.stem}")
            apply = st.form_submit_button("应用文档和范围", disabled=locked, key="apply_document")
        if apply and not locked:
            try:
                selected_pdf = save_uploaded_pdf(upload, base_dir / "data" / "documents") if upload else pdf_text
                selected = prepare_document(
                    selected_pdf, None if scope == "全文" else int(pages), road, index_path,
                    base_dir / "data" / "indexes")
                bind_document(history_path, selected)
            except Exception as error:
                st.error(failure_message(error, "document"))
            else:
                st.rerun()


st.title("📚 PDF RAG Knowledge Assistant")
st.caption("基于 Dense + BM25 + Weighted RRF + DeepSeek")
st.session_state.setdefault("conversation_id", uuid4().hex)
conversations = list_conversations(conversation_dir)
conversation_ids = [item["id"] for item in conversations]
title_counts = Counter(item["title"] for item in conversations)
conversation_labels = {
    item["id"]: item["title"] if title_counts[item["title"]] == 1
    else f"{item['title']} · {item['id'][:8]}"
    for item in conversations
}

with st.sidebar:
    st.button("新建对话", on_click=new_conversation, key="new_conversation")
    st.subheader("历史对话")
    st.caption("按最近使用时间排序")
    if not conversation_ids:
        st.write("暂无历史对话")
    current_id = st.session_state["conversation_id"]
    if current_id in conversation_ids and st.session_state.get("selected_history") is None:
        st.session_state["selected_history"] = current_id
    selected_id = st.selectbox(
        "选择历史对话", conversation_ids, index=None, placeholder="选择已有对话",
        format_func=lambda identity: conversation_labels[identity], key="selected_history",
    )
    if selected_id is not None:
        st.session_state["conversation_id"] = selected_id

history_path = conversation_dir / f"{st.session_state['conversation_id']}.jsonl"
try:
    metadata = recover_request(history_path)
    busy = metadata.get('request', {}).get('status') in ('created', 'running') or metadata.get('title_status') == 'running'
    if busy:
        messages = (st.session_state.get('messages', [])
                    if st.session_state.get('messages_conversation_id') == history_path.stem else [])
    else:
        messages = read_messages(history_path)
except (OSError, ValueError) as error:
    st.error(str(error))
    st.stop()
st.session_state["messages"] = messages
st.session_state["messages_conversation_id"] = history_path.stem
document = metadata.get("document") or {
    "pdf_path": str(road.resolve()), "index_path": str(Path(index_path).resolve()),
    "max_pages": max_pages,
}

with st.sidebar:
    st.divider()
    render_document_form(history_path, metadata, messages, document)
    render_knowledge_base_status(document)

for message in messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        render_sources(message.get("sources"))

request = metadata.get("request", {})
if request:
    with st.expander(f"最近一次提问处理 · {elapsed_text(request.get('elapsed_seconds', 0))}"):
        st.markdown(stage_text(request.get("stages", {})) or "上次处理未记录阶段耗时。")
        st.caption(f"处理次数：{request.get('attempts', 0)}")
if request.get("status") == "failed":
    st.error(request.get("error", "上次请求未完成，可以重试。"))
    if st.button("重试这个问题", key="retry_request", disabled=not metadata.get("document")):
        process_current_request(history_path)
if 'title_elapsed_seconds' in metadata:
    title_state = "完成" if metadata.get("title_status") == "complete" else "失败或未完成"
    st.caption(f"最近一次标题生成：{title_state} · {elapsed_text(metadata['title_elapsed_seconds'])}")
if first_question(messages) and not metadata.get("title"):
    if metadata.get("title_error"):
        st.warning("聊天记录已保留。标题生成失败：" + metadata["title_error"])
    if st.button("重试生成标题" if metadata.get("title_error") else "生成对话标题", key="retry_title", disabled=busy):
        process_title(history_path)

if busy:
    st.info("本会话正在处理，请稍候刷新状态。")
    st.button("刷新处理状态", key="refresh_request")
pending = busy or request.get("status") in ("failed", "created", "running")
unbound_history = bool(messages and not metadata.get("document"))
question = st.chat_input("请输入你想询问的问题", disabled=pending or unbound_history)
if question:
    try:
        if not metadata.get("document"):
            selected = prepare_document(document["pdf_path"], document["max_pages"], road, index_path,
                                        base_dir / "data" / "indexes")
            bind_document(history_path, selected)
        start_request(history_path, question)
    except Exception as error:
        st.error(failure_message(error, "storage"))
    else:
        with st.chat_message("user"):
            st.write(question)
        process_current_request(history_path)
