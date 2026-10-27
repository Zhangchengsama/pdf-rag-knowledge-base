"""网页展示与控件交互。"""
from collections import Counter
from pathlib import Path
from uuid import uuid4

import streamlit as st
import embedding
from config import DEFAULT_MAX_PAGES

from chat_workflow import (failure_message, generate_session_title, recover_request,
                           run_request, start_request)
from conversations import (bind_document, first_question, list_conversations,
                           read_messages)
from document_settings import prepare_document, save_uploaded_pdf
from knowledge_base import get_knowledge_base_status

STAGE_LABELS = {
    "model": "加载嵌入模型", "knowledge_base": "检查与准备知识库",
    "retrieval": "检索参考资料", "generation": "生成回答", "title": "生成对话标题",
}


def render_sources(sources):
    # 显示来源
    # 1. 有来源时打开同一套折叠区。
    # 2. 沿排名显示文件名、页码和原文。
    if not sources:
        return
    with st.expander("📚 查看参考资料"):
        for i, source in enumerate(sources, start=1):
            st.markdown(
                f"**资料 {i}｜{source['file_name']}｜第 {source['page_number']} 页**"
            )
            st.write(source["text"])


@st.cache_resource
def load_model(model_name):
    # 加载模型
    # 1. 使用页面配置加载模型，重跑时复用已加载的资源。
    return embedding.load_embedding_model(model_name)


def record_build_status(state):
    # 记录建库
    # 1. 把本次建库结果放进当前页面状态，供侧栏复用。
    st.session_state["knowledge_base_details"] = state


def new_conversation():
    # 新建对话
    # 1. 生成新的会话 ID，并清空历史选择。
    st.session_state["conversation_id"] = uuid4().hex
    st.session_state["selected_history"] = None


def stage_text(stages):
    # 整理进度
    # 1. 将各阶段状态翻译成中文。
    # 2. 显示已记录耗时，中断且未知时明确说明。
    labels = {"running": "进行中", "complete": "完成", "failed": "失败"}
    return "\n\n".join(
        f"{STAGE_LABELS.get(name, name)}：{'中断' if event.get('interrupted') else labels[event['status']]}"
        f" · {elapsed_text(event['elapsed_seconds'])}"
        for name, event in stages.items()
    )


def elapsed_text(seconds):
    # 显示耗时
    # 1. 把秒数转成文字，未知耗时单独说明。
    return "耗时未完整记录" if seconds is None else f"{seconds:.2f} 秒"


def process_current_request(history_path, settings):
    # 处理提问
    # 1. 创建进度区域，调用会话处理函数。
    # 2. 同步展示阶段和结果，文件错误单独提示。
    # 3. 处理完刷新页面，恢复已保存的消息。
    status = st.status("正在处理问题…", expanded=True)
    output = status.empty()
    stages = {}
    def report(event):
        # 更新进度
        # 1. 将收到的阶段状态和耗时显示到页面。
        stages[event["stage"]] = event
        output.markdown(stage_text(stages))
    try:
        metadata = run_request(history_path, lambda: load_model(settings.embedding_model), on_progress=report,
                               on_build_status=record_build_status,
                               force_rebuild=settings.force_rebuild)
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
    # 处理标题
    # 1. 单独展示标题生成进度。
    # 2. 调用标题流程，结束后刷新会话列表。
    status = st.status("正在根据首个问题生成标题…", expanded=True)
    output = status.empty()
    def report(event):
        # 更新进度
        # 1. 将收到的阶段状态和耗时显示到页面。
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


def render_knowledge_base_status(document, settings):
    # 知识库状态
    # 1. 只读检查当前文档和缓存。
    # 2. 显示读取范围及复用状态。
    # 3. 缓存版本匹配时复用已知块数，否则显示未统计。
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
    if settings.force_rebuild:
        st.caption("当前已开启强制重建，下一次提问将重新建库。")


def render_document_form(history_path, metadata, messages, document, settings):
    # 文档设置
    # 1. 按会话状态决定是否允许更换资料。
    # 2. 读取路径或上传文件，以及页数范围。
    # 3. 点击应用后验证并绑定文档，再刷新页面。
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
                                    value=document["max_pages"] or settings.max_pages or DEFAULT_MAX_PAGES, step=1,
                                    disabled=locked, help="选择全文时忽略此项。",
                                    key=f"pages_{history_path.stem}")
            apply = st.form_submit_button("应用文档和范围", disabled=locked, key="apply_document")
        if apply and not locked:
            try:
                selected_pdf = save_uploaded_pdf(upload, settings.document_dir) if upload else pdf_text
                selected = prepare_document(
                    selected_pdf, None if scope == "全文" else int(pages), settings.pdf_path, settings.index_path,
                    settings.index_dir)
                bind_document(history_path, selected)
            except Exception as error:
                st.error(failure_message(error, "document"))
            else:
                st.rerun()

def initialize_session():
    # 初始化会话
    # 1. 首次进入页面时生成会话 ID，已有 ID 继续复用。
    st.session_state.setdefault("conversation_id", uuid4().hex)


def render_heading():
    # 页面标题
    # 1. 显示页面名称和当前使用的检索流程。
    st.title("📚 PDF RAG Knowledge Assistant")
    st.caption("基于 Dense + BM25 + Weighted RRF + DeepSeek")


def select_conversation(settings):
    # 选择对话
    # 1. 读取按时间排序的会话列表，准备显示标题。
    # 2. 绘制新建按钮和历史选择框，内部始终使用原 ID。
    # 3. 返回当前会话文件的位置。
    conversations = list_conversations(settings.conversation_dir)
    identities = [item["id"] for item in conversations]
    counts = Counter(item["title"] for item in conversations)
    labels = {
        item["id"]: item["title"] if counts[item["title"]] == 1
        else f"{item['title']} · {item['id'][:8]}"
        for item in conversations
    }
    with st.sidebar:
        st.button("新建对话", on_click=new_conversation, key="new_conversation")
        st.subheader("历史对话")
        st.caption("按最近使用时间排序")
        if not identities:
            st.write("暂无历史对话")
        current = st.session_state["conversation_id"]
        if current in identities and st.session_state.get("selected_history") is None:
            st.session_state["selected_history"] = current
        selected = st.selectbox(
            "选择历史对话", identities, index=None, placeholder="选择已有对话",
            format_func=lambda identity: labels[identity], key="selected_history")
        if selected is not None:
            st.session_state["conversation_id"] = selected
    return settings.conversation_dir / f"{st.session_state['conversation_id']}.jsonl"


def session_is_busy(metadata):
    # 判断忙碌
    # 1. 问答或标题仍在处理时，页面不重复提交。
    return (metadata.get("request", {}).get("status") in ("created", "running")
            or metadata.get("title_status") == "running")


def load_conversation(history_path):
    # 加载对话
    # 1. 恢复真正中断的请求，保留其他页面的活动状态。
    # 2. 忙碌时复用当前页面已有消息，否则读取原记录。
    # 3. 更新页面状态，返回会话信息、消息和忙碌标记。
    try:
        metadata = recover_request(history_path)
        busy = session_is_busy(metadata)
        if busy:
            messages = (st.session_state.get("messages", [])
                        if st.session_state.get("messages_conversation_id") == history_path.stem else [])
        else:
            messages = read_messages(history_path)
    except (OSError, ValueError) as error:
        st.error(str(error))
        st.stop()
    st.session_state["messages"] = messages
    st.session_state["messages_conversation_id"] = history_path.stem
    return metadata, messages, busy


def selected_document(metadata, settings):
    # 当前文档
    # 1. 优先使用会话绑定的文档，旧记录和新会话使用默认设置。
    return metadata.get("document") or {
        "pdf_path": str(settings.pdf_path),
        "index_path": str(settings.index_path),
        "max_pages": settings.max_pages,
    }


def render_document_sidebar(history_path, metadata, messages, document, settings):
    # 文档侧栏
    # 1. 绘制文档选择表单和只读知识库状态。
    with st.sidebar:
        st.divider()
        render_document_form(history_path, metadata, messages, document, settings)
        render_knowledge_base_status(document, settings)


def render_messages(messages):
    # 展示消息
    # 1. 按保存顺序显示消息。
    # 2. 对有来源的消息使用同一套来源展示。
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            render_sources(message.get("sources"))


def render_request_status(history_path, metadata, messages, busy, settings):
    # 处理状态
    # 1. 显示最近一次提问和标题生成的状态及耗时。
    # 2. 为失败的问答与标题分别提供重试。
    # 3. 会话忙碌时只提供状态刷新。
    request = metadata.get("request", {})
    if request:
        with st.expander(f"最近一次提问处理 · {elapsed_text(request.get('elapsed_seconds', 0))}"):
            st.markdown(stage_text(request.get("stages", {})) or "上次处理未记录阶段耗时。")
            st.caption(f"处理次数：{request.get('attempts', 0)}")
    if request.get("status") == "failed":
        st.error(request.get("error", "上次请求未完成，可以重试。"))
        if st.button("重试这个问题", key="retry_request", disabled=not metadata.get("document")):
            process_current_request(history_path, settings)
    if "title_elapsed_seconds" in metadata:
        title_state = "完成" if metadata.get("title_status") == "complete" else "失败或未完成"
        st.caption(f"最近一次标题生成：{title_state} · {elapsed_text(metadata['title_elapsed_seconds'])}")
    if first_question(messages) and not metadata.get("title"):
        if metadata.get("title_error"):
            st.warning("聊天记录已保留。标题生成失败：" + metadata["title_error"])
        if st.button("重试生成标题" if metadata.get("title_error") else "生成对话标题",
                     key="retry_title", disabled=busy):
            process_title(history_path)
    if busy:
        st.info("本会话正在处理，请稍候刷新状态。")
        st.button("刷新处理状态", key="refresh_request")


def submit_question(question, history_path, metadata, document, settings):
    # 提交问题
    # 1. 首次提问前确认文档绑定。
    # 2. 保存一次用户问题，再进入可恢复的处理流程。
    try:
        if not metadata.get("document"):
            selected = prepare_document(
                document["pdf_path"], document["max_pages"], settings.pdf_path,
                settings.index_path, settings.index_dir)
            bind_document(history_path, selected)
        start_request(history_path, question)
    except Exception as error:
        st.error(failure_message(error, "storage"))
    else:
        with st.chat_message("user"):
            st.write(question)
        process_current_request(history_path, settings)


def render_question_input(history_path, metadata, messages, document, busy, settings):
    # 问题输入
    # 1. 有未完成请求或旧会话未绑定文档时暂停新提问。
    # 2. 收到问题后交给提交函数处理。
    pending = busy or metadata.get("request", {}).get("status") in ("failed", "created", "running")
    unbound = bool(messages and not metadata.get("document"))
    question = st.chat_input("请输入你想询问的问题", disabled=pending or unbound)
    if question:
        submit_question(question, history_path, metadata, document, settings)


def run_web(settings):
    # 页面流程
    # 1. 初始化并选择当前会话。
    # 2. 加载记录和文档，绘制侧栏及消息。
    # 3. 展示状态，处理新提问或重试。
    initialize_session()
    render_heading()
    history_path = select_conversation(settings)
    metadata, messages, busy = load_conversation(history_path)
    document = selected_document(metadata, settings)
    render_document_sidebar(history_path, metadata, messages, document, settings)
    render_messages(messages)
    render_request_status(history_path, metadata, messages, busy, settings)
    render_question_input(history_path, metadata, messages, document, busy, settings)
