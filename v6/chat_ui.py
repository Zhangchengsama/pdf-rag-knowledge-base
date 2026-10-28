"""问答页面：会话选择、消息、进度与失败重试。"""
from collections import Counter
from pathlib import Path
from uuid import uuid4

import streamlit as st
import embedding

from chat_workflow import (failure_message, generate_session_title, recover_request,
                           run_request, start_request)
from conversations import first_question, list_conversations, read_messages
from document_settings import load_active_document, prepare_document

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
    # 1. 把本次建库结果放进当前页面状态，供知识库管理页复用。
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


def process_current_request(history_path, settings, document):
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
                               force_rebuild=settings.force_rebuild, document=document)
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


def render_messages(messages):
    # 展示消息
    # 1. 按保存顺序显示消息。
    # 2. 对有来源的消息使用同一套来源展示。
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            render_sources(message.get("sources"))


def render_request_status(history_path, metadata, messages, busy, settings, document):
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
        st.caption("重试使用原请求的文档和读取范围。")
        retry_document = request.get("document") or metadata.get("document") or document
        if st.button("重试这个问题", key="retry_request", disabled=not retry_document):
            process_current_request(history_path, settings, document)
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


def submit_question(question, history_path, settings):
    # 提交问题
    # 1. 读取知识库管理页最新保存的文档和范围，检查文件可用。
    # 2. 把设置保存在本次请求中，先保存问题，再处理问答。
    try:
        document = load_active_document(settings)
        selected = prepare_document(
            document["pdf_path"], document["max_pages"], settings.pdf_path,
            settings.index_path, settings.index_dir)
        start_request(history_path, question, document=selected)
    except Exception as error:
        st.error(failure_message(error, "document"))
    else:
        with st.chat_message("user"):
            st.write(question)
        process_current_request(history_path, settings, selected)


def render_question_input(history_path, metadata, document, busy, settings):
    # 问题输入
    # 1. 有未完成请求或知识库设置无法读取时暂停新提问。
    # 2. 收到问题后交给提交函数处理。
    pending = busy or metadata.get("request", {}).get("status") in ("failed", "created", "running")
    question = st.chat_input("请输入你想询问的问题", disabled=pending or document is None)
    if question:
        submit_question(question, history_path, settings)


def run_chat_page(settings):
    # 问答页面
    # 1. 初始化并选择当前会话，读取聊天记录。
    # 2. 读取共用知识库设置；旧会话不再要求绑定资料。
    # 3. 展示消息、状态并处理新提问或重试。
    initialize_session()
    render_heading()
    history_path = select_conversation(settings)
    metadata, messages, busy = load_conversation(history_path)
    try:
        document = load_active_document(settings)
    except (OSError, ValueError) as error:
        st.error(failure_message(error, "document"))
        document = None
    st.caption("提问使用“知识库管理”中当前选择的资料。")
    render_messages(messages)
    render_request_status(history_path, metadata, messages, busy, settings, document)
    render_question_input(history_path, metadata, document, busy, settings)


if __name__ == "__main__":
    run_chat_page(st.session_state["app_settings"])
