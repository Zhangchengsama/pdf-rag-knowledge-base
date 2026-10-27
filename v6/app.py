import json
from collections import Counter
import streamlit as st
from pathlib import Path
from uuid import uuid4
from sentence_transformers import SentenceTransformer

from rag_service import answer_question, get_knowledge_base_status
from conversations import list_conversations


# 当前页面使用的 PDF；派生的 JSON、向量和 BM25 缓存保存在 PDF 同目录。
road = Path(
    r"C:\Users\36143\Desktop\教材资料\化学\普通高中教科书·化学必修 第一册.pdf"
)

# 旧 JSONL 保持原状；标题只用于显示，选择与存储仍使用原 ID。
conversation_dir = Path(__file__).resolve().parent / "data" / "conversations"
conversations = list_conversations(conversation_dir)
conversation_ids = [item["id"] for item in conversations]
title_counts = Counter(item["title"] for item in conversations)
conversation_labels = {
    item["id"]: (item["title"] if title_counts[item["title"]] == 1
                 else f"{item['title']} · {item['id'][:8]}")
    for item in conversations
}

# HNSW 稠密索引单独保存，必须与当前 PDF 的文本块及向量保持对应。
index_path = r"C:\Users\36143\Desktop\Python_work\index\test_index.bin"
max_pages = 30  # 读取前多少页；None 表示全文，实际页数变化时自动重建。
force_rebuild = False  # 强制刷新一次后改回 False，避免每次提问都重新建库。


def save_message(message):
    """保存消息到 JSONL 文件中，每行一个 JSON 对象。"""
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")


def render_sources(sources):
    """按检索排名显示来源；历史消息和当前回答使用同一套展示。"""
    if not sources:
        return
    with st.expander("📚 查看参考资料"):
        for i, source in enumerate(sources, start=1):
            st.markdown(
                f"**资料 {i}｜{source['file_name']}｜第 {source['page_number']} 页**"
            )
            st.write(source["text"])


def record_build_status(state):
    """只保存在当前页面会话，不改变回答字典和原有聊天记录。"""
    st.session_state["knowledge_base_details"] = state


def render_knowledge_base_status():
    """只读展示当前配置和缓存状态；块数复用本次问答已加载的数据。"""
    state = get_knowledge_base_status(road, index_path, max_pages=max_pages)
    st.subheader("当前知识库")
    st.caption(road.name)
    st.write(f"读取上限：{'全文' if max_pages is None else f'前 {max_pages} 页'}")
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
        st.caption(state["error"])
    else:
        st.caption(f"知识库状态：{labels[state['status']]}")
    details = st.session_state.get("knowledge_base_details", {})
    if state["status"] == "ready" and details.get("cache_signature") == state["cache_signature"]:
        st.write(f"文本块数：{details['chunk_count']}")
        action = "重建知识库" if details["rebuilt"] else "复用已有缓存"
        st.caption(f"最近一次提问：{action}")
    else:
        st.write("文本块数：未统计")
    if force_rebuild:
        st.caption("当前已开启强制重建，下一次提问将重新建库。")


@st.cache_resource
def load_model():
    """加载本地多语言嵌入模型；资源缓存让页面重跑时复用同一模型。"""
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        local_files_only=True
    )


model = load_model()


st.title("📚 PDF RAG Knowledge Assistant")
st.caption("基于 Dense + BM25 + Weighted RRF + DeepSeek")

# 提交聊天输入会触发脚本重跑；未提交时不会进入下方问答分支。
question = st.chat_input("请输入你想询问的问题")

#侧边栏，实现对话选择功能
with st.sidebar:
    render_knowledge_base_status()
    st.divider()
    #新建对话
    #1. 使用 uuid4().hex 生成一个全新的 conversation_id。
    #2. 将 st.session_state["messages"] 设置成空列表。
    #3. 清空历史对话选择框的状态，然后重新绘制网页。
    if st.button("新建对话"):
        st.session_state["conversation_id"] = uuid4().hex
        st.session_state.pop("messages", None)
        st.session_state["selected_history"] = None
        st.rerun()

    st.title("历史对话")
    st.caption("按最近使用时间排序")
    if not conversation_ids:
        st.write("暂无历史对话")

    #如果当前对话已经存在对应文件，而且选择框还没有选中历史对话，就自动选中当前对话的 UID
    current_id = st.session_state.get("conversation_id")
    if current_id in conversation_ids and st.session_state.get("selected_history") is None:
        st.session_state["selected_history"] = current_id
    
    selected_id = st.selectbox(
    "选择历史对话",
    conversation_ids,
    format_func=lambda conversation_id: conversation_labels[conversation_id],
    index=None,
    placeholder="选择已有对话",
    key="selected_history"
    #key参数负责在状态中保存历史选择
    )

#加载聊天记录
# 1. 更新 conversation_id
# 2. 根据 selected_id 计算对应文件路径
# 3. 读取文件，替换 session_state["messages"]
# 4. 重新运行页面
with st.spinner("正在加载聊天记录..."):
    if "conversation_id" not in st.session_state:
            st.session_state["conversation_id"] = uuid4().hex

    if selected_id is not None and selected_id != st.session_state["conversation_id"]:
        st.session_state["conversation_id"] = selected_id#更新对话id
        st.session_state.pop("messages", None)#删除已有的消息，重新加载

    history_path = (
    Path(__file__).resolve().parent
    / "data"
    / "conversations"
    / f"{st.session_state['conversation_id']}.jsonl"
    )

    if "messages" not in st.session_state:
        if history_path.exists():
            with history_path.open("r", encoding="utf-8") as f:
                st.session_state["messages"] = [
                    json.loads(line)
                    for line in f
                    if line.strip()
                ]
        else:
            st.session_state["messages"] = []

#加载消息到缓存并显示
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        render_sources(message.get("sources"))

    
#问答模块
if question:
    with st.chat_message("user"):
        st.write(question)
    
    #保存用户消息
    user_message = {"role": "user", "content": question}

    st.session_state["messages"].append(user_message)
    save_message(user_message)

    with st.chat_message("assistant"):

        with st.spinner("正在检索资料并生成回答..."):

            # result 包含回答 answer 和带文件名、页码、原文的 sources。
            result = answer_question(
                road=road,
                index_path=index_path,
                text_input=question,
                model=model,
                max_pages=max_pages,
                force_rebuild=force_rebuild,
                on_build_status=record_build_status,
            )

        st.write(result["answer"])

        render_sources(result["sources"])

    #保存AI消息
    assistant_message = {
    "role": "assistant",
    "content": result["answer"],
    "sources": result["sources"]
    }
    st.session_state["messages"].append(assistant_message)
    save_message(assistant_message)

    # 两条消息保存完成后刷新标题和排序，重跑时不再提交已消费的聊天输入。
    st.rerun()
