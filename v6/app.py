import json
import streamlit as st
from pathlib import Path
from uuid import uuid4
from sentence_transformers import SentenceTransformer

from rag_service import answer_question


# 当前页面使用的 PDF；派生的 JSON、向量和 BM25 缓存保存在 PDF 同目录。
road = Path(
    r"C:\Users\36143\Desktop\教材资料\化学\普通高中教科书·化学必修 第一册.pdf"
)

#读取对话目录ID
conversation_dir = Path(__file__).resolve().parent / "data" / "conversations"
conversation_ids = [
    file.stem
    for file in conversation_dir.glob("*.jsonl")
]
names =conversation_ids  #对话Uid

# HNSW 稠密索引单独保存，必须与当前 PDF 的文本块及向量保持对应。
index_path = r"C:\Users\36143\Desktop\Python_work\index\test_index.bin"
max_pages = 30  # 读取前多少页；None 表示全文，实际页数变化时自动重建。
force_rebuild = False  # 强制刷新一次后改回 False，避免每次提问都重新建库。


def save_message(message):
    """保存消息到 JSONL 文件中，每行一个 JSON 对象。"""
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")

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
    st.write("这里将展示已有的对话")
    if not names:
        st.write("暂无历史对话")

    #如果当前对话已经存在对应文件，而且选择框还没有选中历史对话，就自动选中当前对话的 UID
    current_id = st.session_state.get("conversation_id")
    if current_id in names and st.session_state.get("selected_history") is None:
        st.session_state["selected_history"] = current_id
    
    selected_id = st.selectbox(
    "选择历史对话",
    names,
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

        if message.get("sources"):
            with st.expander("📚 查看参考资料"):
                for i, source in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**资料 {i}｜{source['file_name']}｜第 {source['page_number']} 页**"
                    )
                    st.write(source["text"])

    
#问答模块
if question:
    is_new_conversation = not history_path.exists()

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
                force_rebuild=force_rebuild
            )

        st.write(result["answer"])

        with st.expander("📚 查看参考资料"):

            # 按融合检索的排名展示本次回答使用的原始文本块。
            for i, source in enumerate(result["sources"], start=1):

                st.markdown(
                    f"**资料 {i}｜{source['file_name']}｜第 {source['page_number']} 页**"
                )

                st.write(source["text"])

    #保存AI消息
    assistant_message = {
    "role": "assistant",
    "content": result["answer"],
    "sources": result["sources"]
    }
    st.session_state["messages"].append(assistant_message)
    save_message(assistant_message)

    #如果是新对话，重新运行页面以刷新历史对话选择框
    if is_new_conversation:
        st.rerun()