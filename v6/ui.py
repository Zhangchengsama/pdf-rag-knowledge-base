"""网页入口：在顶部导航中选择问答或知识库管理页面。"""
from pathlib import Path

import streamlit as st


def run_web(settings):
    # 页面导航
    # 1. 将注入的配置交给两个页面，保留跨页面的会话状态。
    # 2. 在主页面顶部绘制两个入口，窄窗口下也不移入侧边栏。
    # 3. 使用原生页面路由，只执行当前选中的页面。
    st.session_state["app_settings"] = settings
    directory = Path(__file__).resolve().parent
    pages = [
        st.Page(directory / "chat_ui.py", title="问答", icon=":material/chat:", default=True),
        st.Page(directory / "knowledge_ui.py", title="知识库管理", icon=":material/library_books:",
                url_path="knowledge"),
    ]
    page = st.navigation(pages, position="hidden")
    with st.container(horizontal=True):
        for destination in pages:
            st.page_link(destination, label=destination.title, icon=destination.icon)
    page.run()
