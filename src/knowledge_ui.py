"""知识库管理页面：文档选择、读取范围和缓存状态。"""
from pathlib import Path

import streamlit as st
from chat_workflow import failure_message
from config import DEFAULT_MAX_PAGES
from document_settings import (load_active_document, prepare_document,
                               save_active_document, save_uploaded_pdf)
from knowledge_base import get_knowledge_base_status


def render_knowledge_base_status(document, settings):
    # 知识库状态
    # 1. 只读检查当前文档和缓存。
    # 2. 显示读取范围及复用状态。
    # 3. 缓存版本匹配时复用已知块数，否则显示未统计。
    state = get_knowledge_base_status(document["pdf_path"], document["index_path"],
                                     max_pages=document["max_pages"], output_path=document.get('output_path'))
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


def render_document_form(document, settings):
    # 文档设置
    # 1. 读取本机路径或上传文件，以及前几页或全文的选择。
    # 2. 点击应用后检查文档，保存所有会话共用的设置。
    # 3. 刷新管理页，新提问使用新设置，已提交的请求继续使用原资料。
    st.subheader("文档与读取范围")
    st.caption("上传文件时优先使用上传的 PDF；修改后请点击“应用文档和范围”。")
    with st.form("knowledge_base_document"):
        pdf_text = st.text_input("本机 PDF 路径", value=document["pdf_path"])
        upload = st.file_uploader("或选择并上传 PDF", type=["pdf"])
        scope = st.radio("读取范围", ["前 N 页", "全文"],
                         index=1 if document["max_pages"] is None else 0)
        pages = st.number_input("读取页数", min_value=1,
                                value=document["max_pages"] or settings.max_pages or DEFAULT_MAX_PAGES,
                                step=1, help="选择全文时忽略此项。")
        apply = st.form_submit_button("应用文档和范围", key="apply_document")
    if apply:
        try:
            selected_pdf = save_uploaded_pdf(upload, settings.document_dir) if upload else pdf_text
            selected = prepare_document(
                selected_pdf, None if scope == "全文" else int(pages), settings.pdf_path,
                settings.index_path, settings.index_dir)
            save_active_document(settings, selected)
        except Exception as error:
            st.error(failure_message(error, "document"))
        else:
            st.session_state["knowledge_settings_saved"] = True
            st.rerun()


def run_knowledge_page(settings):
    # 管理页面
    # 1. 读取当前知识库设置，损坏时仍提供表单用于修复。
    # 2. 在主页面显示文档设置、保存结果和当前知识库状态。
    st.title("知识库管理")
    st.caption("这里的文档和读取范围供所有会话共用，修改后对新提问生效。")
    try:
        document = load_active_document(settings)
    except (OSError, ValueError) as error:
        st.error(failure_message(error, "document"))
        document = {"pdf_path": str(settings.pdf_path), "index_path": str(settings.index_path),
                    "max_pages": settings.max_pages}
    if st.session_state.pop("knowledge_settings_saved", False):
        st.success("文档和读取范围已保存，新提问将使用当前知识库。")
    render_document_form(document, settings)
    st.divider()
    render_knowledge_base_status(document, settings)


if __name__ == "__main__":
    run_knowledge_page(st.session_state["app_settings"])
