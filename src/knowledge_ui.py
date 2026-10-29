"""知识库管理页面：上传目录、文档切换、原页和文本浏览及只读状态。"""
from datetime import datetime
import math
from pathlib import Path

import streamlit as st
import cache_store
import knowledge_view
from chat_workflow import failure_message
from config import DEFAULT_MAX_PAGES
from document_settings import (load_active_document, prepare_document,
                               save_active_document)
from upload_workflow import save_upload_batch


def size_text(size):
    if size is None:
        return '无法确定'
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024 or unit == 'GB':
            return f'{size:.1f} {unit}' if unit != 'B' else f'{size} B'
        size /= 1024


def time_text(nanoseconds):
    return datetime.fromtimestamp(nanoseconds / 1e9).strftime('%Y-%m-%d %H:%M') if nanoseconds else '—'


@st.cache_data(show_spinner=False, max_entries=8, ttl=300)
def cached_preview(pdf_path, page_number, signature):
    # 预览缓存
    # 1. 路径、页序和文件版本共同决定缓存，最多保存八页。
    return knowledge_view.render_pdf_page(pdf_path, page_number)


@st.cache_data(show_spinner=False, max_entries=4, ttl=300)
def cached_chunks(text_path, signature):
    # 文本缓存
    # 1. 文件版本变化后重新读取，避免筛选时反复解析同一 JSON。
    return cache_store.load_chunks(text_path)


def render_knowledge_base_status(document, settings):
    # 知识库状态
    # 1. 只读检查当前文档和缓存。
    # 2. 显示读取范围及复用状态。
    # 3. 缓存版本匹配时复用已知块数，否则显示未统计。
    overview = knowledge_view.document_overview(document)
    state = overview['state']
    st.subheader("当前知识库")
    st.markdown(f'**{Path(document["pdf_path"]).name}**')
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
    with st.container(horizontal=True):
        st.metric('PDF 总页数', overview['total_pages'] if overview['total_pages'] is not None else '—')
        st.metric('当前读取页数', pages if pages is not None else '—')
        st.metric('PDF 大小', size_text(overview['pdf_bytes']))
        st.metric('缓存占用', size_text(overview['cache_bytes']))
    details = st.session_state.get("knowledge_base_details", {})
    text_stats = st.session_state.get('knowledge_text_stats', {})
    count_slot = st.empty()
    if state["status"] == "ready" and details.get("cache_signature") == state["cache_signature"]:
        count_slot.write(f"文本块数：{details['chunk_count']}")
        st.caption("最近一次提问：" + ("重建知识库" if details["rebuilt"] else "复用已有缓存"))
    elif state['status'] == 'ready' and text_stats.get('signature') == state['cache_signature']:
        count_slot.write(f"文本块数：{text_stats['count']}")
    else:
        count_slot.write("文本块数：未统计")
    if settings.force_rebuild:
        st.caption("当前已开启强制重建，下一次提问将重新建库。")
    return overview, count_slot


def document_saved():
    # 保存反馈
    # 1. 清除上一次上传控件的选择，防止后续应用时覆盖刚切换的文档。
    st.session_state['document_form_revision'] = st.session_state.get('document_form_revision', 0) + 1
    st.session_state['knowledge_settings_saved'] = True
    st.session_state.pop('source_preview_document', None)
    st.rerun()


def render_uploaded_documents(document, settings):
    # 文档列表
    # 1. 从新上传目录读取清单，并清楚标记当前资料。
    # 2. 使用完整路径选择，同名文件互不混淆。
    # 3. 点击切换时保留已保存的读取范围，复用原校验和持久化接口。
    st.subheader('已上传文档')
    try:
        entries = knowledge_view.list_uploaded_documents(settings.document_dir)
    except OSError as error:
        st.error(failure_message(error, 'document'))
        return False
    if not entries:
        st.info('还没有上传的 PDF。可在下方上传文件，文件会统一保存在上传目录。')
        return False
    current = str(Path(document['pdf_path']).resolve())
    st.dataframe([{'文件名': row['name'], '大小': size_text(row['size_bytes']),
                   '修改时间': time_text(row['modified_ns']), '当前使用': row['path'] == current}
                  for row in entries], hide_index=True, width='stretch', height=min(280, 38 + 36 * len(entries)))
    by_path = {row['path']: row for row in entries}
    if st.session_state.get('catalog_active_path') != current:
        st.session_state['catalog_active_path'] = current
        if current in by_path:
            st.session_state['uploaded_document'] = current
        else:
            st.session_state.pop('uploaded_document', None)
    def label(path):
        row = by_path[path]
        return f"{row['name']} · {size_text(row['size_bytes'])} · {Path(path).parent.name[:8]}"
    selected = st.selectbox('选择已上传文档', list(by_path), format_func=label, key='uploaded_document',
                            help='可输入文件名搜索；同名文件附带各自的资料标识。')
    with st.container(horizontal=True):
        switch = st.button('设为当前文档', key='use_uploaded_document', disabled=selected == current)
        st.button('刷新列表', key='refresh_uploaded_documents')
    if current not in by_path:
        st.caption('当前资料来自本机路径；列表显示上传目录中的文件。')
    if switch:
        try:
            selected_document = prepare_document(selected, document['max_pages'], settings.pdf_path,
                                                  settings.index_path, settings.index_dir)
            save_active_document(settings, selected_document)
        except Exception as error:
            st.error(failure_message(error, 'document'))
        else:
            document_saved()
    return True


def render_batch_upload(settings):
    # 上传表单
    # 1. 一次接收多份 PDF，通过已有保存接口逐份处理。
    # 2. 保持当前资料和读取范围，保存结果后刷新列表并清空上传选择。
    st.subheader('批量上传 PDF')
    st.caption('上传成功的文件会加入上方列表，可点击“设为当前文档”使用；当前资料和范围保持原设置。')
    revision = st.session_state.get('document_form_revision', 0)
    with st.form('upload_documents_form'):
        uploads = st.file_uploader('选择 PDF（可多选）', type=['pdf'], accept_multiple_files=True,
                                   key=f'pdf_upload_{revision}')
        submit = st.form_submit_button('上传所选 PDF', key='upload_documents')
    if submit:
        if not uploads:
            st.warning('请先选择要上传的 PDF。')
        else:
            st.session_state['last_upload_results'] = save_upload_batch(uploads, settings.document_dir)
            st.session_state['document_form_revision'] = revision + 1
            st.rerun()


def render_upload_results():
    # 上传反馈
    # 1. 在表单外显示汇总，保证上传后即使表单收起也能看到结果。
    results = st.session_state.get('last_upload_results')
    if results:
        success = sum(row['status'] == 'success' for row in results)
        failed = len(results) - success
        st.info(f'上传结果：成功 {success}，失败 {failed}。成功文件已加入文档列表。')
        with st.expander('查看逐份上传结果', expanded=failed > 0):
            st.dataframe([{'文件名': row['name'], '结果': '成功' if row['status'] == 'success' else '失败',
                           '说明': row['message']} for row in results], hide_index=True, width='stretch')


def render_document_form(document, settings):
    # 文档设置
    # 1. 读取本机路径，以及前几页或全文的选择。
    # 2. 点击应用后检查文档，保存所有会话共用的设置。
    # 3. 刷新管理页，新提问使用新设置，已提交的请求继续使用原资料。
    st.subheader("文档与读取范围")
    st.caption('应用本机路径或读取范围后，所有会话的新提问使用这里保存的设置。')
    with st.form("knowledge_base_document"):
        pdf_text = st.text_input("本机 PDF 路径", value=document["pdf_path"])
        scope = st.radio("读取范围", ["前 N 页", "全文"],
                         index=1 if document["max_pages"] is None else 0)
        pages = st.number_input("读取页数", min_value=1,
                                value=document["max_pages"] or settings.max_pages or DEFAULT_MAX_PAGES,
                                step=1, help="选择全文时忽略此项。")
        apply = st.form_submit_button("应用文档和范围", key="apply_document")
    if apply:
        try:
            selected = prepare_document(
                pdf_text, None if scope == "全文" else int(pages), settings.pdf_path,
                settings.index_path, settings.index_dir)
            save_active_document(settings, selected)
        except Exception as error:
            st.error(failure_message(error, "document"))
        else:
            document_saved()


def open_pdf_page(document, page_number):
    # 来源跳转
    # 1. 在控件回调中设置目标页和内容视图，不改写知识库设置。
    st.session_state['source_preview_document'] = dict(document)
    st.session_state[f"preview_page_{document['pdf_path']}"] = page_number
    st.session_state['knowledge_view'] = 'PDF 原页预览'


def render_pdf_preview(document, overview):
    # 原页视图
    # 1. 按 PDF 页序跳转，按需渲染并复用当前页图片。
    # 2. 明示预览页是否位于当前读取范围，预览不会改变范围。
    total = overview['total_pages']
    if not total:
        st.warning('无法预览当前 PDF，请检查文件和访问权限。')
        return
    preview_key = f"preview_page_{document['pdf_path']}"
    if preview_key in st.session_state:
        st.session_state[preview_key] = min(total, max(1, st.session_state[preview_key]))
    page = st.number_input('PDF 页序', min_value=1, max_value=total, step=1, key=preview_key)
    st.caption(f'第 {page} / {total} 页 · 使用 PDF 页序，可能与书内印刷页码不同。')
    covered = overview['state']['actual_pages']
    if covered is not None and page > covered:
        st.info('此页不在当前读取范围内；预览不会修改知识库设置。')
    try:
        info = Path(document['pdf_path']).stat()
        picture = cached_preview(document['pdf_path'], int(page), (info.st_mtime_ns, info.st_size))
        st.image(picture, caption=f"{Path(document['pdf_path']).name} · 第 {page} 页", width='stretch')
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f'原页预览失败：{error}')


def render_indexed_text(document, overview, count_slot):
    # 文本视图
    # 1. 仅在用户打开此视图时，取得文档锁并读取版本匹配的缓存。
    # 2. 按关键词和页码筛选，每次最多展示十块原文。
    if overview['state']['status'] != 'ready':
        st.info('当前文档尚无与读取范围匹配的完整缓存，请先提问完成建库。')
        return
    try:
        chunks, signature = knowledge_view.load_indexed_chunks(document, cached_chunks)
    except Exception as error:
        st.error(f'无法浏览已入库文本：{failure_message(error, "document")}')
        return
    st.session_state['knowledge_text_stats'] = {'signature': signature, 'count': len(chunks)}
    count_slot.write(f'文本块数：{len(chunks)}')
    keyword = st.text_input('关键词', key=f"chunk_keyword_{document['pdf_path']}",
                            help='在已入库原文中查找关键词，不调用检索模型。')
    pages = sorted({row['page_number'] for row in chunks})
    page = st.selectbox('按 PDF 页序筛选', [None, *pages],
                        format_func=lambda value: '全部页码' if value is None else f'第 {value} 页',
                        key=f'chunk_page_filter_{hash(signature)}')
    matches = knowledge_view.filter_chunks(chunks, keyword, page)
    st.caption(f'共 {len(chunks)} 个文本块，匹配 {len(matches)} 个。仅浏览当前入库范围内的原文。')
    if not matches:
        st.info('没有符合条件的文本块。')
        return
    result_page = st.number_input('文本结果页', min_value=1, max_value=max(1, math.ceil(len(matches) / 10)),
                                  value=1, step=1, key=f'chunk_result_page_{hash((signature, keyword, page))}')
    for position, row in enumerate(matches[(result_page - 1) * 10:result_page * 10],
                                    start=(result_page - 1) * 10):
        with st.container(border=True):
            with st.container(horizontal=True):
                st.caption(f"第 {row['page_number']} 页 · 文本块 {row['chunk_id']}")
                st.button('查看原页', key=f'open_pdf_page_{hash(signature)}_{position}',
                          on_click=open_pdf_page, args=(dict(document), row['page_number']),
                          disabled=not 1 <= row['page_number'] <= (overview['total_pages'] or 0))
            st.text(row['text'])


def render_status_details(document, overview):
    # 状态详情
    # 1. 展示原接口给出的错误和缺失缓存，不自动修复或建库。
    # 2. 文件修改时间只作为文件信息展示，不称为建库时间。
    st.button('刷新状态', key='refresh_knowledge_status')
    st.caption(f"当前文档：{document['pdf_path']}")
    if overview['state'].get('error'):
        st.error(overview['state']['error'])
    missing = [row['kind'] for row in overview['files'][:4] if not row['exists'] and not row['error']]
    if missing:
        st.warning('缺少缓存：' + '、'.join(missing))
    st.dataframe([{'类型': row['kind'], '状态': '无法读取' if row['error'] else '存在' if row['exists'] else '缺失',
                   '大小': size_text(row['size_bytes']) if row['exists'] else '—',
                   '修改时间': time_text(row['modified_ns']), '位置': row['path']}
                  for row in overview['files']], hide_index=True, width='stretch')
    st.caption('状态沿用现有缓存复用规则；打开页面和刷新状态不会触发建库。')


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
    overview, count_slot = render_knowledge_base_status(document, settings)
    st.divider()
    has_uploads = render_uploaded_documents(document, settings)
    render_upload_results()
    with st.expander('上传 PDF／本机路径与读取范围', expanded=not has_uploads):
        render_batch_upload(settings)
        st.divider()
        render_document_form(document, settings)
    st.divider()
    st.session_state.setdefault('knowledge_view', '状态详情')
    view = st.segmented_control('查看内容', ['状态详情', 'PDF 原页预览', '已入库文本'],
                                 key='knowledge_view', selection_mode='single',
                                 required=True)
    if view != 'PDF 原页预览':
        st.session_state.pop('source_preview_document', None)
    if view == 'PDF 原页预览':
        source = st.session_state.get('source_preview_document', document)
        if source['pdf_path'] != document['pdf_path']:
            st.info(f"来源文档：{Path(source['pdf_path']).name}；当前知识库已切换，正在只读预览原来源。")
            render_pdf_preview(source, knowledge_view.document_overview(source))
        else:
            render_pdf_preview(document, overview)
    elif view == '已入库文本':
        render_indexed_text(document, overview, count_slot)
    else:
        render_status_details(document, overview)


if __name__ == "__main__":
    run_knowledge_page(st.session_state["app_settings"])
