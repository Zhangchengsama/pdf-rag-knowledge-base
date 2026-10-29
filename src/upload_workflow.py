"""批量上传的页面辅助流程，逐份校验并复用原文档保存接口。"""
from pathlib import Path

import pymupdf
from document_settings import save_uploaded_pdf


def save_upload_batch(uploads, directory):
    # 批量上传
    # 1. 逐份验证文件名和 PDF 内容，损坏文件不落盘。
    # 2. 沿用内容摘要目录及原来的文件复用规则。
    # 3. 收集每份结果，单个失败不影响后续文件，不更改当前知识库设置。
    results = []
    for upload in uploads:
        name = Path(upload.name).name
        try:
            if Path(name).suffix.lower() != '.pdf':
                raise ValueError('请选择 PDF 文件。')
            content = upload.getvalue()
            if not content:
                raise ValueError('上传的 PDF 为空。')
            try:
                with pymupdf.open(stream=content, filetype='pdf') as pdf:
                    if pdf.needs_pass:
                        raise ValueError('PDF 需要密码，当前无法读取。')
                    if pdf.page_count < 1:
                        raise ValueError('PDF 没有可读取的页面。')
            except (OSError, RuntimeError) as error:
                raise ValueError('PDF 无法打开，请检查文件是否损坏。') from error
            path = save_uploaded_pdf(upload, directory)
        except Exception as error:
            results.append({'name': name, 'status': 'failed', 'path': None, 'message': str(error)})
        else:
            results.append({'name': name, 'status': 'success', 'path': str(path.resolve()),
                            'message': '已加入文档列表；已有相同文件时复用。'})
    return results
