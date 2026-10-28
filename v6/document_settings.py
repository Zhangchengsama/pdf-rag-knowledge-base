"""验证文档选择、保存上传文件，并读写所有会话共用的知识库设置。"""
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from get_pdf import get_pdf_page_count


def normalize_document(document):
    # 检查设置
    # 1. 检查文档路径、索引路径和读取范围的基本格式。
    # 2. 返回独立的设置字典，不在这里读取 PDF 或创建缓存。
    if not isinstance(document, dict):
        raise ValueError('知识库设置格式错误，请在知识库管理页重新应用文档和范围。')
    paths = [document.get('pdf_path'), document.get('index_path')]
    limit = document.get('max_pages')
    if ('max_pages' not in document or any(not isinstance(path, str) or not path.strip() for path in paths)
            or (limit is not None and (type(limit) is not int or limit <= 0))):
        raise ValueError('知识库设置格式错误，请在知识库管理页重新应用文档和范围。')
    return {'pdf_path': str(Path(paths[0]).expanduser().resolve()),
            'index_path': str(Path(paths[1]).expanduser().resolve()), 'max_pages': limit}


def load_active_document(settings):
    # 读取知识库
    # 1. 读取共用设置；从未保存过时使用配置中的默认文档和范围。
    # 2. 检查设置格式，损坏时明确提示，不悄悄改用其他资料。
    try:
        document = json.loads(settings.knowledge_settings_path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {'pdf_path': str(settings.pdf_path), 'index_path': str(settings.index_path),
                'max_pages': settings.max_pages}
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError('知识库设置无法读取，请在知识库管理页重新应用文档和范围。') from error
    return normalize_document(document)


def save_active_document(settings, document):
    # 保存知识库
    # 1. 检查设置，创建保存目录。
    # 2. 先写入独立临时文件，再完整替换正式设置，避免其他页面读到半份内容。
    document = normalize_document(document)
    path = settings.knowledge_settings_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.stem}.{uuid4().hex}.tmp')
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    return document


def prepare_document(pdf_path, max_pages, default_pdf, default_index, cache_directory):
    # 选择文档
    # 1. 检查 PDF 路径和读取页数。
    # 2. 默认教材沿用原索引，其他 PDF 按路径分配独立索引。
    # 3. 返回文档设置，不在这里创建知识库。
    pdf = Path(pdf_path).expanduser().resolve()
    if pdf.suffix.lower() != '.pdf':
        raise ValueError('请选择 PDF 文件')
    if not pdf.is_file():
        raise FileNotFoundError(f'找不到 PDF：{pdf}')
    get_pdf_page_count(pdf, max_pages=max_pages)
    if pdf == Path(default_pdf).resolve():
        index = Path(default_index).resolve()
    else:
        identity = hashlib.sha256(str(pdf).casefold().encode('utf-8')).hexdigest()[:24]
        index = Path(cache_directory).resolve() / f'{identity}.bin'
    return {'pdf_path': str(pdf), 'index_path': str(index), 'max_pages': max_pages}


def save_uploaded_pdf(uploaded, directory):
    # 保存上传
    # 1. 检查文件名和内容，按内容摘要确定保存目录。
    # 2. 已有文件直接复用。
    # 3. 新文件先完整写入临时位置，再发布为正式 PDF。
    name = Path(uploaded.name).name
    if Path(name).suffix.lower() != '.pdf':
        raise ValueError('请选择 PDF 文件')
    content = uploaded.getvalue()
    if not content:
        raise ValueError('上传的 PDF 为空')
    identity = hashlib.sha256(content).hexdigest()
    path = Path(directory) / identity / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.uploading')
        temporary.write_bytes(content)
        temporary.replace(path)
    return path
