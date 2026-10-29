"""复用 filelock 的操作系统锁，区分活跃请求和真正中断，不删除锁文件。"""
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout


class ConversationBusyError(RuntimeError):
    pass


class DocumentBusyError(RuntimeError):
    pass


@contextmanager
def conversation_lock(history_path):
    # 锁定会话
    # 1. 为当前会话取得独占锁，结束时释放。
    path = Path(history_path).resolve().with_suffix('.lock')
    with _file_lock(path, ConversationBusyError, '本会话正在处理，请稍后刷新状态，不要重复提交。'):
        yield


@contextmanager
def document_lock(pdf_path):
    # 锁定文档
    # 1. 为同一 PDF 的共享缓存取得独占锁，结束时释放。
    pdf = Path(pdf_path).resolve()
    path = pdf.with_suffix(pdf.suffix + '.rag.lock')
    with _file_lock(path, DocumentBusyError, '该文档正在被另一个请求使用，请稍后重试。'):
        yield


@contextmanager
def _file_lock(path, error_type, message):
    # 文件互斥
    # 1. 创建或复用操作系统文件锁，保留锁文件。
    # 2. 已被占用时立即提示忙碌，不排队重发请求。
    # 3. 处理结束或抛出异常时释放锁。
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path), preserve_lock_file=True, fallback_to_soft=False)
    try:
        lock.acquire(timeout=0)
    except Timeout as error:
        raise error_type(message) from error
    try:
        yield
    finally:
        lock.release()
