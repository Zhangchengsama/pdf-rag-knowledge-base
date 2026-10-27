"""管理一次提问的持久状态，重试复用原问题；标题失败不影响已保存的回答。"""
from time import perf_counter
from uuid import uuid4

from conversations import (append_message, first_question, load_metadata,
                           read_messages, save_metadata)
from progress import track_stage
from conversation_lock import (ConversationBusyError, DocumentBusyError,
                               conversation_lock, document_lock)


def failure_message(error, stage):
    # 失败提示
    # 1. 按错误种类区分文件、模型、密钥和网络问题。
    # 2. 返回用户能采取行动的提示，不展示请求凭据。
    from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                        AuthenticationError, RateLimitError)
    if isinstance(error, (ConversationBusyError, DocumentBusyError)):
        return str(error)
    if isinstance(error, PermissionError):
        return '文件无法读写，请检查 PDF、聊天目录和缓存目录的权限后重试。'
    if isinstance(error, (APITimeoutError, TimeoutError)):
        return '请求超时，请检查网络或稍后重试。'
    if isinstance(error, AuthenticationError) or 'DEEPSEEK_API_KEY' in str(error):
        return '大模型密钥未配置或验证失败，请检查 DEEPSEEK_API_KEY 后重试。'
    if isinstance(error, RateLimitError):
        return '大模型服务限流或额度不足，请检查账户并稍后重试。'
    if isinstance(error, (APIConnectionError, ConnectionError)):
        return '无法连接大模型服务，请检查网络后重试。'
    if stage == 'model':
        return '本地嵌入模型未准备好或加载失败，请检查模型缓存后重试。'
    if isinstance(error, FileNotFoundError):
        return 'PDF 或缓存文件不存在，请检查本会话绑定的文件后重试。'
    if isinstance(error, APIStatusError):
        return f'大模型服务返回错误（HTTP {error.status_code}），请稍后重试。'
    if isinstance(error, ValueError):
        return str(error)
    if stage in ('document', 'knowledge_base'):
        return 'PDF 无法解析或知识库准备失败，请检查文件完整性和文本层后重试。'
    return '处理失败，请检查文档、运行环境和服务状态后重试。'


def _new_request(question, index):
    # 创建请求
    # 1. 给问题生成请求编号，记录对应的消息位置和初始状态。
    return {'id': uuid4().hex, 'question': question, 'user_index': index,
            'status': 'created', 'attempts': 0, 'stages': {}, 'elapsed_seconds': 0}


def recover_request(history_path):
    # 恢复检查
    # 1. 尝试取得会话锁。
    # 2. 拿不到锁说明仍在处理，只读返回当前状态。
    # 3. 拿到锁后再检查真正中断的请求。
    try:
        with conversation_lock(history_path):
            return _recover_request(history_path)
    except ConversationBusyError:
        return load_metadata(history_path)


def _recover_request(history_path):
    # 恢复请求
    # 1. 读取会话信息和原消息，找出没有完成的问题。
    # 2. 把遗留阶段标为中断，无法确定的耗时留空。
    # 3. 保存可重试状态，不自动调用模型。
    metadata = load_metadata(history_path)
    messages = read_messages(history_path)
    request = metadata.get('request')
    if messages and messages[-1]['role'] == 'user':
        index = len(messages) - 1
        if not request or request.get('user_index') != index:
            request = _new_request(messages[-1]['content'], index)
            metadata['request'] = request
    dirty = False
    if request and request['status'] in ('created', 'running'):
        if request['status'] == 'running' and not request.get('elapsed_seconds'):
            request['elapsed_seconds'] = None
        request.update(status='failed', error='上次请求未完成，可以重试同一个问题。')
        dirty = True
    if request:
        for event in request.get('stages', {}).values():
            if event['status'] == 'running':
                event.update(status='failed', interrupted=True, elapsed_seconds=None)
                dirty = True
    if metadata.get('title_status') == 'running':
        metadata.update(title_status='failed', title_elapsed_seconds=None,
                        title_error='上次标题生成中断，请单独重试生成标题。')
        dirty = True
    if dirty:
        save_metadata(history_path, metadata)
    return metadata


def start_request(history_path, question):
    # 开始提问
    # 1. 先取得会话锁，再交给创建流程，防止重复提交。
    with conversation_lock(history_path):
        return _start_request(history_path, question)


def _start_request(history_path, question):
    # 保存问题
    # 1. 检查问题内容和上次未完成请求。
    # 2. 先保存请求信息，再追加一次用户消息。
    if not isinstance(question, str) or not question.strip():
        raise ValueError('请输入非空问题')
    metadata = _recover_request(history_path)
    if metadata.get('request', {}).get('status') == 'failed':
        raise ValueError('请先重试未完成的问题，或新建对话')
    request = _new_request(question, len(read_messages(history_path)))
    metadata['request'] = request
    # 先保存请求，再追加消息；中断后可按 user_index 补写，避免重复追加。
    save_metadata(history_path, metadata)
    append_message(history_path, {'role': 'user', 'content': question})
    return request


def _saved_answer(history_path, request):
    # 核对消息
    # 1. 按请求记录的位置核对原问题，缺失时补写一次。
    # 2. 如果回答已经落盘就复用，避免再次生成。
    # 3. 消息顺序不一致时提示检查，不覆盖原文件。
    messages = read_messages(history_path)
    index = request['user_index']
    expected = {'role': 'user', 'content': request['question']}
    if index == len(messages):
        append_message(history_path, expected)
        return None
    if index >= len(messages) or any(messages[index].get(key) != value for key, value in expected.items()):
        raise ValueError('聊天记录与本次请求不一致，请检查原文件或新建对话')
    if len(messages) > index + 1:
        answer = messages[index + 1]
        if answer['role'] != 'assistant':
            raise ValueError('聊天记录顺序已变化，请检查原文件或新建对话')
        return answer
    return None


def _generate_title(history_path, metadata, on_progress=None):
    # 生成标题
    # 1. 读取完整首问，独立请求大模型概括标题。
    # 2. 保存有效标题；失败只记录标题错误，不回滚回答。
    # 3. 记录本次标题生成耗时。
    if metadata.get('title'):
        return
    question = first_question(read_messages(history_path))
    if not question:
        raise ValueError('还没有首个问题，无法生成标题')
    metadata.update(title_status='running', title_error=None)
    save_metadata(history_path, metadata)
    started = perf_counter()
    try:
        with track_stage('title', on_progress):
            from LLM import generate_title
            title = generate_title(question)
            if not isinstance(title, str) or not title.strip():
                raise ValueError('大模型没有返回有效标题，请重试生成标题')
            title = ' '.join(title.split())[:24]
            if title == ' '.join(question.split()):
                raise ValueError('大模型直接复述了问题，请重试生成概括标题')
            metadata.update(title=title, title_status='complete', title_error=None)
    except Exception as error:
        metadata.update(title_status='failed', title_error=failure_message(error, 'title'))
    metadata['title_elapsed_seconds'] = perf_counter() - started
    save_metadata(history_path, metadata)


def generate_session_title(history_path, on_progress=None):
    # 重试标题
    # 1. 取得会话锁，只处理标题，不重新生成回答。
    with conversation_lock(history_path):
        metadata = load_metadata(history_path)
        _generate_title(history_path, metadata, on_progress)
        return metadata


def run_request(history_path, model_loader, on_progress=None, on_build_status=None,
                force_rebuild=False):
    # 处理请求
    # 1. 取得会话锁后运行一次处理，阻止其他页面重复执行。
    with conversation_lock(history_path):
        return _run_request(history_path, model_loader, on_progress, on_build_status, force_rebuild)


def _run_request(history_path, model_loader, on_progress, on_build_status, force_rebuild):
    # 执行问答
    # 1. 先核对已有回答，再准备本次尝试的状态。
    # 2. 加载模型，锁定共享文档并调用问答服务。
    # 3. 保存回答，再按需生成标题；失败保留原问题供重试。
    # 4. 保存本次总耗时。
    metadata = load_metadata(history_path)
    request = metadata.get('request')
    if not request:
        raise ValueError('没有可以处理的提问')
    saved = _saved_answer(history_path, request)
    if saved is not None:
        # 覆盖“回答已落盘，但完成状态写入前中断”的窗口。
        request.update(status='complete', error=None)
        save_metadata(history_path, metadata)
        if metadata.get('title_status') not in ('failed', 'running'):
            _generate_title(history_path, metadata, on_progress)
        return metadata
    if request['status'] == 'complete':
        raise ValueError('已完成请求的回答记录缺失，请检查原文件')
    started = perf_counter()
    request.update(status='running', attempts=request['attempts'] + 1,
                   error=None, stages={}, elapsed_seconds=0)
    save_metadata(history_path, metadata)

    def report(event):
        # 记录进度
        # 1. 把阶段状态及耗时写入会话信息。
        # 2. 如果页面提供回调，再同步更新显示。
        request['stage'] = event['stage']
        request['stages'][event['stage']] = dict(event)
        save_metadata(history_path, metadata)
        if on_progress is not None:
            on_progress(event)

    try:
        document = metadata.get('document')
        if not document:
            raise ValueError('请先确认并绑定本会话的 PDF 和读取范围')
        with track_stage('model', report):
            model = model_loader()
        from rag_service import answer_question
        # 不同会话可能使用同一 PDF 的不同页数，共用派生文件必须串行访问。
        with document_lock(document['pdf_path']):
            result = answer_question(
                document['pdf_path'], document['index_path'], request['question'], model,
                max_pages=document['max_pages'], force_rebuild=force_rebuild, on_progress=report,
                on_build_status=on_build_status)
        append_message(history_path, {'role': 'assistant', 'content': result['answer'],
                                      'sources': result['sources']})
        request.update(status='complete', error=None)
        save_metadata(history_path, metadata)
        if metadata.get('title_status') not in ('failed', 'running'):
            _generate_title(history_path, metadata, report)
    except Exception as error:
        request.update(status='failed', error=failure_message(error, request.get('stage')))
    finally:
        request['elapsed_seconds'] = perf_counter() - started
        save_metadata(history_path, metadata)
    return metadata
