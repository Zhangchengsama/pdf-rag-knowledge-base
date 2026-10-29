"""保留 JSONL 消息格式，保存标题与请求状态，并兼容旧会话的文档字段。"""
import json
from pathlib import Path
from conversation_lock import conversation_lock


def metadata_path(history_path):
    # 信息位置
    # 1. 由聊天文件名得到旁边的会话信息文件。
    return Path(history_path).with_suffix('.meta.json')


def load_metadata(history_path):
    # 读取信息
    # 1. 旧记录没有信息文件时返回空字典。
    # 2. 读取已保存的信息，损坏时明确报错，避免覆盖。
    try:
        data = json.loads(metadata_path(history_path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError('会话信息损坏，请检查对应的 .meta.json 文件') from error
    if not isinstance(data, dict):
        raise ValueError('会话信息格式错误')
    return data


def save_metadata(history_path, metadata):
    # 保存信息
    # 1. 先把完整信息写入同目录临时文件。
    # 2. 写完再替换正式文件，避免半份记录。
    path = metadata_path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_messages(history_path):
    # 读取记录
    # 1. 逐行读取原 JSONL，忽略空行。
    # 2. 检查消息基本格式，并按原顺序返回。
    # 3. 遇到损坏内容时给出行号，不修改原文件。
    path = Path(history_path)
    if not path.exists():
        return []
    messages = []
    with path.open('r', encoding='utf-8') as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'聊天记录第 {number} 行无法读取，请检查原文件') from error
            if (not isinstance(message, dict) or not isinstance(message.get('role'), str)
                    or not isinstance(message.get('content'), str)):
                raise ValueError(f'聊天记录第 {number} 行格式错误')
            messages.append(message)
    return messages


def append_message(history_path, message):
    # 追加消息
    # 1. 创建需要的目录，把一条消息追加为一行 JSON。
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(message, ensure_ascii=False) + '\n')


def first_question(messages):
    # 取得首问
    # 1. 返回首条非空用户问题的完整内容，供标题提示词使用。
    return next((row['content'] for row in messages
                 if row.get('role') == 'user' and row.get('content', '').strip()), '')


def bind_document(history_path, document):
    # 兼容旧调用
    # 1. 取得会话锁，保留旧业务调用的写入方式；网页已不再调用此函数。
    with conversation_lock(history_path):
        return _bind_document(history_path, document)


def _bind_document(history_path, document):
    # 保存绑定
    # 1. 检查读取页数。
    # 2. 已有消息时保留原绑定，换资料需要新建会话。
    # 3. 把设置保存到额外信息，不修改聊天消息。
    metadata = load_metadata(history_path)
    limit = document['max_pages']
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError('读取页数必须为正整数或全文')
    if (read_messages(history_path) and metadata.get('document')
            and metadata['document'] != document):
        raise ValueError('本会话已绑定文档和读取范围，请新建对话后更换')
    metadata['document'] = document
    save_metadata(history_path, metadata)
    return metadata


def list_conversations(directory, title_length=24):
    # 对话列表
    # 1. 遍历聊天文件，读取已保存的标题。
    # 2. 旧会话显示待生成标题，浏览列表不请求模型。
    # 3. 按聊天文件修改时间排序，同时间再按 ID 排序。
    conversations = []
    for path in Path(directory).glob('*.jsonl'):
        try:
            modified_ns = path.stat().st_mtime_ns
        except FileNotFoundError:
            continue  # 列目录后文件恰好被移走，下次刷新再更新列表。
        try:
            title = load_metadata(path).get('title', '')
            title = ' '.join(title.split()) if isinstance(title, str) else ''
            title = title[:title_length] + ('…' if len(title) > title_length else '')
        except (OSError, ValueError):
            title = '会话信息无法读取'
        conversations.append({
            'id': path.stem,
            'title': title or '待生成标题',
            'modified_ns': modified_ns,
        })
    return sorted(conversations, key=lambda item: (-item['modified_ns'], item['id']))
