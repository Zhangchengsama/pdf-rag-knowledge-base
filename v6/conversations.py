"""保留 JSONL 消息格式，额外元数据保存生成标题、文档绑定和处理状态。"""
import json
from pathlib import Path
from conversation_lock import conversation_lock


def metadata_path(history_path):
    return Path(history_path).with_suffix('.meta.json')


def load_metadata(history_path):
    """旧会话没有元数据时返回空字典；损坏信息不能静默覆盖。"""
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
    """同目录临时文件写完再替换，避免中断留下半份元数据。"""
    path = metadata_path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_messages(history_path):
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
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(message, ensure_ascii=False) + '\n')


def first_question(messages):
    """完整首问作为标题请求的输入，不截取成显示标题。"""
    return next((row['content'] for row in messages
                 if row.get('role') == 'user' and row.get('content', '').strip()), '')


def bind_document(history_path, document):
    with conversation_lock(history_path):
        return _bind_document(history_path, document)


def _bind_document(history_path, document):
    """有消息的会话绑定固定资料，避免切换后混用旧引用。"""
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
    """只显示已生成标题，不调用 API；旧记录显示待生成标题，ID 始终不变。"""
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
