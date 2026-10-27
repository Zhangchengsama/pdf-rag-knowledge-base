"""从原有 JSONL 只读生成会话列表，不改文件名或消息格式。"""
import json
from pathlib import Path


def _read_title(path, title_length):
    """取首条非空用户消息；损坏或无标题的记录使用默认名称。"""
    try:
        with path.open('r', encoding='utf-8') as stream:
            for line in stream:
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict) or message.get('role') != 'user':
                    continue
                content = message.get('content')
                if isinstance(content, str):
                    title = ' '.join(content.split())
                    if title:
                        return title[:title_length] + ('…' if len(title) > title_length else '')
    except (OSError, UnicodeError):
        pass
    return '未命名对话'


def list_conversations(directory, title_length=24):
    """按最近修改时间倒序列出会话，同时间以 ID 排序；ID 始终为原文件名。"""
    conversations = []
    for path in Path(directory).glob('*.jsonl'):
        try:
            modified_ns = path.stat().st_mtime_ns
        except FileNotFoundError:
            continue  # 列目录后文件恰好被移走，下次刷新再更新列表。
        conversations.append({
            'id': path.stem,
            'title': _read_title(path, title_length),
            'modified_ns': modified_ns,
        })
    return sorted(conversations, key=lambda item: (-item['modified_ns'], item['id']))
