"""问答提示词；不加载检索模块或模型。"""
from cache_store import select_chunks


def get_prompt(text_input: str, top_k_indices, output_path, data=None):
    # 兼容提示词
    # 1. 按原来的路径和下标取得来源。
    # 2. 调用纯格式化函数，旧的三参数调用仍可使用。
    return format_prompt(text_input, select_chunks(top_k_indices, output_path, data))


def format_prompt(text_input, chunks):
    # 拼接资料
    # 1. 写入当前问题，不加入历史对话。
    # 2. 沿检索排名拼接文件名、页码和原文。
    # 3. 保留原来的文字、空行和来源顺序。
    result = f"用户问题：{text_input}\n以下是参考资料："
    for chunk in chunks:
        result += f"[{chunk['file_name']}]\n"
        result += f"页码：{chunk['page_number']}\n"
        result += f"文本：{chunk['text']}\n"
        result += "\n"
    return result
