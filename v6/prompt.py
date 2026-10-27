from query import top_k_query_indice2chunk

def get_prompt(text_input:str, top_k_indices, output_path, data=None):
    """按检索排名拼接问题、文件名、PDF 页序和文本，返回大模型提示词。

    top_k_indices 是 JSON 列表下标；output_path 指向保存文本块的 JSON。
    可选 data 为已加载的完整文本块列表；省略时沿用原文件读取方式。
    """
    chunks = top_k_query_indice2chunk(top_k_indices, output_path, data=data)
    return format_prompt(text_input, chunks)


def format_prompt(text_input, chunks):
    """用已选出的来源拼接提示词，保留原有顺序、文字和换行。"""
    # 不包含历史对话；与旧 get_prompt 生成的字符串逐字符一致。
    s =f"用户问题：{text_input}\n以下是参考资料："
    for item in chunks:
        s+=f"[{item['file_name']}]\n"
        s+=f"页码：{item['page_number']}\n"
        s+=f"文本：{item['text']}\n"
        s+="\n"

    return s
