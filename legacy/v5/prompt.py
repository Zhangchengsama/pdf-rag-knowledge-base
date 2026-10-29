from query import top_k_query_indice2chunk

def get_prompt(text_input:str, top_k_indices, output_path):
    s =f"用户问题：{text_input}\n以下是参考资料："

    chunks =top_k_query_indice2chunk(top_k_indices= top_k_indices, output_path=output_path)

    for item in chunks:
        s+=f"[{item['file_name']}]\n"
        s+=f"页码：{item['page_number']}\n"
        s+=f"文本：{item['text']}\n"
        s+="\n"

    return s
    