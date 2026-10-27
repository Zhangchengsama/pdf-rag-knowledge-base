import json
from embedding import get_embedding , get_similarity

def query_text_top_k(text_input, output_path, model, k=5):
    """查询文本与已保存的PDF文本数据的相似度
    text_input: 用户输入的文本字符串
    output_path: 已保存的PDF文本数据的JSON文件路径
    model: 用于生成嵌入的模型
    k: 返回最相似的k个结果
    return: 返回最相似的文本内容和相似度分数
    """

    results = []

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    input_embedding = get_embedding(text_input,model)

    for item in data:
        similarity = get_similarity(input_embedding, item["embedding"])
        results.append((similarity, item))

    results.sort(key=lambda x: x[0], reverse=True)
    top_k_results = results[:k]

    if top_k_results:
        return [item['text'] for _, item in top_k_results], [similarity for similarity, _ in top_k_results]
    else:
        return None, 0



