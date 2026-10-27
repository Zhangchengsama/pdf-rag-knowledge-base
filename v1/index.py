import json
from embedding import get_embedding

def save_embedding(output_path, model):
    data = []

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["embedding"] = get_embedding(item["text"], model).tolist()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        

