import time
from query import query_text_top_k
from get_pdf import get_pdf_text, save_to_json
from index import save_np_embedding, save_index
from pathlib import Path
from sentence_transformers import SentenceTransformer

t0 = time.perf_counter()

road=Path(r"C:\Users\36143\Desktop\教材资料\化学\普通高中教科书·化学必修 第一册.pdf")
output_path = road.with_suffix(".json")
index_path = r"C:\Users\36143\Desktop\Python_work\index\test_index.bin"
Model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    local_files_only=True
    )

print("正在加载 embedding 模型...")

print(f"模型加载完成，用时 {time.perf_counter() - t0:.2f}s")

if output_path.exists():
    print("检测到已有索引，跳过 PDF 解析和 embedding")
    if not road.with_suffix(".npy").exists():
        print("未检测到 embedding, 开始建 embedding")
        save_np_embedding(output_path, Model)

    if not Path(index_path).exists():
        print("未检测到索引，开始建索引")
        save_index(output_path, index_path, Model)
else:
    print("未检测到索引，开始建库")

    data= get_pdf_text(road)
    output_path = save_to_json(data, road)
    save_np_embedding(output_path, Model)
    save_index(output_path, index_path, Model)

text_input = str(input("请输入要查询的文本："))
t2 = time.perf_counter()
result_text, similarity = query_text_top_k(text_input, index_path, output_path, Model, k=5)
t3 = time.perf_counter()


if result_text:
    for i, (text, sim) in enumerate(zip(result_text, similarity), start=1):
        print(f"Top {i}:")
        print(f"文本内容: {text}")
        print(f"相似度: {sim:.4f}")
        print("-" * 40)
else:
    print("未找到相似的文本。")

print(f"查询用时 {t3 - t2:.2f}s")