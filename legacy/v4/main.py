import time
from get_pdf import get_pdf_text, save_to_json
from dense import save_np_embedding ,save_dense_index
from sparse import build_bm25, save_bm25 ,load_bm25 ,build_corpus
from query import RRF_top_k_query_indece, top_k_query_indice2text
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
    if not road.with_suffix(".npy").exists():
        print("未检测到 embedding, 开始建 embedding")
        save_np_embedding(output_path, Model)

    if not Path(index_path).exists():
        print("未检测到dense索引，开始建索引")
        save_dense_index(output_path, index_path, Model)

    if not road.with_suffix(".pkl").exists():
        print("未检测到BM25模型，开始建BM25模型")
        corpus = build_corpus(output_path)
        bm25 = build_bm25(corpus)

        save_bm25(bm25, road)
    else:
        bm25 = load_bm25(road)
else:
    print("未检测到json文件，开始建库")

    data= get_pdf_text(road)
    output_path = save_to_json(data, road)
    save_np_embedding(output_path, Model)
    save_dense_index(output_path, index_path, Model)
    
    bm25 = build_bm25(output_path)
    save_bm25(bm25, road)

text_input = str(input("请输入要查询的文本："))
t2 = time.perf_counter()
top_k_indices, top_k_scores = RRF_top_k_query_indece(text_input, index_path, output_path, Model, bm25, c=60, candidate_k=20, final_k=5, ef=50, space="cosine")
result_text = top_k_query_indice2text(top_k_indices, output_path)
t3 = time.perf_counter()


# if result_text:
#     for i, (text, sim) in enumerate(zip(result_text, top_k_scores), start=1):
#         print(f"Top {i}:")
#         print(f"文本内容: {text}")
#         print(f"相似度: {sim:.4f}")
#         print("-" * 40)
# else:
#     print("未找到相似的文本。")

# print(f"查询用时 {t3 - t2:.2f}s")

from prompt import get_prompt
#print(get_prompt(top_k_indices, output_path))

prompt = get_prompt(
    text_input,
    top_k_indices,
    output_path
)

from LLM import ask_llm

answer = ask_llm(prompt)

print("最终回答：")
print(answer)