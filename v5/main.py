from rag_service import answer_question
from pathlib import Path
from sentence_transformers import SentenceTransformer

road=Path(r"C:\Users\36143\Desktop\教材资料\化学\普通高中教科书·化学必修 第一册.pdf")
output_path = road.with_suffix(".json")
index_path = r"C:\Users\36143\Desktop\Python_work\index\test_index.bin"
Model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    local_files_only=True
    )
text_input = str(input("请输入要查询的文本："))

result= answer_question(road=road, index_path= index_path, text_input=text_input, model= Model)

print("最终回答：")
print(result['answer'])