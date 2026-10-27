from rag_service import answer_question
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 命令行入口：选定 PDF 与对应索引，再读取一个问题并输出回答。
road=Path(r"C:\Users\36143\Desktop\教材资料\化学\普通高中教科书·化学必修 第一册.pdf")
# PDF 同名 JSON 保存文本块及来源信息。
output_path = road.with_suffix(".json")
index_path = r"C:\Users\36143\Desktop\Python_work\index\test_index.bin"
max_pages = 11  # 改成 30 表示读取前 30 页；None 表示全文。
force_rebuild = False  # 同页数内容变化时设为 True；重建后改回 False 以复用缓存。
# 只使用本地缓存的嵌入模型，建库与问题编码共用此实例。
Model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    local_files_only=True
    )
text_input = str(input("请输入要查询的文本："))

# 服务层负责补齐知识库缓存、融合检索、组织提示词并调用大模型。
result= answer_question(road=road, index_path= index_path, text_input=text_input, model= Model,
                       max_pages=max_pages, force_rebuild=force_rebuild)

print("最终回答：")
print(result['answer'])
