from LLM import ask_llm
from prompt import get_prompt
from get_pdf import get_pdf_text, save_to_json
from dense import save_np_embedding ,save_dense_index
from sparse import build_bm25, save_bm25 ,load_bm25 ,build_corpus
from query import RRF_top_k_query_indece ,top_k_query_indice2chunk
from pathlib import Path

def building_data(road,
                index_path,
                model
                  ):
    road=Path(road)
    output_path = road.with_suffix(".json")

    if output_path.exists():
        if not road.with_suffix(".npy").exists():
            print("未检测到 embedding, 开始建 embedding")
            save_np_embedding(output_path= output_path, model= model)

        if not Path(index_path).exists():
            print("未检测到dense索引，开始建索引")
            save_dense_index(output_path, index_path, model)

        if not road.with_suffix(".pkl").exists():
            print("未检测到BM25模型，开始建BM25模型")
            corpus = build_corpus(output_path)
            bm25 = build_bm25(corpus)

            save_bm25(bm25, road)
    else:
        print("未检测到json文件，开始建库")

        data= get_pdf_text(road)
        output_path = save_to_json(data, road)
        save_np_embedding(output_path, model)
        save_dense_index(output_path, index_path, model)
        
        bm25 = build_bm25(output_path)
        save_bm25(bm25, road)
    
    return 


def answer_question(road, index_path, text_input, model):

    building_data(road, index_path=index_path, model=model)

    road=Path(road)
    output_path = road.with_suffix(".json")

    bm25 = load_bm25(road)

    top_k_indices, top_k_scores = RRF_top_k_query_indece(text_input, index_path, output_path, model, bm25, c=60, candidate_k=20, final_k=5, ef=50, space="cosine")

    prompt = get_prompt(
    text_input,
    top_k_indices,
    output_path
    )

    chunks =top_k_query_indice2chunk(top_k_indices= top_k_indices, output_path=output_path)

    answer = ask_llm(prompt)
    result = {"answer":answer,
              "sources":chunks
              }

    return result
