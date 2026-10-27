"""终端问答入口，不保存网页会话。"""
import embedding
import rag_service


def run_cli(settings, input_fn=input, output_fn=print, model_loader=None, answer_fn=None):
    # 终端问答
    # 1. 根据共享配置加载模型，再读取一个问题。
    # 2. 调用与网页相同的问答服务。
    # 3. 输出回答，保留原来的提示文字。
    model = model_loader() if model_loader else embedding.load_embedding_model(settings.embedding_model)
    question = input_fn("请输入要查询的文本：")
    answer = answer_fn or rag_service.answer_question
    result = answer(road=settings.pdf_path, index_path=settings.index_path,
                    text_input=question, model=model, max_pages=settings.max_pages,
                    force_rebuild=settings.force_rebuild)
    output_fn("最终回答：")
    output_fn(result["answer"])
    return result
