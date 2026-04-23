"""ebook_reader_rag.py - 基于《天龙八部》片段的 RAG 问答"""
import asyncio

from milvus_utils import EBOOK_COLLECTION_NAME, create_chat_model, create_embeddings, create_milvus_client, format_search_results, load_collection


async def retrieve_relevant_content(question: str, k: int = 3) -> list[dict]:
    """根据问题从 Milvus 中检索最相关的电子书片段。"""
    client = create_milvus_client()
    embeddings = create_embeddings()

    # 把问题转成向量，用于近邻搜索。
    query_vector = await embeddings.aembed_query(question)
    search_result = client.search(
        collection_name=EBOOK_COLLECTION_NAME,
        data=[query_vector],
        # k 表示最多取回多少个候选片段。
        limit=k,
        # 返回给上层用于展示和构造上下文的字段。
        output_fields=["id", "book_id", "chapter_num", "chunk_index", "content"],
        search_params={"metric_type": "COSINE", "params": {}},
    )
    return format_search_results(search_result[0])


async def answer_ebook_question(question: str, k: int = 3) -> str:
    """先检索，再把检索结果拼进 prompt，让模型回答问题。"""
    # Chat 模型：负责最后的自然语言回答。
    model = create_chat_model(temperature=0.7)
    print("=" * 80)
    print(f"问题: {question}")
    print("=" * 80)

    print("\n【检索相关内容】")
    retrieved_content = await retrieve_relevant_content(question, k)
    if not retrieved_content:
        print("未找到相关内容")
        return "抱歉，我没有找到相关的《天龙八部》内容。"

    for index, item in enumerate(retrieved_content, start=1):
        print(f"\n[片段 {index}] 相似度: {item['score']:.4f}")
        print(f"书籍: {item['book_id']}")
        print(f"章节: 第 {item['chapter_num']} 章")
        print(f"片段索引: {item['chunk_index']}")
        preview = item['content'][:200]
        suffix = '...' if len(item['content']) > 200 else ''
        print(f"内容: {preview}{suffix}")

    # context：把检索到的多个片段拼接成模型可读的上下文。
    context = "\n\n━━━━━\n\n".join(
        f"[片段 {index}]\n章节: 第 {item['chapter_num']} 章\n内容: {item['content']}"
        for index, item in enumerate(retrieved_content, start=1)
    )
    prompt = f"""你是一个专业的《天龙八部》小说助手。基于小说内容回答问题，用准确、详细的语言。

请根据以下《天龙八部》小说片段内容回答问题：
{context}

用户问题: {question}

回答要求：
1. 如果片段中有相关信息，请结合小说内容给出详细、准确的回答
2. 可以综合多个片段的内容，提供完整的答案
3. 如果片段中没有相关信息，请如实告知用户
4. 回答要准确，符合小说的情节和人物设定
5. 可以引用原文内容来支持你的回答

AI 助手的回答:"""

    print("\n【AI 回答】")
    response = await model.ainvoke(prompt)
    print(response.content)
    print()
    return str(response.content)


async def main() -> None:
    print("连接到 Milvus...")
    # 创建客户端只是为了确保连接参数可用。
    client = create_milvus_client()
    print("✓ 已连接\n")

    try:
        # RAG 检索前先加载集合。
        load_collection(EBOOK_COLLECTION_NAME)
        print("✓ 集合已加载\n")
    except Exception as error:
        if "already loaded" not in str(error):
            raise
        print("✓ 集合已处于加载状态\n")

    # question：用户真正提出的问题。
    await answer_ebook_question("鸠摩智会什么武功？", 5)


if __name__ == "__main__":
    asyncio.run(main())
