"""rag2.py - 基于 Milvus 日记数据的 RAG 示例（使用 milvus_utils2 标准版）"""
import asyncio

from milvus_utils2 import (
    DIARY_COLLECTION_NAME,
    create_chat_model,
    create_embeddings,
    create_milvus_client,
    ensure_diary_collection,
    format_search_results,
    load_collection,
)


async def retrieve_relevant_diaries(question: str, k: int = 2) -> list[dict]:
    client = create_milvus_client()
    embeddings = create_embeddings()
    query_vector = await embeddings.aembed_query(question)
    search_result = client.search(
        collection_name=DIARY_COLLECTION_NAME,
        data=[query_vector],
        limit=k,
        output_fields=["id", "content", "date", "mood", "tags"],
        search_params={"metric_type": "COSINE", "params": {}},
    )
    return format_search_results(search_result[0])


async def answer_diary_question(question: str, k: int = 2) -> str:
    model = create_chat_model(temperature=0.7)
    print("=" * 80)
    print(f"问题: {question}")
    print("=" * 80)

    print("\n【检索相关日记】")
    retrieved_diaries = await retrieve_relevant_diaries(question, k)
    if not retrieved_diaries:
        print("未找到相关日记")
        return "抱歉，我没有找到相关的日记内容。"

    for index, diary in enumerate(retrieved_diaries, start=1):
        print(f"\n[日记 {index}] 相似度: {diary['score']:.4f}")
        print(f"日期: {diary['date']}")
        print(f"心情: {diary['mood']}")
        print(f"标签: {', '.join(diary.get('tags', []))}")
        print(f"内容: {diary['content']}")

    context = "\n\n━━━━━\n\n".join(
        f"[日记 {index}]\n日期: {diary['date']}\n心情: {diary['mood']}\n标签: {', '.join(diary.get('tags', []))}\n内容: {diary['content']}"
        for index, diary in enumerate(retrieved_diaries, start=1)
    )
    prompt = f"""你是一个温暖贴心的 AI 日记助手。基于用户的日记内容回答问题，用亲切自然的语言。

请根据以下日记内容回答问题：
{context}

用户问题: {question}

回答要求：
1. 如果日记中有相关信息，请结合日记内容给出详细、温暖的回答
2. 可以总结多篇日记的内容，找出共同点或趋势
3. 如果日记中没有相关信息，请温和地告知用户
4. 用第一人称\"你\"来称呼日记的作者
5. 回答要有同理心，让用户感到被理解和关心

AI 助手的回答:"""

    print("\n【AI 回答】")
    response = await model.ainvoke(prompt)
    print(response.content)
    print()
    return str(response.content)


async def main() -> None:
    client = create_milvus_client()
    ensure_diary_collection(client)
    load_collection(client, DIARY_COLLECTION_NAME)
    await answer_diary_question("我最近做了什么让我感到快乐的事情？", 2)


if __name__ == "__main__":
    asyncio.run(main())
