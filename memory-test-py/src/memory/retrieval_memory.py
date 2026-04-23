"""retrieval_memory.py - 使用 Milvus 检索语义相关历史对话"""
from datetime import datetime

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage

from src.memory_utils import (
    CONVERSATION_COLLECTION_NAME,
    create_chat_model,
    create_embeddings,
    create_milvus_client,
    format_search_results,
    load_collection,
)


# 用于生成最终回复的聊天模型。
model = create_chat_model(temperature=0.0)


async def retrieve_relevant_conversations(query: str, k: int = 2) -> list[dict]:
    """根据当前问题，从 Milvus 检索语义最相关的历史对话。"""
    client = create_milvus_client()
    embeddings = create_embeddings()
    query_vector = await embeddings.aembed_query(query)
    search_result = client.search(
        collection_name=CONVERSATION_COLLECTION_NAME,
        data=[query_vector],
        limit=k,
        output_fields=["id", "content", "round", "timestamp"],
        search_params={"metric_type": "COSINE", "params": {}},
    )
    return format_search_results(search_result[0])


async def retrieval_memory_demo() -> None:
    client = create_milvus_client()
    try:
        print("连接到 Milvus...")
        # 检索前需要先加载集合。
        load_collection(CONVERSATION_COLLECTION_NAME)
        print("✓ 已连接\n")
    except Exception as error:
        print(f"❌ 无法连接到 Milvus: {error}")
        print("请确保 Milvus 服务正在运行（localhost:19530）")
        return

    # 这里的 history 只保存本次进程内的对话。
    history = InMemoryChatMessageHistory()
    embeddings = create_embeddings()
    conversations = [
        {"input": "我之前提到的机器学习项目进展如何？"},
        {"input": "我周末经常做什么？"},
        {"input": "我的职业是什么？"},
    ]

    for index, item in enumerate(conversations, start=1):
        input_text = item["input"]
        user_message = HumanMessage(input_text)
        print(f"\n[第 {index} 轮对话]")
        print(f"用户: {input_text}")

        print("\n【检索相关历史对话】")
        retrieved_conversations = await retrieve_relevant_conversations(input_text, 2)
        relevant_history = ""
        if retrieved_conversations:
            for retrieved_index, conv in enumerate(retrieved_conversations, start=1):
                print(f"\n[历史对话 {retrieved_index}] 相似度: {conv['score']:.4f}")
                print(f"轮次: {conv['round']}")
                print(f"内容: {conv['content']}")
            # relevant_history 会拼进 prompt，作为 RAG 的外部上下文。
            relevant_history = "\n\n━━━━━\n\n".join(
                f"[历史对话 {retrieved_index}]\n轮次: {conv['round']}\n{conv['content']}"
                for retrieved_index, conv in enumerate(retrieved_conversations, start=1)
            )
        else:
            print("未找到相关历史对话")

        context_messages = [
            HumanMessage(f"相关历史对话：\n{relevant_history}\n\n用户问题: {input_text}")
        ] if relevant_history else [user_message]

        print("\n【AI 回答】")
        response = await model.ainvoke(context_messages)
        history.add_message(user_message)
        history.add_message(response)

        # 当前轮对话也继续写回 Milvus，形成可持续增长的长期记忆。
        conversation_text = f"用户: {input_text}\n助手: {response.content}"
        conv_vector = await embeddings.aembed_query(conversation_text)
        conv_id = f"conv_{int(datetime.now().timestamp())}_{index}"
        try:
            client.insert(
                collection_name=CONVERSATION_COLLECTION_NAME,
                data=[{
                    "id": conv_id,
                    "vector": conv_vector,
                    "content": conversation_text,
                    "round": index,
                    "timestamp": datetime.now().isoformat(),
                }],
            )
            print("💾 已保存到 Milvus 向量数据库")
        except Exception as error:
            print(f"保存到向量数据库时出错: {error}")

        print(f"助手: {response.content}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(retrieval_memory_demo())
