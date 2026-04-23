"""insert_conversations.py - 将对话数据写入 Milvus"""
from datetime import datetime

from src.memory_utils import (
    CONVERSATION_COLLECTION_NAME,
    create_embeddings,
    create_milvus_client,
    ensure_conversation_collection,
    load_collection,
)


# 预置的示例对话数据，会先转向量再写入 Milvus。
CONVERSATIONS = [
    {
        "id": "conv_001",
        "content": "用户: 我叫赵六，是一名数据科学家\n助手: 很高兴认识你，赵六！数据科学是一个很有趣的领域。",
        "round": 1,
        "timestamp": datetime.now().isoformat(),
    },
    {
        "id": "conv_002",
        "content": "用户: 我最近在研究机器学习算法\n助手: 机器学习确实很有意思，你在研究哪些算法呢？",
        "round": 2,
        "timestamp": datetime.now().isoformat(),
    },
    {
        "id": "conv_003",
        "content": "用户: 我喜欢打篮球和看电影\n助手: 运动和文化娱乐都是很好的爱好！",
        "round": 3,
        "timestamp": datetime.now().isoformat(),
    },
    {
        "id": "conv_004",
        "content": "用户: 我周末经常去电影院\n助手: 看电影是很好的放松方式。",
        "round": 4,
        "timestamp": datetime.now().isoformat(),
    },
    {
        "id": "conv_005",
        "content": "用户: 我的职业是软件工程师\n助手: 软件工程师是个很有前景的职业！",
        "round": 5,
        "timestamp": datetime.now().isoformat(),
    },
]


async def main() -> None:
    client = create_milvus_client()
    embeddings = create_embeddings()

    print("连接到 Milvus...")
    # 确保 conversations 集合已存在。
    ensure_conversation_collection(client)
    # 加载集合到内存，便于后续插入和检索。
    load_collection(CONVERSATION_COLLECTION_NAME)
    print("✓ 已连接\n")

    print("插入对话数据...")
    # 先把每条对话文本转成向量。
    vectors = await embeddings.aembed_documents([item["content"] for item in CONVERSATIONS])
    rows = [{**item, "vector": vector} for item, vector in zip(CONVERSATIONS, vectors, strict=False)]
    result = client.insert(collection_name=CONVERSATION_COLLECTION_NAME, data=rows)
    print(f"✓ 已插入 {result['insert_count']} 条记录\n")
    print("=" * 60)
    print("说明：已成功将对话数据插入到 Milvus 向量数据库")
    print("这些对话数据将用于后续的 RAG 检索")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
