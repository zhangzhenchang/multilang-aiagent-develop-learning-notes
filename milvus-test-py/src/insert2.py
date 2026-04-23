"""insert2.py - 创建日记集合并插入示例数据（使用 milvus_utils2 标准版）"""
import asyncio

from milvus_utils2 import (
    DIARY_COLLECTION_NAME,
    create_embeddings,
    create_milvus_client,
    ensure_diary_collection,
    load_collection,
)


DIARY_CONTENTS = [
    {
        "id": "diary_001",
        "content": "今天天气很好，去公园散步了，心情愉快。看到了很多花开了，春天真美好。",
        "date": "2026-01-10",
        "mood": "happy",
        "tags": ["生活", "散步"],
    },
    {
        "id": "diary_002",
        "content": "今天工作很忙，完成了一个重要的项目里程碑。团队合作很愉快，感觉很有成就感。",
        "date": "2026-01-11",
        "mood": "excited",
        "tags": ["工作", "成就"],
    },
    {
        "id": "diary_003",
        "content": "周末和朋友去爬山，天气很好，心情也很放松。享受大自然的感觉真好。",
        "date": "2026-01-12",
        "mood": "relaxed",
        "tags": ["户外", "朋友"],
    },
    {
        "id": "diary_004",
        "content": "今天学习了 Milvus 向量数据库，感觉很有意思。向量搜索技术真的很强大。",
        "date": "2026-01-12",
        "mood": "curious",
        "tags": ["学习", "技术"],
    },
    {
        "id": "diary_005",
        "content": "晚上做了一顿丰盛的晚餐，尝试了新菜谱。家人都说很好吃，很有成就感。",
        "date": "2026-01-13",
        "mood": "proud",
        "tags": ["美食", "家庭"],
    },
]


async def main() -> None:
    client = create_milvus_client()
    embeddings = create_embeddings()

    print("Connecting to Milvus...")
    ensure_diary_collection(client)
    load_collection(client, DIARY_COLLECTION_NAME)
    print("✓ Connected and collection ready\n")

    print("Generating embeddings...")
    vectors = await embeddings.aembed_documents([item["content"] for item in DIARY_CONTENTS])
    rows = [{**item, "vector": vector} for item, vector in zip(DIARY_CONTENTS, vectors, strict=False)]

    result = client.insert(collection_name=DIARY_COLLECTION_NAME, data=rows)
    print(f"✓ Inserted {result['insert_count']} records")


if __name__ == "__main__":
    asyncio.run(main())
