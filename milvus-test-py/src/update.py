"""update.py - 通过 upsert 更新日记条目"""
import asyncio

from milvus_utils import DIARY_COLLECTION_NAME, create_embeddings, create_milvus_client, ensure_diary_collection, load_collection


UPDATED_CONTENT = {
    "id": "diary_001",
    "content": "今天下了一整天的雨，心情很糟糕。工作上遇到了很多困难，感觉压力很大。一个人在家，感觉特别孤独。",
    "date": "2026-01-10",
    "mood": "sad",
    "tags": ["生活", "散步", "朋友"],
}


async def main() -> None:
    client = create_milvus_client()
    embeddings = create_embeddings()

    print("Connecting to Milvus...")
    ensure_diary_collection(client)
    load_collection(DIARY_COLLECTION_NAME)
    print("✓ Connected\n")

    print("Updating diary entry...")
    vector = await embeddings.aembed_query(UPDATED_CONTENT["content"])
    result = client.upsert(
        collection_name=DIARY_COLLECTION_NAME,
        data=[{**UPDATED_CONTENT, "vector": vector}],
    )

    print(f"✓ Updated diary entry: {UPDATED_CONTENT['id']}")
    print(f"  Upsert count: {result['upsert_count']}")
    print(f"  New mood: {UPDATED_CONTENT['mood']}")
    print(f"  New tags: {', '.join(UPDATED_CONTENT['tags'])}")


if __name__ == "__main__":
    asyncio.run(main())
