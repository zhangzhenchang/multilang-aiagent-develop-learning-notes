"""query2.py - 搜索相似的日记条目（使用 milvus_utils2 标准版）"""
import asyncio

from milvus_utils2 import (
    DIARY_COLLECTION_NAME,
    create_embeddings,
    create_milvus_client,
    ensure_diary_collection,
    format_search_results,
    load_collection,
)


async def main() -> None:
    client = create_milvus_client()
    embeddings = create_embeddings()

    print("Connecting to Milvus...")
    ensure_diary_collection(client)
    load_collection(client, DIARY_COLLECTION_NAME)
    print("✓ Connected\n")

    query = "我做饭或学习的日记"
    print("Searching for similar diary entries...")
    print(f'Query: "{query}"\n')

    query_vector = await embeddings.aembed_query(query)
    search_result = client.search(
        collection_name=DIARY_COLLECTION_NAME,
        data=[query_vector],
        limit=2,
        output_fields=["id", "content", "date", "mood", "tags"],
        search_params={"metric_type": "COSINE", "params": {}},
    )

    print(f"search_result: {search_result}")

    results = format_search_results(search_result[0])
    print(f"Found {len(results)} results:\n")
    for index, item in enumerate(results, start=1):
        print(f"{index}. [Score: {item['score']:.4f}]")
        print(f"   ID: {item['id']}")
        print(f"   Date: {item['date']}")
        print(f"   Mood: {item['mood']}")
        print(f"   Tags: {', '.join(item.get('tags', []))}")
        print(f"   Content: {item['content']}\n")


if __name__ == "__main__":
    asyncio.run(main())
