"""ebook_query2.py - 搜索电子书中相似的内容片段（使用 milvus_utils2 标准版）"""
import asyncio

from milvus_utils2 import (
    EBOOK_COLLECTION_NAME,
    create_embeddings,
    create_milvus_client,
    format_search_results,
    load_collection,
)


async def main() -> None:
    client = create_milvus_client()
    embeddings = create_embeddings()

    print("Connecting to Milvus...")
    print("✓ Connected\n")

    try:
        load_collection(client, EBOOK_COLLECTION_NAME)
        print("✓ 集合已加载\n")
    except Exception as error:
        if "already loaded" not in str(error):
            raise
        print("✓ 集合已处于加载状态\n")

    query = "鸠摩智会什么武功？"
    print("Searching for similar ebook content...")
    print(f'Query: "{query}"\n')

    query_vector = await embeddings.aembed_query(query)
    search_result = client.search(
        collection_name=EBOOK_COLLECTION_NAME,
        data=[query_vector],
        limit=5,
        output_fields=["id", "book_id", "chapter_num", "chunk_index", "content"],
        search_params={"metric_type": "COSINE", "params": {}},
    )

    results = format_search_results(search_result[0])
    print(f"Found {len(results)} results:\n")
    for index, item in enumerate(results, start=1):
        print(f"{index}. [Score: {item['score']:.4f}]")
        print(f"   ID: {item['id']}")
        print(f"   Book ID: {item['book_id']}")
        print(f"   Chapter: 第 {item['chapter_num']} 章")
        print(f"   Chunk Index: {item['chunk_index']}")
        print(f"   Content: {item['content']}\n")


if __name__ == "__main__":
    asyncio.run(main())
