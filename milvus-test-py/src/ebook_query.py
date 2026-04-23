"""ebook_query.py - 搜索电子书中相似的内容片段"""
import asyncio

from milvus_utils import EBOOK_COLLECTION_NAME, create_embeddings, create_milvus_client, format_search_results, load_collection


async def main() -> None:
    # Milvus 客户端：负责向量检索请求。
    client = create_milvus_client()
    # Embeddings 模型：把用户问题转成查询向量。
    embeddings = create_embeddings()

    print("Connecting to Milvus...")
    print("✓ Connected\n")

    try:
        # 检索前先把集合加载到内存。
        load_collection(EBOOK_COLLECTION_NAME)
        print("✓ 集合已加载\n")
    except Exception as error:
        # 如果已经加载，则直接继续即可。
        if "already loaded" not in str(error):
            raise
        print("✓ 集合已处于加载状态\n")

    # 用户查询文本。
    query = "鸠摩智会什么武功？"
    print("Searching for similar ebook content...")
    print(f'Query: "{query}"\n')

    # 查询向量：用于和库里的 chunk 向量做相似度比较。
    query_vector = await embeddings.aembed_query(query)
    search_result = client.search(
        collection_name=EBOOK_COLLECTION_NAME,
        data=[query_vector],
        # 最多返回前 5 个最相似片段。
        limit=5,
        # 除了相似度分数，还额外返回这些字段。
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
