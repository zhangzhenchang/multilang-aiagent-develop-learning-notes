"""ebook_writer.py - 读取 EPUB、拆分章节并写入 Milvus"""
import asyncio

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from langchain_text_splitters import RecursiveCharacterTextSplitter

from milvus_utils import (
    BOOK_NAME,
    EBOOK_COLLECTION_NAME,
    EBOOK_FILE,
    create_embeddings,
    create_milvus_client,
    ensure_ebook_collection,
    load_collection,
)

# 每个文本块的目标字符数。
CHUNK_SIZE = 500
# 单次调用 embedding 接口的最大批量大小。
EMBEDDING_BATCH_SIZE = 10


def load_epub_documents() -> list[tuple[int, str]]:
    """读取 EPUB，并返回 (章节号, 章节文本) 列表。"""
    book = epub.read_epub(str(EBOOK_FILE))
    chapters: list[tuple[int, str]] = []
    chapter_num = 0

    for item in book.get_items_of_type(ITEM_DOCUMENT):
        # 把章节 HTML 转成纯文本，便于后续切分和向量化。
        text = BeautifulSoup(item.get_body_content(), "html.parser").get_text("\n", strip=True)
        if not text:
            continue
        chapter_num += 1
        chapters.append((chapter_num, text))

    return chapters


async def embed_chunks_in_batches(embeddings, chunks: list[str]) -> list[list[float]]:
    """按批次生成向量，避免单次请求超过服务端 batch 限制。"""
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        # 当前批次的文本块。
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        vectors.extend(await embeddings.aembed_documents(batch))
    return vectors


async def insert_chunks_batch(chunks: list[str], book_id: str, chapter_num: int) -> int:
    """将某一章拆分后的文本块向量化，并批量写入 Milvus。"""
    if not chunks:
        return 0

    client = create_milvus_client()
    embeddings = create_embeddings()

    # 先把当前章节的所有 chunk 分批生成向量。
    vectors = await embed_chunks_in_batches(embeddings, chunks)

    # 构造要插入 Milvus 的记录。
    rows = [
        {
            # 主键：书籍 ID + 章节号 + 片段序号。
            "id": f"{book_id}_{chapter_num}_{chunk_index}",
            # 书籍业务 ID。
            "book_id": book_id,
            # 书名，来自 epub 文件名。
            "book_name": BOOK_NAME,
            # 当前是第几章。
            "chapter_num": chapter_num,
            # 当前 chunk 在章节内的顺序。
            "chunk_index": chunk_index,
            # 原始文本内容。
            "content": chunk,
            # 文本对应的 embedding 向量。
            "vector": vector,
        }
        for chunk_index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=False))
    ]
    result = client.insert(collection_name=EBOOK_COLLECTION_NAME, data=rows)
    return int(result["insert_count"])


async def load_and_process_epub_streaming(book_id: str) -> int:
    """按章节读取、切分、向量化并写入电子书内容。"""
    print(f"\n开始加载 EPUB 文件: {EBOOK_FILE}")
    chapters = load_epub_documents()
    print(f"✓ 加载完成，共 {len(chapters)} 个章节\n")

    # 文本切分器：把长章节切成更适合 embedding 和检索的小片段。
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=50)
    # 记录总插入条数。
    total_inserted = 0

    for chapter_num, chapter_content in chapters:
        print(f"处理第 {chapter_num}/{len(chapters)} 章...")
        chunks = text_splitter.split_text(chapter_content)
        print(f"  拆分为 {len(chunks)} 个片段")
        if not chunks:
            print("  跳过空章节\n")
            continue

        print("  生成向量并插入中...")
        inserted_count = await insert_chunks_batch(chunks, book_id, chapter_num)
        total_inserted += inserted_count
        print(f"  ✓ 已插入 {inserted_count} 条记录（累计: {total_inserted}）\n")

    print(f"\n总共插入 {total_inserted} 条记录\n")
    return total_inserted


async def main() -> None:
    print("=" * 80)
    print("电子书处理程序")
    print("=" * 80)

    client = create_milvus_client()
    print("\n连接 Milvus...")
    # 确保 ebook collection 已存在。
    ensure_ebook_collection(client)
    # 把集合加载到内存，便于后续快速检索。
    load_collection(EBOOK_COLLECTION_NAME)
    print("✓ 已连接\n")

    # 这里的 book_id 是这本书在业务侧的唯一标识。
    await load_and_process_epub_streaming(book_id="1")

    print("=" * 80)
    print("处理完成！")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
