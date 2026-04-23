"""loader_and_splitter2.py - 网页加载、切分并执行 RAG 检索"""
import asyncio
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


async def main() -> None:
    model = ChatOpenAI(
        temperature=0,
        model=os.getenv("MODEL_NAME", "qwen-coder-turbo"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-v3"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        check_embedding_ctx_length=False,
    )

    # 每个 URL 对应一个 Document
    loader = WebBaseLoader(web_path="https://juejin.cn/post/7233327509919547452")
    documents = loader.load()
    assert len(documents) >= 1

    print(f'documents长度{len(documents)}')

    # 让流程更稳定、可控，它先只取第一篇文档
    source_document = documents[0]
    source_document.page_content = "\n".join(
        line.strip() for line in source_document.page_content.splitlines() if line.strip()
    )
    print(f"Total characters: {len(source_document.page_content)}")

    '''
    overloap 只有文本超过 chunk size，文本被打断了才会加，不是所有的块都会有 overlap
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50, # 通常设置为 chunkSize 的 10% - 20%
        separators=["。", "！", "？"],
    )
    split_documents = text_splitter.split_documents([source_document])

    print(split_documents)
    print(f"文档分割完成，共 {len(split_documents)} 个分块\n")

    print("正在创建向量存储...")
    vector_store = InMemoryVectorStore(embeddings)
    await vector_store.aadd_documents(split_documents)
    print("向量存储创建完成\n")

    question = "父亲的去世对作者的人生态度产生了怎样的根本性逆转？"
    print("=" * 80)
    print(f"问题: {question}")
    print("=" * 80)

    scored_results = await vector_store.asimilarity_search_with_score(question, k=2)
    retrieved_docs = [doc for doc, _ in scored_results]

    print("\n【检索到的文档及相似度评分】")
    for index, (doc, score) in enumerate(scored_results, start=1):
        print(f"\n[文档 {index}] 相似度: {1 - score:.4f}")
        print(f"内容: {doc.page_content}")
        if doc.metadata:
            print(f"元数据: {doc.metadata}")

    context = "\n\n━━━━━\n\n".join(
        f"[片段{index}]\n{doc.page_content}" for index, doc in enumerate(retrieved_docs, start=1)
    )
    prompt = (
        "你是一个文章辅助阅读助手，根据文章内容来解答：\n\n"
        f"文章内容：\n{context}\n\n问题: {question}\n\n你的回答:"
    )

    print("\n【AI 回答】")
    response = await model.ainvoke(prompt)
    print(response.content)
    print()


if __name__ == "__main__":
    asyncio.run(main())
