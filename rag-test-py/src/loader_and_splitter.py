"""loader_and_splitter.py - 网页加载并进行文本切分"""
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main() -> None:
    # 加载文档
    loader = WebBaseLoader(
        web_path="https://juejin.cn/post/7233327509919547452",
        # parse_only 控制“只解析哪一部分 HTML”
        # 你这里设成 None，等于不做限制
        bs_kwargs={"parse_only": None},
    )
    documents = loader.load()


    # 把网页文档的 page_content 做一次“去空行 + 去首尾空格 + 重新拼接”的文本清洗。
    # 因为网页加载器抓下来的内容通常很乱，直接拿去切分会导致：
        # chunk 里有很多空白
        # 文本结构不稳定
        # 检索效果变差
    # 清洗后再操作
    for document in documents:
        document.page_content = "\n".join(
            line.strip() for line in document.page_content.splitlines() if line.strip()
        )


    print(f'加载的文档\n{documents}')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["。", "！", "？"],
    )
    split_documents = text_splitter.split_documents(documents)
    for index, document in enumerate(split_documents, start=1):
        print(f"\n[Chunk {index}]")
        print(document)


if __name__ == "__main__":
    main()
