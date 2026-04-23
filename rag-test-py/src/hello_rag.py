"""hello_rag.py - 简单的内存向量库 RAG 示例"""
import asyncio
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


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

    documents = [
        Document(
            page_content="光光是一个活泼开朗的小男孩，他有一双明亮的大眼睛，总是带着灿烂的笑容。光光最喜欢的事情就是和朋友们一起玩耍，他特别擅长踢足球，每次在球场上奔跑时，就像一道阳光一样充满活力。",
            metadata={"chapter": 1, "character": "光光", "type": "角色介绍", "mood": "活泼"},
        ),
        Document(
            page_content="东东是光光最好的朋友，他是一个安静而聪明的男孩。东东喜欢读书和画画，他的画总是充满了想象力。虽然性格不同，但东东和光光从幼儿园就认识了，他们一起度过了无数个快乐的时光。",
            metadata={"chapter": 2, "character": "东东", "type": "角色介绍", "mood": "温馨"},
        ),
        Document(
            page_content="有一天，学校要举办一场足球比赛，光光非常兴奋，他邀请东东一起参加。但是东东从来没有踢过足球，他担心自己会拖累光光。光光看出了东东的担忧，他拍着东东的肩膀说：\"没关系，我们一起练习，我相信你一定能行的！\"",
            metadata={"chapter": 3, "character": "光光和东东", "type": "友情情节", "mood": "鼓励"},
        ),
        Document(
            page_content="接下来的日子里，光光每天放学后都会教东东踢足球。光光耐心地教东东如何控球、传球和射门，而东东虽然一开始总是踢不好，但他从不放弃。东东也用自己的方式回报光光，他画了一幅画送给光光，画上是两个小男孩在球场上一起踢球的场景。",
            metadata={"chapter": 4, "character": "光光和东东", "type": "友情情节", "mood": "互助"},
        ),
        Document(
            page_content="比赛那天终于到了，光光和东东一起站在球场上。虽然东东的技术还不够熟练，但他非常努力，而且他用自己的观察力帮助光光找到了对手的弱点。在关键时刻，东东传出了一个漂亮的球，光光接球后射门得分！他们赢得了比赛，更重要的是，他们的友谊变得更加深厚了。",
            metadata={"chapter": 5, "character": "光光和东东", "type": "高潮转折", "mood": "激动"},
        ),
        Document(
            page_content="从那以后，光光和东东成为了学校里最要好的朋友。光光教东东运动，东东教光光画画，他们互相学习，共同成长。每当有人问起他们的友谊，他们总是笑着说：\"真正的朋友就是互相帮助，一起变得更好的人！\"",
            metadata={"chapter": 6, "character": "光光和东东", "type": "结局", "mood": "欢乐"},
        ),
        Document(
            page_content="多年后，光光成为了一名职业足球运动员，而东东成为了一名优秀的插画师。虽然他们走上了不同的道路，但他们的友谊从未改变。东东为光光设计了球衣上的图案，光光在每场比赛后都会给东东打电话分享喜悦。他们证明了，真正的友情可以跨越时间和距离，永远闪闪发光。",
            metadata={"chapter": 7, "character": "光光和东东", "type": "尾声", "mood": "温馨"},
        ),
    ]

    # 创建向量库
    vector_store = InMemoryVectorStore(embeddings)
    # 添加文档到向量库
    await vector_store.aadd_documents(documents)
    # 查询出余弦相似度最大的 3 个文档
    # 这里 不是立即检索，而是在创建一个“检索器对象”。
    # 意思是：
        # 基于 vector_store 创建一个 retriever
        # 默认检索参数设置成 k=3
        # 但此时还没有真正开始查
        # 真正根据问题查，是后面这句：
    # retrieved_docs = await retriever.ainvoke(question)    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    questions = ["东东和光光是怎么成为朋友的？"]

    for question in questions:
        print("=" * 80)
        print(f"问题: {question}")
        print("=" * 80)

        retrieved_docs = await retriever.ainvoke(question)
        scored_results = await vector_store.asimilarity_search_with_score(question, k=3)

        print("\n【检索到的文档及相似度评分】")
        for index, doc in enumerate(retrieved_docs, start=1):
            matched = next(
                ((scored_doc, score) for scored_doc, score in scored_results if scored_doc.page_content == doc.page_content),
                None,
            )
            similarity = f"{1 - matched[1]:.4f}" if matched is not None else "N/A"
            print(f"\n[文档 {index}] 相似度: {similarity}")
            print(f"内容: {doc.page_content}")
            print(
                "元数据: "
                f"章节={doc.metadata.get('chapter')}, "
                f"角色={doc.metadata.get('character')}, "
                f"类型={doc.metadata.get('type')}, "
                f"心情={doc.metadata.get('mood')}"
            )

        context = "\n\n━━━━━\n\n".join(
            f"[片段{index}]\n{doc.page_content}" for index, doc in enumerate(retrieved_docs, start=1)
        )

        print(f'[拼接后的文档内容]\n{context}')

        prompt = (
            "你是一个讲友情故事的老师。基于以下故事片段回答问题，用温暖生动的语言。"
            '如果故事中没有提到，就说"这个故事里还没有提到这个细节"。\n\n'
            f"故事片段:\n{context}\n\n问题: {question}\n\n老师的回答:"
        )

        print("\n【AI 回答】")
        response = await model.ainvoke(prompt)
        print(response.content)
        print()


if __name__ == "__main__":
    asyncio.run(main())
