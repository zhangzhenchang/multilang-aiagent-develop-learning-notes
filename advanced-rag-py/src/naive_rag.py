"""
朴素 RAG (Naive RAG)
工作流: START → retrieve → generate → END

最简单的 RAG 模式：先向量检索，再把文档喂给 LLM 生成回答。
"""

import asyncio
import sys
import os
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph

load_dotenv()

# ── 常量 ───────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "ebook_collection"  # Milvus 集合名称（与建库时保持一致）
TOP_K = 5                              # 默认每次检索返回的文档数量

# ── 模型初始化（OPENAI_API_KEY / OPENAI_BASE_URL 由环境变量自动注入）────────────
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),  # 通过环境变量切换模型
    temperature=0,                                # 关闭随机性，保证输出确定
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",  # 嵌入模型，维度需与 Milvus 集合一致
    dimensions=1024,            # 向量维度
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
)

# ── 图状态（TypedDict 定义节点间共享的状态字段）─────────────────────────────────
class GraphState(TypedDict):
    question: str          # 用户输入的问题
    k: int                 # 本次检索的文档数量
    documents: List[dict]  # 检索结果列表，每项含 score/content/chapter_num 等
    generation: str        # LLM 最终生成的回答文本


# 全局向量数据库实例（在 main() 中连接后赋值）
vector_store: Optional[Milvus] = None


# ── 工具函数 ──────────────────────────────────────────────────────────────────
async def retrieve_relevant_content(question: str, k: int = TOP_K) -> List[dict]:
    """从 Milvus 向量库检索与 question 最相关的 k 条文档。

    使用 asyncio.to_thread 将同步的 similarity_search_with_score 包装为异步，
    避免阻塞事件循环。
    """
    try:
        docs_with_scores = await asyncio.to_thread(
            vector_store.similarity_search_with_score, question, k
        )
        return [
            {
                "score": float(score),                               # 余弦相似度（越大越相关）
                "content": doc.page_content,                         # 文档正文
                "id": doc.metadata.get("id", "unknown"),
                "book_id": doc.metadata.get("book_id", "未知"),
                "chapter_num": doc.metadata.get("chapter_num", "未知"),
                "index": doc.metadata.get("index", "未知"),          # 章内片段序号
            }
            for doc, score in docs_with_scores
        ]
    except Exception as e:
        print(f"检索内容时出错: {e}", file=sys.stderr)
        return []


# ── 节点函数 ──────────────────────────────────────────────────────────────────
async def retrieve_node(state: GraphState) -> dict:
    """检索节点：根据问题召回相关文档片段。"""
    documents = await retrieve_relevant_content(state["question"], state["k"])
    return {"documents": documents}


async def generate_node(state: GraphState) -> dict:
    """生成节点：将检索文档拼接为上下文，调用 LLM 流式生成回答。

    LCEL 链: ChatPromptTemplate | ChatOpenAI | StrOutputParser
    """
    # 将多条文档片段格式化为结构化上下文
    context = "\n\n━━━━━\n\n".join(
        f"[片段 {i + 1}]\n章节: 第 {item['chapter_num']} 章\n内容: {item['content']}"
        for i, item in enumerate(state["documents"])
    )

    # LCEL 链：Prompt → LLM → 字符串解析器
    prompt = ChatPromptTemplate.from_template(
        "你是一个专业的《天龙八部》小说助手。基于小说内容回答问题，用准确、详细的语言。\n\n"
        "请根据以下《天龙八部》小说片段内容回答问题：\n{context}\n\n"
        "用户问题: {question}\n\n"
        "回答要求：\n"
        "1. 如果片段中有相关信息，请结合小说内容给出详细、准确的回答\n"
        "2. 可以综合多个片段的内容，提供完整的答案\n"
        "3. 如果片段中没有相关信息，请如实告知用户\n"
        "4. 回答要准确，符合小说的情节和人物设定\n"
        "5. 可以引用原文内容来支持你的回答\n\n"
        "AI 助手的回答:"
    )
    chain = prompt | llm | StrOutputParser()

    sys.stdout.write("\n【AI 回答（流式）】\n")
    generation = ""
    # astream 流式逐 chunk 输出，减少首字节延迟
    async for chunk in chain.astream({"context": context, "question": state["question"]}):
        generation += chunk
        sys.stdout.write(chunk)
        sys.stdout.flush()
    sys.stdout.write("\n")

    return {"generation": generation}


# ── 构建 LangGraph 工作流 ──────────────────────────────────────────────────────
builder = StateGraph(GraphState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()


async def main() -> None:
    global vector_store

    question = "阿朱的结局是什么？"
    k = TOP_K

    # 打印 Mermaid 流程图（可粘贴到 https://mermaid.live 可视化）
    print(graph.get_graph().draw_mermaid())

    print("连接到 Milvus...")
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": "localhost", "port": "19530"},  # Milvus 地址
        text_field="content",    # 存储正文的字段名
        primary_field="id",      # 主键字段名
        vector_field="vector",   # 向量字段名
        index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",  # 索引算法
            "params": {
                "M": 16, # 每个节点最多连接 16 个邻居。越大召回越准，内存越多
                "efConstruction": 200 # 建索引时的搜索深度。越大索引质量越高，建库越慢
            },
        },
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        drop_old=False,  # 不删除已有集合
        # False：复用已有集合（生产环境常用） 
        # True：每次启动都删掉重建，用于开发调试 
    )                 

    print("✓ 已连接\n")

    print("=" * 80)
    print(f"问题: {question}")
    print("=" * 80)

    result = await graph.ainvoke({
        "question": question,
        "k": k,
        "documents": [],
        "generation": "",
    })

    print("\n【检索相关内容】")
    if not result["documents"]:
        print("未找到相关内容")
        print("\n【AI 回答】\n抱歉，我没有找到相关的《天龙八部》内容。")
        return

    for i, item in enumerate(result["documents"]):
        print(f"\n[片段 {i + 1}] 相似度: {item['score']:.4f}")
        print(f"书籍: {item['book_id']}")
        print(f"章节: 第 {item['chapter_num']} 章")
        print(f"片段索引: {item['index']}")
        preview = item["content"][:200] + ("..." if len(item["content"]) > 200 else "")
        print(f"内容: {preview}")

    if not result.get("generation"):
        print("\n【AI 回答】\n模型未返回内容。")


if __name__ == "__main__":
    asyncio.run(main())
