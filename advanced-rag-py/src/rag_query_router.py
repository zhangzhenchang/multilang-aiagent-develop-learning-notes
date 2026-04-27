"""
查询路由 RAG (Query Router RAG)
工作流: START → route_question → direct_answer → END
                              ↘ retrieve → rag_generate → END

根据问题复杂度选择处理路径：
- simple（常识/定义类问题）→ 直接调用 LLM 回答，无需检索
- complex（需要小说细节）→ 向量检索 → RAG 生成
"""

import asyncio
import sys
import os
from typing import List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

load_dotenv()

# ── 常量 ───────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "ebook_collection"
TOP_K = 5

# ── 模型初始化 ─────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    dimensions=1024,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
)


# ── Pydantic Schema（用于 with_structured_output 结构化输出）─────────────────
class RouteSchema(BaseModel):
    """路由决策结果：判断问题是否需要外部检索。"""
    strategy: Literal["simple", "complex"]  # simple→直接回答; complex→需要检索
    reason: str                              # 路由判断依据（调试用）


# ── 图状态 ─────────────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    question: str          # 用户问题
    k: int                 # 检索文档数量
    strategy: str          # 路由策略: "simple" | "complex"
    route_reason: str      # 路由原因说明
    documents: List[dict]  # 检索结果
    generation: str        # 最终回答


vector_store: Optional[Milvus] = None


# ── 工具函数 ──────────────────────────────────────────────────────────────────
async def retrieve_relevant_content(question: str, k: int) -> List[dict]:
    """Milvus 向量相似度检索，返回格式化后的文档列表。"""
    try:
        docs_with_scores = await asyncio.to_thread(
            vector_store.similarity_search_with_score, question, k
        )
        return [
            {
                "score": float(score),
                "content": doc.page_content,
                "id": doc.metadata.get("id", "unknown"),
                "book_id": doc.metadata.get("book_id", "未知"),
                "chapter_num": doc.metadata.get("chapter_num", "未知"),
                "index": doc.metadata.get("index", "未知"),
            }
            for doc, score in docs_with_scores
        ]
    except Exception as e:
        print(f"检索内容时出错: {e}", file=sys.stderr)
        return []


# ── 节点函数 ──────────────────────────────────────────────────────────────────
async def route_question_node(state: GraphState) -> dict:
    """路由节点：用结构化输出判断问题属于 simple 还是 complex。

    LCEL 链: llm.with_structured_output(RouteSchema)
    """
    print("---ROUTE_QUESTION---")
    # with_structured_output 会强制 LLM 按 RouteSchema 格式返回 Pydantic 实例
    router = llm.with_structured_output(RouteSchema, method="function_calling")
    route = await router.ainvoke(
        f"你是问答路由器。请判断用户问题是否需要外部检索。\n\n"
        f"规则：\n"
        f"- simple: 常识问答、简短定义、无需特定小说细节即可回答。\n"
        f"- complex: 需要《天龙八部》具体情节、人物关系、章节事实、原文细节或证据支持。\n\n"
        f"用户问题：{state['question']}\n"
    )
    print(f"路由策略: {route.strategy} ({route.reason})")
    return {
        "strategy": route.strategy,
        "route_reason": route.reason,
    }


# ── 条件边函数 ────────────────────────────────────────────────────────────────
def decide_next(state: GraphState) -> str:
    """路由后分支：simple → 直接回答；complex → 先检索。"""
    return "direct_answer" if state["strategy"] == "simple" else "retrieve"



async def retrieve_node(state: GraphState) -> dict:
    """检索节点：向量召回相关文档片段。"""
    print("---RETRIEVE---")
    documents = await retrieve_relevant_content(state["question"], state["k"])
    if not documents:
        print("RETRIEVE结果: 未命中文档")
    else:
        print(f"RETRIEVE结果: 命中 {len(documents)} 条")
        for i, item in enumerate(documents):
            preview = item["content"][:120] + ("..." if len(item["content"]) > 120 else "")
            print(f"[R{i + 1}] score={item['score']:.4f} chapter={item['chapter_num']} index={item['index']}")
            print(f"      {preview}")
    return {"documents": documents}


async def direct_answer_node(state: GraphState) -> dict:
    """直接回答节点：simple 问题无需检索，直接流式输出 LLM 回答。

    LCEL 链: ChatPromptTemplate | ChatOpenAI | StrOutputParser
    """
    print("---DIRECT_ANSWER---")
    sys.stdout.write("\n【AI 回答（流式）】\n")
    generation = ""

    prompt = ChatPromptTemplate.from_template(
        "你是一个中文问答助手，请直接简洁回答问题。\n\n问题：{question}"
    )
    chain = prompt | llm | StrOutputParser()
    async for chunk in chain.astream({"question": state["question"]}):
        generation += chunk
        sys.stdout.write(chunk)
        sys.stdout.flush()
    sys.stdout.write("\n")

    return {"documents": [], "generation": generation}


async def rag_generate_node(state: GraphState) -> dict:
    """RAG 生成节点：基于检索文档，流式生成有依据的回答。

    LCEL 链: ChatPromptTemplate | ChatOpenAI | StrOutputParser
    """
    print("---RAG_GENERATE---")
    context = "\n\n━━━━━\n\n".join(
        f"[片段 {i + 1}]\n章节: 第 {item['chapter_num']} 章\n内容: {item['content']}"
        for i, item in enumerate(state["documents"])
    )
    sys.stdout.write("\n【AI 回答（流式）】\n")
    generation = ""

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
    async for chunk in chain.astream({
        "context": context or "（未检索到相关内容）",
        "question": state["question"],
    }):
        generation += chunk
        sys.stdout.write(chunk)
        sys.stdout.flush()
    sys.stdout.write("\n")

    return {"generation": generation}



# ── 构建 LangGraph 工作流 ──────────────────────────────────────────────────────
builder = StateGraph(GraphState)
builder.add_node("route_question", route_question_node)
builder.add_node("direct_answer", direct_answer_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("rag_generate", rag_generate_node)
builder.add_edge(START, "route_question")
builder.add_conditional_edges(
    "route_question",
    decide_next,
    {"direct_answer": "direct_answer", "retrieve": "retrieve"},
)
builder.add_edge("retrieve", "rag_generate")
builder.add_edge("direct_answer", END)
builder.add_edge("rag_generate", END)
graph = builder.compile()


async def main() -> None:
    global vector_store

    question = "雁门关事件的主谋，他的儿子最终结局是什么？"
    k = TOP_K

    # 打印 Mermaid 流程图
    print(graph.get_graph().draw_mermaid())

    print("连接到 Milvus...")
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": "localhost", "port": "19530"},
        text_field="content",
        primary_field="id",
        vector_field="vector",
        index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        drop_old=False,
    )
    print("✓ 已连接\n")

    print("=" * 80)
    print(f"问题: {question}")
    print("=" * 80)

    result = await graph.ainvoke({
        "question": question,
        "k": k,
        "strategy": "",
        "route_reason": "",
        "documents": [],
        "generation": "",
    })

    if result["strategy"] == "complex":
        print("\n【检索相关内容】")
        if not result["documents"]:
            print("未找到相关内容")
        else:
            for i, item in enumerate(result["documents"]):
                print(f"\n[片段 {i + 1}] 相似度: {item['score']:.4f}")
                print(f"书籍: {item['book_id']}")
                print(f"章节: 第 {item['chapter_num']} 章")
                print(f"片段索引: {item['index']}")
                preview = item["content"][:200] + ("..." if len(item["content"]) > 200 else "")
                print(f"内容: {preview}")

    print(f"\n最终策略: {result['strategy']}")
    if not result.get("generation", "").strip():
        print("模型未返回内容。")


if __name__ == "__main__":
    asyncio.run(main())
