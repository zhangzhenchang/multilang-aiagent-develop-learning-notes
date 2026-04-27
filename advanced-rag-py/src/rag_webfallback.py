"""
Web Fallback RAG
工作流:
  START → route_question → direct_answer → END
                         ↘ local_retrieve → evaluate_local → generate → END
                                                           ↘ web_search ↗（二次评估后必生成）

策略：优先从本地知识库召回，若评估后认为信息不足，则调用 Bocha 联网搜索补充，
     二次评估后无论如何都生成回答（防止死循环）。
"""

import asyncio
import json
import sys
import os
from typing import List, Literal, Optional, TypedDict

import httpx
from dotenv import load_dotenv
from langchain_community.vectorstores import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

# ── 常量 ───────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "ebook_collection"
DEFAULT_K = 8
BOCHA_SEARCH_URL = "https://api.bochaai.com/v1/web-search"  # Bocha 联网搜索 API

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


# ── Pydantic Schema ────────────────────────────────────────────────────────────
class RouteSchema(BaseModel):
    """路由决策。"""
    strategy: Literal["simple", "complex"]
    reason: str


class EvaluateSchema(BaseModel):
    """上下文充分性评估结果。"""
    enough: bool                                      # 当前上下文是否已足够回答问题
    missing: List[str] = Field(default_factory=list)  # 缺失信息点（最多 6 条）
    reason: str                                       # 评估理由
    web_query: Optional[str] = None                   # 建议的联网查询句（首次评估时使用）


# ── 图状态 ─────────────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    question: str              # 用户问题
    k: int                     # 检索文档数量
    strategy: str              # 路由策略
    route_reason: str          # 路由原因
    retrieved_docs: List[dict] # 本地检索文档列表
    local_context: str         # 本地文档拼接的上下文（纯文本）
    web_context: str           # 联网搜索返回的结果文本
    evaluation: str            # 评估结果序列化为 JSON 字符串，节点间传递
    generation: str            # 最终生成的回答


vector_store: Optional[Milvus] = None


# ── 工具函数 ──────────────────────────────────────────────────────────────────
async def retrieve_relevant_content(query: str, k: int) -> List[dict]:
    """Milvus 向量检索。"""
    try:
        docs_with_scores = await asyncio.to_thread(
            vector_store.similarity_search_with_score, query, k
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


async def bocha_web_search(query: str, count: int = 10) -> str:
    """调用 Bocha 联网搜索 API，返回格式化的搜索结果字符串。

    Args:
        query: 搜索查询句（完整中文句，不使用代词）
        count: 返回结果条数

    Returns:
        格式化后的搜索结果，每条包含标题/URL/摘要等信息
    """
    api_key = os.getenv("BOCHA_API_KEY")
    if not api_key:
        raise EnvironmentError("BOCHA_API_KEY 未配置，无法进行联网搜索。")

    payload = {
        "query": query,
        "freshness": "noLimit",  # 不限制时效范围
        "summary": True,         # 返回 AI 摘要
        "count": count,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 使用 httpx 异步客户端，避免阻塞事件循环
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(BOCHA_SEARCH_URL, json=payload, headers=headers)
        except httpx.RequestError as e:
            raise ConnectionError(f"联网搜索请求失败（网络错误）：{e}") from e

    if response.status_code != 200:
        raise RuntimeError(
            f"联网搜索请求失败，状态码: {response.status_code}，错误: {response.text}"
        )

    data = response.json()
    if data.get("code") != 200 or not data.get("data"):
        raise RuntimeError(f"联网搜索 API 返回失败：{data.get('msg', '未知错误')}")

    webpages = data["data"].get("webPages", {}).get("value", [])
    if not webpages:
        return "未找到相关结果。"

    # 将每条结果格式化为可读文本，方便 LLM 引用
    return "\n\n".join(
        f"引用: {idx + 1}\n"
        f"标题: {page['name']}\n"
        f"URL: {page['url']}\n"
        f"摘要: {page['summary']}\n"
        f"网站名称: {page.get('siteName', '')}\n"
        f"发布时间: {page.get('dateLastCrawled', '')}"
        for idx, page in enumerate(webpages)
    )


# ── 节点函数 ──────────────────────────────────────────────────────────────────
async def route_question_node(state: GraphState) -> dict:
    """路由节点：判断是否需要本地检索。"""
    print("---ROUTE_QUESTION---")
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
        "retrieved_docs": [],
        "local_context": "",
        "web_context": "",
        "evaluation": "",
        "generation": "",
    }


async def direct_answer_node(state: GraphState) -> dict:
    """直接回答节点：simple 问题流式输出，无需检索。"""
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
    return {"generation": generation}


async def retrieve_local_node(state: GraphState) -> dict:
    """本地检索节点：向量召回并拼接为纯文本上下文。"""
    print("---LOCAL_RETRIEVE---")
    retrieved_docs = await retrieve_relevant_content(state["question"], state["k"])
    print(f"本地检索命中: {len(retrieved_docs)} 条")
    # 将文档正文拼接为供评估器和生成器使用的上下文字符串
    local_context = "\n\n".join(d["content"] for d in retrieved_docs)
    return {
        "retrieved_docs": retrieved_docs,
        "local_context": local_context,
    }


async def evaluate_node(state: GraphState) -> dict:
    """评估节点：判断当前上下文是否足以回答问题。

    首次调用（无 web_context）：评估本地上下文，并建议联网查询句。
    二次调用（有 web_context）：综合评估，不再建议联网查询（防止再次触发 web_search）。
    """
    has_web = bool(state.get("web_context", "").strip())
    print("---EVALUATE_CONTEXT_WITH_WEB---" if has_web else "---EVALUATE_LOCAL_CONTEXT---")

    # 根据是否有联网结果动态调整 prompt
    web_section = f"\n联网搜索结果：\n{state['web_context']}\n" if has_web else ""
    web_query_hint = (
        ""
        if has_web
        else "- web_query: 若不够，给出一个适合联网搜索的中文查询句（完整句，不用代词；为空也可）"
    )

    evaluator = llm.with_structured_output(EvaluateSchema, method="function_calling")
    out = await evaluator.ainvoke(
        f"你是信息充分性评估器。判断当前上下文是否足以回答用户问题。\n\n"
        f"用户问题：{state['question']}\n\n"
        f"已检索上下文（来自本地知识库）：\n{state.get('local_context') or '（空）'}\n"
        f"{web_section}\n"
        f"输出字段：\n"
        f"- enough: 是否足够回答（true/false）\n"
        f"- missing: 若不够，列出缺失信息点（最多 6 条）\n"
        f"- reason: 简短原因\n"
        f"{web_query_hint}\n"
    )

    print(f"{'二次评估' if has_web else '评估'}: enough={out.enough} ({out.reason})")
    if not out.enough and out.missing:
        for i, m in enumerate(out.missing):
            print(f"  缺失{i + 1}: {m}")

    # 序列化为 JSON 字符串，保留供 web_search_node 读取 web_query 字段
    return {"evaluation": out.model_dump_json()}


async def web_search_node(state: GraphState) -> dict:
    """联网搜索节点：从评估结果中读取 web_query，调用 Bocha API。"""
    print("---WEB_SEARCH---")
    try:
        parsed = json.loads(state.get("evaluation") or "{}")
    except json.JSONDecodeError:
        parsed = {}

    # 优先使用评估器建议的查询句，兜底使用原始问题
    query = (parsed.get("web_query") or "").strip() or state["question"]
    print(f"联网查询: {query}")

    web_context = await bocha_web_search(query, count=8)
    print(f"联网结果长度: {len(web_context)}")
    return {"web_context": web_context}


async def generate_node(state: GraphState) -> dict:
    """生成节点：综合本地上下文和联网结果，流式输出最终回答。"""
    print("---GENERATE---")
    # 合并本地和联网上下文（联网结果以分隔线区分）
    parts = [p for p in [state.get("local_context", ""), state.get("web_context", "")] if p]
    context = "\n\n===== 联网补充 =====\n\n".join(parts)

    sys.stdout.write("\n【AI 回答（流式）】\n")
    generation = ""

    prompt = ChatPromptTemplate.from_template(
        "你是一个严谨的中文问答助手。优先依据上下文作答，不要编造。\n\n"
        "上下文（本地知识库 + 可选联网补充）：\n{context}\n\n"
        "用户问题：{question}\n\n"
        "回答要求：\n"
        "1. 如果上下文足够，给出清晰、可核对的回答；需要时引用「引用: n / URL」或小说片段来支撑。\n"
        "2. 如果上下文仍不足以确定关键事实，明确说明「不确定/无法从上下文确认」，并说明缺失点。\n"
        "3. 不要输出表情符号。\n\n"
        "回答："
    )
    chain = prompt | llm | StrOutputParser()
    async for chunk in chain.astream({"context": context or "（空）", "question": state["question"]}):
        generation += chunk
        sys.stdout.write(chunk)
        sys.stdout.flush()
    sys.stdout.write("\n")
    return {"generation": generation}


# ── 条件边函数 ────────────────────────────────────────────────────────────────
def after_route(state: GraphState) -> str:
    """simple → 直接回答；complex → 本地检索。"""
    return "direct_answer" if state["strategy"] == "simple" else "local_retrieve"


def after_evaluate_local(state: GraphState) -> str:
    """评估后分支逻辑。

    - 若已有 web_context（说明是二次评估），直接生成，避免再次触发 web_search 死循环。
    - 首次评估：enough=True → 生成；enough=False → 联网搜索。
    """
    if state.get("web_context", "").strip():
        # 二次评估：联网结果已加入，无论充不充分都生成
        return "generate"
    try:
        parsed = json.loads(state.get("evaluation") or "{}")
    except json.JSONDecodeError:
        parsed = {}
    return "generate" if parsed.get("enough") is True else "web_search"


# ── 构建 LangGraph 工作流 ──────────────────────────────────────────────────────
builder = StateGraph(GraphState)
builder.add_node("route_question", route_question_node)
builder.add_node("direct_answer", direct_answer_node)
builder.add_node("local_retrieve", retrieve_local_node)
builder.add_node("evaluate_local", evaluate_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "route_question")
builder.add_conditional_edges(
    "route_question", after_route,
    {"direct_answer": "direct_answer", "local_retrieve": "local_retrieve"},
)
builder.add_edge("local_retrieve", "evaluate_local")
builder.add_conditional_edges(
    "evaluate_local", after_evaluate_local,
    {"generate": "generate", "web_search": "web_search"},
)
# web_search → evaluate_local 形成单次回环（二次评估后强制生成，不会再次进入 web_search）
builder.add_edge("web_search", "evaluate_local")
builder.add_edge("direct_answer", END)
builder.add_edge("generate", END)
graph = builder.compile()


async def main() -> None:
    global vector_store

    question = (
        "请回答《天龙八部》小说里「雁门关事件」的主谋是谁，并说明其儿子的最终结局；"
        "另外请补充：在《天龙八部》2013 版电视剧中，这段「雁门关事件」主要出现在哪几集？"
        "请给出可核对的来源链接。"
    )
    k = DEFAULT_K

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
        "retrieved_docs": [],
        "local_context": "",
        "web_context": "",
        "evaluation": "",
        "generation": "",
    })

    print(f"\n最终策略: {result['strategy']}")
    if not result.get("generation", "").strip():
        print("模型未返回内容。")


if __name__ == "__main__":
    asyncio.run(main())
