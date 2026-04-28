"""
多跳 RAG (Multi-hop RAG)
工作流:
  START → route_question → direct_answer → END
                         ↘ decompose_question → retrieve ↔ plan_next_step → generate → END

对于复杂问题先拆解为有序子问题，再按序迭代检索，最终综合所有召回文档生成回答。
"""

import asyncio
import sys
import os
from typing import Annotated, List, Literal, Optional, TypedDict

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
DEFAULT_K = 5
DEFAULT_MAX_RETRIEVALS = 8  # 最大检索轮数，防止子问题过多时无限循环

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
    """路由决策：是否需要多跳检索。"""
    strategy: Literal["simple", "complex"]
    reason: str


class DecomposeSchema(BaseModel):
    """子问题拆解结果：将复杂问题分解为有序的可独立检索子问题。"""
    # 1~8 条有序子问题，每条须是完整独立的中文问句
    sub_questions: Annotated[List[str], Field(min_length=1, max_length=8)]
    reason: str  # 拆解策略说明


class NextStepSchema(BaseModel):
    """规划器决策：继续检索下一个子问题，还是直接生成回答。"""
    next_action: Literal["retrieve", "generate"]
    reason: str


# ── 图状态 ─────────────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    question: str             # 用户原始问题
    k: int                    # 每轮检索返回文档数
    strategy: str             # 路由策略: "simple" | "complex"
    route_reason: str         # 路由原因
    sub_questions: List[str]  # 拆解得到的有序子问题列表
    next_sub_idx: int         # 下一轮 retrieve 要使用的子问题下标
    documents: List[dict]     # 累计去重后的检索文档（按相似度降序）
    current_query: str        # 当前轮实际使用的检索查询
    retrieval_count: int      # 已完成的检索轮数
    max_retrievals: int       # 允许的最大检索轮数
    planned_next: str         # 规划器输出: "retrieve" | "generate"
    generation: str           # 最终生成的回答


vector_store: Optional[Milvus] = None


# ── 工具函数 ──────────────────────────────────────────────────────────────────
async def retrieve_relevant_content(question: str, k: int) -> List[dict]:
    """Milvus 向量检索，返回格式化文档列表。"""
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

# 多路召回融合
def merge_unique(existing: List[dict], new_docs: List[dict]) -> List[dict]:
    """按文档 id 去重合并，同 id 保留更高相似度分数，结果按分数降序排列。"""
    doc_map: dict[str, dict] = {}
    for doc in existing + new_docs:
        key = str(doc["id"])
        if key not in doc_map or doc["score"] > doc_map[key]["score"]:
            doc_map[key] = doc
    return sorted(doc_map.values(), key=lambda d: d["score"], reverse=True)


# ── 节点函数 ──────────────────────────────────────────────────────────────────
async def route_question_node(state: GraphState) -> dict:
    """路由节点：判断是否需要多跳检索。"""
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
        "retrieval_count": 0,
        "max_retrievals": state.get("max_retrievals") or DEFAULT_MAX_RETRIEVALS,
        "documents": [],
        "sub_questions": [],
        "next_sub_idx": 0,
        "current_query": "",
    }


async def decompose_question_node(state: GraphState) -> dict:
    """拆解节点：将复杂问题分解为有序、可独立检索的子问题列表。"""
    print("---DECOMPOSE_QUESTION---")
    decomposer = llm.with_structured_output(DecomposeSchema, method="function_calling")
    out = await decomposer.ainvoke(
        f"你是《天龙八部》多跳问答的「子问题拆解器」。\n\n"
        f"用户原始问题：\n{state['question']}\n\n"
        f"任务：将问题拆成**有序**子问题列表 sub_questions，用于**依次向量检索**。要求：\n"
        f"1. 链式推理、多层关系、因果先后的问题，必须拆成多条；单跳即可答的也可只输出 1 条。\n"
        f"2. 每条子问题必须是**可独立检索**的完整中文问句，**禁止**使用「他/她/此人/上文」等指代；可写全人物名与事件名。\n"
        f"3. 顺序必须符合推理链：先搞清前置实体/事实，再查后续结论。\n"
        f"4. **不要**把整句原题原样复制成唯一一条（除非确实无法拆分）；不要拆成过碎的关键词列表。\n"
        f"5. 输出 1～8 条即可。\n\n"
        f"请输出 sub_questions 与简短 reason。"
    )

    sub_questions = [q.strip() for q in out.sub_questions if q.strip()]
    if not sub_questions:
        raise ValueError("decompose_question: sub_questions 为空")

    print(f"拆解 {len(sub_questions)} 条子问题 ({out.reason})")
    for i, q in enumerate(sub_questions):
        print(f"  [{i + 1}] {q}")

    return {
        "sub_questions": sub_questions,
        "next_sub_idx": 0,
        "current_query": sub_questions[0],  # 首轮查询为第一个子问题
    }


async def retrieve_node(state: GraphState) -> dict:
    """检索节点：取 next_sub_idx 对应的子问题进行向量检索，并与历史文档去重合并。"""
    subs = state.get("sub_questions", [])
    idx = state.get("next_sub_idx", 0)
    if idx >= len(subs) or not subs[idx].strip():
        raise ValueError(f"retrieve: 子问题下标 {idx} 无有效文本（共 {len(subs)} 条）")

    query = subs[idx].strip()
    round_num = state["retrieval_count"] + 1
    print(f"---RETRIEVE (第 {round_num} 轮，子问题 {idx + 1}/{len(subs)})---")
    print(f"查询: {query}")

    new_docs = await retrieve_relevant_content(query, state["k"])
    merged = merge_unique(state.get("documents", []), new_docs)  # 跨轮去重累积

    if not new_docs:
        print("本轮未命中文档")
    else:
        print(f"本轮命中 {len(new_docs)} 条，累计去重后 {len(merged)} 条")
        for i, item in enumerate(new_docs):
            preview = item["content"][:120] + ("..." if len(item["content"]) > 120 else "")
            print(f"[R{i + 1}] score={item['score']:.4f} chapter={item['chapter_num']} index={item['index']}")
            print(f"      {preview}")

    return {
        "documents": merged,
        "retrieval_count": round_num,
        "next_sub_idx": idx + 1,   # 推进指针，下一轮检索下一个子问题
        "current_query": query,
    }


async def plan_next_step_node(state: GraphState) -> dict:
    """规划节点：评估已有文档是否足够，决定继续检索还是生成回答。

    硬性规则（优先于模型判断）：
    - 剩余子问题为 0 → 强制生成
    - 已达最大检索轮数 → 强制生成
    """
    print("---PLAN_NEXT_STEP---")
    subs = state.get("sub_questions", [])
    next_idx = state.get("next_sub_idx", 0)
    remaining = len(subs) - next_idx  # 剩余未检索子问题数量

    # 构建子问题进度列表，标注每条的检索状态
    def sub_status(i: int) -> str:
        if i < next_idx:
            return " （已检索）"
        elif i == next_idx:
            return " （下一轮将检索，若选择继续）"
        return " （未检索）"

    sub_list = "\n".join(f"{i + 1}. {s}{sub_status(i)}" for i, s in enumerate(subs))

    # 取前 6 条文档构建摘要，供规划器参考
    if not state.get("documents"):
        doc_str = "（尚无检索结果）"
    else:
        doc_str = "\n\n".join(
            f"[{i + 1}] score={d['score']:.4f} 第{d['chapter_num']}章: "
            f"{d['content'][:200]}{'...' if len(d['content']) > 200 else ''}"
            for i, d in enumerate(state["documents"][:6])
        )

    planner = llm.with_structured_output(NextStepSchema, method="function_calling")
    out = await planner.ainvoke(
        f"你是多跳 RAG 规划器。检索查询已由前置步骤拆解为**有序子问题**；"
        f"若需继续检索，下一轮将自动使用「下一条子问题」做向量检索，你**不要**自拟新的检索句。\n\n"
        f"用户原始问题：{state['question']}\n\n"
        f"子问题序列：\n{sub_list or '（无）'}\n\n"
        f"已检索轮数：{state['retrieval_count']}；剩余未检索子问题条数：{remaining}\n"
        f"最大检索轮数上限：{state['max_retrievals']}\n\n"
        f"已召回文档摘要：\n{doc_str}\n\n"
        f"请判断下一步：\n"
        f"1) 已有足够依据回答用户原始问题 → next_action=generate\n"
        f"2) 仍缺关键事实、且仍存在未检索的子问题、且未超过轮数上限 → next_action=retrieve\n\n"
        f"硬性规则：\n"
        f"- 若剩余未检索子问题条数为 0，必须 next_action=generate。\n"
        f"- 若已检索轮数已达到或超过最大检索轮数，必须 next_action=generate。"
    )

    # 强制性规则覆盖模型建议
    final_next = out.next_action
    if state["retrieval_count"] >= state["max_retrievals"] or remaining <= 0:
        final_next = "generate"

    print(f"[决策] planned_next={final_next} (模型建议={out.next_action}) ({out.reason})")
    return {"planned_next": final_next}


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


async def generate_node(state: GraphState) -> dict:
    """生成节点：综合所有累积文档，流式生成最终回答。"""
    print("---GENERATE---")
    context = "\n\n━━━━━\n\n".join(
        f"[片段 {i + 1}]\n章节: 第 {item['chapter_num']} 章\n内容: {item['content']}"
        for i, item in enumerate(state.get("documents", []))
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


# ── 条件边函数 ────────────────────────────────────────────────────────────────
def after_route(state: GraphState) -> str:
    """simple → 直接回答；complex → 先拆解子问题。"""
    return "direct_answer" if state["strategy"] == "simple" else "decompose_question"


def after_plan(state: GraphState) -> str:
    """规划器决策边：继续检索或进入生成。"""
    return state.get("planned_next", "generate")


# ── 构建 LangGraph 工作流 ──────────────────────────────────────────────────────
builder = StateGraph(GraphState)
builder.add_node("route_question", route_question_node)
builder.add_node("direct_answer", direct_answer_node)
builder.add_node("decompose_question", decompose_question_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("plan_next_step", plan_next_step_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "route_question")
builder.add_conditional_edges(
    "route_question", after_route,
    {"direct_answer": "direct_answer", "decompose_question": "decompose_question"},
)
builder.add_edge("decompose_question", "retrieve")
builder.add_edge("retrieve", "plan_next_step")
# retrieve ↔ plan_next_step 形成迭代循环，由规划器决定何时退出
builder.add_conditional_edges(
    "plan_next_step", after_plan,
    {"retrieve": "retrieve", "generate": "generate"},
)
builder.add_edge("direct_answer", END)
builder.add_edge("generate", END)
graph = builder.compile()


async def main() -> None:
    global vector_store

    question = (
        "《天龙八部》中「四大恶人」排行第二的是谁？"
        "此人之子在身世揭晓前，其生父在武林中的公开身份是什么？"
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
        "sub_questions": [],
        "next_sub_idx": 0,
        "documents": [],
        "current_query": "",
        "retrieval_count": 0,
        "max_retrievals": DEFAULT_MAX_RETRIEVALS,
        "planned_next": "",
        "generation": "",
    })

    if result["strategy"] == "complex":
        if result.get("sub_questions"):
            print("\n【子问题序列】")
            for i, s in enumerate(result["sub_questions"]):
                print(f"  {i + 1}. {s}")

        print("\n【检索相关内容（累计）】")
        if not result.get("documents"):
            print("未找到相关内容")
        else:
            for i, item in enumerate(result["documents"]):
                print(f"\n[片段 {i + 1}] 相似度: {item['score']:.4f}")
                print(f"书籍: {item['book_id']}")
                print(f"章节: 第 {item['chapter_num']} 章")
                print(f"片段索引: {item['index']}")
                preview = item["content"][:200] + ("..." if len(item["content"]) > 200 else "")
                print(f"内容: {preview}")

        print(f"\n检索轮数: {result['retrieval_count']} / {result['max_retrievals']}")

    print(f"\n最终策略: {result['strategy']}")
    if not result.get("generation", "").strip():
        print("模型未返回内容。")


if __name__ == "__main__":
    asyncio.run(main())
