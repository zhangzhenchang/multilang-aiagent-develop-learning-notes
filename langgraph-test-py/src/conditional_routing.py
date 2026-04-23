"""conditional_routing.py - 条件路由示例：根据输入自动选择数学计算或聊天"""
import asyncio
import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class GraphState(TypedDict):
    """图的状态"""
    query: str      # 用户输入
    route: str      # 路由标签：'math' 或 'chat'
    answer: str     # 最终回答


def router(state: GraphState) -> dict:
    """路由节点：检查 query 里是否包含数学运算符，决定走 math 还是 chat"""
    is_math = bool(re.search(r"[+\-*/]", state["query"]))
    return {"route": "math" if is_math else "chat"}


def math_node(state: GraphState) -> dict:
    """数学计算节点：尝试对 query 做算术运算（仅演示用）"""
    try:
        # 仅允许简单算术表达式
        answer = str(eval(state["query"]))  # noqa: S307
    except Exception:
        answer = "表达式无法计算"
    return {"answer": answer}


def chat_node(state: GraphState) -> dict:
    """聊天节点：原样回复用户的输入"""
    return {"answer": f"你说的是：{state['query']}"}


def route_decision(state: GraphState) -> Literal["math", "chat"]:
    """条件路由函数：返回 state.route 决定下一个节点"""
    return state["route"]


# 构建图：START -> router --(条件)--> math / chat -> END
graph = (
    StateGraph(GraphState)
    .add_node("router", router)
    .add_node("math", math_node)
    .add_node("chat", chat_node)
    .add_edge(START, "router")
    .add_conditional_edges("router", route_decision, {"math": "math", "chat": "chat"})
    .add_edge("math", END)
    .add_edge("chat", END)
    .compile()
)


async def main() -> None:
    # 导出 Mermaid 图
    mermaid = graph.get_graph().draw_mermaid()
    print(mermaid)

    # 测试聊天路径
    result1 = await graph.ainvoke({"query": "你好", "route": "chat", "answer": ""})
    print("result:", result1)

    # 测试数学路径
    result2 = await graph.ainvoke({"query": "10 * 8", "route": "math", "answer": ""})
    print("result:", result2)


if __name__ == "__main__":
    asyncio.run(main())
