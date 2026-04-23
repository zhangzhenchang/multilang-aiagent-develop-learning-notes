"""basic_graph.py - 最简单的 StateGraph 示例：两步顺序节点"""
import asyncio
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class GraphState(TypedDict):
    """图的状态：只有一个 text 字段，每个节点往后面追加文本"""
    text: str


def step1(state: GraphState) -> dict:
    """第一个节点：在 text 后追加 '-> step1'"""
    return {"text": f"{state['text']} -> step1"}


def step2(state: GraphState) -> dict:
    """第二个节点：在 text 后追加 '-> step2'"""
    return {"text": f"{state['text']} -> step2"}


# 构建图：START -> step1 -> step2 -> END
graph = (
    StateGraph(GraphState)
    .add_node("step1", step1)
    .add_node("step2", step2)
    .add_edge(START, "step1")
    .add_edge("step1", "step2")
    .add_edge("step2", END)
    .compile()
)


async def main() -> None:
    # 导出 Mermaid 图：可复制到 https://mermaid.live 查看
    mermaid = graph.get_graph().draw_mermaid()
    print(mermaid)

    # 执行图
    result = await graph.ainvoke({"text": "hello"})
    print("result:", result)


if __name__ == "__main__":
    asyncio.run(main())
