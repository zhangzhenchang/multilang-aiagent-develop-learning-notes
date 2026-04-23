"""loop_retry.py - 循环重试示例：条件路由实现 retry 逻辑"""
import asyncio
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class GraphState(TypedDict):
    """图的状态"""
    tries: int      # 已尝试次数
    ok: bool        # 是否成功
    message: str    # 本轮提示信息


def attempt(state: GraphState) -> dict:
    """
    模拟一次尝试：第 3 次才会成功。
    每次把 tries +1，判断是否达到成功条件。
    """
    tries = state["tries"] + 1
    ok = tries >= 3  # 第 3 次开始算成功
    message = f"第 {tries} 次成功" if ok else f"第 {tries} 次失败，继续重试"
    return {"tries": tries, "ok": ok, "message": message}


def should_retry(state: GraphState) -> Literal["retry", "done"]:
    """条件路由：如果 ok=True 就结束，否则再走一轮 attempt"""
    return "done" if state["ok"] else "retry"


# 构建图：START -> attempt --(条件: retry/done)--> attempt / END
graph = (
    StateGraph(GraphState)
    .add_node("attempt", attempt)
    .add_edge(START, "attempt")
    .add_conditional_edges("attempt", should_retry, {"retry": "attempt", "done": END})
    .compile()
)


async def main() -> None:
    # 导出 Mermaid 图
    mermaid = graph.get_graph().draw_mermaid()
    print(mermaid)

    # 执行：从 tries=0 开始
    result = await graph.ainvoke({"tries": 0, "ok": False, "message": ""})
    print("result:", result)


if __name__ == "__main__":
    asyncio.run(main())
