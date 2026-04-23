"""checkpointer_sqlite.py - SQLite Checkpointer 示例：状态持久化到磁盘"""
import asyncio
import os
from typing import TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph


# SQLite 文件路径
DB_PATH = os.path.join(os.path.dirname(__file__), "checkpointer-demo.sqlite")


class GraphState(TypedDict):
    """图的状态"""
    visit_count: int  # 当前会话的访问次数
    message: str      # 打印消息


def record_visit(state: GraphState) -> dict:
    """每次进入就给访问计数 +1，并生成对应消息"""
    visit_count = state["visit_count"] + 1
    if visit_count == 1:
        message = "这是你在本会话里第 1 次进入。"
    else:
        message = f"这是你在本会话里第 {visit_count} 次进入"
    return {"visit_count": visit_count, "message": message}


async def main() -> None:
    # 先清理旧文件
    if os.path.exists(DB_PATH):
        os.unlink(DB_PATH)

    # 使用异步 SQLite checkpointer
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = (
            StateGraph(GraphState)
            .add_node("record_visit", record_visit)
            .add_edge(START, "record_visit")
            .add_edge("record_visit", END)
            .compile(checkpointer=checkpointer)
        )

        user1_config = {"configurable": {"thread_id": "用户-小张"}}
        user2_config = {"configurable": {"thread_id": "用户-小李"}}

        initial = {"visit_count": 0, "message": ""}

        res1 = await app.ainvoke(initial, user1_config)
        res2 = await app.ainvoke(initial, user1_config)
        res3 = await app.ainvoke(initial, user1_config)
        res4 = await app.ainvoke(initial, user2_config)

        print(res1)
        print(res2)
        print(res3)
        print(res4)


if __name__ == "__main__":
    asyncio.run(main())
