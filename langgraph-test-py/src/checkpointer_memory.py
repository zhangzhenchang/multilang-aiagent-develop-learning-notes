"""checkpointer_memory.py - 内存 Checkpointer 示例：演示同一会话的状态持久化"""
import asyncio
from typing import Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


def _replace(_prev, next_val):
    """Reducer：直接用新值替换旧值（对应 JS 的 (_prev, next) => next）"""
    return next_val


class GraphState(TypedDict):
    visit_count: Annotated[int, _replace]
    message: Annotated[str, _replace]


def record_visit(state: GraphState) -> GraphState:
    """每次进入给当前会话访问次数 +1，并生成对应消息"""
    visit_count = state["visit_count"] + 1
    message = (
        "这是你在本会话里第 1 次进入。"
        if visit_count == 1
        else f"这是你在本会话里第 {visit_count} 次进入。"
    )
    return {"visit_count": visit_count, "message": message}


def build_app() -> object:
    checkpointer = MemorySaver()
    return (
        StateGraph(GraphState, input_schema=GraphState)
        .add_node("record_visit", record_visit)
        .add_edge(START, "record_visit")
        .add_edge("record_visit", END)
        .compile(checkpointer=checkpointer)
    )


async def main() -> None:
    app = build_app()

    '''
    configurable 是 LangGraph 的保留字段，框架会自动读取里面的 thread_id，传给 checkpointer                     
    做存取。这是约定好的协议：                                
                                            
    {"configurable": {"thread_id": "用户-小张"}}
    #  ^^^^^^^^^^^^    ^^^^^^^^^                                                                                
    #  LangGraph 保留  固定 key，框架认识它 
                                                                                                                
    你只需要传不同的 thread_id，框架自动帮你隔离状态。 
    '''
    user1 = {"configurable": {"thread_id": "用户-小张"}}
    user2 = {"configurable": {"thread_id": "用户-小李"}}

    # 统一传 {}，checkpointer 自动用默认值初始化首次状态
    res1 = await app.ainvoke({}, user1)
    res2 = await app.ainvoke({}, user1)
    res3 = await app.ainvoke({}, user1)
    res4 = await app.ainvoke({}, user2)  # 独立会话，从 1 开始

    print(res1)  # visit_count=1
    print(res2)  # visit_count=2
    print(res3)  # visit_count=3
    print(res4)  # visit_count=1


if __name__ == "__main__":
    asyncio.run(main())
