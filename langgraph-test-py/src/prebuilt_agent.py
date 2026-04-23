"""prebuilt_agent.py - 使用 create_react_agent 快速创建工具代理"""
import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from inventory_mock import get_product_by_sku
from utils import create_chat_model


@tool
def get_product_stock(sku: str) -> str:
    """按 SKU 查商品名与库存，SKU 如 SKU-001。"""
    return get_product_by_sku(sku)


# 创建模型
model = create_chat_model()

# 使用 create_agent 快速构建带工具调用的 Agent
agent = create_agent(
    model=model,
    tools=[get_product_stock],
    system_prompt="你是仓库助手。问库存时必须调用 get_product_stock（模拟数据），禁止编造。",
    checkpointer=MemorySaver(),
)


async def main() -> None:
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="SKU-003 还剩多少库存？")]},
        config={"configurable": {"thread_id": "demo-thread"}},
    )

    # 导出 Mermaid 图
    mermaid = agent.get_graph().draw_mermaid()
    print(mermaid)

    # 取最后一条消息
    last_msg = result["messages"][-1]
    print(last_msg.content if hasattr(last_msg, "content") else result["messages"])


if __name__ == "__main__":
    asyncio.run(main())
