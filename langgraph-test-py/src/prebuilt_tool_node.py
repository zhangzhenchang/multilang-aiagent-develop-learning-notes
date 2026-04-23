"""prebuilt_tool_node.py - 使用 ToolNode + toolsCondition 构建工具调用 Agent"""
import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from inventory_mock import get_product_by_sku
from utils import create_chat_model


@tool
def get_product_stock(sku: str) -> str:
    """按 SKU 查商品名与库存，SKU 如 SKU-001。"""
    return get_product_by_sku(sku)


# 所有可用工具列表
tools = [get_product_stock]

# 创建模型并绑定工具
llm = create_chat_model().bind_tools(tools)


async def agent_node(state: MessagesState) -> dict:
    """Agent 节点：把当前消息列表发给 LLM，拿到回复（可能含 tool_calls）"""
    response = await llm.ainvoke(state["messages"])
    print(f'agent_node_response: \n{response}')
    '''
    把 response 的 AIMessage 追加到消息列表末尾
    '''
    return {"messages": [response]}


# 预置的 ToolNode：自动根据 tool_calls 执行对应工具
tool_node = ToolNode(tools)

# 构建图：START -> agent --(条件: 有工具调用? tools : END)--> tools -> agent
'''
tools_condition 固定返回 "tools" 或 "__end__"

  方式二：写字典（映射）                                                                                      
                                                                                                              
  .add_conditional_edges(                                                                                     
      "agent",                                                                                                
      tools_condition,                                                                                        
      {"tools": "tools", "__end__": END},                                                                     
  )                                                                                                           
                                                                                                              
  字典的作用是重命名 — key 是函数返回值，value 是实际跳转的节点名。                                           
                                                                                                              
  适用场景：函数返回 "yes"/"no"，但节点名叫别的：                                                             
                                                                                                              
  .add_conditional_edges(                                                                                     
      "agent",                                                                                                
      my_condition,        # 返回 "yes" 或 "no"                                                               
      {"yes": "tools", "no": END},                                                                            
  ) 
'''
graph = (
    StateGraph(MessagesState)
    .add_node("agent", agent_node)
    .add_node("tools", tool_node)
    .add_edge(START, "agent")
    .add_conditional_edges("agent", tools_condition, ["tools", "__end__"])
    .add_edge("tools", "agent")
    .compile()
)


async def main() -> None:
    # 导出 Mermaid 图
    mermaid = graph.get_graph().draw_mermaid()
    print(mermaid)

    # 执行
    result = await graph.ainvoke({
        "messages": [
            HumanMessage(content="查一下 SKU-002 的库存还有多少，回答里带上商品名和数字。"),
        ],
    })

    # 取最后一条消息的内容
    last_msg = result["messages"][-1]
    print(last_msg.content if hasattr(last_msg, "content") else result["messages"])


if __name__ == "__main__":
    asyncio.run(main())
