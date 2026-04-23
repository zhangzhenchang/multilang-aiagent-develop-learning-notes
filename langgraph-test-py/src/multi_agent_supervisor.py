"""multi_agent_supervisor.py - 多 Agent 协作：Supervisor 调度天气 Agent 和知识 Agent"""
import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

from simple_mock import lookup_city_trivia, lookup_weather
from utils import create_chat_model


model = create_chat_model()


# ---- 工具定义 ----

@tool
def lookup_weather_tool(city: str) -> str:
    """查询某城市当日天气概况（气温区间、天气、空气质量等）。"""
    return lookup_weather(city)


@tool
def lookup_city_trivia_tool(city: str) -> str:
    """查询与某城市相关的一句趣味知识。"""
    return lookup_city_trivia(city)


# ---- 子 Agent A：天气 Agent ----
weather_agent = create_agent(
    model=model,
    tools=[lookup_weather_tool],
    system_prompt="你只处理天气。用户提到城市时，用 lookup_weather_tool 查询后再用中文简短说明。",
    name="weather_agent",
)

# ---- 子 Agent B：城市小知识 Agent ----
trivia_agent = create_agent(
    model=model,
    tools=[lookup_city_trivia_tool],
    system_prompt="你只讲城市小知识。先 lookup_city_trivia_tool，再用人话转述，不要编造工具里没有的内容。",
    name="trivia_agent",
)

# ---- Supervisor：根据问题类型分派到不同子 Agent ----
supervisor = create_supervisor(
    agents=[weather_agent, trivia_agent],
    model=model,
    prompt=(
        "你是调度员，只负责选人，不要自己报气温、也不要自己讲城市百科。\n\n"
        "- 问天气、气温、下不下雨、空气 → 用 weather_agent\n"
        "- 问小知识、名胜、历史、一句介绍 → 用 trivia_agent\n"
    ),
)

# 编译成可执行的图
app = supervisor.compile()


async def main() -> None:
    # 导出 Mermaid 图
    mermaid = app.get_graph().draw_mermaid()
    print(mermaid)

    # 发送任务
    input_msg = {
        "messages": [
            HumanMessage(content="查一下杭州的天气，再讲一条和杭州有关的小知识。"),
        ],
    }

    # 流式执行，跟踪经过的节点路径
    node_path: list[str] = []
    final_state = None

    '''
    对比                                                                                                             
    ┌───────────┬────────────────────────┬────────────────────────────────────┐
    │   模式    │          内容          │                用途                │
    ├───────────┼────────────────────────┼────────────────────────────────────┤
    │ "values"  │ 完整状态               │ 拿最终结果、渲染完整对话           │
    ├───────────┼────────────────────────┼────────────────────────────────────┤
    │ "updates" │ 只有变化部分，带节点名 │ 追踪哪个节点干了什么、记录执行路径 │                                 
    └───────────┴────────────────────────┴────────────────────────────────────┘                                 
                                                
     两个模式配合："updates" 记路径，"values" 拿结果。     
    '''
    async for event in app.astream(input_msg, stream_mode=["updates", "values"]):
        mode, payload = event
        if mode == "updates" and isinstance(payload, dict):
            node_path.extend(payload.keys())
        elif mode == "values":
            final_state = payload

    print("路径:", " → ".join(node_path))

    # 取最后一条消息的内容
    if final_state and "messages" in final_state:
        last_msg = final_state["messages"][-1]
        print(last_msg.content if hasattr(last_msg, "content") else final_state["messages"])


if __name__ == "__main__":
    asyncio.run(main())
