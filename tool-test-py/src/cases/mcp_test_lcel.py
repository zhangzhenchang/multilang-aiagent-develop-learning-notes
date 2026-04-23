"""mcp_test_lcel.py - MCP 客户端测试，使用 LCEL 语法

LangChain 1.x LCEL 核心：
  - create_agent 基于 LangGraph 构建工具调用循环
  - 对每个 MCP tool 设置 handle_tool_error=True，
    等同于原版手动 except ToolException 的处理逻辑：
    工具异常会作为错误消息返回给模型，而不是向上抛出导致崩溃
"""
import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

# ── 模型 ────────────────────────────────────────────────────────────────────
model = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# ── 环境变量 ─────────────────────────────────────────────────────────────────
allowed_paths = [
    path.strip()
    for path in os.getenv("ALLOWED_PATHS", "").split(",")
    if path.strip()
]
amap_api_key = os.getenv("AMAP_MAPS_API_KEY", "")

# ── MCP 客户端 ───────────────────────────────────────────────────────────────
mcp_client = MultiServerMCPClient(
    connections={
        "my-mcp-server": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [os.path.join(os.path.dirname(__file__), "..", "my_mcp_server.py")],
        },
        "amap-maps-streamableHTTP": {
            "transport": "streamable_http",
            "url": f"https://mcp.amap.com/mcp?key={amap_api_key}",
        },
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                *allowed_paths,
            ],
        },
        "chrome-devtools": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "chrome-devtools-mcp@latest",
            ],
        },
    }
)

# ── LCEL Agent ───────────────────────────────────────────────────────────────
# LangChain 1.x 中 create_agent 返回 CompiledStateGraph（LangGraph），
# 内部自动处理"思考 → 调用工具 → 观察"循环，直到模型不再输出 tool_calls。
SYSTEM_PROMPT = (
    "你是一个智能助手，可以使用各种工具来完成用户的任务。"
    "请尽量使用工具获取真实信息，给出准确、有帮助的回答。\n"
    "使用浏览器工具时，必须遵守以下规则：\n"
    "1. 整个任务过程中只允许启动一次浏览器会话，不要重复打开或关闭浏览器。\n"
    "2. 需要展示多个页面时，在同一个浏览器会话里依次新建 tab，不要关掉已有 tab 再重新打开。\n"
    "3. 所有操作完成后不要主动关闭浏览器。"
)


async def run_agent_with_tools(query: str) -> str:
    """用 create_agent (LangChain 1.x LCEL/LangGraph) 构建并执行 agent。"""
    tools = await mcp_client.get_tools()

    # handle_tool_error=True：工具抛出 ToolException 时，
    # 框架自动将错误内容包成 ToolMessage 返回给模型，而不是向上抛出
    for tool in tools:
        tool.handle_tool_error = True

    # create_agent 返回 CompiledStateGraph，支持 .ainvoke / .astream
    agent = create_agent(
        model,
        tools,
        system_prompt=SYSTEM_PROMPT,
    )

    # 输入格式：{"messages": [...]}，输出同样包含 messages 列表
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})

    # 取最后一条消息作为最终回复
    last_message = result["messages"][-1]
    return str(last_message.content)


# ── 入口 ─────────────────────────────────────────────────────────────────────
async def main() -> None:
    result = await run_agent_with_tools(
        "北京南站附近最近的 3 个酒店，拿到每个酒店的图片 URL。"
        "打开浏览器（只启动一次），在同一个浏览器窗口里为每个酒店新建一个 tab，"
        "每个 tab 打开对应酒店图片的 URL，然后把该 tab 的页面标题改为酒店名。"
        "全部完成后不要关闭浏览器。"
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
