"""mcp_test.py - MCP 客户端测试，通过 MultiServerMCPClient 连接多个 MCP 服务器"""
import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools.base import ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI


class Console:
    GREEN_BG = "\033[42m\033[30m"
    BLUE_BG = "\033[44m\033[37m"
    YELLOW_BG = "\033[43m\033[30m"
    RED_BG = "\033[41m\033[37m"
    RESET = "\033[0m"

    @classmethod
    def green(cls, text: str) -> str:
        return f"{cls.GREEN_BG}{text}{cls.RESET}"

    @classmethod
    def blue(cls, text: str) -> str:
        return f"{cls.BLUE_BG}{text}{cls.RESET}"

    @classmethod
    def yellow(cls, text: str) -> str:
        return f"{cls.YELLOW_BG}{text}{cls.RESET}"

    @classmethod
    def red(cls, text: str) -> str:
        return f"{cls.RED_BG}{text}{cls.RESET}"


load_dotenv()

model = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

allowed_paths = [path.strip() for path in os.getenv("ALLOWED_PATHS", "").split(",") if path.strip()]
amap_api_key = os.getenv("AMAP_MAPS_API_KEY", "")

mcp_client = MultiServerMCPClient(
    connections={
        "my-mcp-server": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [os.path.join(os.path.dirname(__file__), "my_mcp_server.py")],
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


async def run_agent_with_tools(query: str, max_iterations: int = 30) -> str:
    tools = await mcp_client.get_tools()
    model_with_tools = model.bind_tools(tools)
    messages = [HumanMessage(query)]

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(Console.green(f"⏳ 正在等待 AI 思考... (第 {iteration} 次)"))
        response = await model_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"\n✨ AI 最终回复:\n{response.content}\n")
            return str(response.content)

        print(
            Console.blue(
                f"🔍 检测到 {len(response.tool_calls)} 个工具调用: "
                f"{[tool_call['name'] for tool_call in response.tool_calls]}"
            )
        )

        for tool_call in response.tool_calls:
            found_tool = next((tool for tool in tools if tool.name == tool_call["name"]), None)
            if found_tool is None:
                messages.append(
                    ToolMessage(
                        content=f"工具未找到: {tool_call['name']}",
                        tool_call_id=tool_call["id"],
                        status="error",
                    )
                )
                continue

            try:
                tool_result = await found_tool.ainvoke(tool_call["args"])
                content_str = (
                    tool_result
                    if isinstance(tool_result, str)
                    else getattr(tool_result, "text", str(tool_result))
                )
                tool_status = "success"
                print(Console.yellow(f"🛠️ 工具执行成功: {tool_call['name']}"))
            except ToolException as exc:
                content_str = f"工具执行失败: {exc}"
                tool_status = "error"
                print(Console.red(f"❌ 工具执行失败: {tool_call['name']} - {exc}"))
            except Exception as exc:
                content_str = f"工具执行异常: {exc}"
                tool_status = "error"
                print(Console.red(f"❌ 工具执行异常: {tool_call['name']} - {exc}"))

            messages.append(
                ToolMessage(
                    content=content_str,
                    tool_call_id=tool_call["id"],
                    status=tool_status,
                )
            )

    return str(messages[-1].content)


async def main() -> None:
    result = await run_agent_with_tools(
        "北京南站附近的酒店，最近的 3 个酒店，拿到酒店图片，打开浏览器，展示每个酒店的图片，每个 tab 一个 url 展示，并且在把那个页面标题改为酒店名"
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
