"""langchain_mcp_test.py - MCP 客户端测试 + 资源读取 + Agent"""
import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI


load_dotenv()

model = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

mcp_client = MultiServerMCPClient(
    connections={
        "my-mcp-server": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [os.path.join(os.path.dirname(__file__), "my_mcp_server.py")],
        },
    }
)


async def load_resource_content() -> str:
    resources = await mcp_client.get_resources()
    contents: list[str] = []

    for resource in resources:
        try:
            text = resource.as_string()
        except Exception:
            text = resource.data.decode("utf-8") if isinstance(resource.data, bytes) else str(resource.data)
        contents.append(text)

    return "\n\n".join(part for part in contents if part)
    
'''
因为要防止 Agent 无限循环。

在“模型调用 -> 工具调用 -> 再喂回模型”这种模式里，理论上可能一直不结束，比如：

模型反复调用同一个工具
工具结果不符合预期，模型一直重试
提示词或工具描述有歧义，模型陷入循环
外部服务异常导致模型持续修正但无收敛
所以 max_iterations 是一个“保险丝”：

到上限就强制停
防止程序卡死、耗时过长、费用失控
'''
async def run_agent_with_tools(query: str, max_iterations: int = 30) -> str:
    tools = await mcp_client.get_tools()
    model_with_tools = model.bind_tools(tools)
    resource_content = await load_resource_content()

    messages = [
        SystemMessage(resource_content) if resource_content else SystemMessage(""),
        HumanMessage(query),
    ]

    for i in range(max_iterations):
        print(f"\n⏳ 正在等待 AI 思考... (第 {i + 1} 次)")
        response = await model_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"\n✨ AI 最终回复:\n{response.content}\n")
            return str(response.content)

        print(
            f"🔍 检测到 {len(response.tool_calls)} 个工具调用: "
            f"{[tool_call['name'] for tool_call in response.tool_calls]}"
        )

        for tool_call in response.tool_calls:
            found_tool = next((tool for tool in tools if tool.name == tool_call["name"]), None)
            if found_tool is None:
                continue

            tool_result = await found_tool.ainvoke(tool_call["args"])
            messages.append(
                ToolMessage(
                    content=(
                        tool_result
                        if isinstance(tool_result, str)
                        else getattr(tool_result, "text", str(tool_result))
                    ),
                    tool_call_id=tool_call["id"],
                )
            )

    return str(messages[-1].content)


async def main() -> None:
    result = await run_agent_with_tools("查一下用户 002 的信息")
    print(result)
    result2 = await run_agent_with_tools("MCP Server 的使用指南是什么")
    print(result2)


if __name__ == "__main__":
    asyncio.run(main())
