"""tool_file_read.py - 使用工具读取文件并解释代码的 Agent 示例"""
import asyncio
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


load_dotenv()

model = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-coder-turbo"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)


@tool
def read_file(file_path: str) -> str:
    """用此工具来读取文件内容。当用户要求读取文件、查看代码、分析文件内容时，调用此工具。"""
    content = Path(file_path).read_text(encoding="utf-8")
    print(f'  [工具调用] read_file("{file_path}") - 成功读取 {len(content)} 字节')
    return f"文件内容:\n{content}"


async def main() -> None:
    tools = [read_file]
    model_with_tools = model.bind_tools(tools)
    messages = [
        SystemMessage(
            "你是一个代码助手，可以使用工具读取文件并解释代码。\n\n"
            "工作流程：\n"
            "1. 用户要求读取文件时，立即调用 read_file 工具\n"
            "2. 等待工具返回文件内容\n"
            "3. 基于文件内容进行分析和解释\n\n"
            "可用工具：\n"
            "- read_file: 读取文件内容"
        ),
        HumanMessage("请读取 ./src/tool_file_read.py 文件内容并解释代码"),
    ]

    # 第一次调用模型，获取工具调用
    response = await model_with_tools.ainvoke(messages)
    messages.append(response)

    print(f"[第一次tool_calls结果]\n{json.dumps(response.tool_calls, indent=2)}")

    while response.tool_calls:
        print(f"\n[检测到 {len(response.tool_calls)} 个工具调用]")

        for tool_call in response.tool_calls:
            # tool_obj 是找到的 tool 对象
            tool_obj = next((tool_item for tool_item in tools if tool_item.name == tool_call["name"]), None)
            
            print(f'[tool_obj结果]\n{tool_obj}\n')
            
            if tool_obj is None:
                result = f"错误: 找不到工具 {tool_call['name']}"
            else:
                print(f"  [执行工具] {tool_call['name']}({tool_call['args']})")
                try:
                    result = await tool_obj.ainvoke(tool_call["args"])
                # 常用写法
                except Exception as exc:
                    result = f"错误: {exc}"

            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                )
            )

        response = await model_with_tools.ainvoke(messages)
        messages.append(response)

    print("\n[最终回复]")
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
