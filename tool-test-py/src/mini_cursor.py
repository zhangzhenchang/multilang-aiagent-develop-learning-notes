"""mini_cursor.py - 使用自定义工具的项目管理 Agent，类似 Cursor 的 AI 编程助手"""
import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from all_tools import TOOLS


load_dotenv()

model = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)


async def run_agent_with_tools(query: str, max_iterations: int = 30) -> str:
    model_with_tools = model.bind_tools(TOOLS)
    messages = [
        SystemMessage(
            f"""你是一个项目管理助手，使用工具完成任务。

当前工作目录: {os.getcwd()}

工具：
1. read_file: 读取文件
2. write_file: 写入文件
3. execute_command: 执行命令（支持 working_directory 参数）
4. list_directory: 列出目录

重要规则 - execute_command：
- working_directory 参数会自动切换到指定目录
- 当使用 working_directory 时，绝对不要在 command 中使用 cd
- 错误示例: command=\"cd react-todo-app && pnpm install\", working_directory=\"react-todo-app\"
  这是错误的！因为 working_directory 已经在 react-todo-app 目录了，再 cd 会找不到目录
- 正确示例: command=\"pnpm install\", working_directory=\"react-todo-app\"
  这样就对了！working_directory 已经切换到 react-todo-app，直接执行命令即可

重要规则 - write_file：
- 当写入 React 组件文件（如 App.tsx）时，如果存在对应的 CSS 文件（如 App.css），在其他 import 语句后加上这个 css 的导入
"""
        ),
        HumanMessage(query),
    ]

    for i in range(max_iterations):
        print(f"\n⏳ 正在等待 AI 思考... (第 {i + 1} 次)")
        response = await model_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"\n✨ AI 最终回复:\n{response.content}\n")
            return str(response.content)

        for tool_call in response.tool_calls:
            found_tool = next((tool for tool in TOOLS if tool.name == tool_call["name"]), None)
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


CASE1 = """创建一个功能丰富的 React TodoList 应用：

1. 创建项目：pnpm create vite react-todo-app --template react-ts
2. 修改 src/App.tsx，实现完整功能的 TodoList：
 - 添加、删除、编辑、标记完成
 - 分类筛选（全部/进行中/已完成）
 - 统计信息显示
 - localStorage 数据持久化
3. 添加复杂样式：
 - 渐变背景（蓝到紫）
 - 卡片阴影、圆角
 - 悬停效果
4. 添加动画：
 - 添加/删除时的过渡动画
 - 使用 CSS transitions
5. 列出目录确认

注意：使用 pnpm，功能要完整，样式要美观，要有动画效果

之后在 react-todo-app 项目中：
1. 使用 pnpm install 安装依赖
2. 使用 pnpm run dev 启动服务器
"""


async def main() -> None:
    try:
        await run_agent_with_tools(CASE1)
    except Exception as exc:
        print(f"\n❌ 错误: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())
