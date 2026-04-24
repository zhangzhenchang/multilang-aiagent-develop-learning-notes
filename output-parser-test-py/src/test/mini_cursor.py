"""
mini_cursor.py - Mini Cursor：基于 LangChain 的自动化代码生成 Agent

核心流程：
  1. 将用户任务加入消息历史
  2. 带工具绑定的模型流式响应（write_file 内容实时预览）
  3. 执行工具调用，将结果追加到历史
  4. 重复直到模型不再调用工具（最终回复）或达到最大迭代次数

依赖：langchain-openai, langchain-core, python-dotenv
运行：需与 all_tools.py 在同一目录
"""
from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.outputs import ChatGeneration
from langchain_openai import ChatOpenAI

# all_tools.py 须与本文件同目录
from all_tools import execute_command, list_directory, read_file, write_file

load_dotenv()

# ──────────────────────────────────────────────────────────
# ANSI 颜色（替代 JS 的 chalk，无需额外依赖）
# ──────────────────────────────────────────────────────────
_G_BG = "\033[42m"   # 绿色背景
_B_BG = "\033[44m"   # 蓝色背景
_GREEN = "\033[32m"  # 绿色字
_RESET = "\033[0m"

# ──────────────────────────────────────────────────────────
# 模型 & 工具初始化
# ──────────────────────────────────────────────────────────
_model = ChatOpenAI(
    model="qwen-plus",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)

TOOLS = [read_file, write_file, execute_command, list_directory]

# bind_tools 将工具 schema 注入模型，模型可按需调用
model_with_tools = _model.bind_tools(TOOLS)

# 便于按名称快速查找工具
_TOOL_MAP = {t.name: t for t in TOOLS}

# JsonOutputToolsParser 使用 parse_partial_json 处理流式不完整 JSON，
# 对应 JS 版的 JsonOutputToolsParser，比手写 regex 健壮得多
_tool_parser = JsonOutputToolsParser()

# ──────────────────────────────────────────────────────────
# 系统提示
# ──────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
你是一个项目管理助手，使用工具完成任务。

当前工作目录: {cwd}

工具：
1. read_file: 读取文件
2. write_file: 写入文件
3. execute_command: 执行命令（支持 working_directory 参数）
4. list_directory: 列出目录

重要规则 - execute_command：
- working_directory 参数会自动切换到指定目录
- 使用 working_directory 时，绝对不要在 command 中包含 cd
- 错误示例: {{ "command": "cd react-todo-app && pnpm install", "working_directory": "react-todo-app" }}
- 正确示例: {{ "command": "pnpm install", "working_directory": "react-todo-app" }}

重要规则 - write_file：
- 写入 React 组件（如 App.tsx）时，若存在对应 CSS 文件（如 App.css），在其他 import 后加上 CSS 导入\
"""


async def run_agent_with_tools(query: str, max_iterations: int = 30) -> str:
    """
    Agent 主循环：流式调用模型 → 执行工具 → 追加历史，直到无工具调用。

    Args:
        query:          用户任务描述
        max_iterations: 最大工具调用轮次，防止死循环

    Returns:
        模型最终文本回复
    """
    history = InMemoryChatMessageHistory()

    # 初始化消息历史
    history.add_messages([
        SystemMessage(content=_SYSTEM_PROMPT.format(cwd=os.getcwd())),
        HumanMessage(content=query),
    ])

    for iteration in range(max_iterations):
        print(f"{_G_BG}⏳ 正在等待 AI 思考...（第 {iteration + 1} 轮）{_RESET}")

        messages = history.messages
        full_ai_message = None

        # write_file 流式预览状态：key = 该工具调用在 parsed_tools 中的下标
        # 值 = 已打印字符数，-1 表示尚未打印过标题
        write_printed: dict[int, int] = {}

        print(f"{_B_BG}\n🚀 Agent 开始思考并生成流...\n{_RESET}")

        # astream 返回 AIMessageChunk 异步生成器
        async for chunk in model_with_tools.astream(messages):
            # 累积 chunk，最终 full_ai_message 包含完整 tool_calls
            full_ai_message = full_ai_message + chunk if full_ai_message else chunk

            # ── write_file 流式预览 ────────────────────────────
            # JsonOutputToolsParser.parse_result(partial=True) 内部调用
            # parse_partial_json，能正确处理未闭合的 JSON 及转义字符，
            # 比 regex 解析原始 args 字符串健壮得多。
            # 返回值: [{"type": tool_name, "args": dict(partial), ...}, ...]
            try:
                partial_tools = _tool_parser.parse_result(
                    [ChatGeneration(message=full_ai_message)], partial=True
                )
            except Exception:
                partial_tools = []

            if partial_tools:
                for i, tool in enumerate(partial_tools):
                    if tool["type"] == "write_file":
                        content: str = tool.get("args", {}).get("content") or ""
                        prev = write_printed.get(i, -1)

                        if prev < 0 and content:
                            # 首次出现内容时打印标题
                            fp = tool.get("args", {}).get("file_path") or "unknown"
                            print(
                                f"\n{_B_BG}[工具调用] write_file(\"{fp}\")"
                                f" - 开始写入（流式预览）\n{_RESET}"
                            )
                            write_printed[i] = 0

                        # 仅打印本次新增的字符
                        if content and len(content) > write_printed.get(i, 0):
                            sys.stdout.write(content[write_printed[i]:])
                            sys.stdout.flush()
                            write_printed[i] = len(content)

            # ── 无工具调用时，直接输出文本内容 ───────────────
            elif chunk.content and not chunk.tool_call_chunks:
                text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                sys.stdout.write(text)
                sys.stdout.flush()

        # 将完整 AI 消息存入历史（包含完整 tool_calls）
        history.add_messages([full_ai_message])
        print(f"\n{_GREEN}✅ 消息已完整存入历史{_RESET}")

        # ── 检查是否还有工具调用 ──────────────────────────────
        tool_calls = getattr(full_ai_message, "tool_calls", None) or []
        if not tool_calls:
            # 模型不再调用工具，返回最终回复
            final_content = full_ai_message.content
            print(f"\n✨ AI 最终回复:\n{final_content}\n")
            return final_content

        # ── 执行工具调用，将结果追加到历史 ───────────────────
        for tc in tool_calls:
            tool = _TOOL_MAP.get(tc["name"])
            if tool:
                # invoke 同步执行工具（subprocess/文件 IO 均为阻塞操作）
                tool_result = tool.invoke(tc["args"])
                history.add_messages([
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tc["id"],
                    )
                ])

    # 超出最大迭代次数，返回历史中最后一条内容
    last = history.messages[-1]
    return last.content if hasattr(last, "content") else str(last)


# ──────────────────────────────────────────────────────────
# 示例任务：创建 React TodoList 应用
# ──────────────────────────────────────────────────────────
_CASE1 = """\
创建一个功能丰富的 React TodoList 应用：

1. 创建项目：echo -e "n\\nn" | pnpm create vite react-todo-app --template react-ts
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

去掉 main.tsx 里的 index.css 导入

之后在 react-todo-app 项目中：
1. 使用 pnpm install 安装依赖
2. 使用 pnpm run dev 启动服务器
"""


if __name__ == "__main__":
    try:
        asyncio.run(run_agent_with_tools(_CASE1))
    except KeyboardInterrupt:
        print("\n⏹  已手动中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}\n")
        raise SystemExit(1)
