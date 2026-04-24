"""
all_tools.py - LangChain 工具定义模块

提供四个基础工具供 Agent 使用：
  - read_file    读取文件内容
  - write_file   写入文件内容（自动创建目录）
  - execute_command  执行 Shell 命令（实时输出到终端）
  - list_directory   列出目录内容
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional

from langchain_core.tools import tool


# ────────────────────────────────────────────────────────────
# 1. 读取文件工具
# ────────────────────────────────────────────────────────────
@tool
def read_file(file_path: str) -> str:
    """读取指定路径的文件内容"""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        print(f'  [工具调用] read_file("{file_path}") - 成功读取 {len(content)} 字节')
        return f"文件内容:\n{content}"
    except Exception as e:
        print(f'  [工具调用] read_file("{file_path}") - 错误: {e}')
        return f"读取文件失败: {e}"


# ────────────────────────────────────────────────────────────
# 2. 写入文件工具
# ────────────────────────────────────────────────────────────
@tool
def write_file(file_path: str, content: str) -> str:
    """向指定路径写入文件内容，目录不存在时自动创建"""
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            # exist_ok=True 等价于 mkdir({ recursive: true })
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f'  [工具调用] write_file("{file_path}") - 成功写入 {len(content)} 字节')
        return f"文件写入成功: {file_path}"
    except Exception as e:
        print(f'  [工具调用] write_file("{file_path}") - 错误: {e}')
        return f"写入文件失败: {e}"


# ────────────────────────────────────────────────────────────
# 3. 执行命令工具（stdio 继承父进程，实时显示输出）
# ────────────────────────────────────────────────────────────
@tool
def execute_command(command: str, working_directory: Optional[str] = None) -> str:
    """执行系统命令，支持通过 working_directory 参数指定工作目录，输出实时显示在终端。
    使用 working_directory 时，command 中不要包含 cd 命令。"""
    cwd = working_directory or os.getcwd()
    dir_info = f" - 工作目录: {working_directory}" if working_directory else ""
    print(f'  [工具调用] execute_command("{command}"){dir_info}')

    # shell=True 支持管道、环境变量等完整 Shell 特性
    # 不捕获 stdout/stderr，让输出实时流向终端（等价于 stdio: 'inherit'）
    result = subprocess.run(command, shell=True, cwd=cwd)

    if result.returncode == 0:
        print(f'  [工具调用] execute_command("{command}") - 执行成功')
        cwd_hint = (
            f'\n\n重要提示：命令在目录 "{working_directory}" 中执行成功。'
            f'如需继续在此目录执行命令，请使用 working_directory: "{working_directory}" 参数，不要使用 cd 命令。'
            if working_directory
            else ""
        )
        return f"命令执行成功: {command}{cwd_hint}"
    else:
        print(f'  [工具调用] execute_command("{command}") - 执行失败，退出码: {result.returncode}')
        return f"命令执行失败，退出码: {result.returncode}"


# ────────────────────────────────────────────────────────────
# 4. 列出目录内容工具
# ────────────────────────────────────────────────────────────
@tool
def list_directory(directory_path: str) -> str:
    """列出指定目录下的所有文件和文件夹"""
    try:
        entries = os.listdir(directory_path)
        print(f'  [工具调用] list_directory("{directory_path}") - 找到 {len(entries)} 个项目')
        return "目录内容:\n" + "\n".join(f"- {e}" for e in entries)
    except Exception as e:
        print(f'  [工具调用] list_directory("{directory_path}") - 错误: {e}')
        return f"列出目录失败: {e}"
