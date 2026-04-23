"""all_tools.py - 自定义工具：读取文件、写入文件、执行命令、列出目录"""
import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool


@tool
def read_file(file_path: str) -> str:
    """读取指定路径的文件内容"""
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        print(f'  [工具调用] read_file("{file_path}") - 成功读取 {len(content)} 字节')
        return f"文件内容:\n{content}"
    except Exception as exc:
        print(f'  [工具调用] read_file("{file_path}") - 错误: {exc}')
        return f"读取文件失败: {exc}"


@tool
def write_file(file_path: str, content: str) -> str:
    """向指定路径写入文件内容，自动创建目录"""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f'  [工具调用] write_file("{file_path}") - 成功写入 {len(content)} 字节')
        return f"文件写入成功: {file_path}"
    except Exception as exc:
        print(f'  [工具调用] write_file("{file_path}") - 错误: {exc}')
        return f"写入文件失败: {exc}"


@tool
def execute_command(command: str, working_directory: str | None = None) -> str:
    """执行系统命令，支持指定工作目录，实时显示输出"""
    cwd = working_directory or os.getcwd()
    print(f'  [工具调用] execute_command("{command}") - 工作目录: {cwd}')

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=False,
            text=True,
        )
        if result.returncode == 0:
            print(f'  [工具调用] execute_command("{command}") - 执行成功')
            cwd_info = ""
            if working_directory:
                cwd_info = (
                    f'\n\n重要提示：命令在目录 "{working_directory}" 中执行成功。'
                    "如果需要在这个项目目录中继续执行命令，请继续传入 working_directory 参数。"
                )
            return f"命令执行成功: {command}{cwd_info}"

        print(f'  [工具调用] execute_command("{command}") - 执行失败，退出码: {result.returncode}')
        return f"命令执行失败，退出码: {result.returncode}"
    except Exception as exc:
        print(f'  [工具调用] execute_command("{command}") - 错误: {exc}')
        return f"执行命令出错: {exc}"


@tool
def list_directory(directory_path: str) -> str:
    """列出指定目录下的所有文件和文件夹"""
    try:
        files = os.listdir(directory_path)
        print(f'  [工具调用] list_directory("{directory_path}") - 找到 {len(files)} 个项目')
        return "目录内容:\n" + "\n".join(f"- {name}" for name in files)
    except Exception as exc:
        print(f'  [工具调用] list_directory("{directory_path}") - 错误: {exc}')
        return f"列出目录失败: {exc}"


TOOLS = [read_file, write_file, execute_command, list_directory]


def main() -> None:
    print("可用工具:")
    for tool_obj in TOOLS:
        print(f"- {tool_obj.name}: {tool_obj.description}")


if __name__ == "__main__":
    main()
