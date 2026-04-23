"""node_exec.py - 使用 subprocess 执行命令的示例"""
import os
import subprocess


def main() -> int:
    command = 'echo "test"'
    cwd = os.getcwd()

    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=False,
        text=True,
    )

    if result.returncode == 0:
        print("命令执行成功")
        return 0

    print(f"命令执行失败，退出码: {result.returncode}")
    return result.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
