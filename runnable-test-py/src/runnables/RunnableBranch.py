"""RunnableBranch.py - 根据条件分支执行不同逻辑"""
import asyncio

from langchain_core.runnables import RunnableBranch, RunnableLambda


# 分支处理函数：根据不同情况返回不同字符串
handle_positive = RunnableLambda(lambda x: f"正数: {x} + 10 = {x + 10}")
handle_negative = RunnableLambda(lambda x: f"负数: {x} - 10 = {x - 10}")
handle_even     = RunnableLambda(lambda x: f"偶数: {x} * 2 = {x * 2}")
handle_default  = RunnableLambda(lambda x: f"默认: {x}")

# RunnableBranch：按顺序检查条件，命中第一个为真的条件后执行对应分支
# 最后一个参数为默认分支（无条件执行）
branch = RunnableBranch(
    (lambda x: x > 0, handle_positive),   # 正数分支
    (lambda x: x < 0, handle_negative),   # 负数分支
    (lambda x: x % 2 == 0, handle_even),  # 偶数分支（此处 x==0 时命中）
    handle_default,                        # 默认分支
)


async def main() -> None:
    test_cases = [5, -3, 4, 0]
    for val in test_cases:
        result = await branch.ainvoke(val)
        print(f"输入: {val} => {result}")


if __name__ == "__main__":
    asyncio.run(main())
