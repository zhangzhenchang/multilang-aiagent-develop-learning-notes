"""RunnableLambda.py - 将普通函数包装为 Runnable 并串联成管道"""
import asyncio

from langchain_core.runnables import RunnableLambda


def add_one(x: int) -> int:
    print(f"输入: {x}")
    return x + 1


def multiply_two(x: int) -> int:
    print(f"输入: {x}")
    return x * 2


# RunnableLambda：把普通 Python 函数包装成可组合的 Runnable
add_one_runnable      = RunnableLambda(add_one)
multiply_two_runnable = RunnableLambda(multiply_two)

# 串联：先 +1，再 *2，再 +1  =>  (5+1)*2+1 = 13
# | 运算符等价于 JS 的 RunnableSequence.from([...])
chain = add_one_runnable | multiply_two_runnable | add_one_runnable


async def main() -> None:
    result = await chain.ainvoke(5)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
