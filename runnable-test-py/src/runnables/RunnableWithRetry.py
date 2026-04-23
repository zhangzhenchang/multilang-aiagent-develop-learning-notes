"""RunnableWithRetry.py - 失败时自动重试"""
import asyncio
import random

from langchain_core.runnables import RunnableLambda


# 用于记录当前尝试次数（全局可变状态）
attempt = 0


async def unstable_func(input_val: str) -> str:
    """模拟一个 70% 概率失败的不稳定操作"""
    global attempt
    attempt += 1
    print(f"第 {attempt} 次尝试，输入: {input_val}")

    if random.random() < 0.7:
        print("本次尝试失败，抛出错误。")
        raise RuntimeError("模拟的随机错误")

    print("本次尝试成功。")
    return f"成功处理: {input_val}"


unstable_runnable = RunnableLambda(unstable_func)

# with_retry()：失败时自动重试，最多尝试 stop_after_attempt 次
# wait_exponential_jitter=True 表示每次重试前加随机抖动延迟
runnable_with_retry = unstable_runnable.with_retry(
    stop_after_attempt=5,
)


async def main() -> None:
    try:
        result = await runnable_with_retry.ainvoke("演示 withRetry")
        print("✅ 最终结果:", result)
    except Exception as err:
        print("❌ 重试多次后仍然失败:", err)


if __name__ == "__main__":
    asyncio.run(main())
