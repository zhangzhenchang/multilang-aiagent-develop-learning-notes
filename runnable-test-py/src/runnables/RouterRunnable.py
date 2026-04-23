"""RouterRunnable.py - 根据 key 动态路由到不同 Runnable"""
import asyncio

from langchain_core.runnables import RouterRunnable, RunnableLambda


# 两个简单的处理函数
to_upper_case = RunnableLambda(lambda text: text.upper())      # 转大写
reverse_text  = RunnableLambda(lambda text: text[::-1])        # 反转字符串

# RouterRunnable：根据输入的 key 字段选择对应的 Runnable 执行
# 输入格式：{"key": "<runnable名>", "input": <实际输入>}
router = RouterRunnable(runnables={
    "toUpperCase": to_upper_case,
    "reverseText": reverse_text,
})


async def main() -> None:
    # 测试：调用 reverseText
    result1 = await router.ainvoke({"key": "reverseText", "input": "Hello World"})
    print("reverseText 结果:", result1)

    # 测试：调用 toUpperCase
    result2 = await router.ainvoke({"key": "toUpperCase", "input": "Hello World"})
    print("toUpperCase 结果:", result2)


if __name__ == "__main__":
    asyncio.run(main())
