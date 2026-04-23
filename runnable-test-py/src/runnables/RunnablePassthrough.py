"""RunnablePassthrough.py - 透传输入并按需追加新字段"""
import asyncio

from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# RunnablePassthrough.assign()：透传所有现有字段，同时追加新字段
# 等价于 {...input, newKey: fn(input)}
chain = (
    # 步骤 1：将裸字符串包装成字典
    RunnableLambda(lambda x: {"concept": x})
    # 步骤 2：透传 concept，同时追加 original 和 processed 两个字段
    | RunnablePassthrough.assign(
        # original：直接透传整个输入字典（RunnablePassthrough 不做任何修改）
        original=RunnablePassthrough(),
        # processed：基于 concept 派生出大写和长度信息
        processed=RunnableLambda(lambda obj: {
            "concept": obj["concept"],
            "upper":   obj["concept"].upper(),
            "length":  len(obj["concept"]),
        }),
    )
)


async def main() -> None:
    result = await chain.ainvoke("神说要有光")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
