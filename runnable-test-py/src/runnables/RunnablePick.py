"""RunnablePick.py - 从字典中挑选指定字段"""
import asyncio

from langchain_core.runnables import RunnableLambda, RunnablePick


# 原始输入：包含多个字段
input_data = {
    "name":    "神光",
    "age":     30,
    "city":    "北京",
    "country": "中国",
    "email":   "shenguang@example.com",
    "phone":   "+86-13800138000",
}

chain = (
    # 步骤 1：在原字典基础上追加一个聚合字段 fullInfo
    RunnableLambda(lambda x: {
        **x,
        "fullInfo": f"{x['name']}，{x['age']}岁，来自{x['city']}",
    })
    # 步骤 2：RunnablePick 只保留指定的 key，其余字段丢弃
    | RunnablePick(["name", "fullInfo"])
)


async def main() -> None:
    result = await chain.ainvoke(input_data)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
