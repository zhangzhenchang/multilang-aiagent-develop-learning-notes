"""RunnableEach.py - 对列表中的每个元素依次应用同一条链"""
import asyncio

from langchain_core.runnables import RunnableLambda


to_upper_case  = RunnableLambda(lambda x: x.upper())          # 转大写
add_greeting   = RunnableLambda(lambda x: f"你好，{x}！")      # 加问候语

# 先转大写，再加问候语；| 运算符创建 RunnableSequence
process_item = to_upper_case | add_greeting

# .map() 返回一个新 Runnable，对输入列表中的每个元素调用 process_item
# 等价于旧版 RunnableEach(bound=process_item)，langchain-core 1.0 中已移除 RunnableEach
chain = process_item.map()


async def main() -> None:
    input_list = ["alice", "bob", "carol"]
    result = await chain.ainvoke(input_list)

    print("✅ RunnableEach - 数组元素处理:")
    print("输入:", input_list)
    print("输出:", result)


if __name__ == "__main__":
    asyncio.run(main())
