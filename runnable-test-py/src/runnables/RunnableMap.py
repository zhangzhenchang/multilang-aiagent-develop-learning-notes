"""RunnableMap.py - 并行执行多个 Runnable，将结果合并为一个字典"""
import asyncio

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel


# 数学运算：从输入字典中取 num 字段进行计算
add_one     = RunnableLambda(lambda x: x["num"] + 1)
multiply_two = RunnableLambda(lambda x: x["num"] * 2)
square      = RunnableLambda(lambda x: x["num"] * x["num"])

# PromptTemplate：直接作为 Runnable 使用，用输入字典格式化模板
greet_template   = PromptTemplate.from_template("你好，{name}！")
weather_template = PromptTemplate.from_template("今天天气{weather}。")

# RunnableParallel（即 RunnableMap）：并行执行所有子 Runnable
# 每个子 Runnable 接收相同的完整输入，结果以对应 key 汇总返回
# 注意：用关键字参数而非 steps={} 字典，避免结果被额外包一层 steps key
runnable_map = RunnableParallel(
    add      = add_one,           # 数值 +1
    multiply = multiply_two,      # 数值 *2
    square   = square,            # 数值 平方
    greeting = greet_template,    # 格式化问候语
    weather  = weather_template,  # 格式化天气语
)

# 输入字典同时满足所有子 Runnable 的需要
input_data = {
    "name":    "神光",
    "weather": "多云",
    "num":     5,
}


async def main() -> None:
    result = await runnable_map.ainvoke(input_data)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
