"""runnable.py - RunnableSequence 基础示例：翻译 + 结构化输出"""
import asyncio
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from utils import create_chat_model


# ---------- 模型 ----------
model = create_chat_model()

# ---------- 输出结构（对应 JS 的 Zod schema）----------
class TranslationResult(BaseModel):
    translation: str = Field(description="翻译后的英文文本")
    keywords: List[str] = Field(description="3个关键词")

# PydanticOutputParser：把模型纯文本输出解析成 Python 对象
output_parser = PydanticOutputParser(pydantic_object=TranslationResult)

# PromptTemplate：带占位符的提示模板
prompt_template = PromptTemplate.from_template(
    "将以下文本翻译成英文，然后总结为3个关键词。\n\n文本：{text}\n\n{format_instructions}"
)

# 用管道运算符 | 将多个 Runnable 串联成管道 prompt → model → parser
# Python 中等价于 JS 的 RunnableSequence.from([...]) 或 .pipe()
chain = prompt_template | model | output_parser

# 输入字典：text 为待翻译文本，format_instructions 告知模型输出格式
input_data = {
    "text": "LangChain 是一个强大的 AI 应用开发框架",
    "format_instructions": output_parser.get_format_instructions(),
}


async def main() -> None:
    result = await chain.ainvoke(input_data)
    print("✅ 最终结果:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
