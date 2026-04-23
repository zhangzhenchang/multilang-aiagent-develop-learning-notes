"""structured_output_parser2_demo.py - Pydantic 结构化输出解析示例"""
from typing import Optional

from pydantic import BaseModel, Field
from langchain_classic.output_parsers import PydanticOutputParser

from utils import create_chat_model, strip_markdown_fence


class ScientistInfo(BaseModel):
    name: str = Field(description="科学家的全名")
    birth_year: int = Field(description="出生年份")
    death_year: Optional[int] = Field(default=None, description="去世年份，如果还在世则不填")
    nationality: str = Field(description="国籍")
    fields: list[str] = Field(description="研究领域列表")
    major_achievements: list[str] = Field(description="主要成就列表")
    biography: str = Field(description="简短传记，100字以内")


model = create_chat_model()
parser = PydanticOutputParser(pydantic_object=ScientistInfo)
question = f"""请介绍一下居里夫人（Marie Curie）的详细信息，包括她的教育背景、研究领域、获得的奖项、主要成就和著名理论。

{parser.get_format_instructions()}"""


async def main() -> None:
    try:
        print("📋 生成的提示词:\n")
        print(question)
        print("🤔 正在调用大模型（使用 Pydantic Schema）...\n")
        response = await model.ainvoke(question)
        print("📤 模型原始响应:\n")
        print(response.content)
        result = parser.parse(strip_markdown_fence(str(response.content)))
        print("✅ 解析并验证后的结果:\n")
        print(result.model_dump_json(indent=2))
    except Exception as error:
        print(f"❌ 错误: {error}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
