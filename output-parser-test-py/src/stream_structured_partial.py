"""stream_structured_partial.py - 先流式文本再整体解析结构化结果"""
from pydantic import BaseModel, Field
from langchain_classic.output_parsers import PydanticOutputParser

from utils import create_chat_model, strip_markdown_fence


class MozartInfo(BaseModel):
    name: str = Field(description="姓名")
    birth_year: int = Field(description="出生年份")
    death_year: int = Field(description="去世年份")
    nationality: str = Field(description="国籍")
    occupation: str = Field(description="职业")
    famous_works: list[str] = Field(description="著名作品列表")
    biography: str = Field(description="简短传记")


model = create_chat_model()
parser = PydanticOutputParser(pydantic_object=MozartInfo)
prompt = f"""详细介绍莫扎特的信息。\n\n{parser.get_format_instructions()}"""


async def main() -> None:
    print("🌊 流式结构化输出演示\n")
    try:
        stream = model.astream(prompt)
        full_content = ""
        chunk_count = 0
        print("📡 接收流式数据:\n")
        async for chunk in stream:
            chunk_count += 1
            content = str(chunk.content)
            full_content += content
            print(content, end="")
        print(f"\n\n✅ 共接收 {chunk_count} 个数据块\n")
        result = parser.parse(strip_markdown_fence(full_content))
        print("📊 解析后的结构化结果:\n")
        print(result.model_dump_json(indent=2))
    except Exception as error:
        print(f"\n❌ 错误: {error}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
