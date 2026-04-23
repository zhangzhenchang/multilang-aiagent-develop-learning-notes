"""stream_tool_calls_parser.py - 使用工具解析器解析流式工具结果"""
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from pydantic import BaseModel, Field

from utils import create_chat_model


class ScientistInfo(BaseModel):
    name: str = Field(description="科学家的全名")
    birth_year: int = Field(description="出生年份")
    death_year: int | None = Field(default=None, description="去世年份")
    nationality: str = Field(description="国籍")
    fields: list[str] = Field(description="研究领域列表")
    achievements: list[str] = Field(description="主要成就")
    biography: str = Field(description="简短传记")


model = create_chat_model()
model_with_tool = model.bind_tools([ScientistInfo])
parser = JsonOutputToolsParser()
chain = model_with_tool | parser


async def main() -> None:
    try:
        stream = chain.astream("详细介绍牛顿的生平和成就")
        last_content = ""
        print("📡 实时输出流式内容:\n")
        async for chunk in stream:
            if chunk:
                tool_call = chunk[0]
                current_content = str(tool_call.get("args", {}))
                if len(current_content) > len(last_content):
                    print(current_content[len(last_content):], end="")
                    last_content = current_content
        print("\n\n✅ 流式输出完成")
    except Exception as error:
        print(f"\n❌ 错误: {error}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
