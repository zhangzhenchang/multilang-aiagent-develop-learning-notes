"""stream_tool_calls_raw.py - 流式输出原始工具参数片段"""
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


async def main() -> None:
    print("🌊 流式 Tool Calls 演示 - 直接打印原始 tool_call_chunks\n")
    try:
        stream = model_with_tool.astream("详细介绍牛顿的生平和成就")
        print("📡 实时输出流式 tool_call_chunks:\n")
        async for chunk in stream:
            if chunk.tool_call_chunks:
                args_chunk = chunk.tool_call_chunks[0].get("args")
                if args_chunk:
                    print(args_chunk, end="")
        print("\n\n✅ 流式输出完成")
    except Exception as error:
        print(f"\n❌ 错误: {error}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
