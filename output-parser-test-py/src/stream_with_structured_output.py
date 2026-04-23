"""stream_with_structured_output.py - 结构化流式输出示例"""
from pydantic import BaseModel, Field

from utils import create_chat_model


class MozartInfo(BaseModel):
    name: str | None = Field(default=None, description="姓名")
    birth_year: int | None = Field(default=None, description="出生年份")
    death_year: int | None = Field(default=None, description="去世年份")
    nationality: str | None = Field(default=None, description="国籍")
    occupation: str | None = Field(default=None, description="职业")
    famous_works: list[str] = Field(default_factory=list, description="著名作品列表")
    biography: str | None = Field(default=None, description="简短传记")


model = create_chat_model()
structured_model = model.with_structured_output(MozartInfo, method="function_calling")
prompt = "请结构化介绍莫扎特的信息"


async def main() -> None:
    print("🌊 流式结构化输出演示（with_structured_output）\n")
    try:
        stream = structured_model.astream(prompt)
        chunk_count = 0
        result = None
        print("📡 接收流式数据:\n")
        async for chunk in stream:
            chunk_count += 1
            result = chunk
            print(f"[Chunk {chunk_count}]")
            print(chunk)
        print(f"\n✅ 共接收 {chunk_count} 个数据块\n")
        if result is not None:
            print("📊 最终结构化结果:\n")
            print(result)
    except Exception as error:
        print(f"\n❌ 错误: {error}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
