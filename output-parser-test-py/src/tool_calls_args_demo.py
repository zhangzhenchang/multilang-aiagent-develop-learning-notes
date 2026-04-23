"""tool_calls_args_demo.py - 工具调用参数解析示例"""
from pydantic import BaseModel, Field

from utils import create_chat_model


class ScientistInfo(BaseModel):
    name: str = Field(description="科学家的全名")
    birth_year: int = Field(description="出生年份")
    nationality: str = Field(description="国籍")
    fields: list[str] = Field(description="研究领域列表")


model = create_chat_model()
model_with_tool = model.bind_tools([ScientistInfo])


async def main() -> None:
    response = await model_with_tool.ainvoke("介绍一下爱因斯坦")
    print("response.tool_calls:", response.tool_calls)
    if response.tool_calls:
        result = response.tool_calls[0]["args"]
        print("结构化结果:", result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
