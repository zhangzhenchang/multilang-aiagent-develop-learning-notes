"""with_structured_output_demo.py - with_structured_output 示例"""
from pydantic import BaseModel, Field

from utils import create_chat_model


class ScientistInfo(BaseModel):
    name: str = Field(description="科学家的全名")
    birth_year: int = Field(description="出生年份")
    nationality: str = Field(description="国籍")
    fields: list[str] = Field(description="研究领域列表")


model = create_chat_model()
'''
现在获取结构化数据一般会用 withStructuredOutput 这个 api
它会判断模型是否支持 tool calls，
支持的话就用 tool 的方式获取结构化数据，否则用 output parser 的方式，不用我们自己去处理
'''
structured_model = model.with_structured_output(ScientistInfo, method="function_calling")


async def main() -> None:
    prompt = "请结构化介绍一下爱因斯坦"
    result = await structured_model.ainvoke(prompt)
    print("结构化结果:", result.model_dump_json(indent=2))
    print(f"\n姓名: {result.name}")
    print(f"出生年份: {result.birth_year}")
    print(f"国籍: {result.nationality}")
    print(f"研究领域: {', '.join(result.fields)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
