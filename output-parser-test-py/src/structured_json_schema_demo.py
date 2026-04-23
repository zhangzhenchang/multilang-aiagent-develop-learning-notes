"""structured_json_schema_demo.py - 原生 JSON Schema 约束输出示例"""
from pydantic import BaseModel, Field

from utils import create_chat_model


class ScientistInfo(BaseModel):
    name: str = Field(description="科学家的全名")
    birth_year: int = Field(description="出生年份")
    field: str = Field(description="主要研究领域")
    achievements: list[str] = Field(description="主要成就列表")


model = create_chat_model(model_name="qwen-max")
structured_model = model.with_structured_output(ScientistInfo, method="function_calling")


async def main() -> None:
    print("🧪 测试原生 JSON Schema 模式...\n")
    prompt = "请结构化介绍一下杨振宁，返回姓名、出生年份、主要研究领域和主要成就。"
    result = await structured_model.ainvoke(prompt)
    print("✅ 收到响应 (结构化对象):")
    print(result.model_dump())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
