"""structured_output_parser_demo.py - StructuredOutputParser 基础示例"""
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser

from utils import create_chat_model, strip_markdown_fence


model = create_chat_model()
parser = StructuredOutputParser.from_response_schemas(
    [
        ResponseSchema(name="name", description="姓名"),
        ResponseSchema(name="birth_year", description="出生年份"),
        ResponseSchema(name="nationality", description="国籍"),
        ResponseSchema(name="major_achievements", description="主要成就，用逗号分隔的字符串"),
        ResponseSchema(name="famous_theory", description="著名理论"),
    ]
)
question = f"""请介绍一下爱因斯坦的信息。

{parser.get_format_instructions()}"""


async def main() -> None:
    try:
        print("question:", question)
        response = await model.ainvoke(question)
        print("📤 模型原始响应:\n")
        print(response.content)
        result = parser.parse(strip_markdown_fence(str(response.content)))
        print("\n✅ StructuredOutputParser 自动解析的结果:\n")
        print(result)
    except Exception as error:
        print(f"❌ 错误: {error}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
