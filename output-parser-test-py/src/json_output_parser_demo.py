"""json_output_parser_demo.py - JsonOutputParser 示例"""
from langchain_core.output_parsers import JsonOutputParser

from utils import create_chat_model, strip_markdown_fence


model = create_chat_model()
parser = JsonOutputParser()
question = f"""请介绍一下爱因斯坦的信息。请以 JSON 格式返回，包含以下字段：name（姓名）、birth_year（出生年份）、nationality（国籍）、major_achievements（主要成就，数组）、famous_theory（著名理论）。

{parser.get_format_instructions()}"""


async def main() -> None:
    try:
        print("question:", question)
        print("🤔 正在调用大模型（使用 JsonOutputParser）...\n")
        response = await model.ainvoke(question)
        print("📤 模型原始响应:\n")
        print(response.content)
        result = parser.parse(strip_markdown_fence(str(response.content)))
        print("✅ JsonOutputParser 自动解析的结果:\n")
        print(result)
    except Exception as error:
        print(f"❌ 错误: {error}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
