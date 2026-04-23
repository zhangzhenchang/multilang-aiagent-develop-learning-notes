"""xml_output_parser_demo.py - XMLOutputParser 示例"""
from langchain_core.output_parsers import XMLOutputParser

from utils import create_chat_model, strip_markdown_fence


model = create_chat_model()
parser = XMLOutputParser()
question = f"""请提取以下文本中的人物信息：阿尔伯特·爱因斯坦出生于 1879 年，是一位伟大的物理学家。

{parser.get_format_instructions()}"""


async def main() -> None:
    try:
        print("question:", question)
        print("🤔 正在调用大模型（使用 XMLOutputParser）...\n")
        response = await model.ainvoke(question)
        print("📤 模型原始响应:\n")
        print(response.content)
        result = parser.parse(strip_markdown_fence(str(response.content)))
        print("\n✅ XMLOutputParser 自动解析的结果:\n")
        print(result)
    except Exception as error:
        print(f"❌ 错误: {error}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
