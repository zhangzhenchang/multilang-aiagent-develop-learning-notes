"""normal.py - 普通文本输出后手动解析 JSON"""
import json

from utils import create_chat_model


model = create_chat_model()
question = "请介绍一下爱因斯坦的信息。请以 JSON 格式返回，包含以下字段：name（姓名）、birth_year（出生年份）、nationality（国籍）、major_achievements（主要成就，数组）、famous_theory（著名理论）。"


async def main() -> None:
    try:
        print("🤔 正在调用大模型...\n")
        response = await model.ainvoke(question)
        print("✅ 收到响应:\n")
        print(response.content)
        json_result = json.loads(str(response.content))
        print("\n📋 解析后的 JSON 对象:")
        print(json_result)
    except Exception as error:
        print(f"❌ 错误: {error}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
