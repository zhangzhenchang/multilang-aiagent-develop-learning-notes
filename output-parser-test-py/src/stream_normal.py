"""stream_normal.py - 普通流式输出示例"""
from utils import create_chat_model


model = create_chat_model()
prompt = "详细介绍莫扎特的信息。"


async def main() -> None:
    print("🌊 普通流式输出演示（无结构化）\n")
    try:
        stream = model.astream(prompt)
        full_content = ""
        chunk_count = 0
        print("📡 接收流式数据:\n")
        async for chunk in stream:
            chunk_count += 1
            content = str(chunk.content)
            full_content += content
            print(content, end="")
        print(f"\n\n✅ 共接收 {chunk_count} 个数据块\n")
        print(f"📝 完整内容长度: {len(full_content)} 字符")
    except Exception as error:
        print(f"\n❌ 错误: {error}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
