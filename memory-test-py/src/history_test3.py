"""history_test3.py - 从 JSON 文件恢复消息历史示例"""
from langchain_core.messages import HumanMessage, SystemMessage

from memory_utils import CHAT_HISTORY_FILE, JSONChatMessageHistory, create_chat_model


# 负责生成回复的聊天模型。
model = create_chat_model(temperature=0.0)


async def file_history_restore_demo() -> None:
    # 从之前保存过的 JSON 文件中恢复历史。
    history = JSONChatMessageHistory(file_path=CHAT_HISTORY_FILE, session_id="user_session_001")
    system_message = SystemMessage("你是一个友好、幽默的做菜助手，喜欢分享美食和烹饪技巧。")

    restored_messages = history.messages
    print(f"从文件恢复了 {len(restored_messages)} 条历史消息：")
    for index, msg in enumerate(restored_messages, start=1):
        prefix = "用户" if msg.type == "human" else "助手"
        print(f"  {index}. [{prefix}]: {str(msg.content)[:50]}...")
    print()

    print("[第三轮对话]")
    user_message_3 = HumanMessage("需要哪些食材？")
    history.add_message(user_message_3)
    messages_3 = [system_message, *history.messages]
    response_3 = await model.ainvoke(messages_3)
    history.add_message(response_3)
    print(f"用户: {user_message_3.content}")
    print(f"助手: {response_3.content}")
    print("✓ 对话已保存到文件\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(file_history_restore_demo())
