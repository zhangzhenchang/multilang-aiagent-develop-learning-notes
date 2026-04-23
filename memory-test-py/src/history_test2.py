"""history_test2.py - 基于 JSON 文件的消息历史示例"""
from langchain_core.messages import HumanMessage, SystemMessage

from memory_utils import CHAT_HISTORY_FILE, JSONChatMessageHistory, create_chat_model


# 负责多轮对话回复的聊天模型。
model = create_chat_model(temperature=0.0)


async def file_history_demo() -> None:
    file_path = CHAT_HISTORY_FILE
    # session_id 表示一组独立会话。
    session_id = "user_session_001"
    system_message = SystemMessage("你是一个友好的做菜助手，喜欢分享美食和烹饪技巧。")
    # JSON 文件版历史消息：程序结束后仍会保留。
    history = JSONChatMessageHistory(file_path=file_path, session_id=session_id)

    print("[第一轮对话]")
    user_message_1 = HumanMessage("红烧肉怎么做")
    history.add_message(user_message_1)
    messages_1 = [system_message, *history.messages]
    response_1 = await model.ainvoke(messages_1)
    history.add_message(response_1)
    print(f"用户: {user_message_1.content}")
    print(f"助手: {response_1.content}")
    print(f"✓ 对话已保存到文件: {file_path}\n")

    print("[第二轮对话]")
    user_message_2 = HumanMessage("好吃吗？")
    history.add_message(user_message_2)
    messages_2 = [system_message, *history.messages]
    response_2 = await model.ainvoke(messages_2)
    history.add_message(response_2)
    print(f"用户: {user_message_2.content}")
    print(f"助手: {response_2.content}")
    print("✓ 对话已更新到文件\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(file_history_demo())
