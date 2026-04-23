"""history_test.py - 内存消息历史示例"""
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage

from memory_utils import create_chat_model


# 负责生成回复的聊天模型。
model = create_chat_model(temperature=0.0)


async def in_memory_demo() -> None:
    # 内存版历史消息：程序结束后就会丢失。
    history = InMemoryChatMessageHistory()
    system_message = SystemMessage("你是一个友好、幽默的做菜助手，喜欢分享美食和烹饪技巧。")

    print("[第一轮对话]")
    user_message_1 = HumanMessage("你今天吃的什么？")
    history.add_message(user_message_1)
    messages_1 = [system_message, *history.messages]
    response_1 = await model.ainvoke(messages_1)
    history.add_message(response_1)
    print(f"用户: {user_message_1.content}")
    print(f"助手: {response_1.content}\n")
    print(f'history\n{history}')

    print("[第二轮对话 - 基于历史记录]")
    user_message_2 = HumanMessage("好吃吗？")
    history.add_message(user_message_2)
    messages_2 = [system_message, *history.messages]
    response_2 = await model.ainvoke(messages_2)
    history.add_message(response_2)
    print(f"用户: {user_message_2.content}")
    print(f"助手: {response_2.content}\n")
    print(f'history\n{history}')

    print("[历史消息记录]")
    print(f"共保存了 {len(history.messages)} 条消息：")
    for index, msg in enumerate(history.messages, start=1):
        prefix = "用户" if msg.type == "human" else "助手"
        print(f"  {index}. [{prefix}]: {str(msg.content)[:50]}...")


if __name__ == "__main__":
    import asyncio

    asyncio.run(in_memory_demo())
