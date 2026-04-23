"""RunnableWithMessageHistory.py - 为链自动注入多轮对话历史"""
import asyncio
import os
from typing import Dict

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

# ---------- 模型 ----------
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-coder-turbo"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.3,
)

# ---------- Prompt ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个简洁、有帮助的中文助手，会用 1-2 句话回答用户问题，重点给出明确、有用的信息。"),
    # MessagesPlaceholder：历史消息的占位符，由 RunnableWithMessageHistory 自动填充
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# 基础链：prompt → model → 字符串解析
simple_chain = prompt | model | StrOutputParser()

# ---------- 消息历史存储（按 session_id 隔离）----------
# 每个 session_id 对应一个独立的 InMemoryChatMessageHistory 实例
message_histories: Dict[str, InMemoryChatMessageHistory] = {}


def get_message_history(session_id: str) -> InMemoryChatMessageHistory:
    """根据 session_id 获取或创建对应的消息历史"""
    if session_id not in message_histories:
        message_histories[session_id] = InMemoryChatMessageHistory()
    return message_histories[session_id]


# RunnableWithMessageHistory：包装基础链，每次 invoke 时自动读写历史
# - get_session_history：通过 session_id 获取历史存储
# - input_messages_key：输入字典中人类问题的 key
# - history_messages_key：历史消息注入到 prompt 的 key（对应 MessagesPlaceholder）
chain = RunnableWithMessageHistory(
    runnable=simple_chain,
    get_session_history=get_message_history,
    input_messages_key="question",
    history_messages_key="history",
)

# session_id 用于区分不同用户/对话
SESSION_CONFIG = {"configurable": {"session_id": "user-123"}}


async def main() -> None:
    print("--- 第一次对话（提供信息） ---")
    q1 = "我的名字是神光，我来自山东，我喜欢编程、写作、金铲铲。"
    result1 = await chain.ainvoke({"question": q1}, config=SESSION_CONFIG)
    print(f"问题: {q1}")
    print("回答:", result1)
    print()

    print("--- 第二次对话（询问之前的信息） ---")
    q2 = "我刚才说我来自哪里？"
    result2 = await chain.ainvoke({"question": q2}, config=SESSION_CONFIG)
    print(f"问题: {q2}")
    print("回答:", result2)
    print()

    print("--- 第三次对话（继续询问） ---")
    q3 = "我的爱好是什么？"
    result3 = await chain.ainvoke({"question": q3}, config=SESSION_CONFIG)
    print(f"问题: {q3}")
    print("回答:", result3)
    print()


if __name__ == "__main__":
    asyncio.run(main())
