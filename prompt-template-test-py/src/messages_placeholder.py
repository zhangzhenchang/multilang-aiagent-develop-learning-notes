"""messages_placeholder.py - MessagesPlaceholder 示例"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


chat_prompt_with_history = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一名资深工程效率顾问，善于在多轮对话上下文中给出建议。"),
        MessagesPlaceholder("history"),
        ("human", "这是用户本轮的新问题：{current_input}\n\n请结合历史对话给出建议。"),
    ]
)


async def main() -> None:
    history_messages = [
        ("human", "我们团队最近在做一个内部的周报自动生成工具。"),
        ("ai", "可以先把数据源梳理清楚，再考虑 Prompt 模块化设计。"),
        ("human", "我们已经把 Prompt 拆成了人设、背景、任务、格式四块。"),
        ("ai", "很好，接下来可以考虑做成可复用的 PipelinePromptTemplate。"),
    ]
    formatted_messages = await chat_prompt_with_history.aformat_prompt(
        history=history_messages,
        current_input="现在我们想再优化一下多人协同编辑周报的流程，有什么建议？",
    )
    print("包含历史对话的消息数组：")
    print(formatted_messages.to_messages())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
