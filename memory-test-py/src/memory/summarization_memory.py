"""summarization_memory.py - 基于消息数量阈值的总结策略演示"""
from langchain_core.messages import get_buffer_string

from src.memory_utils import create_chat_model, make_sample_messages


# 用于生成“历史摘要”的聊天模型。
model = create_chat_model(temperature=0.0)


async def summarize_history(messages) -> str:
    """把一段较长的旧对话压缩成摘要。"""
    if not messages:
        return ""

    '''
    get_buffer_string 把一组聊天消息（Human/AI）拼成一段可读的纯文本对话记录。
    在你这个场景里，它主要解决两件事：

    消息对象转文本
    messages 原本是 HumanMessage/AIMessage 对象，不是直接给模型看的字符串。

    加角色前缀
    你指定了：

    human_prefix="用户"
    ai_prefix="助手"
    所以输出文本更清楚“谁说的什么”。

    一句话：
    get_buffer_string 就是把“结构化消息历史”格式化成“带角色标记的连续对话文本”。
    '''
    conversation_text = get_buffer_string(messages, human_prefix="用户", ai_prefix="助手")
    print(f'conversation_text\n{conversation_text}')
    summary_prompt = f"请总结以下对话的核心内容，保留重要信息：\n\n{conversation_text}\n\n总结："
    summary_response = await model.ainvoke(summary_prompt)
    return str(summary_response.content)


async def summarization_memory_demo() -> None:
    # 先构造一段较长的示例历史。
    history = make_sample_messages(
        [
            {"type": "human", "content": "我想学做红烧肉，你能教我吗？"},
            {"type": "ai", "content": "当然可以！红烧肉是一道经典的中式菜肴。首先需要准备五花肉、冰糖、生抽、老抽、料酒等材料。"},
            {"type": "human", "content": "五花肉需要切多大块？"},
            {"type": "ai", "content": "建议切成3-4厘米见方的块，这样既容易入味，口感也更好。切好后可以用开水焯一下去除血沫。"},
            {"type": "human", "content": "炒糖色的时候有什么技巧吗？"},
            {"type": "ai", "content": "炒糖色是关键步骤。用小火慢慢炒，等冰糖完全融化变成焦糖色，冒小泡时就可以下肉了。注意不要炒过头，否则会发苦。"},
            {"type": "human", "content": "需要炖多长时间？"},
            {"type": "ai", "content": "一般需要炖40-60分钟，用小火慢炖，直到肉变得软糯入味。可以用筷子戳一下，能轻松戳透就说明好了。"},
            {"type": "human", "content": "最后收汁的时候要注意什么？"},
            {"type": "ai", "content": "收汁时要用大火，不断翻动，让汤汁均匀包裹在肉块上。看到汤汁变得浓稠，颜色红亮就可以出锅了。"},
        ]
    )
    # 当消息数量达到阈值时，就对旧消息做摘要。
    max_messages = 6
    all_messages = history.messages
    print(f"原始消息数量: {len(all_messages)}")
    print("原始消息:", "\n  ".join(f"{msg.__class__.__name__}: {msg.content}" for msg in all_messages))

    if len(all_messages) >= max_messages:
        # 只保留最近两条，其余旧消息压缩成 summary。
        keep_recent = 2
        recent_messages = all_messages[-keep_recent:]
        messages_to_summarize = all_messages[:-keep_recent]
        print("\n💡 历史消息过多，开始总结...")
        print(f"📝 将被总结的消息数量: {len(messages_to_summarize)}")
        print(f"📝 将被保留的消息数量: {len(recent_messages)}")
        summary = await summarize_history(messages_to_summarize)
        history.clear()
        for msg in recent_messages:
            history.add_message(msg)
        print(f"\n保留消息数量: {len(recent_messages)}")
        print("保留的消息:", "\n  ".join(f"{msg.__class__.__name__}: {msg.content}" for msg in recent_messages))
        print(f"\n总结内容（不包含保留的消息）: {summary}")
    else:
        print("\n消息数量未超过阈值，无需总结")


if __name__ == "__main__":
    import asyncio

    asyncio.run(summarization_memory_demo())
