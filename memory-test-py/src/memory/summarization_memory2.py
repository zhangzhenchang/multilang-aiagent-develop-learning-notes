"""summarization_memory2.py - 基于 token 阈值的总结策略演示"""
import tiktoken
from langchain_core.messages import get_buffer_string

from src.memory_utils import create_chat_model, make_sample_messages


# 用于生成摘要的聊天模型。
model = create_chat_model(temperature=0.0)
# cl100k_base 是常见 OpenAI/Qwen 兼容 tokenizer 编码。
encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages) -> int:
    """统计一组消息大约占用多少 token。"""
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += len(encoder.encode(content))
    return total


async def summarize_history(messages) -> str:
    if not messages:
        return ""
    conversation_text = get_buffer_string(messages, human_prefix="用户", ai_prefix="助手")
    summary_prompt = f"请总结以下对话的核心内容，保留重要信息：\n\n{conversation_text}\n\n总结："
    summary_response = await model.ainvoke(summary_prompt)
    return str(summary_response.content)


async def summarization_memory_demo() -> None:
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
    # 基于 token 数而不是消息条数来判断是否要摘要，更贴近真实上下文窗口。
    max_tokens = 200
    keep_recent_tokens = 80
    all_messages = history.messages
    total_tokens = count_tokens(all_messages)

    if total_tokens >= max_tokens:
        recent_messages = []
        recent_tokens = 0
        # 从后往前保留最近消息，直到达到保留 token 上限。
        for msg in reversed(all_messages):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            msg_tokens = len(encoder.encode(content))
            if recent_tokens + msg_tokens <= keep_recent_tokens:
                recent_messages.insert(0, msg)
                recent_tokens += msg_tokens
            else:
                break
        messages_to_summarize = all_messages[: len(all_messages) - len(recent_messages)]
        summarize_tokens = count_tokens(messages_to_summarize)
        print("\n💡 Token 数量超过阈值，开始总结...")
        print(f"📝 将被总结的消息数量: {len(messages_to_summarize)} ({summarize_tokens} tokens)")
        print(f"📝 将被保留的消息数量: {len(recent_messages)} ({recent_tokens} tokens)")
        summary = await summarize_history(messages_to_summarize)
        history.clear()
        for msg in recent_messages:
            history.add_message(msg)
        print(f"\n保留消息数量: {len(recent_messages)}")
        print(
            "保留的消息:",
            "\n  ".join(
                f"{msg.__class__.__name__} ({len(encoder.encode(msg.content if isinstance(msg.content, str) else str(msg.content)))} tokens): {msg.content}"
                for msg in recent_messages
            ),
        )
        print(f"\n总结内容（不包含保留的消息）: {summary}")
    else:
        print(f"\nToken 数量 ({total_tokens}) 未超过阈值 ({max_tokens})，无需总结")


if __name__ == "__main__":
    import asyncio

    asyncio.run(summarization_memory_demo())
