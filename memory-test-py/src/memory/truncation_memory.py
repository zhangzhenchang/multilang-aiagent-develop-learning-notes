"""truncation_memory.py - 按消息数量和 token 数量截断历史消息"""
import tiktoken
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, trim_messages


# 用于估算 token 数量的 tokenizer。
encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages) -> int:
    """统计一组消息的 token 数。"""
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += len(encoder.encode(content))
    return total


async def message_count_truncation() -> None:
    """按消息条数截断，只保留最近几条。"""
    history = InMemoryChatMessageHistory()
    max_messages = 4
    messages = [
        {"type": "human", "content": "我叫张三"},
        {"type": "ai", "content": "你好张三，很高兴认识你！"},
        {"type": "human", "content": "我今年25岁"},
        {"type": "ai", "content": "25岁正是青春年华，有什么我可以帮助你的吗？"},
        {"type": "human", "content": "我喜欢编程"},
        {"type": "ai", "content": "编程很有趣！你主要用什么语言？"},
        {"type": "human", "content": "我住在北京"},
        {"type": "ai", "content": "北京是个很棒的城市！"},
        {"type": "human", "content": "我的职业是软件工程师"},
        {"type": "ai", "content": "软件工程师是个很有前景的职业！"},
    ]
    for item in messages:
        if item["type"] == "human":
            history.add_message(HumanMessage(item["content"]))
        else:
            history.add_message(AIMessage(item["content"]))

    trimmed_messages = history.messages[-max_messages:]
    print(f"保留消息数量: {len(trimmed_messages)}")
    print("保留的消息:", "\n  ".join(f"{msg.__class__.__name__}: {msg.content}" for msg in trimmed_messages))


async def token_count_truncation() -> None:
    """按 token 数截断，更贴近模型真实上下文限制。"""
    history = InMemoryChatMessageHistory()
    max_tokens = 100
    messages = [
        {"type": "human", "content": "我叫李四"},
        {"type": "ai", "content": "你好李四，很高兴认识你！"},
        {"type": "human", "content": "我是一名设计师"},
        {"type": "ai", "content": "设计师是个很有创造力的职业！你主要做什么类型的设计？"},
        {"type": "human", "content": "我喜欢艺术和音乐"},
        {"type": "ai", "content": "艺术和音乐都是很好的爱好，它们能激发创作灵感。"},
        {"type": "human", "content": "我擅长 UI/UX 设计"},
        {"type": "ai", "content": "UI/UX 设计非常重要，好的用户体验能让产品更成功！"},
    ]
    for item in messages:
        if item["type"] == "human":
            history.add_message(HumanMessage(item["content"]))
        else:
            history.add_message(AIMessage(item["content"]))

    '''
     trim_messages 的 api，可以根据 token 来截断消息
     
     lambda 是一个“匿名函数”（inline function），相当于你临时定义了一个函数，但不需要写 def
        token_counter=lambda msgs: count_tokens(msgs)
        等价于：
        def token_counter(msgs):
        return count_tokens(msgs)
    '''
    # 按 token 上限裁剪消息：
    trimmed_messages = trim_messages(
        # - 输入是完整历史 history.messages
        history.messages,
        # - max_tokens 限制裁剪后总 token 数
        max_tokens=max_tokens,
        # - token_counter 指定“如何计算 token”
        token_counter=lambda msgs: count_tokens(msgs),
        # - strategy="last" 表示优先保留最近消息，丢弃更早消息
        strategy="last",
    )
    total_tokens = count_tokens(trimmed_messages)
    print(f"总 token 数: {total_tokens}/{max_tokens}")
    print(f"保留消息数量: {len(trimmed_messages)}")
    print(
        "保留的消息:",
        "\n  ".join(
            f"{msg.__class__.__name__} ({len(encoder.encode(msg.content if isinstance(msg.content, str) else str(msg.content)))} tokens): {msg.content}"
            for msg in trimmed_messages
        ),
    )


async def run_all() -> None:
    await message_count_truncation()
    await token_count_truncation()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_all())
