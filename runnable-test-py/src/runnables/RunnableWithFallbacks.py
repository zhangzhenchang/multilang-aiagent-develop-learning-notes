"""RunnableWithFallbacks.py - 主链失败时自动降级到备用链"""
import asyncio

from langchain_core.runnables import RunnableLambda


# ---------- 三个模拟翻译服务，优先级从高到低 ----------

async def premium_translator(text: str) -> str:
    """高级翻译服务（模拟超时失败）"""
    print("[Premium] 尝试翻译...")
    raise RuntimeError("Premium 服务超时")


async def standard_translator(text: str) -> str:
    """标准翻译服务（模拟可用）"""
    print("[Standard] 尝试翻译...")
    return "xxx"  # 模拟返回结果；改成 raise 可演示继续降级


async def local_translator(text: str) -> str:
    """本地词典翻译（最终兜底）"""
    print("[Local] 使用本地词典翻译...")
    dictionary = {"hello": "你好", "world": "世界", "goodbye": "再见"}
    words = text.lower().split()
    return "".join(dictionary.get(w, w) for w in words)


premium  = RunnableLambda(premium_translator)
standard = RunnableLambda(standard_translator)
local    = RunnableLambda(local_translator)

# with_fallbacks()：当 premium 抛出异常时，依次尝试 standard → local
# exceptions_to_handle 默认捕获所有 Exception
translator = premium.with_fallbacks(
    fallbacks=[standard, local],
)


async def main() -> None:
    result = await translator.ainvoke("hello world")
    print("翻译结果:", result)


if __name__ == "__main__":
    asyncio.run(main())
