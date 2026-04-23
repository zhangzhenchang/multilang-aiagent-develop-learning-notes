"""RunnableWithCallbacks.py - 通过 callbacks 观测链每一步的执行过程"""
import asyncio
import json
import re
from typing import Any, Dict, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda


# ---------- 文本处理链：清洗 → 分词 → 统计 ----------
# re.sub 折叠连续空白，等价于 JS 的 /\s+/g
clean    = RunnableLambda(lambda text: re.sub(r"\s+", " ", text.strip()))  # 折叠空白
tokenize = RunnableLambda(lambda text: text.split())                        # 按空白分词
count    = RunnableLambda(lambda tokens: {"tokens": tokens, "wordCount": len(tokens)})

chain = clean | tokenize | count


# ---------- 自定义回调处理器 ----------
class StepObserver(BaseCallbackHandler):
    """打印链每一步的启动、结束和错误事件"""

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        # langchain-core 1.0 中 RunnableLambda 不再填充 serialized，
        # 优先用 kwargs["name"]（run_name），其次从 serialized["id"] 取末段
        step = (
            kwargs.get("name")
            or (serialized or {}).get("name")
            or ((serialized or {}).get("id") or ["unknown"])[-1]
        )
        print(f"[START] {step}")

    def on_chain_end(
        self,
        outputs: Union[Dict[str, Any], Any],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        print(f"[END]   output={json.dumps(outputs, ensure_ascii=False)}\n")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        print(f"[ERROR] {error}\n")


async def main() -> None:
    # 通过 config 的 callbacks 字段传入观测器
    result = await chain.ainvoke(
        "  hello   world   from   langchain  ",
        config={"callbacks": [StepObserver()]},
    )
    print("结果:", result)


if __name__ == "__main__":
    asyncio.run(main())
