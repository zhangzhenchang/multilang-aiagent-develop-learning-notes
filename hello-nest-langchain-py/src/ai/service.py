"""AI 服务 — 封装 LangChain LCEL 链的构建与调用。"""

from collections.abc import AsyncGenerator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI


class AiService:
    """AI 对话服务，持有并管理一条 LCEL 链。

    通过 Depends(_get_ai_service) 注入，每请求重建，无共享状态。
    """

    def __init__(self, model: ChatOpenAI) -> None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个助手，请简洁准确地用中文回答用户的问题。"),
            ("human", "{query}"),
        ])
        # LCEL：prompt → model → 纯字符串解析
        self.chain: Runnable = prompt | model | StrOutputParser()

    async def run_chain(self, query: str) -> str:
        """单次调用，等待完整响应。"""
        return await self.chain.ainvoke({"query": query})

    async def stream_chain(self, query: str) -> AsyncGenerator[str, None]:
        """流式调用，逐 token yield，适合 SSE 推送。"""
        async for chunk in self.chain.astream({"query": query}):
            yield chunk
