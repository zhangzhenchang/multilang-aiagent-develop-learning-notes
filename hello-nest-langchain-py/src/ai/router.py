"""AI 路由 — LLM 对话端点。

路由前缀 /ai：
  GET /ai/chat          同步，等待完整响应返回 JSON
  GET /ai/chat/stream   流式，SSE 实时推送
"""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from langchain_openai import ChatOpenAI
from starlette.responses import StreamingResponse

from src.ai.service import AiService

router = APIRouter(prefix="/ai", tags=["AI"])


def _get_model(request: Request) -> ChatOpenAI:
    """从 app.state 取出 lifespan 创建的 ChatOpenAI 单例，复用 HTTP 连接池。"""
    return request.app.state.chat_model  # type: ignore[return-value]


def _get_ai_service(model: Annotated[ChatOpenAI, Depends(_get_model)]) -> AiService:
    """创建 AiService 并注入 ChatOpenAI。AiService 无状态，每请求重建开销可忽略。"""
    return AiService(model)


_AiService = Annotated[AiService, Depends(_get_ai_service)]


@router.get("/chat", summary="非流式对话")
async def chat(
    query: Annotated[str, Query(description="用户问题")],
    service: _AiService,
) -> dict[str, str]:
    """同步对话：等待 LLM 生成完整回答，以 JSON 返回。

    响应示例：{"answer": "LangChain 是一个用于构建 LLM 应用的框架..."}
    """
    answer = await service.run_chain(query)
    return {"answer": answer}


@router.get("/chat/stream", summary="SSE 流式对话")
async def chat_stream(
    query: Annotated[str, Query(description="用户问题")],
    service: _AiService,
) -> StreamingResponse:
    """流式对话：通过 Server-Sent Events 实时推送 LLM 逐 token 响应。

    SSE 格式：
      data: <文本片段>\\n\\n      — LLM 响应 chunk
      event: done\\ndata: \\n\\n  — 流结束信号
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        async for chunk in service.stream_chain(query):
            yield f"data: {chunk}\n\n"
        yield "event: done\ndata: \n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
