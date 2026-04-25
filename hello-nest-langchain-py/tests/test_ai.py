"""
测试 AI 端点（mock AiService，不发起真实 LLM 请求）。

测试策略：
  - 通过 app.dependency_overrides 替换 _get_ai_service，
    注入 AsyncMock，避免调用真实 OpenAI API。
  - 每个测试用 try/finally 清理 dependency_overrides，
    防止 mock 泄漏影响其他测试。

覆盖场景：
  GET /ai/chat           → 200 + {"answer": ...}
  GET /ai/chat           → 422，缺少 query 参数
  GET /ai/chat/stream    → 200 text/event-stream，SSE 事件格式
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.ai.router import _get_ai_service
from src.ai.service import AiService
from src.main import app


def _make_mock_service(answer: str = "这是测试回答") -> AiService:
    """
    创建 AiService 的 AsyncMock。

    run_chain  — 协程，返回字符串
    stream_chain — 异步生成器，逐块 yield 字符串
    """
    mock = AsyncMock(spec=AiService)
    mock.run_chain.return_value = answer

    # 模拟异步生成器：将 answer 拆成单字符逐步 yield
    async def _fake_stream(query: str):
        for char in answer:
            yield char

    mock.stream_chain.side_effect = _fake_stream
    return mock


# ── GET /ai/chat ───────────────────────────────────────────────────────────

def test_chat_returns_answer():
    """
    /ai/chat?query=xxx 应返回 200 及 {"answer": "..."} JSON。
    """
    mock_service = _make_mock_service("LangChain 是构建 LLM 应用的框架")

    # 用 lambda 覆盖依赖，FastAPI 的 Depends 机制会调用此函数
    app.dependency_overrides[_get_ai_service] = lambda: mock_service
    try:
        with TestClient(app) as client:
            response = client.get("/ai/chat", params={"query": "什么是 LangChain？"})
        assert response.status_code == 200
        assert response.json() == {"answer": "LangChain 是构建 LLM 应用的框架"}
    finally:
        app.dependency_overrides.clear()    # 清理，防止 mock 污染其他测试


def test_chat_missing_query():
    """
    /ai/chat 不传 query 参数应返回 422（FastAPI Query(...) 必填校验）。
    """
    with TestClient(app) as client:
        response = client.get("/ai/chat")
    assert response.status_code == 422


# ── GET /ai/chat/stream ────────────────────────────────────────────────────

def test_chat_stream_sse_format():
    """
    /ai/chat/stream 应返回 200，Content-Type 为 text/event-stream，
    响应体应包含 SSE 格式的 data 行和 done 事件。
    """
    mock_service = _make_mock_service("你好")

    app.dependency_overrides[_get_ai_service] = lambda: mock_service
    try:
        with TestClient(app) as client:
            response = client.get(
                "/ai/chat/stream",
                params={"query": "你好"},
            )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        body = response.text
        # 每个字符应各自出现在一个 "data: " 行中
        assert "data: 你" in body
        assert "data: 好" in body
        # 流结束应有 "event: done" 信号
        assert "event: done" in body
    finally:
        app.dependency_overrides.clear()
