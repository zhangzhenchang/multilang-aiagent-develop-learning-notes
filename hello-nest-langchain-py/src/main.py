"""FastAPI 应用入口。

启动方式：
  uv run uvicorn src.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI

from src.ai.router import router as ai_router
from src.book.router import router as book_router
from src.config import get_settings
from src.exceptions import BookNotFoundError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理器。

    startup：创建 ChatOpenAI 单例并挂载到 app.state，全程复用同一 HTTP 连接池。
    shutdown：如有数据库连接等资源，在此处 await close()。
    """
    settings = get_settings()
    app.state.chat_model = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    yield


def create_app() -> FastAPI:
    """应用工厂函数。

    将 FastAPI 实例化逻辑封装在工厂函数中，便于测试时独立创建实例，
    同时将所有路由和中间件注册集中在一处，避免模块级副作用。
    """
    application = FastAPI(
        title="Hello LangChain API",
        description="FastAPI + LangChain 1.0 最佳实践",
        version="2.0.0",
        lifespan=lifespan,
    )

    # ── 异常处理器 ─────────────────────────────────────────────────────────
    # 服务层抛业务异常，这里统一映射为 HTTP 状态码，保持路由层干净。

    @application.exception_handler(BookNotFoundError)
    async def book_not_found_handler(
        request: Request, exc: BookNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    # ── 路由注册 ───────────────────────────────────────────────────────────
    application.include_router(ai_router)
    application.include_router(book_router)

    # ── 根端点 ─────────────────────────────────────────────────────────────
    @application.get("/", summary="健康检查")
    async def get_hello() -> str:
        return "Hello World!"

    # ── 静态文件 ───────────────────────────────────────────────────────────
    # mount 必须在 API 路由注册完毕后执行，避免通配路径遮蔽 API。
    static_dir = Path(__file__).parent / "static"
    application.mount(
        "/static",
        StaticFiles(directory=str(static_dir), html=True),
        name="static",
    )

    return application


app = create_app()
