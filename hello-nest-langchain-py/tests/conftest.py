"""
pytest 通用 fixtures。

执行顺序：
  1. 设置测试环境变量（必须在导入 src 之前，否则 pydantic-settings 读不到值）
  2. 导入 app
  3. 用 TestClient 发起请求（TestClient 会触发 lifespan，即 startup/shutdown）

注意：OPENAI_API_KEY 设置为占位符，ChatOpenAI 实例化不需要真实密钥，
      真实 API 调用在 test_ai.py 中通过 mock 绕过。
"""

import os

# ── 必须在导入 src 前设置环境变量 ─────────────────────────────────────────
# pydantic-settings 在模块首次导入时读取 .env / 环境变量；
# 若未设置 OPENAI_API_KEY，Settings 实例化会抛 ValidationError
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("MODEL_NAME", "test-model")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    """
    模块级 FastAPI 测试客户端。

    scope="module"：同一测试文件内复用同一客户端（含 lifespan），
    避免重复触发 startup/shutdown，提升速度。

    TestClient 上下文管理器会自动调用 lifespan，
    即在 __enter__ 时执行 startup，在 __exit__ 时执行 shutdown。
    """
    with TestClient(app) as c:
        yield c
