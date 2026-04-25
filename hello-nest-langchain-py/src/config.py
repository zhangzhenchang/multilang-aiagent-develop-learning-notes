"""
应用配置 — pydantic-settings 类型安全配置管理。

等价于 NestJS ConfigModule.forRoot({ isGlobal: true })，优势：
  ✓ 字段有明确类型，配置错误在启动时立即报错，而非运行到一半才崩溃
  ✓ IDE 类型提示完整，无需记忆环境变量名
  ✓ 测试友好：通过 app.dependency_overrides[get_settings] 注入测试配置

加载优先级（高到低）：
  1. 进程环境变量
  2. .env 文件（相对于启动时的 CWD）
  3. 字段默认值
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    全局配置模型。

    每个字段自动对应同名环境变量（大小写不敏感）：
      openai_api_key  ↔  OPENAI_API_KEY
      model_name      ↔  MODEL_NAME
      ...
    """

    model_config = SettingsConfigDict(
        env_file=".env",            # 从项目根 .env 加载（相对于 CWD）
        env_file_encoding="utf-8",
        case_sensitive=False,       # 环境变量名大小写不敏感
        extra="ignore",             # 忽略 .env 中未声明的多余字段
    )

    # ── OpenAI / 兼容 API ────────────────────────────────────────────────────

    # 必填字段：无默认值，启动时未设置则直接抛 ValidationError
    openai_api_key: str = Field(
        description="OpenAI 或兼容 API 密钥（如阿里云 DashScope 的 sk-xxxx）",
    )

    # 可填字段：指向 OpenAI 兼容端点（DashScope、Azure、本地 vLLM 等）
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="API 端点基础 URL，配置后可对接任意 OpenAI 兼容服务",
    )

    # 模型标识符，与 API 端点对应（DashScope 用 qwen-plus 等）
    model_name: str = Field(
        default="qwen-plus",
        description="LLM 模型名称，如 gpt-4o、qwen-plus",
    )

    # ── 服务器 ───────────────────────────────────────────────────────────────

    port: int = Field(
        default=8000,
        description="HTTP 监听端口",
        ge=1024,    # 1024 以下为特权端口，普通用户无权监听
        le=65535,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    获取配置单例（惰性初始化 + 缓存）。

    @lru_cache(maxsize=1) 保证全程只实例化一次 Settings，
    等价于 NestJS ConfigModule isGlobal: true 的全局单例语义。

    测试中可通过依赖覆盖注入假配置：
        app.dependency_overrides[get_settings] = lambda: Settings(openai_api_key="test")
    """
    return Settings()
