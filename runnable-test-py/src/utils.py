"""utils.py - 共享的模型工厂函数"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def create_chat_model(model_name: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    """创建并返回一个 ChatOpenAI 实例，从环境变量读取配置"""
    return ChatOpenAI(
        model=model_name or os.getenv("MODEL_NAME", "qwen-coder-turbo"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=temperature,
    )
