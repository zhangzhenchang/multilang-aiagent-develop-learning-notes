"""shared model helpers for output parser demos"""
from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def create_chat_model(model_name: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name or os.getenv("MODEL_NAME", "qwen-coder-turbo"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=temperature,
    )


def strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()
