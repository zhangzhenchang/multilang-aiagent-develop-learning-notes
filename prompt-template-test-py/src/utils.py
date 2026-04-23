"""shared prompt-template helpers"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

VECTOR_DIM = 1024


def create_chat_model(temperature: float = 0.0, model_name: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name or os.getenv("MODEL_NAME", "qwen-coder-turbo"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=temperature,
    )


def create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-v3"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        dimensions=VECTOR_DIM,
        check_embedding_ctx_length=False,
    )
