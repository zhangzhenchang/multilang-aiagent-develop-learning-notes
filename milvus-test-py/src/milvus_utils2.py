"""milvus_utils2.py - Milvus 标准实践版工具函数"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pymilvus import DataType, MilvusClient


load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
VECTOR_DIM = 1024
DIARY_COLLECTION_NAME = "ai_diary"
EBOOK_COLLECTION_NAME = "ebook_collection"
EBOOK_FILE = Path(__file__).resolve().parent.parent / "天龙八部.epub"
BOOK_NAME = EBOOK_FILE.stem


def create_embeddings() -> OpenAIEmbeddings:
    """创建 OpenAI Embeddings 实例"""
    return OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-v3"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        dimensions=VECTOR_DIM,
        check_embedding_ctx_length=False,
    )


def create_chat_model(temperature: float = 0.7) -> ChatOpenAI:
    """创建 ChatOpenAI 实例"""
    return ChatOpenAI(
        temperature=temperature,
        model=os.getenv("MODEL_NAME", "qwen-coder-turbo"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def create_milvus_client() -> MilvusClient:
    """创建 Milvus 客户端"""
    return MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)


def ensure_diary_collection(client: MilvusClient) -> None:
    """
    确保日记集合存在，不存在则创建

    标准流程：
    1. 检查 collection 是否存在
    2. 创建 schema
    3. 创建 collection
    4. 准备索引参数
    5. 创建索引
    """
    if client.has_collection(collection_name=DIARY_COLLECTION_NAME):
        return

    # 1. 创建 schema
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        description="AI diary collection",
    )

    # 2. 添加字段
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=50)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=5000)
    schema.add_field(field_name="date", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="mood", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(
        field_name="tags",
        datatype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=10,
        max_length=50,
    )

    # 3. 准备索引参数
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 1024},
    )

    # 4. 创建 collection（同时创建索引）
    client.create_collection(
        collection_name=DIARY_COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )


def ensure_ebook_collection(client: MilvusClient) -> None:
    """
    确保电子书集合存在，不存在则创建

    标准流程同 ensure_diary_collection
    """
    if client.has_collection(collection_name=EBOOK_COLLECTION_NAME):
        return

    # 1. 创建 schema
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        description="ebook chunk collection",
    )

    # 2. 添加字段
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
    schema.add_field(field_name="book_id", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="book_name", datatype=DataType.VARCHAR, max_length=200)
    schema.add_field(field_name="chapter_num", datatype=DataType.INT32)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT32)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=10000)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    # 3. 准备索引参数
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 1024},
    )

    # 4. 创建 collection（同时创建索引）
    client.create_collection(
        collection_name=EBOOK_COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )


def load_collection(client: MilvusClient, collection_name: str) -> None:
    """
    加载集合到内存

    注意：
    - 只有加载后才能执行 search 操作
    - 如果已经加载，重复调用不会报错
    """
    client.load_collection(collection_name=collection_name)


def release_collection(client: MilvusClient, collection_name: str) -> None:
    """释放集合，从内存中卸载"""
    if client.has_collection(collection_name=collection_name):
        client.release_collection(collection_name=collection_name)


def format_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    格式化搜索结果

    将 Milvus 返回的结果格式转换为更易用的格式
    """
    formatted: list[dict[str, Any]] = []
    for item in results:
        entity = item.get("entity", {})
        formatted.append({**entity, "score": item.get("distance")})
    return formatted
