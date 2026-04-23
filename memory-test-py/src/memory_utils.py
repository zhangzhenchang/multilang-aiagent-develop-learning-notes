"""memory_utils.py - memory 示例共享工具函数"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, message_to_dict, messages_from_dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, connections


# 读取 .env 中的配置。
load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN") or None
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-coder-turbo")
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-v3")
VECTOR_DIM = 1024
# 对话记忆在 Milvus 里的集合名。
CONVERSATION_COLLECTION_NAME = "conversations"
# 基于文件的消息历史默认保存位置。
CHAT_HISTORY_FILE = Path(__file__).resolve().parent.parent / "chat_history.json"


def create_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    """创建聊天模型。"""
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def create_embeddings() -> OpenAIEmbeddings:
    """创建 embedding 模型。"""
    return OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=EMBEDDINGS_MODEL_NAME,
        base_url=os.getenv("OPENAI_BASE_URL"),
        dimensions=VECTOR_DIM,
        check_embedding_ctx_length=False,
    )


def create_milvus_client() -> MilvusClient:
    """创建 Milvus 客户端，token 为可选。"""
    kwargs: dict[str, Any] = {"uri": MILVUS_URI}
    if MILVUS_TOKEN:
        kwargs["token"] = MILVUS_TOKEN
    return MilvusClient(**kwargs)


def connect_default() -> None:
    """为 ORM 风格的 Collection API 创建默认连接。"""
    kwargs: dict[str, Any] = {"alias": "default", "uri": MILVUS_URI}
    if MILVUS_TOKEN:
        kwargs["token"] = MILVUS_TOKEN
    connections.connect(**kwargs)


def ensure_conversation_collection(client: MilvusClient) -> None:
    """确保 conversations 集合存在，不存在则创建。"""
    connect_default()
    if client.has_collection(collection_name=CONVERSATION_COLLECTION_NAME):
        return

    # schema = 集合结构定义，类似关系型数据库里的表结构。
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=50),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="round", dtype=DataType.INT64),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=100),
        ],
        description="Conversation memory collection",
        auto_id=False,
        enable_dynamic_field=False,
    )
    collection = Collection(name=CONVERSATION_COLLECTION_NAME, schema=schema)
    # 在向量字段上建立 IVF_FLAT + COSINE 索引，提高语义检索速度。
    index_params = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}}
    collection.create_index(field_name="vector", index_params=index_params)


def load_collection(name: str) -> None:
    """把指定集合加载到内存中，便于后续检索。"""
    connect_default()
    Collection(name).load()


def format_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 pymilvus 的原始搜索结果整理成更易用的结构。"""
    formatted: list[dict[str, Any]] = []
    for item in results:
        entity = item.get("entity", {})
        formatted.append({**entity, "score": item.get("distance")})
    return formatted


class JSONChatMessageHistory(BaseChatMessageHistory):
    """一个简化版的 JSON 文件消息历史实现。"""

    def __init__(self, file_path: str | Path, session_id: str):
        self.file_path = Path(file_path)
        self.session_id = session_id
        if not self.file_path.exists():
            self.file_path.write_text("{}", encoding="utf-8")

    def _read_store(self) -> dict[str, Any]:
        return json.loads(self.file_path.read_text(encoding="utf-8") or "{}")

    def _write_store(self, data: dict[str, Any]) -> None:
        self.file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # @property = 让方法像属性一样访问（obj.xxx 而不是 obj.xxx()）
    @property
    def messages(self) -> list[BaseMessage]:
        store = self._read_store()
        raw_messages = store.get(self.session_id, [])
        return messages_from_dict(raw_messages)

    def add_message(self, message: BaseMessage) -> None:
        store = self._read_store()
        session_messages = store.get(self.session_id, [])
        session_messages.append(message_to_dict(message))
        store[self.session_id] = session_messages
        self._write_store(store)

    def clear(self) -> None:
        store = self._read_store()
        store[self.session_id] = []
        self._write_store(store)


def make_sample_messages(items: list[dict[str, str]]) -> InMemoryChatMessageHistory:
    """把简单字典列表转成 InMemoryChatMessageHistory。"""
    history = InMemoryChatMessageHistory()
    for item in items:
        if item["type"] == "human":
            history.add_message(HumanMessage(item["content"]))
        else:
            history.add_message(AIMessage(item["content"]))
    return history
