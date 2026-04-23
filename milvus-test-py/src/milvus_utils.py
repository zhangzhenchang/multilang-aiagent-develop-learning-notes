"""milvus_utils.py - Milvus 示例共享工具函数"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, connections, utility


load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
VECTOR_DIM = 1024
DIARY_COLLECTION_NAME = "ai_diary"
EBOOK_COLLECTION_NAME = "ebook_collection"
EBOOK_FILE = Path(__file__).resolve().parent.parent / "天龙八部.epub"
BOOK_NAME = EBOOK_FILE.stem


def create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-v3"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        
        # 也要把嵌入模型指定为 1024 的维度
        dimensions=VECTOR_DIM,
        
        check_embedding_ctx_length=False,
    )


def create_chat_model(temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        temperature=temperature,
        model=os.getenv("MODEL_NAME", "qwen-coder-turbo"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

# 创建 Milvus 客户端
def create_milvus_client() -> MilvusClient:
    return MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

# 连接默认的 Milvus 客户端
def connect_default() -> None:
    connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)

# 确保日记集合存在
def ensure_diary_collection(client: MilvusClient) -> None:
    connect_default()
    # 意思：如果集合已经存在，则直接返回，不存在，则创建集合
    if client.has_collection(collection_name=DIARY_COLLECTION_NAME):
        return

    '''
        你可以先用这个心智模型入门：

        collection ≈ table
        field ≈ column
        record/entity ≈ row
    '''
    # 和 mysql 的表差不多，唯一的区别是 vector 这个字段，我们设置了 FloatVector 类型
    # 除了 vector，其他都是元信息
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=50),
            
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="mood", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10, max_length=50),
        ],
        description="AI diary collection",
        # 意思：主键不自动生成，需要你自己传 id
        auto_id=False,
        # 意思：不允许未在 schema 声明的额外字段
        enable_dynamic_field=False,
    )
    # 创建集合
    collection = Collection(name=DIARY_COLLECTION_NAME, schema=schema)
    '''
    创建索引的作用：让向量检索更快。
    不建索引也能查（暴力扫描），但数据一多会很慢。

    field_name="vector"
    指定给哪个字段建索引
    这里是向量字段 vector

    index_type="IVF_FLAT"
    索引类型
    IVF 思路：先把向量空间分成很多“桶/簇”，搜索时只查部分桶，不全表扫
    FLAT：桶内仍是精确比对（不再二次压缩）
    特点：速度快于全扫，召回质量通常也比较稳

    metric_type="COSINE"
    相似度度量方式：余弦相似度
    适合文本 embedding（常见默认）
    向量方向越接近，分数越高（越相似）

    params={"nlist": 1024}
    IVF 的核心超参数：聚类桶数量
    nlist 越大：
    索引更细
    通常召回潜力更高
    但建索引和内存成本更高
    1024 是一个中等偏常用的起点值（具体要看数据量）
    补充一个常配套参数（在 search 阶段）：

    nprobe：查询时探测多少个桶
    大：更准但更慢
    小：更快但可能漏召回
    一句话总结：
    你这组配置是在 vector 字段上建立 IVF_FLAT 余弦索引，用 nlist=1024 把向量空间分桶，以换取更高效的检索。

    '''
    # 创建索引 metric_type 指定用余弦相似度作为距离度量
    index_params = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}}
    collection.create_index(field_name="vector", index_params=index_params)

# 确保电子书集合存在
def ensure_ebook_collection(client: MilvusClient) -> None:
    connect_default()
    if client.has_collection(collection_name=EBOOK_COLLECTION_NAME):
        return

    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="book_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="book_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="chapter_num", dtype=DataType.INT32),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        ],
        description="ebook chunk collection",
        auto_id=False,
        enable_dynamic_field=False,
    )
    # 创建集合
    collection = Collection(name=EBOOK_COLLECTION_NAME, schema=schema)
    # 创建索引
    index_params = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}}
    collection.create_index(field_name="vector", index_params=index_params)
    
# 加载集合
# 把集合加载到内存才能做快速语义检索
def load_collection(name: str) -> None:
    connect_default()
    collection = Collection(name)
    collection.load()

# 释放集合
def release_collection(name: str) -> None:
    connect_default()
    if utility.has_collection(name):
        Collection(name).release()

# 格式化搜索结果
def format_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for item in results:
        entity = item.get("entity", {})
        formatted.append({**entity, "score": item.get("distance")})
    return formatted
