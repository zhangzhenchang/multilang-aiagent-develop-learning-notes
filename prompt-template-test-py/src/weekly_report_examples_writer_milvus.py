"""weekly_report_examples_writer_milvus.py - 把周报示例写入 Milvus"""
from __future__ import annotations

import os

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, connections

from utils import VECTOR_DIM, create_embeddings


COLLECTION_NAME = "weekly_report_examples"
EXAMPLES = [
    {
        "scenario": "支付系统稳定性治理，强调风险防控、告警收敛和应急预案完善。",
        "report_snippet": "- 本周聚焦支付链路稳定性，处理事故并优化告警；\n- 完成关键接口超时阈值优化。",
    },
    {
        "scenario": "新功能首发，更强调对外展示亮点。",
        "report_snippet": "- 上线运营实时看板；\n- 打通埋点到实时服务链路；\n- 组织跨部门分享。",
    },
]


def connect_default() -> None:
    connections.connect(alias="default", uri=os.getenv("MILVUS_URI", "http://localhost:19530"), token=os.getenv("MILVUS_TOKEN") or None)


def ensure_collection(client: MilvusClient) -> None:
    connect_default()
    if client.has_collection(collection_name=COLLECTION_NAME):
        return
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="scenario", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="report_snippet", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        ],
        auto_id=False,
        enable_dynamic_field=False,
    )
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    collection.create_index(field_name="vector", index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}})


async def main() -> None:
    client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"), token=os.getenv("MILVUS_TOKEN") or None)
    embeddings = create_embeddings()
    print("连接 Milvus...")
    ensure_collection(client)
    connect_default()
    Collection(COLLECTION_NAME).load()
    vectors = await embeddings.aembed_documents([item["scenario"] + item["report_snippet"] for item in EXAMPLES])
    rows = [
        {
            "id": f"weekly_{index + 1}",
            "scenario": item["scenario"],
            "report_snippet": item["report_snippet"],
            "vector": vector,
        }
        for index, (item, vector) in enumerate(zip(EXAMPLES, vectors, strict=False))
    ]
    result = client.insert(collection_name=COLLECTION_NAME, data=rows)
    print(f"✓ 已插入 {result['insert_count']} 条记录")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
