"""example_selector2.py - SemanticSimilarityExampleSelector + Milvus 示例"""
import os

from langchain_community.vectorstores import Milvus
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

from utils import create_embeddings


COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "weekly_report_examples")
example_prompt = PromptTemplate.from_template("用户场景：{scenario}\n生成的周报片段：\n{report_snippet}\n---")


async def main() -> None:
    embeddings = create_embeddings()
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": os.getenv("MILVUS_URI", "http://localhost:19530")},
    )
    example_selector = SemanticSimilarityExampleSelector(vectorstore=vector_store, k=2)
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="下面是一些不同类型的周报示例：\n",
        suffix="\n\n现在请根据上面的示例风格，为下面这个场景写一份新的周报：\n场景描述：{current_scenario}",
        input_variables=["current_scenario"],
    )
    final_prompt = await few_shot_prompt.aformat(
        current_scenario="本周主要清理历史技术债：重构老旧订单模块、补齐核心接口单测。"
    )
    print(final_prompt)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
