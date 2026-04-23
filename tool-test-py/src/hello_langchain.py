"""hello_langchain.py - 简单的 LangChain ChatOpenAI 调用"""
import os
import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


async def main() -> None:
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "qwen-coder-turbo"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    response = await model.ainvoke("介绍下自己")
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
