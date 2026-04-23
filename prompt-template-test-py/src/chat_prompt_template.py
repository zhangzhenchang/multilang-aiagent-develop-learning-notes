"""chat_prompt_template.py - ChatPromptTemplate 基础示例"""
from langchain_core.prompts import ChatPromptTemplate

from utils import create_chat_model


model = create_chat_model()
chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名资深工程团队负责人，擅长用结构化、易读的方式写技术周报。写作风格要求：{tone}。",
        ),
        (
            "human",
            "本周信息如下：\n公司名称：{company_name}\n团队名称：{team_name}\n直接汇报对象：{manager_name}\n本周时间范围：{week_range}\n\n本周团队核心目标：\n{team_goal}\n\n本周开发数据：\n{dev_activities}",
        ),
    ]
)


async def main() -> None:
    chat_messages = await chat_prompt.aformat_messages(
        tone="专业、清晰、略带鼓励",
        company_name="星航科技",
        team_name="智能应用平台组",
        manager_name="王总",
        week_range="2025-05-05 ~ 2025-05-11",
        team_goal="完成内部 AI 助手灰度上线，并确保核心链路稳定。",
        dev_activities="- 小李：完成工单流转能力\n- 小张：接入日志检索和知识库查询",
    )
    print("ChatPromptTemplate 生成的消息:")
    print(chat_messages)
    response = await model.ainvoke(chat_messages)
    print("\nAI 生成的周报草稿:")
    print(response.content)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
