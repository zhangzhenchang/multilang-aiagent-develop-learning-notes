"""pipeline_prompt_template3.py - 分块 PromptTemplate + ChatPromptTemplate 组合示例"""
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from pipeline_prompt_template import context_prompt, persona_prompt


weekly_task_prompt = PromptTemplate.from_template(
    "以下是本周与你所在团队相关的关键事实与数据：\n{dev_activities}\n\n请生成一份技术周报，包含整体达成、关键成果、问题风险和下周计划。"
)
weekly_format_prompt = PromptTemplate.from_template(
    "请用 Markdown 写周报，结构包含本周概览、详细拆分和关键指标表格。语气要求：{tone}。"
)
final_chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一名资深工程团队负责人，擅长把复杂技术细节总结成结构化周报。"),
        (
            "human",
            "人设与写作风格：\n{persona_block}\n\n团队与本周背景：\n{context_block}\n\n任务与输入数据：\n{task_block}\n\n输出格式要求：\n{format_block}",
        ),
    ]
)


async def format_weekly_chat_prompt(**kwargs: str):
    persona_block = await persona_prompt.aformat(tone=kwargs["tone"])
    context_block = await context_prompt.aformat(
        company_name=kwargs["company_name"],
        team_name=kwargs["team_name"],
        manager_name=kwargs["manager_name"],
        week_range=kwargs["week_range"],
        team_goal=kwargs["team_goal"],
    )
    task_block = await weekly_task_prompt.aformat(dev_activities=kwargs["dev_activities"])
    format_block = await weekly_format_prompt.aformat(tone=kwargs["tone"])
    return await final_chat_prompt.aformat_prompt(
        persona_block=persona_block,
        context_block=context_block,
        task_block=task_block,
        format_block=format_block,
    )


async def main() -> None:
    prompt_value = await format_weekly_chat_prompt(
        tone="专业、清晰、略带鼓励",
        company_name="星航科技",
        team_name="AI 平台组",
        manager_name="王总",
        week_range="2025-05-12 ~ 2025-05-18",
        team_goal="完成周报自动生成能力的灰度验证。",
        dev_activities="- Git：合并 4 个主要特性分支\n- Jira：关闭 9 个 Story / 5 个 Bug\n- 运维：P1 事故 0 起",
    )
    print(prompt_value.to_messages())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
