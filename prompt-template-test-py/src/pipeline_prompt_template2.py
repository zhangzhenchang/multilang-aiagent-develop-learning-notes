"""pipeline_prompt_template2.py - 复用分块 PromptTemplate 生成季度 OKR 回顾"""
from langchain_core.prompts import PromptTemplate

from pipeline_prompt_template import context_prompt, persona_prompt


okr_review_task_prompt = PromptTemplate.from_template(
    "以下是本季度与你所在团队相关的关键事实与数据：\n{okr_facts}\n\n请整理一份发给 {manager_name} 的季度 OKR 回顾邮件。"
)
okr_review_format_prompt = PromptTemplate.from_template(
    "请用 Markdown 写邮件，结构包含邮件开头、整体概览、逐条 OKR 回顾、问题风险和下季度计划。"
)
okr_review_final_prompt = PromptTemplate.from_template(
    "{persona_block}\n\n{context_block}\n\n{task_block}\n\n{format_block}\n\n现在请生成季度 OKR 回顾邮件："
)


async def format_okr_review_prompt(**kwargs: str) -> str:
    persona_block = await persona_prompt.aformat(tone=kwargs["tone"])
    context_block = await context_prompt.aformat(
        company_name=kwargs["company_name"],
        team_name=kwargs["team_name"],
        manager_name=kwargs["manager_name"],
        week_range=kwargs["week_range"],
        team_goal=kwargs["team_goal"],
    )
    task_block = await okr_review_task_prompt.aformat(
        okr_facts=kwargs["okr_facts"], manager_name=kwargs["manager_name"]
    )
    format_block = await okr_review_format_prompt.aformat()
    return await okr_review_final_prompt.aformat(
        persona_block=persona_block,
        context_block=context_block,
        task_block=task_block,
        format_block=format_block,
    )


async def main() -> None:
    prompt_for_review = await format_okr_review_prompt(
        tone="专业、真诚、偏书面表达",
        company_name="星航科技",
        team_name="AI 平台组",
        manager_name="王总",
        week_range="2025 Q1",
        team_goal="支撑公司核心 AI 能力建设。",
        okr_facts="- O1：完成在线特征平台 V1 上线\n- O2：首页 CTR 提升 6.3%\n- O3：GPU 利用率提升到 67%",
    )
    print(prompt_for_review)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
