"""pipeline_prompt_template.py - 用分块 PromptTemplate 组合生成最终提示词"""
from langchain_core.prompts import PromptTemplate


persona_prompt = PromptTemplate.from_template("你是一名资深工程团队负责人，写作风格：{tone}。")
context_prompt = PromptTemplate.from_template(
    "公司：{company_name}\n部门：{team_name}\n直接汇报对象：{manager_name}\n本周时间范围：{week_range}\n本周部门核心目标：{team_goal}"
)
task_prompt = PromptTemplate.from_template(
    "以下是本周团队的开发活动：\n{dev_activities}\n\n请提炼本周亮点、风险和下周计划。"
)
format_prompt = PromptTemplate.from_template(
    "请用 Markdown 输出周报，包含 Summary、详细拆分和关键指标表格。语气符合 {company_values}。"
)
final_weekly_prompt = PromptTemplate.from_template(
    "{persona_block}\n\n{context_block}\n\n{task_block}\n\n{format_block}\n\n现在请生成本周的最终周报："
)


async def format_pipeline_prompt(**kwargs: str) -> str:
    persona_block = await persona_prompt.aformat(tone=kwargs["tone"])
    context_block = await context_prompt.aformat(
        company_name=kwargs["company_name"],
        team_name=kwargs["team_name"],
        manager_name=kwargs["manager_name"],
        week_range=kwargs["week_range"],
        team_goal=kwargs["team_goal"],
    )
    task_block = await task_prompt.aformat(dev_activities=kwargs["dev_activities"])
    format_block = await format_prompt.aformat(company_values=kwargs["company_values"])
    return await final_weekly_prompt.aformat(
        persona_block=persona_block,
        context_block=context_block,
        task_block=task_block,
        format_block=format_block,
    )


async def main() -> None:
    pipeline_formatted = await format_pipeline_prompt(
        tone="专业、清晰、略带幽默",
        company_name="星航科技",
        team_name="AI 平台组",
        manager_name="王总",
        week_range="2025-02-03 ~ 2025-02-09",
        team_goal="完成智能周报 Agent 的 MVP 版本。",
        dev_activities="- Git: 58 次提交\n- Jira: 完成 12 个 Story\n- 完成 Prompt 拆分",
        company_values="极致、开放、靠谱",
    )
    print(pipeline_formatted)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
