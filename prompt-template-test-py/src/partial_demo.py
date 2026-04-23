"""partial_demo.py - partial 预填充示例"""
from langchain_core.prompts import PromptTemplate

from pipeline_prompt_template import context_prompt, final_weekly_prompt, format_prompt, persona_prompt, task_prompt


async def main() -> None:
    persona_with_partial = persona_prompt.partial(tone="偏正式但不僵硬")
    context_with_partial = context_prompt.partial(company_name="星航科技")
    format_with_partial = format_prompt.partial(company_values="极致、开放、靠谱")

    persona_block = await persona_with_partial.aformat()
    context_block = await context_with_partial.aformat(
        team_name="AI 平台组",
        manager_name="刘东",
        week_range="2025-02-10 ~ 2025-02-16",
        team_goal="上线周报 Agent 到内部试用环境，并收集反馈。",
    )
    task_block = await task_prompt.aformat(
        dev_activities="- 小明：完成 Git/Jira 集成\n- 小红：实现 Prompt 配置化加载"
    )
    format_block = await format_with_partial.aformat()
    partial_formatted = await final_weekly_prompt.aformat(
        persona_block=persona_block,
        context_block=context_block,
        task_block=task_block,
        format_block=format_block,
    )
    print(partial_formatted)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
