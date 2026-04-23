"""prompt_template1.py - PromptTemplate 基础示例"""
from langchain_core.prompts import PromptTemplate

from utils import create_chat_model


model = create_chat_model()
naive_template = PromptTemplate.from_template(
    """
你是一名严谨但不失人情味的工程团队负责人，需要根据本周数据写一份周报。

公司名称：{company_name}
部门名称：{team_name}
直接汇报对象：{manager_name}
本周时间范围：{week_range}

本周团队核心目标：
{team_goal}

本周开发数据（Git 提交 / Jira 任务）：
{dev_activities}

请根据以上信息生成一份【Markdown 周报】，要求：
- 有简短的整体 summary（两三句话）
- 有按模块/项目拆分的小结
- 用一个 Markdown 表格列出关键指标（字段示例：模块 / 亮点 / 风险 / 下周计划）
- 语气专业但有一点人情味，适合作为给老板和团队抄送的周报。
"""
)


async def main() -> None:
    prompt = await naive_template.aformat(
        company_name="极光云科技",
        team_name="订单结算后端组",
        manager_name="陈总",
        week_range="2025-04-07 ~ 2025-04-13",
        team_goal="本周以稳定性为主，集中清理历史技术债和高频告警。",
        dev_activities="- 老王：修复高优先级线上 Bug 7 个\n- 小何：重构结算批任务调度逻辑\n- 小陈：梳理告警策略\n- 实习生小刘：补齐历史接口单测",
    )
    print("格式化后的提示词:")
    print(prompt)
    stream = model.astream(prompt)
    print("\nAI 回答:")
    async for chunk in stream:
        print(chunk.content, end="")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
