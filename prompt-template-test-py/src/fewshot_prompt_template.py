"""fewshot_prompt_template.py - FewShotPromptTemplate 示例"""
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate


example_prompt = PromptTemplate.from_template(
    "用户输入：{user_requirement}\n期望周报结构：{expected_style}\n模型示例输出片段：\n{report_snippet}\n---"
)
examples = [
    {
        "user_requirement": "重点突出稳定性治理，本周主要在修 Bug 和清理技术债。",
        "expected_style": "语气稳健、偏保守，多强调风险识别和兜底动作。",
        "report_snippet": "- 支付链路本周共处理线上 P1 Bug 2 个、P2 Bug 3 个；\n- 完成 3 个核心接口的超时阈值优化。",
    },
    {
        "user_requirement": "偏向对外展示成果，希望多写一些亮点。",
        "expected_style": "语气积极、突出成果，对技术细节做适度抽象。",
        "report_snippet": "- 新上线运营实时看板；\n- 打通埋点 → 数据仓库 → 实时服务链路。",
    },
]
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="下面是几条已经写好的【周报示例】：\n",
    suffix="\n基于上面的示例风格，请帮我写一份新的周报。",
    input_variables=[],
)


async def main() -> None:
    few_shot_block = await few_shot_prompt.aformat()
    print(few_shot_block)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
