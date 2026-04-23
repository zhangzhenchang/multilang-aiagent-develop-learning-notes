"""example_selector1.py - LengthBasedExampleSelector 示例"""
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate


example_prompt = PromptTemplate.from_template("用户需求：{user_requirement}\n周报片段示例：\n{report_snippet}\n---")
examples = [
    {"user_requirement": "稳定性治理", "report_snippet": "- 处理故障并优化告警。"},
    {"user_requirement": "对外展示成果", "report_snippet": "- 上线新看板并组织分享。"},
    {"user_requirement": "非常简短的周报", "report_snippet": "本周整体运行平稳，核心指标正常。"},
]
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=300,
    get_text_length=len,
)
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="下面是一些不同风格和长度的周报片段示例：\n",
    suffix="\n\n现在请根据上面的示例风格，为下面这个场景写一份新的周报：\n场景描述：{current_requirement}",
    input_variables=["current_requirement"],
)


async def main() -> None:
    final_prompt = await few_shot_prompt.aformat(
        current_requirement="我们本周在做内部 AI 助手项目，既有稳定性保障，也有新功能上线。"
    )
    print(final_prompt)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
