"""fewshot_chat_prompt_template.py - FewShotChatMessagePromptTemplate 示例"""
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from utils import create_chat_model


model = create_chat_model(temperature=0.3)
examples = [
    {
        "input": "本周主要推进支付稳定性治理，做了事故处置、告警优化和演练。",
        "output": "- 本周围绕支付链路稳定性开展治理工作；\n- 完成故障排查修复并优化告警规则；\n- 组织 1 次故障应急演练。",
    },
    {
        "input": "本周交付了新运营看板，并给业务同学做了多场分享。",
        "output": "- 上线新一代运营实时看板；\n- 打通数据链路；\n- 面向业务团队组织 2 场培训。",
    },
]
few_shot_examples = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "下面是本周的工作概述：\n{input}\n\n请帮我整理成适合发在团队周报里的要点列表。"), ("ai", "{output}")]
    ),
    examples=examples,
)
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一名资深技术负责人，请参考示例，帮我写结构清晰的周报片段。"),
        few_shot_examples,
        ("human", "这是我本周的实际工作内容，请帮我整理成周报：\n{current_work}"),
    ]
)


async def main() -> None:
    messages = await chat_prompt.aformat_messages(
        current_work="本周完成订单模块重构，补齐核心路径单测，并修复两起线上性能问题。"
    )
    print("===== 发送给模型的消息 =====\n")
    print(messages)
    stream = model.astream(messages)
    print("\n===== 模型输出 =====\n")
    async for chunk in stream:
        print(chunk.content, end="")
    print()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
