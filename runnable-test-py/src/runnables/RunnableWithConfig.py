"""RunnableWithConfig.py - 通过 with_config 为链绑定运行时配置"""
import asyncio
import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig, RunnableLambda

load_dotenv()

# 模拟用户数据库
MOCK_USERS: Dict[str, Dict[str, str]] = {
    "user-123": {"id": "user-123", "name": "神光", "email": "guang@example.com"},
}

# 允许发送系统通知的角色集合
ALLOWED_ROLES = {"管理员", "运营", "系统"}


# ---------- 节点 1：从 config.configurable 中读取 userId，查询用户 ----------
async def fetch_user_from_config(input_val: str, config: RunnableConfig) -> Dict[str, Any]:
    # config["configurable"] 存放业务自定义参数
    configurable = (config or {}).get("configurable", {})
    user_id = configurable.get("userId")

    print("【节点1】收到了通知内容:", input_val)
    print("【节点1】从 config 里拿到 userId:", user_id)

    user = MOCK_USERS.get(user_id) if user_id else None
    if not user:
        raise ValueError("未找到用户，无法发送通知")

    return {"user": user, "notification": input_val}


# ---------- 节点 2：从 config.configurable 中读取 role，做权限校验 ----------
async def check_permission_by_role(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    configurable = (config or {}).get("configurable", {})
    role = configurable.get("role", "普通用户")

    print("【节点2】当前角色:", role)

    if role not in ALLOWED_ROLES:
        raise PermissionError(f"角色「{role}」无权限发送系统通知")

    return {**state, "role": role}


# ---------- 节点 3：从 config.configurable 中读取 locale，生成通知文案 ----------
async def format_notification_by_locale(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    configurable = (config or {}).get("configurable", {})
    locale = configurable.get("locale", "zh-CN")

    print("【节点3】locale:", locale)

    if locale == "en-US":
        content = (
            f"Dear {state['user']['name']},\n\n"
            f"{state['notification']}\n\n"
            f"(from role: {state['role']})"
        )
    else:
        content = (
            f"亲爱的 {state['user']['name']}，\n\n"
            f"{state['notification']}\n\n"
            f"（发送人角色：{state['role']}）"
        )

    return {**state, "locale": locale, "finalContent": content}


# ---------- 将三个节点串联成链 ----------
chain = (
    RunnableLambda(fetch_user_from_config)
    | RunnableLambda(check_permission_by_role)
    | RunnableLambda(format_notification_by_locale)
)

# with_config()：为链预绑定配置，每次 invoke 时自动合并这些配置
# configurable 字段传递业务参数，tags/metadata 用于追踪观测
chain_with_config = chain.with_config(
    tags=["demo", "withConfig", "notification"],
    metadata={"demoName": "RunnableWithConfig"},
    configurable={
        "userId": "user-123",
        "role":   "管理员",
        "locale": "zh-CN",
    },
)

# 第二个配置变体：英文 locale + 运营角色
chain_with_config2 = chain.with_config(
    tags=["demo", "withConfig", "notification-en"],
    metadata={"demoName": "RunnableWithConfig2"},
    configurable={
        "userId": "user-123",
        "role":   "运营",
        "locale": "en-US",
    },
)


async def main() -> None:
    result = await chain_with_config.ainvoke("你有一条新的系统通知，请及时查看。")
    print("✅ 最终通知内容:\n", result["finalContent"])

    print("\n--- chain_with_config2 ---\n")

    result2 = await chain_with_config2.ainvoke("System maintenance scheduled tonight.")
    print("✅ 最终通知内容:\n", result2["finalContent"])


if __name__ == "__main__":
    asyncio.run(main())
