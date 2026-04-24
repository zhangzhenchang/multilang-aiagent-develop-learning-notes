"""
smart_import2.py - 智能文本提取 + MySQL 批量写入（LCEL 重构版）

与 smart_import.py 的区别：
  - 用 ChatPromptTemplate | model.with_structured_output() 组成 LCEL chain
  - 将 AI 提取与 DB 写入拆成两个职责单一的函数
  - 模板变量替代手动 .format()，chain 在模块级声明，可复用

依赖：langchain-openai, langchain-core, pydantic, pymysql, python-dotenv
"""
from __future__ import annotations

import json
import os
from typing import Optional

import pymysql
import pymysql.cursors
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


# ──────────────────────────────────────────────────────────
# Pydantic 模型
# ──────────────────────────────────────────────────────────
class FriendInfo(BaseModel):
    name: str = Field(description="姓名")
    gender: str = Field(description="性别（男/女）")
    birth_date: str = Field(description="出生日期，格式 YYYY-MM-DD；无具体日期时根据年龄估算")
    company: Optional[str] = Field(default=None, description="公司名称，无则为 null")
    title: Optional[str] = Field(default=None, description="职位/头衔，无则为 null")
    phone: Optional[str] = Field(default=None, description="手机号，无则为 null")
    wechat: Optional[str] = Field(default=None, description="微信号，无则为 null")


class FriendList(BaseModel):
    friends: list[FriendInfo] = Field(description="从文本中提取到的所有好友信息数组")


# ──────────────────────────────────────────────────────────
# LCEL chain：prompt template | structured model
# 在模块级构建，避免每次调用重复初始化
# ──────────────────────────────────────────────────────────
_prompt = ChatPromptTemplate.from_template("""\
请从以下文本中提取所有好友信息，文本中可能包含一个或多个人的信息。

{text}

要求：
1. 为每个人创建单独的对象，字段包括 name/gender/birth_date/company/title/phone/wechat
2. birth_date 格式 YYYY-MM-DD，无具体日期时根据年龄估算
3. 找不到的字段返回 null，结果放在 friends 数组中\
""")

_model = ChatOpenAI(
    model=os.environ["MODEL_NAME"],
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)

# | 是 LCEL 管道运算符：prompt 渲染后直接流入结构化模型
# method="function_calling" 避免 json_mode 要求 prompt 含 "json" 的限制
extraction_chain = _prompt | _model.with_structured_output(FriendList, method="function_calling")

# ──────────────────────────────────────────────────────────
# 数据库配置
# ──────────────────────────────────────────────────────────
_DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "admin",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}


# ──────────────────────────────────────────────────────────
# 业务函数：职责分离
# ──────────────────────────────────────────────────────────
def extract_friends(text: str) -> list[FriendInfo]:
    """用 LCEL chain 从自然语言中提取结构化好友列表"""
    print("🤔 正在从文本中提取信息...\n")
    # chain.invoke 接收模板变量字典，输出 FriendList 实例
    result: FriendList = extraction_chain.invoke({"text": text})
    friends = result.friends
    print(f"✅ 提取到 {len(friends)} 条结构化信息:")
    print(json.dumps([f.model_dump() for f in friends], ensure_ascii=False, indent=2))
    return friends


def insert_friends(friends: list[FriendInfo]) -> dict:
    """将好友列表批量写入 MySQL，返回写入结果"""
    if not friends:
        print("⚠️  没有可插入的数据")
        return {"count": 0, "insert_ids": []}

    conn = pymysql.connect(**_DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute("USE hello")
            cursor.executemany(
                "INSERT INTO friends (name, gender, birth_date, company, title, phone, wechat) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [(f.name, f.gender, f.birth_date or None,
                  f.company, f.title, f.phone, f.wechat)
                 for f in friends],
            )
        conn.commit()

        count = cursor.rowcount
        first_id = cursor.lastrowid   # pymysql executemany 后为首条插入 ID
        print(f"\n✅ 成功批量插入 {count} 条，ID：{first_id} ~ {first_id + count - 1}")
        return {"count": count, "insert_ids": list(range(first_id, first_id + count))}

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def extract_and_insert(text: str) -> dict:
    """顶层入口：提取 + 写入"""
    friends = extract_friends(text)
    return insert_friends(friends)


# ──────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────
def main() -> None:
    text = (
        "我最近认识了几个新朋友。第一个是张总，女的，看起来30出头，在腾讯做技术总监，"
        "手机13800138000，微信是zhangzong2024。第二个是李工，男，大概28岁，在阿里云做架构师，"
        "电话15900159000，微信号lee_arch。还有一个是陈经理，女，35岁左右，在美团做产品经理，"
        "手机号是18800188000，微信chenpm2024。"
    )

    print("📝 输入文本:")
    print(text, "\n")

    try:
        result = extract_and_insert(text)
        print(f"\n🎉 处理完成！共插入 {result['count']} 条，"
              f"ID：{', '.join(str(i) for i in result['insert_ids'])}")
    except Exception as e:
        print(f"❌ 处理失败：{e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
