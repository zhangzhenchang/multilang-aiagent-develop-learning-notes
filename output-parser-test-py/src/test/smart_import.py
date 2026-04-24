"""
smart_import.py - 智能文本提取 + MySQL 批量写入

流程：
  1. 接收自然语言文本（可含多个人的联系信息）
  2. 使用 LangChain with_structured_output 让模型按 Pydantic 模型结构化输出
  3. 将提取结果批量插入 MySQL friends 表

依赖：langchain-openai, pydantic, pymysql, python-dotenv
"""
from __future__ import annotations

import json
import os
from typing import Optional

import pymysql
import pymysql.cursors
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


# ──────────────────────────────────────────────────────────
# Pydantic 模型：对应 friends 表结构（替代 JS 中的 Zod schema）
# ──────────────────────────────────────────────────────────
class FriendInfo(BaseModel):
    """单个好友的结构化信息"""

    name: str = Field(description="姓名")
    gender: str = Field(description="性别（男/女）")
    birth_date: str = Field(
        description="出生日期，格式 YYYY-MM-DD；无具体日期时根据年龄估算"
    )
    company: Optional[str] = Field(default=None, description="公司名称，无则返回 null")
    title: Optional[str] = Field(default=None, description="职位/头衔，无则返回 null")
    phone: Optional[str] = Field(default=None, description="手机号，无则返回 null")
    wechat: Optional[str] = Field(default=None, description="微信号，无则返回 null")


class FriendList(BaseModel):
    """好友信息列表（作为 with_structured_output 的顶层容器）"""

    friends: list[FriendInfo] = Field(description="从文本中提取到的所有好友信息数组")


# ──────────────────────────────────────────────────────────
# 模型初始化
# ──────────────────────────────────────────────────────────
_model = ChatOpenAI(
    model=os.environ["MODEL_NAME"],
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)

# method="function_calling" 使用工具调用协议而非 json_object response_format。
# 默认的 json_mode 要求 prompt 包含 "json" 关键字，部分模型（如 Qwen）会报 400 错误。
structured_model = _model.with_structured_output(FriendList, method="function_calling")

# ──────────────────────────────────────────────────────────
# 数据库连接配置
# ──────────────────────────────────────────────────────────
_DB_CONFIG: dict = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "admin",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

_EXTRACT_PROMPT = """\
请从以下文本中提取所有好友信息，文本中可能包含一个或多个人的信息。请将每个人的信息分别提取出来。

{text}

要求：
1. 文本中有多个人时，为每个人创建单独的对象
2. 字段说明：
   - name：人名
   - gender：性别（男/女）
   - birth_date：出生日期（格式 YYYY-MM-DD，无具体日期时根据年龄估算）
   - company：公司名称
   - title：职位/头衔
   - phone：手机号
   - wechat：微信号
3. 找不到的字段返回 null
4. 结果必须是数组，即使只有一个人也放在数组中\
"""


def extract_and_insert(text: str) -> dict:
    """
    从自然语言文本中提取好友信息并批量写入数据库。

    Args:
        text: 包含好友信息的自然语言文本

    Returns:
        {"count": int, "insert_ids": list[int]}
    """
    conn = pymysql.connect(**_DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute("USE hello")

            # ── Step 1: AI 结构化提取 ──────────────────────────
            print("🤔 正在从文本中提取信息...\n")
            prompt = _EXTRACT_PROMPT.format(text=text)
            result: FriendList = structured_model.invoke(prompt)
            friends = result.friends

            print(f"✅ 提取到 {len(friends)} 条结构化信息:")
            # model_dump() 是 Pydantic v2 的序列化方法（v1 用 .dict()）
            print(json.dumps([f.model_dump() for f in friends], ensure_ascii=False, indent=2))
            print()

            if not friends:
                print("⚠️  没有提取到任何信息")
                return {"count": 0, "insert_ids": []}

            # ── Step 2: 批量插入数据库 ─────────────────────────
            insert_sql = """
                INSERT INTO friends (name, gender, birth_date, company, title, phone, wechat)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            # 将 Pydantic 对象转为元组列表，None 值原样传入（MySQL 写为 NULL）
            values = [
                (f.name, f.gender, f.birth_date or None,
                 f.company, f.title, f.phone, f.wechat)
                for f in friends
            ]

            # executemany 批量插入，比逐条 execute 效率更高
            cursor.executemany(insert_sql, values)
            conn.commit()

            count = cursor.rowcount
            # lastrowid 在 pymysql executemany 后返回第一条插入行的 ID
            first_id = cursor.lastrowid
            insert_ids = list(range(first_id, first_id + count))

            print(f"✅ 成功批量插入 {count} 条数据")
            print(f"   插入的ID范围：{first_id} - {first_id + count - 1}")

            return {"count": count, "insert_ids": insert_ids}

    except Exception as e:
        conn.rollback()
        print(f"❌ 执行出错：{e}")
        raise

    finally:
        conn.close()


def main() -> None:
    sample_text = (
        "我最近认识了几个新朋友。第一个是张总，女的，看起来30出头，在腾讯做技术总监，"
        "手机13800138000，微信是zhangzong2024。第二个是李工，男，大概28岁，在阿里云做架构师，"
        "电话15900159000，微信号lee_arch。还有一个是陈经理，女，35岁左右，在美团做产品经理，"
        "手机号是18800188000，微信chenpm2024。"
    )

    print("📝 输入文本:")
    print(sample_text)
    print()

    try:
        result = extract_and_insert(sample_text)
        print(f"\n🎉 处理完成！成功插入 {result['count']} 条记录")
        print(f"   插入的ID：{', '.join(str(i) for i in result['insert_ids'])}")
    except Exception as e:
        print(f"❌ 处理失败：{e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
