"""
create_table.py - MySQL 数据库初始化脚本

创建 hello 数据库及 friends 表，并插入一条 demo 数据。
依赖：pymysql
"""
from __future__ import annotations

import pymysql
import pymysql.cursors


# ──────────────────────────────────────────────────────────
# 数据库连接配置（不指定 database，由脚本内部切换）
# ──────────────────────────────────────────────────────────
CONNECTION_CONFIG: dict = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "admin",
    "charset": "utf8mb4",
    # DictCursor 让查询结果以字典形式返回，更直观
    "cursorclass": pymysql.cursors.DictCursor,
}


def main() -> None:
    # 建立连接（不指定 db，稍后通过 SQL 切换）
    conn = pymysql.connect(**CONNECTION_CONFIG)

    try:
        with conn.cursor() as cursor:
            # 创建数据库（utf8mb4 支持全字符集，包括 emoji）
            cursor.execute(
                "CREATE DATABASE IF NOT EXISTS hello "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cursor.execute("USE hello")

            # 创建好友表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS friends (
                    id         INT AUTO_INCREMENT PRIMARY KEY,
                    name       VARCHAR(50)  NOT NULL,
                    gender     VARCHAR(10),          -- 性别
                    birth_date DATE,                 -- 出生日期
                    company    VARCHAR(100),         -- 公司
                    title      VARCHAR(100),         -- 职位
                    phone      VARCHAR(20),          -- 当前手机号
                    wechat     VARCHAR(50)           -- 微信号
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

            # 插入 demo 数据（使用参数化查询防止 SQL 注入）
            insert_sql = """
                INSERT INTO friends (name, gender, birth_date, company, title, phone, wechat)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            demo_values = (
                "王经理",            # name
                "男",               # gender
                "1990-01-01",       # birth_date
                "字节跳动",          # company
                "产品经理/产品总监",  # title
                "18612345678",      # phone
                "wangjingli2024",   # wechat
            )
            cursor.execute(insert_sql, demo_values)

        conn.commit()
        print(f"成功创建数据库和表，并插入 demo 数据，插入ID：{cursor.lastrowid}")

    except Exception as e:
        conn.rollback()
        print(f"执行出错：{e}")
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()
