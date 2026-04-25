"""
测试 Book CRUD 端点（无需 mock，内存仓库无外部依赖）。

覆盖场景：
  POST   /book         创建 → 返回含 id 的 book
  GET    /book         获取初始 3 条
  GET    /book/{id}    存在 → 200；不存在 → 404
  PATCH  /book/{id}    更新标题；不存在 → 404
  DELETE /book/{id}    删除；不存在 → 404

注意：由于 BookService 是模块级单例（@lru_cache），
各测试间共享同一内存仓库状态，测试按声明顺序执行。
"""

import pytest


# ── POST /book ─────────────────────────────────────────────────────────────

def test_create_book(client):
    """创建新 Book，返回 201 及包含 id 和 title 的 JSON。"""
    response = client.post("/book", json={"title": "Clean Code"})
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Clean Code"
    assert "id" in data               # 自动生成 id

def test_create_book_empty_title(client):
    """title 为空字符串时应返回 422 Unprocessable Entity（Pydantic min_length 校验）。"""
    response = client.post("/book", json={"title": ""})
    assert response.status_code == 422


# ── GET /book ──────────────────────────────────────────────────────────────

def test_find_all_books(client):
    """获取 Book 列表，初始数据包含 3 条（+ 上面 create 测试新增的 1 条）。"""
    response = client.get("/book")
    assert response.status_code == 200
    books = response.json()
    assert isinstance(books, list)
    assert len(books) >= 3            # 至少初始 3 条


# ── GET /book/{id} ─────────────────────────────────────────────────────────

def test_find_one_existing(client):
    """按已存在 ID 查询应返回 200 及对应 book。"""
    response = client.get("/book/1")
    assert response.status_code == 200
    assert response.json()["id"] == 1

def test_find_one_not_found(client):
    """按不存在 ID 查询应返回 404。"""
    response = client.get("/book/999")
    assert response.status_code == 404


# ── PATCH /book/{id} ───────────────────────────────────────────────────────

def test_update_book(client):
    """更新已存在 Book 的标题，返回更新后的值。"""
    response = client.patch("/book/2", json={"title": "Updated Title"})
    assert response.status_code == 200
    assert response.json()["title"] == "Updated Title"

def test_update_book_partial(client):
    """PATCH 不传 title 时，返回原标题不变（PATCH 语义）。"""
    # 先查出当前标题
    current = client.get("/book/3").json()["title"]
    # PATCH 空 body
    response = client.patch("/book/3", json={})
    assert response.status_code == 200
    assert response.json()["title"] == current   # 保持不变

def test_update_book_not_found(client):
    """更新不存在 Book 应返回 404。"""
    response = client.patch("/book/999", json={"title": "x"})
    assert response.status_code == 404


# ── DELETE /book/{id} ──────────────────────────────────────────────────────

def test_delete_book(client):
    """删除已存在 Book 应返回 200 及 message。"""
    response = client.delete("/book/1")
    assert response.status_code == 200
    assert "message" in response.json()

def test_delete_book_not_found(client):
    """删除不存在 Book（或已删除）应返回 404。"""
    response = client.delete("/book/1")   # 刚刚已删除
    assert response.status_code == 404
