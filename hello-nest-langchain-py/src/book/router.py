"""Book 路由 — RESTful CRUD 端点。

路由前缀 /book：POST /book, GET /book, GET /book/{id},
PATCH /book/{id}, DELETE /book/{id}
"""

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends

from src.book.schemas import Book, BookCreate, BookUpdate
from src.book.service import BookService

router = APIRouter(prefix="/book", tags=["Books"])


@lru_cache(maxsize=1)
def _get_book_service() -> BookService:
    """BookService 依赖工厂，lru_cache 保证全局单例，跨请求共享 BookRepository 状态。"""
    return BookService()


_BookService = Annotated[BookService, Depends(_get_book_service)]


@router.post("", response_model=Book, status_code=201, summary="创建 Book")
async def create(body: BookCreate, service: _BookService) -> dict:
    return service.create(body)


@router.get("", response_model=list[Book], summary="获取所有 Book")
async def find_all(service: _BookService) -> list[dict]:
    return service.find_all()


@router.get("/{book_id}", response_model=Book, summary="按 ID 获取 Book")
async def find_one(book_id: int, service: _BookService) -> dict:
    return service.find_one(book_id)


@router.patch("/{book_id}", response_model=Book, summary="更新 Book")
async def update(book_id: int, body: BookUpdate, service: _BookService) -> dict:
    return service.update(book_id, body)


@router.delete("/{book_id}", summary="删除 Book")
async def delete(book_id: int, service: _BookService) -> dict:
    return service.delete(book_id)
