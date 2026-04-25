"""Book 服务层 — 业务逻辑。"""

from src.book.repository import BookRepository
from src.book.schemas import BookCreate, BookUpdate
from src.exceptions import BookNotFoundError


class BookService:
    """Book CRUD 服务。"""

    def __init__(self) -> None:
        self._repo = BookRepository()

    def create(self, body: BookCreate) -> dict:
        return self._repo.create(body.title)

    def find_all(self) -> list[dict]:
        return self._repo.find_all()

    def find_one(self, book_id: int) -> dict:
        book = self._repo.find_by_id(book_id)
        if book is None:
            raise BookNotFoundError(book_id)
        return book

    def update(self, book_id: int, body: BookUpdate) -> dict:
        book = self._repo.update(book_id, body.title)
        if book is None:
            raise BookNotFoundError(book_id)
        return book

    def delete(self, book_id: int) -> dict:
        if not self._repo.delete(book_id):
            raise BookNotFoundError(book_id)
        return {"message": f"Book #{book_id} removed"}
