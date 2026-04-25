"""Book 仓库层 — 内存数据访问。"""


class BookRepository:
    """内存 Book 仓库，支持完整 CRUD。"""

    def __init__(self) -> None:
        self._books: list[dict] = [
            {"id": 1, "title": "Book 1"},
            {"id": 2, "title": "Book 2"},
            {"id": 3, "title": "Book 3"},
        ]

    def find_all(self) -> list[dict]:
        return [dict(b) for b in self._books]

    def find_by_id(self, book_id: int) -> dict | None:
        for book in self._books:
            if book["id"] == book_id:
                return dict(book)
        return None

    def create(self, title: str) -> dict:
        new_id = max((b["id"] for b in self._books), default=0) + 1
        book: dict = {"id": new_id, "title": title}
        self._books.append(book)
        return dict(book)

    def update(self, book_id: int, title: str | None) -> dict | None:
        """PATCH 语义：title=None 时保持原值不变。"""
        for book in self._books:
            if book["id"] == book_id:
                if title is not None:
                    book["title"] = title
                return dict(book)
        return None

    def delete(self, book_id: int) -> bool:
        for i, book in enumerate(self._books):
            if book["id"] == book_id:
                del self._books[i]
                return True
        return False
