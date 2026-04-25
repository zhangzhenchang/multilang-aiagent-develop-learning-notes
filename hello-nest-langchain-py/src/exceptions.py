"""业务层异常定义。

服务层只抛业务异常，HTTP 状态码映射在 main.py 的 exception_handler 中集中管理。
"""


class BookNotFoundError(Exception):
    def __init__(self, book_id: int) -> None:
        self.book_id = book_id
        super().__init__(f"Book #{book_id} not found")
