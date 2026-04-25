"""Book Pydantic 模型 — 请求体/响应体的数据校验与序列化。"""

from pydantic import BaseModel, Field


class Book(BaseModel):
    """Book 响应模型，用于 response_model。"""

    id: int = Field(..., description="Book 唯一 ID（自增）")
    title: str = Field(..., description="Book 标题")


class BookCreate(BaseModel):
    """POST /book 请求体。"""

    title: str = Field(
        ...,
        min_length=1,
        description="Book 标题",
        examples=["The Pragmatic Programmer"],
    )


class BookUpdate(BaseModel):
    """PATCH /book/{id} 请求体，所有字段可选。"""

    title: str | None = Field(
        None,
        min_length=1,
        description="Book 标题（不传则保持原值）",
        examples=["Clean Architecture"],
    )
