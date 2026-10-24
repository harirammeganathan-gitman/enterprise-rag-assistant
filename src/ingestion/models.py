"""Typed document and chunk models shared by ingestion and retrieval."""

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    text: str = Field(min_length=1)
    source: str
    file_hash: str
    chunk_index: int = Field(ge=0)
    page_number: int | None = Field(default=None, ge=1)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
