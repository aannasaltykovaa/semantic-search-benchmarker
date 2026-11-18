"""Typed data passed between chunking, indexing, and evaluation."""
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    parent_doc: str
    metadata: dict = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float


class Evidence(BaseModel):
    parent_doc: str
    text: str


class EvalQuery(BaseModel):
    query_id: str
    query: str
    kind: str
    evidence: list[Evidence] = Field(min_length=1)
