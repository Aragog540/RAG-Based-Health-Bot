"""
app/models.py
─────────────
Request and response schemas used by the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The medical question to ask",
        examples=["What are the symptoms of Type 2 Diabetes?"],
    )
    language: str = Field(
        default="en",
        description="Target language for the answer, using a language code or name",
        examples=["en", "es", "fr"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for multi-turn conversations",
    )


class SourceDocument(BaseModel):
    content: str = Field(description="Chunk text used to generate the answer")
    page: Optional[int] = Field(default=None, description="Page number in the PDF")
    source: Optional[str] = Field(default=None, description="Source filename")


class ChatResponse(BaseModel):
    answer: str = Field(description="LLM-generated answer grounded in the book")
    language: str = Field(
        default="en",
        description="Language used for the answer",
    )
    sources: List[SourceDocument] = Field(
        default_factory=list,
        description="Document chunks that supported the answer",
    )
    grounded: bool = Field(
        default=True,
        description="Whether the answer was grounded in retrieved documents",
    )
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    llm_model: str
    embedding_model: str
    collection: str
    total_chunks: int


class IngestResponse(BaseModel):
    message: str
    chunks_added: int
    total_chunks: int
