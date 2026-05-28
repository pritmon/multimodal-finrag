"""Pydantic v2 request and response schemas for the FinRAG API."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ── Shared ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    index_loaded: bool


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the async ingestion job")
    filename: str
    s3_key: str
    status: str = Field(default="queued", description="queued | processing | done | error")
    message: str = ""


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="Natural-language question")
    top_k: int = Field(default=8, ge=1, le=50, description="Number of source nodes to retrieve")
    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Metadata filters applied to retrieved nodes, e.g. {\"source\": \"report.pdf\"}",
    )
    include_charts: bool = Field(default=True, description="Whether to include chart images in the response")

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class SourceNode(BaseModel):
    text: str = Field(..., description="Excerpt of the retrieved passage (truncated to 500 chars)")
    score: float
    page_number: Optional[int] = None
    source: Optional[str] = None
    is_heading: Optional[bool] = None
    node_type: Optional[str] = None


class ChartResult(BaseModel):
    caption: str
    chart_type: str
    clip_score: float
    page_number: int
    source: str
    width: int
    height: int
    image_b64: str = Field(..., description="Base64-encoded PNG of the chart image")


class QueryResponse(BaseModel):
    answer: str
    query: str
    model_id: str
    sources: list[SourceNode]
    charts: list[ChartResult] = []
    latency_ms: float


# ── Entities ──────────────────────────────────────────────────────────────────

class EntityRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000, description="Text to extract entities from")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be blank")
        return v


class EntityResult(BaseModel):
    text: str
    label: str = Field(..., description="Entity type: ORG, MONEY, DATE, PERCENT")
    start: int
    end: int
    confidence: float


class EntityResponse(BaseModel):
    entities: list[EntityResult]
    entity_count: int
    text_length: int
    model_path: str
