"""Pydantic v2 request and response schemas for the FinRAG API.

HOW IT WORKS (simple analogy):
  Think of these as the "argument types" and "return types" for each API endpoint.
  Like declaring variable types in UiPath — except Pydantic also validates
  the values automatically and generates API documentation.

  When a request comes in:
    1. Pydantic reads the JSON body
    2. Validates each field against the schema (type, length, range, etc.)
    3. If invalid → returns HTTP 422 Unprocessable Entity automatically
    4. If valid → creates a Python object you can work with in the route handler

  When a response goes out:
    1. The route handler returns a Python object
    2. Pydantic serialises it to JSON
    3. Only fields declared in the schema are included (sensitive internals stay hidden)

SCHEMA GROUPS:
  HealthResponse     → GET /health response
  IngestResponse     → POST /ingest response
  QueryRequest       → POST /query request body
  QueryResponse      → POST /query response
  SourceNode         → one cited passage in a query response
  ChartResult        → one chart image in a query response
  EntityRequest      → POST /entities request body
  EntityResponse     → POST /entities response
  EntityResult       → one detected named entity
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response schema for GET /health.

    Returns the API version and whether the index is ready to serve queries.
    """
    status: str = "ok"          # always "ok" — if the API is down, no response arrives
    version: str                 # API version string (e.g. "0.1.0")
    index_loaded: bool           # True if the vector index is loaded and ready


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response schema for POST /ingest.

    Returned immediately (HTTP 202) — indexing continues in the background.
    Use job_id to poll GET /ingest/status/{job_id}.
    """
    job_id: str = Field(..., description="Unique identifier for the async ingestion job")
    filename: str                # original filename of the uploaded PDF
    s3_key: str                  # full S3 key where the PDF was stored
    status: str = Field(default="queued", description="queued | processing | done | error")
    message: str = ""            # human-readable status message


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body schema for POST /query.

    Pydantic validates all fields before the route handler runs.
    Invalid requests (too short, out of range) get auto-rejected with HTTP 422.
    """
    question: str = Field(
        ...,
        min_length=3,       # reject single-word or empty questions
        max_length=2000,    # prevent huge inputs that could slow the LLM
        description="Natural-language question",
    )
    top_k: int = Field(
        default=8,
        ge=1,               # must be at least 1
        le=50,              # cap at 50 — more than this overwhelms the LLM context
        description="Number of source nodes to retrieve",
    )
    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Metadata filters applied to retrieved nodes, e.g. {\"source\": \"report.pdf\"}",
    )
    include_charts: bool = Field(
        default=True,
        description="Whether to include chart images in the response",
    )

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        """Reject questions that are all whitespace after stripping."""
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class SourceNode(BaseModel):
    """One retrieved passage from the indexed documents.

    Included in QueryResponse.sources — represents a cited source chunk.
    The text is truncated to 500 characters to keep responses compact.
    """
    text: str = Field(..., description="Excerpt of the retrieved passage (truncated to 500 chars)")
    score: float               # relevance score from the retriever (0.0 to 1.0)
    page_number: Optional[int] = None   # which page of the PDF this came from
    source: Optional[str] = None        # which file this came from (filename or S3 key)
    is_heading: Optional[bool] = None   # whether this chunk was a heading in the PDF
    node_type: Optional[str] = None     # "text" or "chart_caption"


class ChartResult(BaseModel):
    """One detected chart image with its AI-generated caption.

    Included in QueryResponse.charts when the retrieved passages include
    pages that contained charts.
    """
    caption: str               # AI-generated description of the chart (from Bedrock)
    chart_type: str            # top CLIP label (e.g. "a bar chart")
    clip_score: float          # CLIP confidence that this is a chart (0.0 to 1.0)
    page_number: int           # which page the chart appeared on (0-indexed)
    source: str                # which PDF file the chart came from
    width: int                 # chart image width in pixels
    height: int                # chart image height in pixels
    image_b64: str = Field(..., description="Base64-encoded PNG of the chart image")


class QueryResponse(BaseModel):
    """Response schema for POST /query.

    The complete RAG response: answer + sources + charts + timing.
    """
    answer: str                # the LLM-generated answer to the question
    query: str                 # the original question (echoed back for confirmation)
    model_id: str              # which Bedrock model was used to generate the answer
    sources: list[SourceNode]  # cited source passages used to generate the answer
    charts: list[ChartResult] = []  # relevant chart images from the same pages
    latency_ms: float          # total query time in milliseconds (retrieval + generation)


# ── Entities ──────────────────────────────────────────────────────────────────

class EntityRequest(BaseModel):
    """Request body schema for POST /entities.

    The text to extract financial named entities from.
    Can be a full paragraph, multiple sentences, or a whole page of text.
    """
    text: str = Field(
        ...,
        min_length=1,         # must have at least one character
        max_length=50_000,    # ~50KB of text — large enough for a full page
        description="Text to extract entities from",
    )

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Reject text that is all whitespace after stripping."""
        v = v.strip()
        if not v:
            raise ValueError("text must not be blank")
        return v


class EntityResult(BaseModel):
    """One detected named entity in the input text.

    Includes the exact matched text, entity type, character position,
    and the model's confidence score.

    Example:
      text="Goldman Sachs", label="ORG", start=0, end=13, confidence=0.9821
    """
    text: str                  # exact string matched in the input
    label: str = Field(..., description="Entity type: ORG, MONEY, DATE, PERCENT")
    start: int                 # character offset where entity starts (inclusive)
    end: int                   # character offset where entity ends (exclusive)
    confidence: float          # model confidence (0.0 = uncertain, 1.0 = very confident)


class EntityResponse(BaseModel):
    """Response schema for POST /entities.

    Returns all detected entities plus summary statistics.
    """
    entities: list[EntityResult]  # all detected entities, sorted by start position
    entity_count: int             # total number of entities found
    text_length: int              # length of the input text (in characters)
    model_path: str               # path to the LoRA model weights that were used
