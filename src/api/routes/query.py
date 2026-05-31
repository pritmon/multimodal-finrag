"""POST /query — run a RAG query against the indexed financial documents.

HOW IT WORKS (simple analogy):
  This is the "Ask a Question" endpoint — the core of the RAG system.
  The user sends a question; we return an AI-generated answer with citations.

  Think of it like a UiPath process that:
  1. Receives a question from Action Center
  2. Passes it to the orchestrator workflow (FinRAGPipeline.query())
  3. Returns the answer + sources back to the user

  The heavy lifting is all done in FinRAGPipeline.query():
    - Hybrid retrieval (BM25 + vector + reranker)
    - Bedrock Nova generation
  This route is just the HTTP interface that wraps that pipeline.
"""

from __future__ import annotations

import logging
import time  # for measuring query latency
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.schemas import ChartResult, QueryRequest, QueryResponse, SourceNode
from src.config import Settings, get_settings

# TYPE_CHECKING: only import for type hints (avoids circular imports at runtime)
if TYPE_CHECKING:
    from src.rag.pipeline import FinRAGPipeline

logger = logging.getLogger(__name__)

# All endpoints in this router will be under /query with tag "rag"
router = APIRouter(prefix="/query", tags=["rag"])


def _get_pipeline(request: Request) -> "FinRAGPipeline":
    """Dependency function: get the shared RAG pipeline from app.state.

    FastAPI calls this automatically when a route handler declares it as
    a dependency (via Depends or direct function call).

    Raises HTTP 503 if the pipeline isn't ready (still starting up,
    or failed to initialise at startup).
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialised; please wait for startup to complete",
        )
    return pipeline


@router.post(
    "",                                     # path: POST /query
    response_model=QueryResponse,           # validates + documents the response shape
    status_code=status.HTTP_200_OK,
    summary="Query indexed financial documents",
    description=(
        "Submit a natural-language question. The RAG pipeline retrieves relevant "
        "passages and chart captions from the indexed documents using hybrid BM25 + "
        "vector search with cross-encoder reranking, then generates an answer using "
        "AWS Bedrock Claude 3."
    ),
)
async def query_documents(
    body: QueryRequest,           # the request body (validated by Pydantic)
    request: Request,             # the raw FastAPI request (needed to access app.state)
    settings: Settings = Depends(get_settings),  # injected settings
) -> QueryResponse:
    """Handle a RAG query request.

    Validates the request body with Pydantic (question length, top_k range, etc.),
    runs the query through the RAG pipeline, measures latency, and returns
    a structured response with the answer, source citations, and optional charts.
    """
    # Get the shared pipeline from app.state (raises 503 if not available)
    pipeline = _get_pipeline(request)

    logger.info(
        "RAG query: %r (top_k=%d, filters=%s)",
        body.question,
        body.top_k,
        body.filters,
    )

    # Measure total query latency (retrieval + reranking + LLM generation)
    t0 = time.perf_counter()
    try:
        # Run the full RAG pipeline: retrieve → rerank → generate
        result = pipeline.query(
            question=body.question,
            top_k=body.top_k,
            filters=body.filters,
        )
    except Exception as exc:
        logger.exception("RAG pipeline query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        )
    latency_ms = (time.perf_counter() - t0) * 1000  # convert seconds to milliseconds

    # ── Build the response ─────────────────────────────────────────────────────
    # Convert internal NodeWithScore objects to API-friendly SourceNode schemas
    sources = [
        SourceNode(
            text=n.node.get_content()[:500],              # truncate to 500 chars for response size
            score=round(n.score or 0.0, 4),               # relevance score (0.0 to 1.0)
            page_number=n.node.metadata.get("page_number"),  # which page it came from
            source=n.node.metadata.get("source"),            # which file it came from
            is_heading=n.node.metadata.get("is_heading"),    # was this chunk a heading?
            node_type=n.node.metadata.get("node_type"),      # "text" or "chart_caption"
        )
        for n in result.source_nodes
    ]

    # Convert ChartNode objects to API ChartResult schemas (only if requested)
    charts: list[ChartResult] = []
    if body.include_charts:
        for c in result.chart_nodes:
            d = c.to_dict()  # ChartNode → dict (includes base64-encoded image)
            charts.append(
                ChartResult(
                    caption=d["caption"],
                    chart_type=d["chart_type"],
                    clip_score=round(d["clip_score"], 4),
                    page_number=d["page_number"],
                    source=d["source"],
                    width=d["width"],
                    height=d["height"],
                    image_b64=d["image_b64"],  # base64-encoded PNG for display in browser
                )
            )

    return QueryResponse(
        answer=result.answer,           # the LLM-generated answer
        query=result.query,             # the original question (echoed back)
        model_id=result.model_id,       # which Bedrock model was used
        sources=sources,                # cited source passages
        charts=charts,                  # relevant chart images (if any)
        latency_ms=round(latency_ms, 1),  # total time in milliseconds
    )
