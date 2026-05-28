"""POST /query — run a RAG query against the indexed financial documents."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.schemas import ChartResult, QueryRequest, QueryResponse, SourceNode
from src.config import Settings, get_settings

if TYPE_CHECKING:
    from src.rag.pipeline import FinRAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["rag"])


def _get_pipeline(request: Request) -> "FinRAGPipeline":
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialised; please wait for startup to complete",
        )
    return pipeline


@router.post(
    "",
    response_model=QueryResponse,
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
    body: QueryRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    pipeline = _get_pipeline(request)

    logger.info(
        "RAG query: %r (top_k=%d, filters=%s)",
        body.question,
        body.top_k,
        body.filters,
    )

    t0 = time.perf_counter()
    try:
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
    latency_ms = (time.perf_counter() - t0) * 1000

    # Build response
    sources = [
        SourceNode(
            text=n.node.get_content()[:500],
            score=round(n.score or 0.0, 4),
            page_number=n.node.metadata.get("page_number"),
            source=n.node.metadata.get("source"),
            is_heading=n.node.metadata.get("is_heading"),
            node_type=n.node.metadata.get("node_type"),
        )
        for n in result.source_nodes
    ]

    charts: list[ChartResult] = []
    if body.include_charts:
        for c in result.chart_nodes:
            d = c.to_dict()
            charts.append(
                ChartResult(
                    caption=d["caption"],
                    chart_type=d["chart_type"],
                    clip_score=round(d["clip_score"], 4),
                    page_number=d["page_number"],
                    source=d["source"],
                    width=d["width"],
                    height=d["height"],
                    image_b64=d["image_b64"],
                )
            )

    return QueryResponse(
        answer=result.answer,
        query=result.query,
        model_id=result.model_id,
        sources=sources,
        charts=charts,
        latency_ms=round(latency_ms, 1),
    )
