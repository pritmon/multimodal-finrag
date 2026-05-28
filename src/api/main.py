"""FastAPI application for the FinRAG system.

Startup lifespan:
1. Initialise the RAG pipeline (load/build vector index).
2. Initialise the NER inference engine (load LoRA adapter).

Endpoints:
- GET  /health
- POST /ingest
- POST /query
- POST /entities
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import entities, ingest, query
from src.api.schemas import HealthResponse
from src.config import get_settings

logger = logging.getLogger(__name__)

# Configure logging format
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=get_settings().log_level,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialise heavy components on startup."""
    settings = get_settings()
    logger.info("FinRAG API starting up (region=%s, model=%s)", settings.aws_region, settings.bedrock_model_id)

    # ── RAG Pipeline ──────────────────────────────────────────────────────────
    try:
        from src.rag.pipeline import FinRAGPipeline
        pipeline = FinRAGPipeline(settings=settings)
        pipeline.load_or_build_index(force_rebuild=False)
        app.state.pipeline = pipeline
        logger.info("RAG pipeline initialised")
    except Exception as exc:
        logger.error("Failed to initialise RAG pipeline: %s", exc, exc_info=True)
        app.state.pipeline = None

    # ── NER Engine ────────────────────────────────────────────────────────────
    try:
        from src.finetune.inference import NERInferenceEngine
        lora_path = settings.lora_model_path
        if lora_path.exists():
            ner_engine = NERInferenceEngine(
                model_path=lora_path,
                base_model_name=settings.base_ner_model,
            )
            app.state.ner_engine = ner_engine
            logger.info("NER engine initialised from %s", lora_path)
        else:
            logger.warning(
                "LORA_MODEL_PATH %s does not exist; NER endpoint will be unavailable",
                lora_path,
            )
            app.state.ner_engine = None
    except Exception as exc:
        logger.error("Failed to initialise NER engine: %s", exc, exc_info=True)
        app.state.ner_engine = None

    logger.info("Startup complete")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("FinRAG API shutting down")
    app.state.pipeline = None
    app.state.ner_engine = None


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="FinRAG — Multimodal Financial Document Intelligence",
        description=(
            "Multimodal RAG system for financial documents. "
            "Supports PDF ingestion, chart understanding, hybrid retrieval, "
            "and fine-tuned financial NER."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(entities.router)

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["ops"],
        summary="Health check",
        description="Returns service health and component readiness.",
    )
    async def health() -> HealthResponse:
        pipeline_ready = getattr(app.state, "pipeline", None) is not None
        return HealthResponse(
            status="ok",
            version="0.1.0",
            index_loaded=pipeline_ready,
        )

    # ── Global error handler ──────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    return app


app = create_app()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        reload=False,
    )
