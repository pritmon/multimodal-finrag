"""FastAPI application for the FinRAG system.

HOW IT WORKS (simple analogy):
  FastAPI is like UiPath Action Center — it receives requests from the outside
  world and routes them to the right workflow (route handler).

  This file is the entry point of the web application. It:
  1. Creates the FastAPI app with all its configuration
  2. Registers all route handlers (ingest, query, entities)
  3. Sets up startup/shutdown logic (initialise heavy models on start)
  4. Provides a health check endpoint

Startup lifespan:
1. Initialise the RAG pipeline (load/build vector index).
2. Initialise the NER inference engine (load LoRA adapter).

Endpoints:
- GET  /health    → check if the service is running and index is loaded
- POST /ingest    → upload a PDF and index it in the background
- POST /query     → ask a question, get an AI-generated answer
- POST /entities  → extract financial entities (ORG, MONEY, DATE, PERCENT)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager  # for defining startup/shutdown logic
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # allows browser-based API calls
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import entities, ingest, query  # route modules
from src.api.schemas import HealthResponse
from src.config import get_settings

logger = logging.getLogger(__name__)

# Configure global logging format once at module load
# Format: "2024-01-15 10:30:00 INFO src.api.main Starting up..."
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=get_settings().log_level,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager — runs on startup and shutdown.

    FastAPI's lifespan replaces the old @app.on_event("startup") pattern.
    Code BEFORE the yield runs at startup.
    Code AFTER the yield runs at shutdown.

    Think of this like UiPath's Initialize State Machine — it sets up
    all the shared resources (pipeline, NER engine) that route handlers need.

    Resources are stored on app.state so all route handlers can access them:
      app.state.pipeline    → the FinRAGPipeline instance
      app.state.ner_engine  → the NERInferenceEngine instance
    """
    settings = get_settings()
    logger.info("FinRAG API starting up (region=%s, model=%s)", settings.aws_region, settings.bedrock_model_id)

    # ── RAG Pipeline Initialisation ────────────────────────────────────────────
    # Load the vector index from disk (or build a new one if it doesn't exist)
    # This is the most important component — all /query requests need this
    try:
        from src.rag.pipeline import FinRAGPipeline
        pipeline = FinRAGPipeline(settings=settings)
        pipeline.load_or_build_index(force_rebuild=False)  # False = use cached index if available
        app.state.pipeline = pipeline  # store on app.state for route handlers to access
        logger.info("RAG pipeline initialised")
    except Exception as exc:
        # Don't crash on startup — set to None and let /health report the issue
        logger.error("Failed to initialise RAG pipeline: %s", exc, exc_info=True)
        app.state.pipeline = None

    # ── NER Engine Initialisation ──────────────────────────────────────────────
    # Load the LoRA fine-tuned BERT model for financial entity extraction
    # This is optional — if the model file doesn't exist, /entities will be unavailable
    try:
        from src.finetune.inference import NERInferenceEngine
        lora_path = settings.lora_model_path
        if lora_path.exists():
            # Model weights found — load the NER engine
            ner_engine = NERInferenceEngine(
                model_path=lora_path,
                base_model_name=settings.base_ner_model,
            )
            app.state.ner_engine = ner_engine
            logger.info("NER engine initialised from %s", lora_path)
        else:
            # No model file — warn but continue (NER is optional)
            logger.warning(
                "LORA_MODEL_PATH %s does not exist; NER endpoint will be unavailable",
                lora_path,
            )
            app.state.ner_engine = None
    except Exception as exc:
        logger.error("Failed to initialise NER engine: %s", exc, exc_info=True)
        app.state.ner_engine = None

    logger.info("Startup complete")

    # ── Application runs here ──────────────────────────────────────────────────
    yield  # FastAPI serves requests here; everything after yield is shutdown

    # ── Shutdown ───────────────────────────────────────────────────────────────
    # Release resources to allow clean shutdown (important in containers)
    logger.info("FinRAG API shutting down")
    app.state.pipeline = None
    app.state.ner_engine = None


# ── Application Factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Factory pattern — returns a configured FastAPI instance.
    Using a factory (instead of a module-level app variable) makes it easy to
    create the app with different settings for testing.
    """
    settings = get_settings()

    # Create the FastAPI app with metadata (appears in auto-generated /docs)
    app = FastAPI(
        title="FinRAG — Multimodal Financial Document Intelligence",
        description=(
            "Multimodal RAG system for financial documents. "
            "Supports PDF ingestion, chart understanding, hybrid retrieval, "
            "and fine-tuned financial NER."
        ),
        version="0.1.0",
        lifespan=lifespan,        # hook in our startup/shutdown logic
        docs_url="/docs",         # Swagger UI at /docs
        redoc_url="/redoc",       # ReDoc UI at /redoc
        openapi_url="/openapi.json",
    )

    # ── CORS Middleware ────────────────────────────────────────────────────────
    # CORS (Cross-Origin Resource Sharing) — allows browsers on different domains
    # to call this API. Required when the frontend is hosted separately from the API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,  # e.g. ["http://localhost:3000"]
        allow_credentials=True,
        allow_methods=["*"],   # allow all HTTP methods (GET, POST, etc.)
        allow_headers=["*"],   # allow all request headers
    )

    # ── Static Files ───────────────────────────────────────────────────────────
    # Serve the frontend HTML/CSS/JS from the "static" directory
    # This is how the demo UI (index.html) is served
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)  # create the directory if it doesn't exist
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ── Route Registration ─────────────────────────────────────────────────────
    # Each router handles a group of related endpoints
    # The prefix and tags defined in each router are used here
    app.include_router(ingest.router)    # POST /ingest, GET /ingest/status/{job_id}
    app.include_router(query.router)     # POST /query
    app.include_router(entities.router)  # POST /entities

    # ── Root Endpoint ──────────────────────────────────────────────────────────
    @app.get("/")
    async def root():
        """Serve the demo UI HTML page."""
        return FileResponse(str(Path(__file__).parent / "static" / "index.html"))

    # ── Health Check Endpoint ──────────────────────────────────────────────────
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["ops"],
        summary="Health check",
        description="Returns service health and component readiness.",
    )
    async def health() -> HealthResponse:
        """Return the health status of the API and its components.

        Checks if the RAG pipeline is initialised and the index is loaded.
        Returns {"status": "ok", "index_loaded": true} when healthy.
        """
        pipeline_ready = getattr(app.state, "pipeline", None) is not None
        return HealthResponse(
            status="ok",
            version="0.1.0",
            index_loaded=pipeline_ready,
        )

    # ── Global Error Handler ───────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc: Exception):
        """Catch any unhandled exception and return a clean 500 error response.

        Without this, unhandled exceptions would return an ugly stack trace.
        This handler returns a structured JSON error instead.
        """
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    return app


# Create the app instance used by uvicorn to serve requests
app = create_app()


# ── Entry Point ───────────────────────────────────────────────────────────────
# This block only runs when you execute: python src/api/main.py
# When deployed with uvicorn (normal operation), uvicorn imports the app variable directly
if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",          # module:variable path to the FastAPI app
        host=settings.api_host,      # default: "0.0.0.0" (all interfaces)
        port=settings.api_port,      # default: 8000
        workers=settings.api_workers,  # default: 4 worker processes
        log_level=settings.log_level.lower(),
        reload=False,                # False in production (True for development)
    )
