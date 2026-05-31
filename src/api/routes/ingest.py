"""POST /ingest — upload a PDF instantly, index in background.

HOW IT WORKS (simple analogy):
  Think of this like a UiPath process that accepts a file, saves it immediately,
  and then processes it asynchronously (background job).

  Instead of making the user wait 30+ seconds for indexing to complete,
  we return a job ID immediately. The user can then poll /ingest/status/{job_id}
  to see when indexing is done.

  Why async?
    - PDF parsing + CLIP chart detection + Bedrock captioning = ~10-30 seconds
    - Holding an HTTP connection open that long is bad practice
    - Background processing keeps the API responsive

  Flow:
    1. Validate the uploaded file (is it a PDF?)
    2. Upload bytes to S3 immediately (fast, ~1 second)
    3. Register a job record in memory with status="indexing"
    4. Return job_id + s3_key to the caller IMMEDIATELY (HTTP 202 Accepted)
    5. Background thread: parse PDF → extract charts → index into RAG
    6. Update job record to status="done" when indexing completes

  The in-memory job store (_jobs dict) is reset when the server restarts.
  For production, replace with Redis or DynamoDB for persistent job tracking.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated

import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.api.schemas import IngestResponse
from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Router: groups all /ingest endpoints under one prefix
# The "ingestion" tag groups these endpoints in the Swagger UI (/docs)
router = APIRouter(prefix="/ingest", tags=["ingestion"])

# In-memory job status store — maps job_id → job status dict
# Example: {"abc123": {"status": "done", "text_nodes": 42, ...}}
# NOTE: This is reset on server restart — use Redis/DynamoDB for persistence
_jobs: dict[str, dict] = {}


@router.get("/status/{job_id}", tags=["ingestion"], summary="Poll ingestion job status")
async def ingest_status(job_id: str):
    """Return the current status of an ingestion job.

    Call this after POST /ingest to check if indexing is complete.
    Returns 404 if the job_id is not found.

    Possible status values:
      "indexing" → still processing (poll again in a few seconds)
      "done"     → indexing complete, document is now searchable
      "failed"   → something went wrong (check "message" field for details)
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job  # return the full job dict as JSON


@router.post(
    "",                                     # path is just "/ingest" (prefix + "")
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,   # 202 = "request accepted, processing async"
    summary="Upload a PDF — returns instantly, indexes in background",
)
async def ingest_document(
    file: Annotated[UploadFile, File(description="PDF file to ingest")],
    settings: Settings = Depends(get_settings),  # inject settings via FastAPI DI
) -> IngestResponse:
    """Accept a PDF upload, store it in S3, and start background indexing.

    Returns immediately (HTTP 202) with a job_id.
    Poll GET /ingest/status/{job_id} to track indexing progress.

    Validates:
      - File must be a PDF (by extension or content type)
      - File must not be empty
    """
    # ── Step 1: Validate the upload ────────────────────────────────────────────
    # Check file extension first, then fall back to content type check
    if not (file.filename or "").lower().endswith(".pdf"):
        if file.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only PDF files are accepted",
            )

    # Use original filename, or generate a UUID-based name if none provided
    filename = file.filename or f"upload_{uuid.uuid4().hex}.pdf"
    job_id = uuid.uuid4().hex  # unique ID for this ingestion job
    # S3 key: "documents/<job_id>/<filename>" — scoped by job_id to avoid collisions
    s3_key = f"{settings.s3_prefix}{job_id}/{filename}"

    # Read the full file content into memory
    content = await file.read()  # async read — non-blocking

    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    # ── Step 2: Upload to S3 immediately ───────────────────────────────────────
    # This is fast (~1 second for typical PDFs) — we do it synchronously
    # before returning the job ID so the file is safely stored
    try:
        session = boto3.Session(**settings.boto3_session_kwargs)
        s3 = session.client("s3")
        s3.put_object(
            Bucket=settings.s3_bucket,
            Key=s3_key,
            Body=content,
            ContentType="application/pdf",
            # Custom metadata stored with the S3 object — useful for tracking
            Metadata={
                "job_id": job_id,
                "original_filename": filename,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info("Uploaded to s3://%s/%s", settings.s3_bucket, s3_key)
    except ClientError as exc:
        # S3 upload failed — don't proceed with indexing, return error
        logger.error("S3 upload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"S3 upload failed: {exc.response['Error']['Message']}",
        )

    # ── Step 3: Register the job and kick off background indexing ──────────────
    # Store initial job status in memory (will be updated by the background thread)
    _jobs[job_id] = {
        "job_id": job_id,
        "filename": filename,
        "status": "indexing",
        "message": "Uploaded to S3. Indexing text and charts in background…",
        "text_nodes": 0,
        "chart_nodes": 0,
    }

    # Run indexing in a thread pool — it's a CPU/IO heavy operation
    # run_in_executor submits a synchronous function to the thread pool
    # without blocking the async event loop
    asyncio.get_event_loop().run_in_executor(
        None,  # use the default thread pool
        _index_document,  # the function to run in the background
        job_id, content, filename, settings  # arguments to pass to it
    )

    # ── Step 4: Return immediately (HTTP 202) ──────────────────────────────────
    return IngestResponse(
        job_id=job_id,
        filename=filename,
        s3_key=s3_key,
        status="indexing",
        message=f"'{filename}' uploaded. Indexing in background — poll /ingest/status/{job_id}",
    )


def _index_document(job_id: str, content: bytes, filename: str, settings: Settings) -> None:
    """Background worker: parse PDF, extract charts, and index into RAG.

    This runs in a separate thread (not the async event loop thread).
    Updates _jobs[job_id] with progress and final status.

    Gets the shared pipeline from app.state — the same pipeline instance
    used by the /query endpoint. This means newly indexed documents are
    immediately available for querying once indexing is done.
    """
    # Import the app to access app.state.pipeline
    # This is a circular import but safe here — this function runs AFTER startup
    import src.api.main as _main_module
    pipeline = getattr(_main_module.app.state, "pipeline", None)

    if pipeline is None:
        # Pipeline wasn't initialised (startup error) — can't index
        _jobs[job_id].update({"status": "failed", "message": "Pipeline not available"})
        return

    try:
        # Update status to show we're actively processing
        _jobs[job_id]["message"] = "Extracting text and detecting charts…"

        # This is the main work:
        # parse_bytes → chunk → embed → detect charts → caption → insert into index
        result = pipeline.add_pdf_bytes(content, source=filename, generate_chart_captions=True)

        # Extract counts from the result dict
        text_nodes = result.get("text_nodes", 0) if isinstance(result, dict) else result
        chart_nodes = result.get("chart_nodes", 0) if isinstance(result, dict) else 0

        # Build a human-readable summary message
        parts = [f"{text_nodes} text chunks"]
        if chart_nodes:
            parts.append(f"{chart_nodes} charts captioned")

        # Update job to "done" with the final counts
        _jobs[job_id].update({
            "status": "done",
            "message": f"Indexed {' + '.join(parts)} from '{filename}'",
            "text_nodes": text_nodes,
            "chart_nodes": chart_nodes,
        })
        logger.info("Background indexing complete for job %s", job_id)

    except Exception as exc:
        # Something went wrong — store the error message so the user can see it
        logger.error("Background indexing failed for job %s: %s", job_id, exc, exc_info=True)
        _jobs[job_id].update({"status": "failed", "message": str(exc)})
