"""POST /ingest — upload a PDF instantly, index in background."""

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
router = APIRouter(prefix="/ingest", tags=["ingestion"])

# In-memory job status store  {job_id: {status, message, ...}}
_jobs: dict[str, dict] = {}


@router.get("/status/{job_id}", tags=["ingestion"], summary="Poll ingestion job status")
async def ingest_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF — returns instantly, indexes in background",
)
async def ingest_document(
    file: Annotated[UploadFile, File(description="PDF file to ingest")],
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    # ── Validate ──────────────────────────────────────────────────────────────
    if not (file.filename or "").lower().endswith(".pdf"):
        if file.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only PDF files are accepted",
            )

    filename = file.filename or f"upload_{uuid.uuid4().hex}.pdf"
    job_id = uuid.uuid4().hex
    s3_key = f"{settings.s3_prefix}{job_id}/{filename}"
    content = await file.read()

    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    # ── Upload to S3 (fast, synchronous) ──────────────────────────────────────
    try:
        session = boto3.Session(**settings.boto3_session_kwargs)
        s3 = session.client("s3")
        s3.put_object(
            Bucket=settings.s3_bucket,
            Key=s3_key,
            Body=content,
            ContentType="application/pdf",
            Metadata={
                "job_id": job_id,
                "original_filename": filename,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info("Uploaded to s3://%s/%s", settings.s3_bucket, s3_key)
    except ClientError as exc:
        logger.error("S3 upload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"S3 upload failed: {exc.response['Error']['Message']}",
        )

    # ── Register job and kick off background indexing ─────────────────────────
    _jobs[job_id] = {
        "job_id": job_id,
        "filename": filename,
        "status": "indexing",
        "message": "Uploaded to S3. Indexing text and charts in background…",
        "text_nodes": 0,
        "chart_nodes": 0,
    }

    asyncio.get_event_loop().run_in_executor(
        None, _index_document, job_id, content, filename, settings
    )

    return IngestResponse(
        job_id=job_id,
        filename=filename,
        s3_key=s3_key,
        status="indexing",
        message=f"'{filename}' uploaded. Indexing in background — poll /ingest/status/{job_id}",
    )


def _index_document(job_id: str, content: bytes, filename: str, settings: Settings) -> None:
    """Run in a thread pool — parse PDF, extract charts, index into RAG."""
    import src.api.main as _main_module
    pipeline = getattr(_main_module.app.state, "pipeline", None)

    if pipeline is None:
        _jobs[job_id].update({"status": "failed", "message": "Pipeline not available"})
        return

    try:
        _jobs[job_id]["message"] = "Extracting text and detecting charts…"
        result = pipeline.add_pdf_bytes(content, source=filename, generate_chart_captions=True)

        text_nodes = result.get("text_nodes", 0) if isinstance(result, dict) else result
        chart_nodes = result.get("chart_nodes", 0) if isinstance(result, dict) else 0

        parts = [f"{text_nodes} text chunks"]
        if chart_nodes:
            parts.append(f"{chart_nodes} charts captioned")

        _jobs[job_id].update({
            "status": "done",
            "message": f"Indexed {' + '.join(parts)} from '{filename}'",
            "text_nodes": text_nodes,
            "chart_nodes": chart_nodes,
        })
        logger.info("Background indexing complete for job %s", job_id)

    except Exception as exc:
        logger.error("Background indexing failed for job %s: %s", job_id, exc, exc_info=True)
        _jobs[job_id].update({"status": "failed", "message": str(exc)})
