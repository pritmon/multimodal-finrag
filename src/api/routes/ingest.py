"""POST /ingest — upload a PDF, extract text + charts, index into RAG pipeline."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from typing import Annotated

from src.api.schemas import IngestResponse
from src.config import Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF and ingest (text + charts)",
    description=(
        "Accepts a PDF upload, stores it in S3, extracts text blocks and chart images, "
        "captions charts with Claude Vision, then indexes everything into the RAG pipeline."
    ),
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

    logger.info("Ingesting %s (%d bytes)", filename, len(content))

    # ── Upload to S3 ──────────────────────────────────────────────────────────
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

    # ── Full multimodal ingestion (text + charts) ─────────────────────────────
    from fastapi import Request
    from starlette.requests import Request as StarletteRequest

    # Access app state via a workaround — get pipeline from app state
    import src.api.main as _main_module
    pipeline = getattr(_main_module.app.state, "pipeline", None)

    text_nodes = 0
    chart_nodes = 0
    chart_captions = []

    if pipeline is not None:
        try:
            result = pipeline.add_pdf_bytes(
                content,
                source=filename,
                generate_chart_captions=True,
            )
            if isinstance(result, dict):
                text_nodes = result.get("text_nodes", 0)
                chart_nodes = result.get("chart_nodes", 0)
                chart_captions = result.get("chart_captions", [])
            else:
                text_nodes = result
            logger.info(
                "Indexed %d text nodes + %d chart nodes from %s",
                text_nodes, chart_nodes, filename,
            )
        except Exception as exc:
            logger.error("Indexing failed: %s", exc, exc_info=True)
            return IngestResponse(
                job_id=job_id,
                filename=filename,
                s3_key=s3_key,
                status="upload_ok_index_failed",
                message=f"Uploaded to S3 but indexing failed: {exc}",
            )
    else:
        logger.warning("Pipeline not available — document stored in S3 only")

    parts = [f"{text_nodes} text chunks"]
    if chart_nodes:
        parts.append(f"{chart_nodes} charts captioned by Claude Vision")
    summary = " + ".join(parts)

    return IngestResponse(
        job_id=job_id,
        filename=filename,
        s3_key=s3_key,
        status="indexed",
        message=f"Successfully indexed {summary} from '{filename}'",
    )
