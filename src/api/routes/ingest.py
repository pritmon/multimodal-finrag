"""POST /ingest — upload a PDF document to S3 and trigger async Lambda processing."""

from __future__ import annotations

import json
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


def _get_s3_client(settings: Settings = Depends(get_settings)):
    session = boto3.Session(**settings.boto3_session_kwargs)
    return session.client("s3")


def _get_lambda_client(settings: Settings = Depends(get_settings)):
    session = boto3.Session(**settings.boto3_session_kwargs)
    return session.client("lambda")


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF and trigger async ingestion",
    description=(
        "Accepts a multipart/form-data PDF upload, stores it in S3, then "
        "asynchronously invokes the Lambda document processor. Returns a job_id "
        "that can be used to poll status (via DynamoDB or a future status endpoint)."
    ),
)
async def ingest_document(
    file: Annotated[UploadFile, File(description="PDF file to ingest")],
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    # ── Validate file type ────────────────────────────────────────────────────
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only PDF files are accepted",
            )

    filename = file.filename or f"upload_{uuid.uuid4().hex}.pdf"
    job_id = uuid.uuid4().hex
    s3_key = f"{settings.s3_prefix}{job_id}/{filename}"

    # ── Read file content ─────────────────────────────────────────────────────
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    logger.info("Ingesting file: %s (%d bytes) → %s", filename, len(content), s3_key)

    # ── Upload to S3 ──────────────────────────────────────────────────────────
    s3 = _get_s3_client(settings)
    try:
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

    # ── Invoke Lambda asynchronously ──────────────────────────────────────────
    lambda_payload = json.dumps({"bucket": settings.s3_bucket, "key": s3_key})
    lambda_client = _get_lambda_client(settings)
    try:
        lambda_client.invoke(
            FunctionName=settings.lambda_function_name,
            InvocationType="Event",  # async (fire-and-forget)
            Payload=lambda_payload.encode(),
        )
        logger.info("Lambda invoked asynchronously for job %s", job_id)
        trigger_status = "queued"
        message = "Document uploaded and processing queued"
    except ClientError as exc:
        # Lambda trigger failure is non-fatal: document is in S3
        logger.warning("Lambda invocation failed (non-fatal): %s", exc)
        trigger_status = "uploaded_only"
        message = "Document uploaded to S3; Lambda trigger failed — manual processing may be required"

    return IngestResponse(
        job_id=job_id,
        filename=filename,
        s3_key=s3_key,
        status=trigger_status,
        message=message,
    )
