"""AWS Lambda handler for asynchronous financial document processing.

HOW IT WORKS (simple analogy):
  Think of this Lambda function like a UiPath robot triggered by a file
  appearing in a folder (S3 bucket).

  When a new PDF is uploaded to S3:
    1. S3 sends an event notification → Lambda is triggered automatically
    2. Lambda downloads the PDF from S3
    3. Parses text + extracts charts → indexes into the RAG vector store
    4. Saves document metadata to DynamoDB (title, page count, job status)
    5. Returns a summary of what was processed

  This runs separately from the API server — it's invoked by AWS
  when files land in S3, not by user HTTP requests.

  Event formats supported:
    - Direct S3 notification: triggered by S3 bucket event
    - SQS-wrapped S3 notification: S3 → SQS queue → Lambda
    - Direct invocation: {"bucket": "...", "key": "..."} (for testing)

Environment variables (injected by Lambda / ECS):
- All variables from src/config.py (AWS_REGION, S3_BUCKET, etc.)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus  # S3 keys can have URL-encoded characters

import boto3

# Configure structured logging early — Lambda captures stdout as CloudWatch logs
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=os.environ.get("LOG_LEVEL", "INFO"),
)
logger = logging.getLogger(__name__)


def _get_dynamodb_client():
    """Create and return a DynamoDB client.

    DynamoDB stores document metadata (filename, page count, index status).
    Like a database table in UiPath's Data Service.
    """
    return boto3.client("dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def _get_s3_client():
    """Create and return an S3 client for downloading PDF files."""
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def _lazy_pipeline():
    """Import and initialise the RAG pipeline (called on cold start).

    Lazy import avoids loading heavy ML libraries (torch, sentence-transformers)
    until they're actually needed. Lambda cold starts are slow enough already.
    """
    from src.config import get_settings
    from src.rag.pipeline import FinRAGPipeline

    settings = get_settings()
    pipeline = FinRAGPipeline(settings=settings)
    # Load existing index from disk if available (avoids rebuilding from scratch)
    pipeline.load_or_build_index(force_rebuild=False)
    return pipeline


# Module-level singleton: Lambda reuses the same pipeline across warm invocations
# On a warm invocation (Lambda container already running), this is already initialised
# On a cold start (new container), this is None and gets created by _get_pipeline()
_PIPELINE = None


def _get_pipeline():
    """Return the shared RAG pipeline, initialising it on cold start.

    Lambda warm start optimisation: the pipeline (with loaded ML models and
    vector index) stays in memory between invocations of the same container.
    This means subsequent invocations skip the ~10-second initialisation.
    """
    global _PIPELINE
    if _PIPELINE is None:
        logger.info("Cold start: initialising RAG pipeline")
        _PIPELINE = _lazy_pipeline()
    return _PIPELINE


# ── Lambda Handler ────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    """Main Lambda entry point — called by AWS for every invocation.

    Accepts S3 PutObject event notifications (single or batched via SQS).
    Processes each PDF file found in the event records.
    Returns a summary dict with processing results.

    AWS passes two arguments:
      event: dict containing the trigger data (S3 notification, SQS message, etc.)
      context: Lambda runtime context (remaining time, function name, etc.)
    """
    logger.info("Received event: %s", json.dumps(event))
    # Parse the event to extract {bucket, key} records for each PDF
    records = _extract_s3_records(event)

    if not records:
        logger.warning("No S3 records found in event; skipping")
        return {"statusCode": 200, "processed": 0}

    processed = 0
    errors: list[dict] = []

    # Process each PDF file — continue even if one fails
    for record in records:
        bucket = record["bucket"]
        key = record["key"]
        try:
            result = _process_document(bucket, key)
            logger.info("Processed s3://%s/%s → %d nodes", bucket, key, result["nodes_added"])
            processed += 1
        except Exception as exc:
            # Log the error but continue with remaining files
            logger.exception("Failed to process s3://%s/%s: %s", bucket, key, exc)
            errors.append({"bucket": bucket, "key": key, "error": str(exc)})

    # HTTP 200 if all succeeded, HTTP 207 (Multi-Status) if some failed
    response = {
        "statusCode": 200 if not errors else 207,
        "processed": processed,
        "errors": errors,
    }
    logger.info("Lambda complete: %s", response)
    return response


def _process_document(bucket: str, key: str) -> dict:
    """Download, parse, index a single PDF and store metadata in DynamoDB.

    This is the main per-document processing function:
      1. Download the PDF bytes from S3
      2. Pass bytes to the RAG pipeline (parse + chunk + embed + index)
      3. Write document metadata to DynamoDB for tracking
      4. Return processing summary

    Returns a dict with bucket, key, etag, nodes_added, processing_seconds.
    """
    s3 = _get_s3_client()

    # ── Step 1: Download the PDF from S3 ──────────────────────────────────────
    logger.info("Downloading s3://%s/%s", bucket, key)
    start_ts = time.time()
    response = s3.get_object(Bucket=bucket, Key=key)
    pdf_bytes = response["Body"].read()  # download all bytes into memory
    s3_metadata = response.get("Metadata", {})  # any custom metadata stored with the file
    etag = response.get("ETag", "").strip('"')  # ETag is the S3 checksum (MD5)
    content_length = len(pdf_bytes)
    logger.info("Downloaded %d bytes (ETag=%s)", content_length, etag)

    # ── Step 2: Index the document through the RAG pipeline ───────────────────
    pipeline = _get_pipeline()  # get or create the shared pipeline
    filename = key.split("/")[-1]  # extract just the filename from the full S3 key
    nodes_added = pipeline.add_pdf_bytes(
        pdf_bytes,
        source=f"s3://{bucket}/{key}",  # use the full S3 path as the document identifier
        generate_chart_captions=True,    # run CLIP + Bedrock captioning on charts
    )

    elapsed = time.time() - start_ts  # total time for download + indexing

    # ── Step 3: Store document metadata in DynamoDB ────────────────────────────
    # DynamoDB uses a specific format: each value is {"S": "..."} or {"N": "..."}
    # S = String, N = Number (stored as string)
    doc_metadata = {
        "document_id": {"S": etag or _make_doc_id(key)},  # use ETag as primary key
        "s3_key": {"S": key},
        "s3_bucket": {"S": bucket},
        "filename": {"S": filename},
        "size_bytes": {"N": str(content_length)},
        "nodes_added": {"N": str(nodes_added)},
        "status": {"S": "indexed"},
        "indexed_at": {"S": datetime.now(timezone.utc).isoformat()},
        "processing_seconds": {"N": str(round(elapsed, 2))},
        # Include any custom S3 metadata (job_id, uploaded_by, etc.) with a "meta_" prefix
        **{f"meta_{k}": {"S": v} for k, v in s3_metadata.items()},
    }
    _put_dynamodb_item(doc_metadata)

    return {
        "bucket": bucket,
        "key": key,
        "etag": etag,
        "nodes_added": nodes_added,
        "processing_seconds": round(elapsed, 2),
    }


def _put_dynamodb_item(item: dict) -> None:
    """Write a document metadata record to the DynamoDB table.

    Non-fatal — if DynamoDB is unavailable, we log a warning but don't
    fail the whole indexing operation. The document is still indexed in
    the vector store even if metadata storage fails.
    """
    table_name = os.environ.get("DYNAMODB_TABLE_NAME", "finrag-document-metadata")
    try:
        ddb = _get_dynamodb_client()
        ddb.put_item(TableName=table_name, Item=item)
        logger.info("Wrote metadata to DynamoDB table %s", table_name)
    except Exception as exc:
        # Non-fatal: warn but continue — don't fail the Lambda just for metadata
        logger.warning("DynamoDB write failed (non-fatal): %s", exc)


def _extract_s3_records(event: dict) -> list[dict]:
    """Parse various Lambda event formats into a flat list of {bucket, key} dicts.

    Lambda can be triggered from different sources with different event shapes:

    1. Direct S3 event notification (S3 → Lambda directly):
       {"Records": [{"s3": {"bucket": {"name": "..."}, "object": {"key": "..."}}}]}

    2. SQS-wrapped S3 notification (S3 → SQS → Lambda):
       {"Records": [{"body": '{"Records": [{"s3": {...}}]}'}]}

    3. Direct invocation (e.g. for testing from CLI or Step Functions):
       {"bucket": "my-bucket", "key": "documents/report.pdf"}

    This function handles all three and returns a normalised list of
    {"bucket": "...", "key": "..."} dicts for only .pdf files.
    """
    records: list[dict] = []

    for record in event.get("Records", []):
        if "s3" in record:
            # Format 1: Direct S3 event notification
            bucket = record["s3"]["bucket"]["name"]
            # unquote_plus decodes URL-encoded characters in S3 keys
            # (spaces become %20, etc. — this converts them back)
            key = unquote_plus(record["s3"]["object"]["key"])
            if key.lower().endswith(".pdf"):
                records.append({"bucket": bucket, "key": key})

        elif "body" in record:
            # Format 2: SQS-wrapped S3 notification — the S3 event is JSON inside "body"
            try:
                body = json.loads(record["body"])  # parse the inner JSON string
                for inner in body.get("Records", []):
                    if "s3" in inner:
                        bucket = inner["s3"]["bucket"]["name"]
                        key = unquote_plus(inner["s3"]["object"]["key"])
                        if key.lower().endswith(".pdf"):
                            records.append({"bucket": bucket, "key": key})
            except (json.JSONDecodeError, KeyError):
                # Skip malformed SQS messages — don't fail the whole batch
                pass

    # Format 3: Direct invocation with bucket + key at the top level
    if not records and "bucket" in event and "key" in event:
        key = event["key"]
        if key.lower().endswith(".pdf"):
            records.append({"bucket": event["bucket"], "key": key})

    return records


def _make_doc_id(key: str) -> str:
    """Generate a deterministic document ID from the S3 key.

    Used as a fallback when ETag is not available.
    SHA-256 of the key → first 16 hex characters.
    Same key always produces the same ID — useful for deduplication.
    """
    import hashlib
    return hashlib.sha256(key.encode()).hexdigest()[:16]
