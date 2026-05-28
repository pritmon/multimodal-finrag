"""AWS Lambda handler for asynchronous financial document processing.

Triggered by S3 PutObject events (via EventBridge or S3 notification).
For each new PDF:
1. Download the PDF from S3.
2. Parse text + extract charts.
3. Index all nodes into the LlamaIndex vector store.
4. Persist document metadata to DynamoDB.
5. Update the persisted index on S3 (optional: write index back).

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
from urllib.parse import unquote_plus

import boto3

# Configure structured logging early (Lambda captures stdout)
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=os.environ.get("LOG_LEVEL", "INFO"),
)
logger = logging.getLogger(__name__)


def _get_dynamodb_client():
    return boto3.client("dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def _get_s3_client():
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def _lazy_pipeline():
    """Import and initialise the RAG pipeline lazily (warm-start caching)."""
    from src.config import get_settings
    from src.rag.pipeline import FinRAGPipeline

    settings = get_settings()
    pipeline = FinRAGPipeline(settings=settings)
    pipeline.load_or_build_index(force_rebuild=False)
    return pipeline


# Module-level singleton: Lambda reuses this across warm invocations
_PIPELINE = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        logger.info("Cold start: initialising RAG pipeline")
        _PIPELINE = _lazy_pipeline()
    return _PIPELINE


# ── Lambda handler ────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    """Main Lambda entry point.

    Accepts S3 PutObject event notifications (single or batched via SQS).

    Returns a summary dict with processing results.
    """
    logger.info("Received event: %s", json.dumps(event))
    records = _extract_s3_records(event)

    if not records:
        logger.warning("No S3 records found in event; skipping")
        return {"statusCode": 200, "processed": 0}

    processed = 0
    errors: list[dict] = []

    for record in records:
        bucket = record["bucket"]
        key = record["key"]
        try:
            result = _process_document(bucket, key)
            logger.info("Processed s3://%s/%s → %d nodes", bucket, key, result["nodes_added"])
            processed += 1
        except Exception as exc:
            logger.exception("Failed to process s3://%s/%s: %s", bucket, key, exc)
            errors.append({"bucket": bucket, "key": key, "error": str(exc)})

    response = {
        "statusCode": 200 if not errors else 207,
        "processed": processed,
        "errors": errors,
    }
    logger.info("Lambda complete: %s", response)
    return response


def _process_document(bucket: str, key: str) -> dict:
    """Download, parse, index a single PDF document and record metadata."""
    s3 = _get_s3_client()

    # ── Download PDF ──────────────────────────────────────────────────────────
    logger.info("Downloading s3://%s/%s", bucket, key)
    start_ts = time.time()
    response = s3.get_object(Bucket=bucket, Key=key)
    pdf_bytes = response["Body"].read()
    s3_metadata = response.get("Metadata", {})
    etag = response.get("ETag", "").strip('"')
    content_length = len(pdf_bytes)
    logger.info("Downloaded %d bytes (ETag=%s)", content_length, etag)

    # ── Build/index document ──────────────────────────────────────────────────
    pipeline = _get_pipeline()
    filename = key.split("/")[-1]
    nodes_added = pipeline.add_pdf_bytes(
        pdf_bytes,
        source=f"s3://{bucket}/{key}",
        generate_chart_captions=True,
    )

    elapsed = time.time() - start_ts

    # ── Store metadata in DynamoDB ────────────────────────────────────────────
    doc_metadata = {
        "document_id": {"S": etag or _make_doc_id(key)},
        "s3_key": {"S": key},
        "s3_bucket": {"S": bucket},
        "filename": {"S": filename},
        "size_bytes": {"N": str(content_length)},
        "nodes_added": {"N": str(nodes_added)},
        "status": {"S": "indexed"},
        "indexed_at": {"S": datetime.now(timezone.utc).isoformat()},
        "processing_seconds": {"N": str(round(elapsed, 2))},
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
    """Write a document metadata record to DynamoDB."""
    table_name = os.environ.get("DYNAMODB_TABLE_NAME", "finrag-document-metadata")
    try:
        ddb = _get_dynamodb_client()
        ddb.put_item(TableName=table_name, Item=item)
        logger.info("Wrote metadata to DynamoDB table %s", table_name)
    except Exception as exc:
        logger.warning("DynamoDB write failed (non-fatal): %s", exc)


def _extract_s3_records(event: dict) -> list[dict]:
    """Parse various event formats into a flat list of {bucket, key} dicts."""
    records: list[dict] = []

    # Direct S3 event notification
    for record in event.get("Records", []):
        if "s3" in record:
            bucket = record["s3"]["bucket"]["name"]
            key = unquote_plus(record["s3"]["object"]["key"])
            if key.lower().endswith(".pdf"):
                records.append({"bucket": bucket, "key": key})
        elif "body" in record:
            # SQS-wrapped S3 notification
            try:
                body = json.loads(record["body"])
                for inner in body.get("Records", []):
                    if "s3" in inner:
                        bucket = inner["s3"]["bucket"]["name"]
                        key = unquote_plus(inner["s3"]["object"]["key"])
                        if key.lower().endswith(".pdf"):
                            records.append({"bucket": bucket, "key": key})
            except (json.JSONDecodeError, KeyError):
                pass

    # Direct invocation format: {"bucket": "...", "key": "..."}
    if not records and "bucket" in event and "key" in event:
        key = event["key"]
        if key.lower().endswith(".pdf"):
            records.append({"bucket": event["bucket"], "key": key})

    return records


def _make_doc_id(key: str) -> str:
    import hashlib
    return hashlib.sha256(key.encode()).hexdigest()[:16]
