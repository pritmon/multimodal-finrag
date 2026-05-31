"""S3 document loader: upload, list, download PDFs and generate presigned URLs.

HOW IT WORKS (simple analogy):
  Think of S3 as a cloud file server — like UiPath Storage Buckets.
  This class is a helper that wraps the AWS SDK (boto3) with clean methods
  for the specific operations this project needs:

  - Upload a PDF (from bytes or from disk)
  - List all PDFs in the bucket
  - Download a PDF back to bytes or to a local file
  - Generate presigned URLs (temporary links for browser-based downloads/uploads)

  All operations use the boto3 S3 client under the hood.
  boto3 is the official AWS Python SDK — like using the UiPath AWS connector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urljoin

import boto3
from botocore.exceptions import ClientError  # AWS SDK error type

logger = logging.getLogger(__name__)


@dataclass
class S3Document:
    """Metadata for a PDF document stored in S3.

    Like a DataRow that describes one file in the bucket — its name,
    size, when it was last modified, and a checksum (etag).
    Does NOT contain the actual file content (download separately).
    """

    key: str             # full S3 path, e.g. "documents/report.pdf"
    bucket: str          # S3 bucket name, e.g. "pritam-finrag-docs"
    size_bytes: int      # file size in bytes
    last_modified: datetime  # when the file was last uploaded/changed
    etag: str            # MD5 checksum of the file (AWS's version of a hash)
    content_type: str = "application/pdf"  # MIME type (default: PDF)
    custom_metadata: dict = None  # type: ignore[assignment]  # any custom key-value metadata

    def __post_init__(self) -> None:
        # Ensure custom_metadata is always a dict, never None
        if self.custom_metadata is None:
            self.custom_metadata = {}

    @property
    def filename(self) -> str:
        """Extract just the filename from the full S3 key path.

        Example: "documents/job123/report.pdf" → "report.pdf"
        Same as Path(key).name in Python.
        """
        return self.key.split("/")[-1]


class S3Loader:
    """High-level S3 client for the FinRAG document store.

    Wraps boto3 with project-specific methods.
    All methods operate on the configured bucket and prefix by default.

    Parameters
    ----------
    bucket:
        Default S3 bucket name (e.g. "pritam-finrag-docs").
    prefix:
        Key prefix — like a "folder" in S3, e.g. "documents/".
        All documents are stored under this prefix.
    region:
        AWS region where the bucket lives (e.g. "us-east-1").
    session_kwargs:
        Extra kwargs forwarded to boto3.Session — used for passing
        AWS credentials when not using IAM roles.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "documents/",
        region: str = "us-east-1",
        **session_kwargs,
    ) -> None:
        self.bucket = bucket
        # Ensure prefix always ends with "/" (S3 "folders" are just key prefixes)
        self.prefix = prefix.rstrip("/") + "/"
        # Create a boto3 session (handles auth) and an S3 client (handles API calls)
        session = boto3.Session(**session_kwargs)
        self._s3 = session.client("s3", region_name=region)
        self._region = region

    # ── Upload Methods ─────────────────────────────────────────────────────────

    def upload_pdf(
        self,
        pdf_bytes: bytes,
        filename: str,
        metadata: Optional[dict[str, str]] = None,
        content_type: str = "application/pdf",
    ) -> str:
        """Upload raw PDF bytes directly to S3.

        Use this when you have the PDF in memory (e.g. from a web upload).
        The S3 key will be: <prefix>/<filename>
        e.g. "documents/annual_report_2024.pdf"

        Custom metadata is stored as S3 object tags — key-value pairs
        that travel with the file (like UiPath Storage Bucket metadata).

        Returns the full S3 key of the uploaded file.
        """
        key = f"{self.prefix}{filename}"

        # Build ExtraArgs for the S3 put — content type + optional metadata
        extra_args: dict = {"ContentType": content_type}
        if metadata:
            # S3 metadata values must all be strings (boto3 requirement)
            extra_args["Metadata"] = {k: str(v) for k, v in metadata.items()}

        logger.info("Uploading %d bytes to s3://%s/%s", len(pdf_bytes), self.bucket, key)
        # put_object sends bytes directly to S3 (no temp file needed)
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=pdf_bytes,
            **extra_args,
        )
        logger.info("Upload complete: s3://%s/%s", self.bucket, key)
        return key

    def upload_file(self, local_path: str | Path, s3_key: Optional[str] = None) -> str:
        """Upload a file from disk to S3.

        Use this when the PDF is already saved as a local file.
        If s3_key is not specified, uses "<prefix>/<filename>".

        boto3's upload_file() is more efficient than reading bytes manually
        — it handles multipart uploads for large files automatically.

        Returns the full S3 key of the uploaded file.
        """
        local_path = Path(local_path)
        key = s3_key or f"{self.prefix}{local_path.name}"
        logger.info("Uploading file %s → s3://%s/%s", local_path, self.bucket, key)
        self._s3.upload_file(str(local_path), self.bucket, key)
        return key

    # ── List Methods ──────────────────────────────────────────────────────────

    def list_documents(self, sub_prefix: str = "") -> list[S3Document]:
        """Return metadata for all PDFs stored under the configured prefix.

        Uses S3 pagination — S3 returns results in pages of 1000 objects max.
        The paginator handles multiple pages automatically (like looping in UiPath).

        Only returns objects whose keys end in ".pdf" (case-insensitive filter).

        sub_prefix: optional extra folder within the main prefix.
        """
        full_prefix = f"{self.prefix}{sub_prefix}"
        # Paginator automatically handles multiple pages of S3 results
        paginator = self._s3.get_paginator("list_objects_v2")
        documents: list[S3Document] = []

        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            # "Contents" is the list of objects in this page of results
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if not key.lower().endswith(".pdf"):
                    continue  # skip non-PDF files (index files, logs, etc.)
                documents.append(
                    S3Document(
                        key=key,
                        bucket=self.bucket,
                        size_bytes=obj["Size"],
                        last_modified=obj["LastModified"],
                        etag=obj.get("ETag", "").strip('"'),  # AWS wraps ETag in quotes
                    )
                )

        logger.info("Found %d PDFs under s3://%s/%s", len(documents), self.bucket, full_prefix)
        return documents

    # ── Download Methods ──────────────────────────────────────────────────────

    def download_pdf(self, key: str) -> bytes:
        """Download an S3 object and return its content as raw bytes.

        Use this when you need the PDF content in memory
        (e.g. to pass directly to PDFParser.parse_bytes()).
        """
        logger.info("Downloading s3://%s/%s", self.bucket, key)
        response = self._s3.get_object(Bucket=self.bucket, Key=key)
        # response["Body"] is a streaming body — .read() gets all bytes at once
        data: bytes = response["Body"].read()
        logger.info("Downloaded %d bytes from s3://%s/%s", len(data), self.bucket, key)
        return data

    def download_to_file(self, key: str, local_path: str | Path) -> Path:
        """Download an S3 object and save it to a local file.

        Creates parent directories if they don't exist.
        Use this when you need to save a PDF to disk first.
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading s3://%s/%s → %s", self.bucket, key, local_path)
        self._s3.download_file(self.bucket, key, str(local_path))
        return local_path

    def iter_documents(self, sub_prefix: str = "") -> Iterator[tuple[S3Document, bytes]]:
        """Yield (S3Document, pdf_bytes) for each PDF in the prefix.

        A generator — downloads one PDF at a time to avoid loading everything
        into memory at once. Like a UiPath For Each that processes files lazily.

        Usage:
            for doc, pdf_bytes in loader.iter_documents():
                parsed = parser.parse_bytes(pdf_bytes, source=doc.key)
        """
        for doc in self.list_documents(sub_prefix):
            data = self.download_pdf(doc.key)
            yield doc, data  # caller gets one (metadata, bytes) pair at a time

    # ── Presigned URL Methods ─────────────────────────────────────────────────

    def generate_presigned_url(
        self, key: str, expiration_seconds: int = 3600, method: str = "get_object"
    ) -> str:
        """Generate a time-limited URL for direct S3 access without AWS credentials.

        Presigned URLs are like temporary access passes — the URL itself proves
        the holder has permission to download/upload for the specified time window.
        Commonly used so browsers can download files directly from S3
        without going through your API server.

        method:
            "get_object"  → a download URL (the default)
            "put_object"  → an upload URL
        expiration_seconds:
            How long the URL is valid (default: 1 hour = 3600 seconds)

        Raises ClientError if the URL cannot be generated (e.g. invalid key).
        """
        try:
            url: str = self._s3.generate_presigned_url(
                ClientMethod=method,
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expiration_seconds,
            )
            logger.debug("Generated presigned URL for %s (expires %ds)", key, expiration_seconds)
            return url
        except ClientError as exc:
            logger.error("Failed to generate presigned URL for %s: %s", key, exc)
            raise  # re-raise so caller can handle the error

    def generate_presigned_post(
        self, key: str, expiration_seconds: int = 3600, max_content_length: int = 100_000_000
    ) -> dict:
        """Generate a presigned POST form for browser-based file uploads.

        Instead of routing uploads through your API server, users can upload
        directly from a browser to S3 using an HTML form.

        Returns a dict with:
          - "url": the S3 endpoint to POST to
          - "fields": form fields that must be included in the multipart POST

        max_content_length: maximum allowed file size in bytes (default: 100MB)
        """
        # Conditions restrict what the uploader is allowed to send
        conditions = [
            ["content-length-range", 1, max_content_length],  # file must be 1 byte to 100MB
            {"Content-Type": "application/pdf"},               # must be a PDF
        ]
        result: dict = self._s3.generate_presigned_post(
            Bucket=self.bucket,
            Key=key,
            Conditions=conditions,
            ExpiresIn=expiration_seconds,
        )
        return result

    # ── Metadata / Utility Methods ────────────────────────────────────────────

    def get_object_metadata(self, key: str) -> dict:
        """Return S3 object metadata WITHOUT downloading the file body.

        Uses HTTP HEAD request — gets size, timestamps, ETag, etc.
        Fast because no file content is transferred.
        Useful for checking if a file exists and getting its properties.
        """
        response = self._s3.head_object(Bucket=self.bucket, Key=key)
        return {
            "key": key,
            "bucket": self.bucket,
            "size_bytes": response.get("ContentLength", 0),
            "last_modified": response.get("LastModified"),
            "etag": response.get("ETag", "").strip('"'),
            "content_type": response.get("ContentType", ""),
            "metadata": response.get("Metadata", {}),
        }

    def object_exists(self, key: str) -> bool:
        """Check whether an S3 object exists without downloading it.

        Uses head_object (HTTP HEAD) — much faster than downloading.
        Returns False for 404 errors; re-raises any other AWS errors.

        Like UiPath's File.Exists() but for S3.
        """
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True  # no exception = object exists
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                return False  # 404 = not found = doesn't exist
            raise  # any other error (403 access denied, etc.) is re-raised

    def delete_object(self, key: str) -> None:
        """Permanently delete an object from S3.

        WARNING: This is irreversible unless versioning is enabled on the bucket.
        """
        logger.info("Deleting s3://%s/%s", self.bucket, key)
        self._s3.delete_object(Bucket=self.bucket, Key=key)
