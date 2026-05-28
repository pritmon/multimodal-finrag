"""S3 document loader: upload, list, download PDFs and generate presigned URLs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urljoin

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass
class S3Document:
    """Metadata for a PDF document stored in S3."""

    key: str
    bucket: str
    size_bytes: int
    last_modified: datetime
    etag: str
    content_type: str = "application/pdf"
    custom_metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.custom_metadata is None:
            self.custom_metadata = {}

    @property
    def filename(self) -> str:
        return self.key.split("/")[-1]


class S3Loader:
    """High-level S3 client for the FinRAG document store.

    Parameters
    ----------
    bucket:
        Default S3 bucket name.
    prefix:
        Key prefix (folder) within the bucket.
    region:
        AWS region.
    session_kwargs:
        Extra kwargs forwarded to ``boto3.Session`` (credentials, etc.).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "documents/",
        region: str = "us-east-1",
        **session_kwargs,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        session = boto3.Session(**session_kwargs)
        self._s3 = session.client("s3", region_name=region)
        self._region = region

    # ── Upload ────────────────────────────────────────────────────────────────

    def upload_pdf(
        self,
        pdf_bytes: bytes,
        filename: str,
        metadata: Optional[dict[str, str]] = None,
        content_type: str = "application/pdf",
    ) -> str:
        """Upload PDF bytes to S3 and return the S3 key.

        Parameters
        ----------
        pdf_bytes:
            Raw PDF content.
        filename:
            Destination filename (appended to ``self.prefix``).
        metadata:
            Custom S3 object metadata (values must be strings).

        Returns
        -------
        str
            The full S3 key of the uploaded object.
        """
        key = f"{self.prefix}{filename}"
        extra_args: dict = {"ContentType": content_type}
        if metadata:
            # S3 metadata keys must be ASCII strings
            extra_args["Metadata"] = {k: str(v) for k, v in metadata.items()}

        logger.info("Uploading %d bytes to s3://%s/%s", len(pdf_bytes), self.bucket, key)
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=pdf_bytes,
            **extra_args,
        )
        logger.info("Upload complete: s3://%s/%s", self.bucket, key)
        return key

    def upload_file(self, local_path: str | Path, s3_key: Optional[str] = None) -> str:
        """Upload a local file to S3.

        Parameters
        ----------
        local_path:
            Path to the local file.
        s3_key:
            Override the S3 key; defaults to ``prefix + filename``.

        Returns
        -------
        str
            The full S3 key.
        """
        local_path = Path(local_path)
        key = s3_key or f"{self.prefix}{local_path.name}"
        logger.info("Uploading file %s → s3://%s/%s", local_path, self.bucket, key)
        self._s3.upload_file(str(local_path), self.bucket, key)
        return key

    # ── List ──────────────────────────────────────────────────────────────────

    def list_documents(self, sub_prefix: str = "") -> list[S3Document]:
        """Return metadata for all PDFs under ``prefix + sub_prefix``."""
        full_prefix = f"{self.prefix}{sub_prefix}"
        paginator = self._s3.get_paginator("list_objects_v2")
        documents: list[S3Document] = []

        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if not key.lower().endswith(".pdf"):
                    continue
                documents.append(
                    S3Document(
                        key=key,
                        bucket=self.bucket,
                        size_bytes=obj["Size"],
                        last_modified=obj["LastModified"],
                        etag=obj.get("ETag", "").strip('"'),
                    )
                )

        logger.info("Found %d PDFs under s3://%s/%s", len(documents), self.bucket, full_prefix)
        return documents

    # ── Download ──────────────────────────────────────────────────────────────

    def download_pdf(self, key: str) -> bytes:
        """Download an S3 object and return its raw bytes."""
        logger.info("Downloading s3://%s/%s", self.bucket, key)
        response = self._s3.get_object(Bucket=self.bucket, Key=key)
        data: bytes = response["Body"].read()
        logger.info("Downloaded %d bytes from s3://%s/%s", len(data), self.bucket, key)
        return data

    def download_to_file(self, key: str, local_path: str | Path) -> Path:
        """Download an S3 object to a local file."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading s3://%s/%s → %s", self.bucket, key, local_path)
        self._s3.download_file(self.bucket, key, str(local_path))
        return local_path

    def iter_documents(self, sub_prefix: str = "") -> Iterator[tuple[S3Document, bytes]]:
        """Yield (S3Document, pdf_bytes) for each PDF in the prefix."""
        for doc in self.list_documents(sub_prefix):
            data = self.download_pdf(doc.key)
            yield doc, data

    # ── Presigned URLs ────────────────────────────────────────────────────────

    def generate_presigned_url(
        self, key: str, expiration_seconds: int = 3600, method: str = "get_object"
    ) -> str:
        """Generate a presigned URL for GET or PUT access.

        Parameters
        ----------
        key:
            S3 object key.
        expiration_seconds:
            URL validity window (default: 1 hour).
        method:
            ``"get_object"`` or ``"put_object"``.

        Returns
        -------
        str
            The presigned URL.
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
            raise

    def generate_presigned_post(
        self, key: str, expiration_seconds: int = 3600, max_content_length: int = 100_000_000
    ) -> dict:
        """Generate a presigned POST form for browser-based uploads.

        Returns a dict with ``url`` and ``fields`` keys suitable for an
        HTML form or ``requests.post(url, data=fields, files=...)``.
        """
        conditions = [
            ["content-length-range", 1, max_content_length],
            {"Content-Type": "application/pdf"},
        ]
        result: dict = self._s3.generate_presigned_post(
            Bucket=self.bucket,
            Key=key,
            Conditions=conditions,
            ExpiresIn=expiration_seconds,
        )
        return result

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_object_metadata(self, key: str) -> dict:
        """Return S3 object metadata without downloading the body."""
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
        """Return True if the S3 key exists."""
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                return False
            raise

    def delete_object(self, key: str) -> None:
        """Delete an object from S3."""
        logger.info("Deleting s3://%s/%s", self.bucket, key)
        self._s3.delete_object(Bucket=self.bucket, Key=key)
