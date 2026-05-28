#!/usr/bin/env python3
"""CLI: Download PDFs from S3, parse them, and build/update the vector index.

Usage examples:
    python scripts/build_index.py --s3-prefix documents/2024/ --output-dir ./index_store
    python scripts/build_index.py --local-dir /data/pdfs --output-dir ./index_store
    python scripts/build_index.py --s3-prefix documents/ --force-rebuild
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make sure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.s3_loader import S3Loader
from src.rag.pipeline import FinRAGPipeline

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("build_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or update the FinRAG vector index from PDFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--s3-prefix",
        default=None,
        help="S3 key prefix to download PDFs from (uses S3_BUCKET from config)",
    )
    source_group.add_argument(
        "--local-dir",
        default=None,
        help="Local directory containing PDF files to index",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to persist the index (overrides INDEX_PERSIST_DIR in config)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        default=False,
        help="Delete any existing index and rebuild from scratch",
    )
    parser.add_argument(
        "--no-chart-captions",
        action="store_true",
        default=False,
        help="Skip Bedrock chart captioning (faster, no vision API calls)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    settings = get_settings()

    # Override output dir if provided
    if args.output_dir:
        settings.index_persist_dir = Path(args.output_dir)
        settings.index_persist_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== FinRAG Index Builder ===")
    logger.info("Output dir: %s", settings.index_persist_dir)
    logger.info("Force rebuild: %s", args.force_rebuild)

    # Initialise pipeline
    pipeline = FinRAGPipeline(settings=settings)
    pipeline.load_or_build_index(force_rebuild=args.force_rebuild)

    generate_captions = not args.no_chart_captions
    total_docs = 0
    total_nodes = 0
    errors: list[str] = []

    t0 = time.time()

    # ── Source: S3 ────────────────────────────────────────────────────────────
    if args.s3_prefix is not None:
        loader = S3Loader(
            bucket=settings.s3_bucket,
            prefix=args.s3_prefix,
            **settings.boto3_session_kwargs,
        )
        docs = loader.list_documents()
        logger.info("Found %d PDFs in s3://%s/%s", len(docs), settings.s3_bucket, args.s3_prefix)

        if args.max_docs:
            docs = docs[: args.max_docs]
            logger.info("Processing first %d documents", len(docs))

        for doc_meta in docs:
            try:
                logger.info("[%d/%d] Processing %s", total_docs + 1, len(docs), doc_meta.key)
                pdf_bytes = loader.download_pdf(doc_meta.key)
                nodes = pipeline.add_pdf_bytes(
                    pdf_bytes,
                    source=f"s3://{doc_meta.bucket}/{doc_meta.key}",
                    generate_chart_captions=generate_captions,
                )
                total_nodes += nodes
                total_docs += 1
            except Exception as exc:
                logger.error("Failed to process %s: %s", doc_meta.key, exc)
                errors.append(f"{doc_meta.key}: {exc}")

    # ── Source: local directory ───────────────────────────────────────────────
    elif args.local_dir is not None:
        local_dir = Path(args.local_dir)
        if not local_dir.is_dir():
            logger.error("Local directory does not exist: %s", local_dir)
            return 1

        pdf_files = sorted(local_dir.rglob("*.pdf"))
        logger.info("Found %d PDFs in %s", len(pdf_files), local_dir)

        if args.max_docs:
            pdf_files = pdf_files[: args.max_docs]

        for pdf_path in pdf_files:
            try:
                logger.info("[%d/%d] Processing %s", total_docs + 1, len(pdf_files), pdf_path)
                nodes = pipeline.add_pdf_file(
                    pdf_path,
                    generate_chart_captions=generate_captions,
                )
                total_nodes += nodes
                total_docs += 1
            except Exception as exc:
                logger.error("Failed to process %s: %s", pdf_path, exc)
                errors.append(f"{pdf_path}: {exc}")

    else:
        logger.error("Specify either --s3-prefix or --local-dir")
        return 1

    elapsed = time.time() - t0
    logger.info("=== Build Complete ===")
    logger.info("Documents processed: %d", total_docs)
    logger.info("Nodes indexed:       %d", total_nodes)
    logger.info("Errors:              %d", len(errors))
    logger.info("Time elapsed:        %.1fs", elapsed)

    if errors:
        logger.warning("Failed documents:")
        for err in errors:
            logger.warning("  %s", err)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
