#!/usr/bin/env python3
"""CLI: Package and deploy the Lambda document processor.

Supports two deployment strategies:
1. Container image (recommended for large ML dependencies):
   Build a Docker image, push to ECR, update the Lambda function.

2. ZIP deployment (lightweight, no ML deps — suitable for simple Lambda):
   Create a deployment ZIP, upload to S3, update the Lambda function.

Usage examples:
    # Container image deployment
    python scripts/deploy_lambda.py container \\
        --ecr-repo finrag-lambda \\
        --function-name finrag-document-processor \\
        --region us-east-1

    # ZIP deployment
    python scripts/deploy_lambda.py zip \\
        --function-name finrag-document-processor \\
        --s3-bucket my-lambda-packages \\
        --region us-east-1
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("deploy_lambda")

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Sub-commands ──────────────────────────────────────────────────────────────

def deploy_container(args: argparse.Namespace) -> int:
    """Build Docker image → push to ECR → update Lambda."""
    region = args.region
    account_id = _get_aws_account_id(region)
    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{args.ecr_repo}"
    image_tag = args.image_tag or f"build-{int(time.time())}"
    full_image = f"{ecr_uri}:{image_tag}"

    logger.info("=== Container Image Deployment ===")
    logger.info("ECR URI:   %s", full_image)
    logger.info("Function:  %s", args.function_name)

    # Authenticate Docker with ECR
    logger.info("Authenticating Docker with ECR...")
    _run(
        f"aws ecr get-login-password --region {region} | "
        f"docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com",
        shell=True,
    )

    # Build image
    dockerfile = _REPO_ROOT / "src" / "lambda_handler" / "Dockerfile"
    logger.info("Building Docker image...")
    _run([
        "docker", "build",
        "-t", full_image,
        "-f", str(dockerfile),
        str(_REPO_ROOT),
    ])

    # Ensure ECR repository exists
    _ensure_ecr_repo(args.ecr_repo, region)

    # Push image
    logger.info("Pushing image to ECR...")
    _run(["docker", "push", full_image])
    logger.info("Pushed: %s", full_image)

    # Update Lambda function
    logger.info("Updating Lambda function %s...", args.function_name)
    lam = boto3.client("lambda", region_name=region)
    response = lam.update_function_code(
        FunctionName=args.function_name,
        ImageUri=full_image,
        Publish=True,
    )
    version = response.get("Version", "?")
    logger.info("Lambda updated to version %s", version)

    # Optionally update configuration
    if args.memory_mb or args.timeout_secs:
        update_kwargs: dict = {"FunctionName": args.function_name}
        if args.memory_mb:
            update_kwargs["MemorySize"] = args.memory_mb
        if args.timeout_secs:
            update_kwargs["Timeout"] = args.timeout_secs
        lam.update_function_configuration(**update_kwargs)
        logger.info("Updated Lambda configuration")

    _wait_for_lambda_active(lam, args.function_name)
    logger.info("=== Deployment complete ===")
    return 0


def deploy_zip(args: argparse.Namespace) -> int:
    """Create a ZIP package → upload to S3 → update Lambda."""
    region = args.region
    logger.info("=== ZIP Deployment ===")
    logger.info("Function: %s", args.function_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        pkg_dir = tmpdir_path / "package"
        pkg_dir.mkdir()

        # Copy source code
        src_dst = pkg_dir / "src"
        shutil.copytree(_REPO_ROOT / "src", src_dst)

        # Install dependencies into package
        logger.info("Installing dependencies into package directory...")
        _run([
            sys.executable, "-m", "pip", "install",
            "--target", str(pkg_dir),
            "--no-deps",  # assumes deps already baked in Lambda layer
            "boto3", "PyMuPDF", "Pillow", "rank_bm25",
            "pydantic-settings", "tenacity",
        ])

        # Create ZIP
        zip_path = tmpdir_path / "finrag-lambda.zip"
        logger.info("Creating ZIP archive...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in pkg_dir.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(pkg_dir))

        zip_size_mb = zip_path.stat().st_size / 1_000_000
        logger.info("ZIP size: %.1f MB", zip_size_mb)

        # Upload to S3
        s3_key = f"lambda-packages/finrag-lambda-{int(time.time())}.zip"
        logger.info("Uploading ZIP to s3://%s/%s...", args.s3_bucket, s3_key)
        s3 = boto3.client("s3", region_name=region)
        s3.upload_file(str(zip_path), args.s3_bucket, s3_key)

        # Update Lambda
        logger.info("Updating Lambda function %s...", args.function_name)
        lam = boto3.client("lambda", region_name=region)
        lam.update_function_code(
            FunctionName=args.function_name,
            S3Bucket=args.s3_bucket,
            S3Key=s3_key,
            Publish=True,
        )

    _wait_for_lambda_active(lam, args.function_name)
    logger.info("=== Deployment complete ===")
    return 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd, shell: bool = False) -> None:
    result = subprocess.run(cmd, shell=shell, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def _get_aws_account_id(region: str) -> str:
    sts = boto3.client("sts", region_name=region)
    return sts.get_caller_identity()["Account"]


def _ensure_ecr_repo(repo_name: str, region: str) -> None:
    ecr = boto3.client("ecr", region_name=region)
    try:
        ecr.describe_repositories(repositoryNames=[repo_name])
        logger.info("ECR repository %s already exists", repo_name)
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "RepositoryNotFoundException":
            logger.info("Creating ECR repository %s...", repo_name)
            ecr.create_repository(
                repositoryName=repo_name,
                imageScanningConfiguration={"scanOnPush": True},
                imageTagMutability="MUTABLE",
            )
        else:
            raise


def _wait_for_lambda_active(lam, function_name: str, timeout: int = 120) -> None:
    """Poll until the Lambda function state is Active."""
    logger.info("Waiting for Lambda function to become active...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = lam.get_function(FunctionName=function_name)
        state = resp["Configuration"].get("State", "Unknown")
        last_update = resp["Configuration"].get("LastUpdateStatus", "Unknown")
        logger.debug("State: %s, LastUpdateStatus: %s", state, last_update)
        if state == "Active" and last_update in ("Successful", "Unknown"):
            logger.info("Lambda is Active")
            return
        time.sleep(5)
    logger.warning("Lambda did not become Active within %ds — check AWS console", timeout)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy the FinRAG Lambda document processor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--function-name", default="finrag-document-processor")
    parser.add_argument("--memory-mb", type=int, default=None, help="Lambda memory (MB)")
    parser.add_argument("--timeout-secs", type=int, default=None, help="Lambda timeout (seconds)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="strategy", required=True)

    # Container image
    container_parser = subparsers.add_parser("container", help="Deploy via ECR container image")
    container_parser.add_argument("--ecr-repo", default="finrag-lambda")
    container_parser.add_argument("--image-tag", default=None, help="Docker image tag (default: build-<timestamp>)")

    # ZIP
    zip_parser = subparsers.add_parser("zip", help="Deploy via S3 ZIP package")
    zip_parser.add_argument("--s3-bucket", required=True, help="S3 bucket for the ZIP package")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    if args.strategy == "container":
        return deploy_container(args)
    elif args.strategy == "zip":
        return deploy_zip(args)
    else:
        logger.error("Unknown deployment strategy: %s", args.strategy)
        return 1


if __name__ == "__main__":
    sys.exit(main())
