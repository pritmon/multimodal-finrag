"""Application configuration using Pydantic BaseSettings.

HOW IT WORKS (simple analogy):
  Think of this like UiPath's Config.xlsx or Orchestrator Assets.
  All configuration values (AWS region, model IDs, API settings, etc.)
  are defined here with default values.

  Any value can be overridden by:
    1. Environment variable (e.g. export AWS_REGION=us-west-2)
    2. A .env file (loaded automatically from the project root)

  Pydantic BaseSettings reads environment variables automatically —
  no manual os.environ.get() calls needed anywhere in the code.

  The @lru_cache decorator on get_settings() ensures only ONE Settings
  instance is ever created — like a singleton in UiPath.

  Example .env file:
    AWS_REGION=us-east-1
    S3_BUCKET=my-bucket
    BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
"""

from __future__ import annotations

from functools import lru_cache  # for singleton caching
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the FinRAG system.

    All values can be overridden via environment variables or a .env file.
    Environment variables are case-insensitive (AWS_REGION = aws_region).

    Grouped by concern: AWS, S3, Bedrock, API, RAG, etc.
    """

    # Pydantic settings config — where to read values from
    model_config = SettingsConfigDict(
        env_file=".env",           # read from .env file in the project root
        env_file_encoding="utf-8",
        case_sensitive=False,      # AWS_REGION and aws_region are treated the same
        extra="ignore",            # silently ignore unknown environment variables
    )

    # ── AWS Core ──────────────────────────────────────────────────────────────
    # When running on ECS/EKS with an IAM role, access_key and secret are NOT needed
    # AWS SDK automatically uses the instance role in that case
    aws_region: str = Field(default="us-east-1", description="AWS region for all services")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key (omit for IAM role)")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret key")
    aws_session_token: Optional[str] = Field(default=None, description="AWS session token for temporary credentials")

    # ── S3 ────────────────────────────────────────────────────────────────────
    s3_bucket: str = Field(default="finrag-documents", description="S3 bucket for document storage")
    s3_prefix: str = Field(default="documents/", description="S3 key prefix (folder) for documents")

    # ── Bedrock Models ────────────────────────────────────────────────────────
    # The actual running project uses "amazon.nova-lite-v1:0" (set in .env)
    # These defaults are fallbacks if no .env is present
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        description="Bedrock model ID for generation",
    )
    bedrock_embed_model_id: str = Field(
        default="amazon.titan-embed-text-v1",
        description="Bedrock model ID for embeddings (not used — local embeddings are used instead)",
    )
    bedrock_max_tokens: int = Field(default=4096, description="Max tokens for Bedrock generation")
    bedrock_temperature: float = Field(
        default=0.1,
        description="Temperature for generation (0.0=deterministic, 1.0=creative)",
        ge=0.0,   # must be >= 0.0
        le=1.0,   # must be <= 1.0
    )

    # ── Lambda ────────────────────────────────────────────────────────────────
    lambda_function_name: str = Field(
        default="finrag-document-processor",
        description="Lambda function name for async document processing",
    )

    # ── DynamoDB ──────────────────────────────────────────────────────────────
    # Stores document metadata (filename, index status, page count, etc.)
    dynamodb_table_name: str = Field(
        default="finrag-document-metadata",
        description="DynamoDB table for document metadata",
    )

    # ── PostgreSQL / pgvector ─────────────────────────────────────────────────
    # Not used in the current deployment (using in-memory FAISS instead)
    # Kept for future migration to persistent vector storage
    postgres_url: str = Field(
        default="postgresql://finrag:password@localhost:5432/finrag_db",
        description="PostgreSQL connection URL (with pgvector extension)",
    )

    # ── Index Storage ─────────────────────────────────────────────────────────
    # The FAISS vector index is saved to disk in this directory
    # On container restart, the index is loaded from here instead of being rebuilt
    index_persist_dir: Path = Field(
        default=Path("./index_store"),
        description="Directory to persist the LlamaIndex vector index",
    )

    # ── LoRA Fine-tuned Model ─────────────────────────────────────────────────
    # Path to the saved LoRA adapter weights for the financial NER model
    # The NER endpoint is disabled if this path doesn't exist
    lora_model_path: Path = Field(
        default=Path("./models/finrag-ner-lora"),
        description="Path to the saved LoRA adapter weights",
    )
    base_ner_model: str = Field(
        default="bert-base-uncased",
        description="HuggingFace model ID for the base NER model",
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="API server host (0.0.0.0 = all interfaces)")
    api_port: int = Field(default=8000, description="API server port", ge=1, le=65535)
    api_workers: int = Field(default=4, description="Number of uvicorn worker processes")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins (comma-separated string or list)",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Python logging level (DEBUG/INFO/WARNING/ERROR)")

    # ── WandB (optional experiment tracking) ─────────────────────────────────
    # WandB logs training metrics during LoRA fine-tuning
    # Not needed for the main API — only used by lora_trainer.py
    wandb_project: Optional[str] = Field(default=None, description="Weights & Biases project name")
    wandb_api_key: Optional[str] = Field(default=None, description="Weights & Biases API key")

    # ── RAG Tuning ────────────────────────────────────────────────────────────
    # These control the quality/speed tradeoff of the retrieval pipeline
    retriever_top_k: int = Field(default=8, description="Number of nodes to retrieve from each retriever")
    reranker_top_n: int = Field(default=4, description="Number of nodes to keep after cross-encoder reranking")
    chunk_size: int = Field(default=512, description="Maximum token size per text chunk")
    chunk_overlap: int = Field(default=64, description="Token overlap between adjacent chunks")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Accept CORS origins as either a list or a comma-separated string.

        This allows setting CORS_ORIGINS="http://a.com,http://b.com" in .env
        OR passing a Python list directly (useful in tests).
        """
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("index_persist_dir", "lora_model_path", mode="before")
    @classmethod
    def coerce_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects.

        Pydantic can't auto-coerce strings to Path, so we do it manually.
        """
        return Path(v)

    @model_validator(mode="after")
    def ensure_dirs_exist(self) -> "Settings":
        """Create required directories if they don't already exist.

        Called after all fields are validated.
        Creates index_persist_dir and the parent of lora_model_path
        so the application doesn't crash when trying to write to them.
        """
        self.index_persist_dir.mkdir(parents=True, exist_ok=True)
        self.lora_model_path.parent.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def boto3_session_kwargs(self) -> dict:
        """Build kwargs for boto3.Session(), omitting None values.

        If running on ECS/EKS with an IAM role, access_key and secret_key
        will be None — boto3 picks up the role credentials automatically.
        Only explicit credentials are included in the kwargs dict.

        Example output (with explicit credentials):
          {"region_name": "us-east-1", "aws_access_key_id": "AKI...", ...}

        Example output (with IAM role — no credentials needed):
          {"region_name": "us-east-1"}
        """
        kwargs: dict = {"region_name": self.aws_region}
        if self.aws_access_key_id:
            kwargs["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            kwargs["aws_session_token"] = self.aws_session_token
        return kwargs


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    @lru_cache(maxsize=1) means this function is called once and the result
    is cached forever. Every subsequent call returns the same Settings object.

    This is used as a FastAPI dependency — all route handlers call get_settings()
    and get the same instance with the same configuration.
    """
    return Settings()
