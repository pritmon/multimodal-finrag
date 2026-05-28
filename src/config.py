"""Application configuration using Pydantic BaseSettings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the FinRAG system.

    All values can be overridden via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── AWS Core ──────────────────────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1", description="AWS region for all services")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key (omit for IAM role)")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret key")
    aws_session_token: Optional[str] = Field(default=None, description="AWS session token for temporary credentials")

    # ── S3 ────────────────────────────────────────────────────────────────────
    s3_bucket: str = Field(default="finrag-documents", description="S3 bucket for document storage")
    s3_prefix: str = Field(default="documents/", description="S3 key prefix for documents")

    # ── Bedrock Models ────────────────────────────────────────────────────────
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        description="Bedrock model ID for generation",
    )
    bedrock_embed_model_id: str = Field(
        default="amazon.titan-embed-text-v1",
        description="Bedrock model ID for embeddings",
    )
    bedrock_max_tokens: int = Field(default=4096, description="Max tokens for Bedrock generation")
    bedrock_temperature: float = Field(default=0.1, description="Temperature for generation", ge=0.0, le=1.0)

    # ── Lambda ────────────────────────────────────────────────────────────────
    lambda_function_name: str = Field(
        default="finrag-document-processor",
        description="Lambda function name for async document processing",
    )

    # ── DynamoDB ──────────────────────────────────────────────────────────────
    dynamodb_table_name: str = Field(
        default="finrag-document-metadata",
        description="DynamoDB table for document metadata",
    )

    # ── PostgreSQL / pgvector ─────────────────────────────────────────────────
    postgres_url: str = Field(
        default="postgresql://finrag:password@localhost:5432/finrag_db",
        description="PostgreSQL connection URL (with pgvector extension)",
    )

    # ── Index Storage ─────────────────────────────────────────────────────────
    index_persist_dir: Path = Field(
        default=Path("./index_store"),
        description="Directory to persist the LlamaIndex vector index",
    )

    # ── LoRA Fine-tuned Model ─────────────────────────────────────────────────
    lora_model_path: Path = Field(
        default=Path("./models/finrag-ner-lora"),
        description="Path to the saved LoRA adapter weights",
    )
    base_ner_model: str = Field(
        default="bert-base-uncased",
        description="HuggingFace model ID for the base NER model",
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port", ge=1, le=65535)
    api_workers: int = Field(default=4, description="Number of uvicorn workers")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Python logging level")

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_project: Optional[str] = Field(default=None, description="Weights & Biases project name")
    wandb_api_key: Optional[str] = Field(default=None, description="Weights & Biases API key")

    # ── RAG Tuning ────────────────────────────────────────────────────────────
    retriever_top_k: int = Field(default=8, description="Number of nodes to retrieve")
    reranker_top_n: int = Field(default=4, description="Number of nodes after reranking")
    chunk_size: int = Field(default=512, description="Token chunk size for text splitting")
    chunk_overlap: int = Field(default=64, description="Token overlap between chunks")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("index_persist_dir", "lora_model_path", mode="before")
    @classmethod
    def coerce_path(cls, v: str | Path) -> Path:
        return Path(v)

    @model_validator(mode="after")
    def ensure_dirs_exist(self) -> "Settings":
        self.index_persist_dir.mkdir(parents=True, exist_ok=True)
        self.lora_model_path.parent.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def boto3_session_kwargs(self) -> dict:
        """Build kwargs for boto3 Session, omitting None values."""
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
    """Return a cached singleton Settings instance."""
    return Settings()
