"""LlamaIndex BaseEmbedding subclass using AWS Bedrock Titan Embeddings.

Features:
- Batch embedding with configurable batch size
- In-memory LRU cache to avoid re-embedding identical texts
- Automatic retry on transient Bedrock throttling errors
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any, Optional

import boto3
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_EMBED_DIMS = {
    "amazon.titan-embed-text-v1": 1536,
    "amazon.titan-embed-text-v2:0": 1024,
}


class BedrockTitanEmbedding(BaseEmbedding):
    """Bedrock Titan Embeddings model wrapped as a LlamaIndex BaseEmbedding.

    Parameters
    ----------
    model_id:
        Bedrock embedding model identifier.
    aws_region:
        AWS region for the Bedrock client.
    batch_size:
        Number of texts to embed per Bedrock call. Titan processes one text
        per call, so this controls the concurrency of sequential calls.
    cache_size:
        Number of (text → embedding) pairs to cache in memory.
    session_kwargs:
        Extra kwargs forwarded to ``boto3.Session``.
    """

    model_id: str = "amazon.titan-embed-text-v1"
    aws_region: str = "us-east-1"
    batch_size: int = 32
    cache_size: int = 2048

    _client: Any = None
    _cache: dict = {}  # simple dict cache keyed by SHA-256 of text

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        session_kwargs = kwargs.get("session_kwargs", {})
        session = boto3.Session(**session_kwargs)
        object.__setattr__(
            self,
            "_client",
            session.client("bedrock-runtime", region_name=self.aws_region),
        )
        object.__setattr__(self, "_cache", {})

    @classmethod
    def class_name(cls) -> str:
        return "BedrockTitanEmbedding"

    # ── Public interface ──────────────────────────────────────────────────────

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._embed_single(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._embed_single(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._embed_batch(texts)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        # Async not natively supported; fall back to sync
        return self._embed_single(query)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return self._embed_single(text)

    # ── Core embedding logic ──────────────────────────────────────────────────

    def _embed_single(self, text: str) -> Embedding:
        """Embed a single text string, using cache if available."""
        cache_key = _sha256(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self._invoke_bedrock(text)

        # Evict oldest entry if cache is full
        if len(self._cache) >= self.cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        self._cache[cache_key] = embedding
        return embedding

    def _embed_batch(self, texts: list[str]) -> list[Embedding]:
        """Embed a list of texts, leveraging the cache for already-seen texts."""
        results: list[Optional[Embedding]] = [None] * len(texts)
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            cache_key = _sha256(text)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)

        logger.debug(
            "Embedding batch: %d total, %d cached, %d to fetch",
            len(texts),
            len(texts) - len(uncached_indices),
            len(uncached_indices),
        )

        for idx in uncached_indices:
            text = texts[idx]
            embedding = self._invoke_bedrock(text)
            results[idx] = embedding

            cache_key = _sha256(text)
            if len(self._cache) >= self.cache_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[cache_key] = embedding

        return results  # type: ignore[return-value]

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _invoke_bedrock(self, text: str) -> Embedding:
        """Call Bedrock Titan Embeddings API for a single text.

        Titan v1 accepts ``{"inputText": "..."}`` and returns
        ``{"embedding": [...], "inputTextTokenCount": N}``.
        """
        # Truncate to ~8000 chars to stay within token limit
        truncated = text[:8000]
        body = json.dumps({"inputText": truncated})

        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            embedding: list[float] = result["embedding"]
            return embedding
        except Exception as exc:
            logger.warning("Bedrock embedding call failed: %s", exc)
            raise

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension for the configured model."""
        return _EMBED_DIMS.get(self.model_id, 1536)

    def clear_cache(self) -> None:
        """Clear the in-memory embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    @property
    def cache_stats(self) -> dict:
        return {"size": len(self._cache), "capacity": self.cache_size}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
