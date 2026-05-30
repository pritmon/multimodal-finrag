"""Local sentence-transformers embeddings — fast, free, no API calls.

Replaces BedrockTitanEmbedding with all-MiniLM-L6-v2 running on CPU/GPU locally.
800 chunks in ~3s vs ~4 minutes with Bedrock Titan on free tier.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding

logger = logging.getLogger(__name__)


class LocalEmbedding(BaseEmbedding):
    """LlamaIndex BaseEmbedding backed by sentence-transformers (local, no API).

    Parameters
    ----------
    model_name:
        HuggingFace model ID. Default: all-MiniLM-L6-v2 (384-dim, very fast).
    batch_size:
        Number of texts to encode per forward pass.
    cache_size:
        Number of (text → embedding) pairs to cache in memory.
    """

    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    cache_size: int = 4096

    _model: Any = None
    _cache: dict = {}

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from sentence_transformers import SentenceTransformer
        object.__setattr__(self, "_model", SentenceTransformer(self.model_name))
        object.__setattr__(self, "_cache", {})
        logger.info("LocalEmbedding loaded: %s (dim=%d)", self.model_name, self.get_embedding_dim())

    @classmethod
    def class_name(cls) -> str:
        return "LocalEmbedding"

    def get_embedding_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    # ── LlamaIndex interface ──────────────────────────────────────────────────

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._embed_single(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._embed_single(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._embed_batch(texts)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._embed_single(query)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return self._embed_single(text)

    # ── Core logic ────────────────────────────────────────────────────────────

    def _embed_single(self, text: str) -> Embedding:
        key = _sha256(text)
        if key in self._cache:
            return self._cache[key]
        vec = self._model.encode([text], batch_size=1)[0].tolist()
        self._evict_and_store(key, vec)
        return vec

    def _embed_batch(self, texts: list[str]) -> list[Embedding]:
        results: list[Optional[Embedding]] = [None] * len(texts)
        uncached = [i for i, t in enumerate(texts) if _sha256(t) not in self._cache]
        cached = [i for i in range(len(texts)) if i not in uncached]

        for i in cached:
            results[i] = self._cache[_sha256(texts[i])]

        if uncached:
            batch_texts = [texts[i] for i in uncached]
            vecs = self._model.encode(
                batch_texts, batch_size=self.batch_size, show_progress_bar=False
            ).tolist()
            for idx, vec in zip(uncached, vecs):
                results[idx] = vec
                self._evict_and_store(_sha256(texts[idx]), vec)

        logger.debug("Embedded %d texts (%d cached, %d new)", len(texts), len(cached), len(uncached))
        return results  # type: ignore[return-value]

    def _evict_and_store(self, key: str, vec: Embedding) -> None:
        if len(self._cache) >= self.cache_size:
            del self._cache[next(iter(self._cache))]
        self._cache[key] = vec

    def clear_cache(self) -> None:
        self._cache.clear()


# Keep BedrockTitanEmbedding as a fallback alias for backward compat
BedrockTitanEmbedding = LocalEmbedding


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
