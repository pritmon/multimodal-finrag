"""Local sentence-transformers embeddings — fast, free, no API calls.

HOW IT WORKS (simple analogy):
  Imagine you want to search a library of books by meaning, not just keywords.
  Embeddings convert text into a list of numbers (a "vector") that captures
  the meaning of the text.

  Two sentences with similar meaning → similar vectors → close together in space.
  Example:
    "What was the profit?"   → [0.12, -0.45, 0.78, ...]
    "How much did we earn?"  → [0.11, -0.44, 0.79, ...]  ← very similar!
    "The sky is blue."       → [-0.92, 0.23, -0.11, ...]  ← very different

  When a user asks a question, we convert it to a vector and find the
  document chunks with the most similar vectors — those are the relevant passages.

WHY LOCAL (not Bedrock Titan):
  - Bedrock Titan has a rate limit of 5 requests/second on the free tier
  - 800 chunks × ~0.2s/chunk = 4+ hours to embed a whole document
  - all-MiniLM-L6-v2 runs locally: 800 chunks in ~3 seconds — 5000x faster
  - It's also completely free with no API costs

This class extends LlamaIndex's BaseEmbedding so it plugs into the
LlamaIndex RAG pipeline like any other embedding model.
"""

from __future__ import annotations

import hashlib  # for generating cache keys
import logging
from typing import Any, Optional

# LlamaIndex base class — we must implement certain methods to plug in
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding

logger = logging.getLogger(__name__)


class LocalEmbedding(BaseEmbedding):
    """LlamaIndex BaseEmbedding backed by sentence-transformers (local, no API).

    Inherits from LlamaIndex's BaseEmbedding so it can be used anywhere
    LlamaIndex expects an embedding model — no code changes needed in the pipeline.

    Includes an in-memory LRU cache: if the same text is embedded twice,
    the cached vector is returned instantly (no re-computation).

    Parameters
    ----------
    model_name:
        HuggingFace model ID. Default: all-MiniLM-L6-v2
        - 384-dimensional output vectors (compact, fast)
        - Strong semantic understanding for English text
        - ~22MB model size — fits comfortably in RAM
    batch_size:
        Number of texts encoded per forward pass.
        Larger = faster but uses more memory. 64 is a good default.
    cache_size:
        Maximum number of (text → embedding) pairs to hold in memory.
        When full, oldest entries are evicted (FIFO eviction policy).
    """

    # Pydantic field declarations (required by LlamaIndex's BaseEmbedding)
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    cache_size: int = 4096

    # Private attributes — set after __init__ via object.__setattr__ (Pydantic workaround)
    _model: Any = None   # the SentenceTransformer model instance
    _cache: dict = {}    # text hash → embedding vector cache

    class Config:
        arbitrary_types_allowed = True  # allows storing non-Pydantic types like SentenceTransformer

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Import here to avoid loading sentence-transformers at module import time
        # (saves startup time when this class isn't needed)
        from sentence_transformers import SentenceTransformer
        # Use object.__setattr__ because Pydantic v2 models are normally immutable
        object.__setattr__(self, "_model", SentenceTransformer(self.model_name))
        object.__setattr__(self, "_cache", {})  # start with an empty cache
        logger.info("LocalEmbedding loaded: %s (dim=%d)", self.model_name, self.get_embedding_dim())

    @classmethod
    def class_name(cls) -> str:
        """LlamaIndex requires this for serialisation/identification."""
        return "LocalEmbedding"

    def get_embedding_dim(self) -> int:
        """Return the number of dimensions in each embedding vector.

        all-MiniLM-L6-v2 produces 384-dimensional vectors.
        (Some models produce 768, 1536, or 3072 dimensions.)
        """
        return self._model.get_sentence_embedding_dimension()

    # ── LlamaIndex Interface Methods ──────────────────────────────────────────
    # LlamaIndex calls these methods internally. We implement them to plug in
    # our local model. The sync and async versions do the same thing
    # (sentence-transformers is synchronous — no true async needed).

    def _get_query_embedding(self, query: str) -> Embedding:
        """Embed a single search query (called when user asks a question)."""
        return self._embed_single(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        """Embed a single text chunk (called during document indexing)."""
        return self._embed_single(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """Embed a list of text chunks in one batch (faster than one at a time)."""
        return self._embed_batch(texts)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Async version of _get_query_embedding (runs synchronously — see note above)."""
        return self._embed_single(query)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Async version of _get_text_embedding."""
        return self._embed_single(text)

    # ── Core Embedding Logic ──────────────────────────────────────────────────

    def _embed_single(self, text: str) -> Embedding:
        """Embed one text string, using the cache if available.

        Cache lookup:
          1. Hash the text to a key (SHA-256 → 64-char hex string)
          2. If the key is in the cache, return the cached vector immediately
          3. Otherwise, run the model and store the result before returning
        """
        key = _sha256(text)  # hash the text to use as a cache key
        if key in self._cache:
            return self._cache[key]  # cache hit — return immediately
        # Cache miss — compute the embedding
        vec = self._model.encode([text], batch_size=1)[0].tolist()
        self._evict_and_store(key, vec)  # store in cache for future calls
        return vec

    def _embed_batch(self, texts: list[str]) -> list[Embedding]:
        """Embed a batch of texts efficiently, only computing uncached ones.

        Separates texts into:
          - cached: already have vectors, return from cache
          - uncached: need to run through the model

        Only the uncached texts are sent to the model — this avoids
        redundant computation and saves time when texts repeat across documents.
        """
        results: list[Optional[Embedding]] = [None] * len(texts)

        # Split into cached and uncached indices
        uncached = [i for i, t in enumerate(texts) if _sha256(t) not in self._cache]
        cached = [i for i in range(len(texts)) if i not in uncached]

        # Fill cached results immediately from memory
        for i in cached:
            results[i] = self._cache[_sha256(texts[i])]

        # Batch-encode only the texts that aren't cached
        if uncached:
            batch_texts = [texts[i] for i in uncached]
            # encode() with a batch processes multiple texts in parallel on GPU/CPU
            vecs = self._model.encode(
                batch_texts, batch_size=self.batch_size, show_progress_bar=False
            ).tolist()
            # Store each new vector in the cache and in results
            for idx, vec in zip(uncached, vecs):
                results[idx] = vec
                self._evict_and_store(_sha256(texts[idx]), vec)

        logger.debug("Embedded %d texts (%d cached, %d new)", len(texts), len(cached), len(uncached))
        return results  # type: ignore[return-value]

    def _evict_and_store(self, key: str, vec: Embedding) -> None:
        """Add a new entry to the cache, evicting the oldest entry if full.

        This is a simple FIFO (First In First Out) eviction policy.
        When the cache reaches cache_size, we delete the oldest entry
        (Python dicts maintain insertion order since Python 3.7).
        """
        if len(self._cache) >= self.cache_size:
            # Delete the first (oldest) key — next(iter(dict)) returns the first key
            del self._cache[next(iter(self._cache))]
        self._cache[key] = vec  # store the new vector

    def clear_cache(self) -> None:
        """Empty the in-memory embedding cache.

        Useful when you want to free memory or force re-computation.
        """
        self._cache.clear()


# Backward compatibility alias — old code may reference BedrockTitanEmbedding
# (we switched from Bedrock Titan to local embeddings for speed/cost reasons)
BedrockTitanEmbedding = LocalEmbedding


# ── Utility Functions ─────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    """Compute a SHA-256 hash of a text string.

    Used as the cache key — a unique fingerprint of the text content.
    Two identical strings will always produce the same hash.
    A 64-character hex string is returned (e.g. "a3f2c1...").
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
