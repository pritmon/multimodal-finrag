"""Hybrid BM25 + vector retriever with cross-encoder reranking.

Architecture:
1. BM25Retriever  – sparse keyword-based retrieval
2. VectorIndexRetriever – dense semantic retrieval from the LlamaIndex index
3. Score fusion (Reciprocal Rank Fusion)
4. Cross-encoder reranking (sentence-transformers)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class BM25Retriever:
    """Thin wrapper around rank_bm25 for LlamaIndex node collections."""

    def __init__(self, nodes: list[TextNode], top_k: int = 10) -> None:
        self.nodes = nodes
        self.top_k = top_k
        tokenized = [_tokenize(n.get_content()) for n in nodes]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str) -> list[NodeWithScore]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k]
        return [
            NodeWithScore(node=self.nodes[i], score=float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]


class HybridRetriever(BaseRetriever):
    """Combine BM25 + vector retrieval with Reciprocal Rank Fusion and reranking.

    Parameters
    ----------
    vector_retriever:
        A LlamaIndex ``BaseRetriever`` backed by a vector store.
    nodes:
        All indexed nodes (needed to build the BM25 index).
    top_k:
        Number of nodes to fetch from each sub-retriever.
    reranker_model:
        HuggingFace model ID for the cross-encoder reranker.
    reranker_top_n:
        Final number of nodes to return after reranking.
    rrf_k:
        RRF constant (typically 60).
    use_reranker:
        Set to False to skip cross-encoder reranking (faster, lower quality).
    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        nodes: list[TextNode],
        top_k: int = 10,
        reranker_model: str = _DEFAULT_CROSS_ENCODER,
        reranker_top_n: int = 4,
        rrf_k: int = 60,
        use_reranker: bool = True,
    ) -> None:
        super().__init__()
        self._vector_retriever = vector_retriever
        self._bm25 = BM25Retriever(nodes=nodes, top_k=top_k)
        self._top_k = top_k
        self._reranker_top_n = reranker_top_n
        self._rrf_k = rrf_k
        self._use_reranker = use_reranker
        self._reranker: Optional[CrossEncoder] = None

        if use_reranker:
            logger.info("Loading cross-encoder reranker: %s", reranker_model)
            self._reranker = CrossEncoder(reranker_model)

    # ── BaseRetriever interface ───────────────────────────────────────────────

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_str = query_bundle.query_str

        # Step 1: Fetch from both retrievers
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self._bm25.retrieve(query_str)

        logger.debug(
            "Hybrid retrieve: %d vector nodes, %d BM25 nodes",
            len(vector_nodes),
            len(bm25_nodes),
        )

        # Step 2: Reciprocal Rank Fusion
        fused = _reciprocal_rank_fusion(
            [vector_nodes, bm25_nodes], k=self._rrf_k
        )

        # Keep top_k before reranking
        candidates = fused[: self._top_k * 2]

        # Step 3: Cross-encoder reranking
        if self._use_reranker and self._reranker and candidates:
            candidates = self._rerank(query_str, candidates)

        return candidates[: self._reranker_top_n]

    # ── Reranking ─────────────────────────────────────────────────────────────

    def _rerank(
        self, query: str, nodes: list[NodeWithScore]
    ) -> list[NodeWithScore]:
        """Score (query, passage) pairs with the cross-encoder and re-sort."""
        pairs = [(query, node.node.get_content()) for node in nodes]
        scores: list[float] = self._reranker.predict(pairs).tolist()

        reranked = [
            NodeWithScore(node=nws.node, score=float(score))
            for nws, score in zip(nodes, scores)
        ]
        reranked.sort(key=lambda x: x.score, reverse=True)
        logger.debug("Reranked %d → top score %.4f", len(reranked), reranked[0].score if reranked else 0)
        return reranked


# ── Utilities ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lower-case tokenizer for BM25."""
    return text.lower().split()


def _reciprocal_rank_fusion(
    ranked_lists: list[list[NodeWithScore]],
    k: int = 60,
) -> list[NodeWithScore]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = Σ_r 1 / (k + rank(r, d))
    """
    scores: dict[str, float] = defaultdict(float)
    node_map: dict[str, NodeWithScore] = {}

    for ranked in ranked_lists:
        for rank, nws in enumerate(ranked, start=1):
            node_id = nws.node.node_id
            scores[node_id] += 1.0 / (k + rank)
            if node_id not in node_map:
                node_map[node_id] = nws

    result = [
        NodeWithScore(node=node_map[node_id], score=score)
        for node_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return result
