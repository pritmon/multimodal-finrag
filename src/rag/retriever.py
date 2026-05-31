"""Hybrid BM25 + vector retriever with cross-encoder reranking.

HOW IT WORKS (simple analogy):
  Imagine you're searching a library for books relevant to your question.
  We use THREE search strategies, then pick the best results:

  1. BM25 (keyword search):
     Like a traditional search engine — finds chunks containing the exact words
     in your question. Fast and reliable for specific terms like "revenue Q3 2023".
     Think of it as Filter DataTable by keyword in UiPath.

  2. Vector search (semantic search):
     Understands meaning, not just words. "profit" and "earnings" are treated
     as similar even if they're different words.
     Finds the chunks whose meaning is closest to your question.

  3. Reciprocal Rank Fusion (RRF):
     Merges the two result lists into one ranked list.
     Documents that appear high in BOTH lists get a big boost.
     Formula: score = 1/(k + rank_in_bm25) + 1/(k + rank_in_vector)
     k=60 is a standard constant that prevents top-ranked items from dominating.

  4. Cross-encoder reranking:
     A final quality check — takes the top N merged results and re-scores each
     (question, passage) pair using a more powerful but slower model.
     Like a human expert reviewing the shortlist and picking the best answer.

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
from rank_bm25 import BM25Okapi       # BM25 library — keyword-based search
from sentence_transformers import CrossEncoder  # cross-encoder for reranking

logger = logging.getLogger(__name__)

# Default cross-encoder model — fast, small, trained on MS MARCO passage ranking
_DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class BM25Retriever:
    """Thin wrapper around rank_bm25 for LlamaIndex node collections.

    BM25 (Best Match 25) is a classic information retrieval algorithm.
    It scores documents by how often your query terms appear in them,
    adjusted for document length (longer docs aren't unfairly boosted).

    Works like a simple SQL LIKE/CONTAINS search, but smarter — it also
    considers how rare a term is across the whole document collection
    (rare words are more informative than common words like "the").
    """

    def __init__(self, nodes: list[TextNode], top_k: int = 10) -> None:
        self.nodes = nodes
        self.top_k = top_k
        # Tokenize all nodes at construction time — BM25 needs pre-tokenized corpus
        # This happens once when the retriever is built, not on every query
        tokenized = [_tokenize(n.get_content()) for n in nodes]
        self._bm25 = BM25Okapi(tokenized)  # build the BM25 index

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """Return the top_k most relevant nodes for a keyword query.

        Steps:
          1. Tokenize the query the same way we tokenized the corpus
          2. BM25 scores every document in the corpus against the query
          3. Sort by score descending, return top_k with score > 0

        Nodes with score 0 (no matching keywords) are excluded.
        """
        tokens = _tokenize(query)  # tokenize the query
        scores = self._bm25.get_scores(tokens)  # BM25 score for each node

        # Sort node indices by score (descending), take top_k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k]

        # Return as NodeWithScore objects (LlamaIndex's standard result format)
        return [
            NodeWithScore(node=self.nodes[i], score=float(scores[i]))
            for i in top_indices
            if scores[i] > 0  # only return nodes with at least one matching keyword
        ]


class HybridRetriever(BaseRetriever):
    """Combine BM25 + vector retrieval with Reciprocal Rank Fusion and reranking.

    Extends LlamaIndex's BaseRetriever — can be dropped into any LlamaIndex
    pipeline that uses a retriever.

    Parameters
    ----------
    vector_retriever:
        A LlamaIndex BaseRetriever backed by a vector store (FAISS/in-memory).
        Handles semantic similarity search.
    nodes:
        All indexed nodes (needed to build the BM25 index).
    top_k:
        Number of nodes to fetch from each sub-retriever (BM25 and vector).
    reranker_model:
        HuggingFace model ID for the cross-encoder reranker.
    reranker_top_n:
        Final number of nodes to return after reranking.
        Usually 4-8 — enough context without overwhelming the LLM.
    rrf_k:
        RRF constant. Standard value is 60. Higher = less weight on top ranks.
    use_reranker:
        Set to False to skip reranking (faster response, lower quality).
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
        self._vector_retriever = vector_retriever  # semantic search (FAISS/dense)
        self._bm25 = BM25Retriever(nodes=nodes, top_k=top_k)  # keyword search (BM25/sparse)
        self._top_k = top_k
        self._reranker_top_n = reranker_top_n
        self._rrf_k = rrf_k
        self._use_reranker = use_reranker
        self._reranker: Optional[CrossEncoder] = None

        # Load the cross-encoder only if reranking is enabled
        # (cross-encoder is a heavier model — skip it if speed matters more)
        if use_reranker:
            logger.info("Loading cross-encoder reranker: %s", reranker_model)
            self._reranker = CrossEncoder(reranker_model)

    # ── BaseRetriever Interface ───────────────────────────────────────────────

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Main retrieval method — called by LlamaIndex for every query.

        Full pipeline:
          Step 1: Get top_k nodes from vector search (semantic similarity)
          Step 2: Get top_k nodes from BM25 (keyword matching)
          Step 3: Merge both lists with Reciprocal Rank Fusion
          Step 4: Rerank the merged list with the cross-encoder
          Step 5: Return top reranker_top_n nodes
        """
        query_str = query_bundle.query_str

        # Step 1: Fetch from both retrievers independently
        vector_nodes = self._vector_retriever.retrieve(query_bundle)   # semantic results
        bm25_nodes = self._bm25.retrieve(query_str)                     # keyword results

        logger.debug(
            "Hybrid retrieve: %d vector nodes, %d BM25 nodes",
            len(vector_nodes),
            len(bm25_nodes),
        )

        # Step 2: Merge both ranked lists with Reciprocal Rank Fusion
        # RRF rewards documents that appear high in BOTH lists
        fused = _reciprocal_rank_fusion(
            [vector_nodes, bm25_nodes], k=self._rrf_k
        )

        # Keep double the top_k before reranking — give the reranker enough candidates
        candidates = fused[: self._top_k * 2]

        # Step 3: Re-score candidates with the cross-encoder
        # (slower but much more accurate — evaluates each (query, passage) pair together)
        if self._use_reranker and self._reranker and candidates:
            candidates = self._rerank(query_str, candidates)

        # Return only the top reranker_top_n results
        return candidates[: self._reranker_top_n]

    # ── Reranking ─────────────────────────────────────────────────────────────

    def _rerank(
        self, query: str, nodes: list[NodeWithScore]
    ) -> list[NodeWithScore]:
        """Re-score (query, passage) pairs with the cross-encoder model.

        A cross-encoder reads the query AND passage together (not separately),
        so it understands their relationship much better than dot-product similarity.
        It returns a raw logit score (can be any real number).

        We apply sigmoid to normalise logits to [0, 1] range:
          sigmoid(x) = 1 / (1 + e^(-x))
          Large positive x → close to 1.0 (very relevant)
          Large negative x → close to 0.0 (not relevant)
          x = 0 → 0.5 (uncertain)

        Then sort by the new scores and return.
        """
        import math
        # Build (query, passage) pairs — the cross-encoder reads both together
        pairs = [(query, node.node.get_content()) for node in nodes]
        # Run all pairs through the cross-encoder in one batch
        raw_scores: list[float] = self._reranker.predict(pairs).tolist()

        # Sigmoid normalisation: convert raw logits to [0, 1] probabilities
        scores = [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]

        # Rebuild NodeWithScore objects with the new cross-encoder scores
        reranked = [
            NodeWithScore(node=nws.node, score=float(score))
            for nws, score in zip(nodes, scores)
        ]
        # Sort by score descending — best match first
        reranked.sort(key=lambda x: x.score, reverse=True)
        logger.debug("Reranked %d → top score %.4f", len(reranked), reranked[0].score if reranked else 0)
        return reranked


# ── Utility Functions ─────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lower-case tokenizer for BM25.

    Splits text into words by whitespace and lowercases everything.
    Example: "Revenue Q3 2023" → ["revenue", "q3", "2023"]

    This must be applied identically to both the corpus (at index time)
    and the query (at search time) — otherwise BM25 won't find matches.
    """
    return text.lower().split()


def _reciprocal_rank_fusion(
    ranked_lists: list[list[NodeWithScore]],
    k: int = 60,
) -> list[NodeWithScore]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score for document d = Σ_r 1 / (k + rank(r, d))

    Where:
      - rank(r, d) = position of document d in ranked list r (1-indexed)
      - k = smoothing constant (typically 60) — prevents top ranks from dominating

    Example with k=60:
      Doc A is rank 1 in vector, rank 3 in BM25:
        RRF = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323

      Doc B is rank 2 in vector, rank 1 in BM25:
        RRF = 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325

      Doc B wins slightly — it ranked high in both lists.

    Documents not appearing in a list simply get no contribution from that list.
    """
    scores: dict[str, float] = defaultdict(float)  # node_id → accumulated RRF score
    node_map: dict[str, NodeWithScore] = {}         # node_id → NodeWithScore object

    for ranked in ranked_lists:
        for rank, nws in enumerate(ranked, start=1):  # rank is 1-indexed
            node_id = nws.node.node_id
            # Add this list's contribution: 1 / (k + rank)
            scores[node_id] += 1.0 / (k + rank)
            # Keep one copy of the node itself (for building the final result)
            if node_id not in node_map:
                node_map[node_id] = nws.node

    # Build the merged result list, sorted by RRF score descending
    result = [
        NodeWithScore(node=node_map[node_id], score=score)
        for node_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return result
