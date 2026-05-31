"""LlamaIndex multimodal RAG pipeline for financial documents.

HOW IT WORKS (simple analogy — end to end):

  Think of this as the MASTER WORKFLOW in UiPath that orchestrates everything.
  It calls the PDF parser, chart extractor, embedder, retriever, and LLM
  in the right order.

  INDEXING (one-time setup):
    1. Parse the PDF → get TextBlocks + EmbeddedImages
    2. Group text by page → one Document per page
    3. Split each Document into chunks (SentenceSplitter)
    4. Embed each chunk (all-MiniLM-L6-v2) → 384-dim vectors
    5. Store vectors in a FAISS index (persisted to disk)
    6. Detect charts with CLIP → caption with Bedrock Nova
    7. Add chart captions as extra text nodes (so they're searchable)

  QUERYING (every user request):
    1. Get the user's question
    2. BM25 search → top 10 keyword-matching chunks
    3. Vector search → top 10 semantically similar chunks
    4. Reciprocal Rank Fusion → merge into one ranked list
    5. Cross-encoder reranking → pick the best 4 chunks
    6. Find any charts from the same pages as the top chunks
    7. Build a prompt: context chunks + chart captions + question
    8. Call Bedrock Nova → get the answer
    9. Return the answer + source citations

Orchestrates:
- Document ingestion (text + chart nodes)
- VectorStoreIndex construction
- Hybrid retrieval (BM25 + vector + reranking)
- Bedrock Nova generation with source citations
- Index persistence and incremental updates
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from llama_index.core import (
    Settings as LlamaSettings,   # global LlamaIndex config
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter  # splits documents into chunks
from llama_index.core.schema import (
    Document,       # LlamaIndex's document type
    ImageDocument,
    NodeWithScore,  # a retrieved chunk with its relevance score
    TextNode,       # a chunk of text in the index
)

from src.config import Settings, get_settings
from src.ingestion.chart_extractor import ChartExtractor, ChartNode
from src.ingestion.pdf_parser import ParsedDocument, PDFParser

from .bedrock_llm import BedrockLLM
from .embeddings import BedrockTitanEmbedding   # actually LocalEmbedding (alias kept for compat)
from .retriever import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Structured result returned from the RAG pipeline for one question.

    Contains everything the API needs to build a response:
    - The generated answer text
    - The source chunks that were used to generate the answer
    - Any chart images that were relevant
    - The original question and model used
    """

    answer: str                         # the LLM-generated answer
    source_nodes: list[NodeWithScore]   # the retrieved text chunks with relevance scores
    chart_nodes: list[ChartNode]        # relevant chart images from the same pages
    query: str                          # the original question
    model_id: str                       # which Bedrock model generated the answer

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON API responses."""
        return {
            "answer": self.answer,
            "query": self.query,
            "model_id": self.model_id,
            # Include first 500 chars of each source chunk + its score + metadata
            "sources": [
                {
                    "text": n.node.get_content()[:500],  # truncate long chunks
                    "score": n.score,
                    "metadata": n.node.metadata,          # page number, source file, etc.
                }
                for n in self.source_nodes
            ],
            "charts": [c.to_dict() for c in self.chart_nodes],
        }


class FinRAGPipeline:
    """End-to-end multimodal RAG pipeline for financial document intelligence.

    This is the central coordinator — it owns the index, the retriever,
    the LLM, and the chart extractor. All API endpoints interact through
    this class.

    The pipeline is stateful:
      - _index: the FAISS vector index (persisted to disk)
      - _all_nodes: all text chunks (needed to build the BM25 index)
      - _chart_nodes: all detected chart images with captions
      - _retriever: the HybridRetriever (rebuilt when new docs are added)

    Parameters
    ----------
    settings:
        Application settings (uses cached singleton if None).
    pdf_parser:
        PDF parser instance (created from settings if None).
    chart_extractor:
        Chart extractor instance — lazily created on first use to avoid
        loading CLIP at startup when no PDFs need processing.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        pdf_parser: Optional[PDFParser] = None,
        chart_extractor: Optional[ChartExtractor] = None,
    ) -> None:
        # Load settings from the singleton cache (reads .env + environment variables)
        self.cfg = settings or get_settings()
        self._pdf_parser = pdf_parser or PDFParser()
        self._chart_extractor = chart_extractor  # None = lazy initialisation

        # ── Configure the LLM ────────────────────────────────────────────────
        # BedrockLLM wraps the Bedrock API; Nova Lite is fast and free-tier compatible
        self._llm = BedrockLLM(
            model_id=self.cfg.bedrock_model_id,
            aws_region=self.cfg.aws_region,
            max_tokens=self.cfg.bedrock_max_tokens,
            temperature=self.cfg.bedrock_temperature,
            session_kwargs=self.cfg.boto3_session_kwargs,
        )

        # ── Configure the embedding model ─────────────────────────────────────
        # LocalEmbedding uses all-MiniLM-L6-v2 — fast local embeddings (no API calls)
        self._embed_model = BedrockTitanEmbedding()

        # ── Set LlamaIndex global settings ────────────────────────────────────
        # LlamaIndex reads these globals internally — must be set before building index
        LlamaSettings.llm = self._llm
        LlamaSettings.embed_model = self._embed_model
        LlamaSettings.chunk_size = self.cfg.chunk_size     # e.g. 512 tokens per chunk
        LlamaSettings.chunk_overlap = self.cfg.chunk_overlap  # e.g. 64 tokens of overlap

        # SentenceSplitter: splits a Document into overlapping chunks
        # Overlap ensures context isn't lost at chunk boundaries
        self._node_parser = SentenceSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )

        # ── Index state ───────────────────────────────────────────────────────
        self._index: Optional[VectorStoreIndex] = None  # FAISS vector index
        self._all_nodes: list[TextNode] = []             # all chunks (for BM25)
        self._chart_nodes: list[ChartNode] = []          # detected charts
        self._retriever: Optional[HybridRetriever] = None  # hybrid BM25+vector+reranker

    # ── Index Management ──────────────────────────────────────────────────────

    def load_or_build_index(self, force_rebuild: bool = False) -> None:
        """Load the vector index from disk if it exists, otherwise create a new one.

        The index is persisted as JSON files in index_persist_dir (e.g. ./index_store/).
        On startup, this is called to restore a previously built index.

        If force_rebuild=True, the existing index is ignored and a fresh one is built.
        Use this when you want to completely re-index all documents.
        """
        persist_dir = self.cfg.index_persist_dir
        index_file = persist_dir / "docstore.json"  # this file exists if index was saved

        if index_file.exists() and not force_rebuild:
            # Index exists on disk — load it (much faster than rebuilding)
            logger.info("Loading existing index from %s", persist_dir)
            storage_ctx = StorageContext.from_defaults(persist_dir=str(persist_dir))
            self._index = load_index_from_storage(storage_ctx)
            # Rebuild the in-memory BM25+vector retriever from the loaded nodes
            self._rebuild_retriever()
        else:
            # No index found, or force rebuild requested — start fresh
            logger.info("Building new index (force_rebuild=%s)", force_rebuild)
            self._index = VectorStoreIndex(nodes=[], embed_model=self._embed_model)
            self._persist_index()  # save the empty index to disk

    def add_parsed_document(
        self, parsed_doc: ParsedDocument, generate_chart_captions: bool = True
    ) -> int:
        """Add a pre-parsed document to the vector index.

        This is the main indexing method. It:
          1. Groups text blocks by page → one Document per page
          2. Splits each Document into chunks with SentenceSplitter
          3. Detects charts with CLIP and generates captions with Bedrock
          4. Adds chart captions as extra searchable text nodes
          5. Embeds all chunks in parallel using local embeddings
          6. Inserts all nodes into the vector index
          7. Rebuilds the hybrid retriever with the new nodes
          8. Saves the updated index to disk

        Returns the total number of nodes added to the index.
        """
        # Ensure the index exists (load from disk or create new)
        if self._index is None:
            self.load_or_build_index()

        # Step 1: Group text blocks by page number
        # Each page becomes one Document — avoids 11,000+ tiny block-level chunks
        from collections import defaultdict
        page_texts: dict = defaultdict(list)  # page_number → [block_text, block_text, ...]
        for block in parsed_doc.text_blocks:
            if block.text.strip():
                page_texts[block.page_number].append(block.text)

        # Build one LlamaIndex Document per page, with metadata
        documents: list[Document] = []
        for page_num in sorted(page_texts):
            documents.append(
                Document(
                    text="\n".join(page_texts[page_num]),  # join all blocks on this page
                    metadata={
                        "source": parsed_doc.source,       # filename / S3 key
                        "page_number": page_num,            # 0-indexed page number
                        **parsed_doc.metadata,              # title, author, etc.
                    },
                )
            )

        # Step 2: Split pages into smaller chunks (SentenceSplitter)
        # e.g. a 3-page document → ~15 chunks of ~512 tokens each
        nodes = self._node_parser.get_nodes_from_documents(documents)
        logger.info("Merged %d pages → %d nodes for %s", len(documents), len(nodes), parsed_doc.source)

        # Step 3: Extract charts and generate captions (if images exist in the PDF)
        if parsed_doc.images:
            extractor = self._get_chart_extractor()
            new_charts = extractor.extract_charts(
                parsed_doc.images, generate_captions=generate_chart_captions
            )
            self._chart_nodes.extend(new_charts)  # keep for query-time retrieval

            # Step 4: Add chart captions as text nodes so they're searchable
            # A user asking "What does the revenue chart show?" can match this node
            for chart in new_charts:
                if chart.caption:
                    nodes.append(
                        TextNode(
                            text=f"[Chart on page {chart.page_number + 1}] {chart.caption}",
                            metadata={
                                "source": parsed_doc.source,
                                "page_number": chart.page_number,
                                "node_type": "chart_caption",  # mark it as a chart caption
                                "chart_type": chart.chart_type,
                            },
                        )
                    )

        # Step 5: Pre-embed all nodes in one batch (faster than LlamaIndex's sequential embed)
        # LlamaIndex normally embeds one node at a time — we batch them for speed
        texts = [n.get_content() for n in nodes]
        embeddings = LlamaSettings.embed_model._get_text_embeddings(texts)
        for node, emb in zip(nodes, embeddings):
            node.embedding = emb  # attach the vector directly to the node

        # Step 6: Insert all nodes into the FAISS vector index
        self._index.insert_nodes(nodes)  # type: ignore[arg-type]

        # Step 7: Add nodes to the BM25 corpus and rebuild the retriever
        self._all_nodes.extend(nodes)  # type: ignore[arg-type]
        self._rebuild_retriever()

        # Step 8: Persist the updated index to disk
        self._persist_index()

        return len(nodes)  # return total nodes added (text + chart caption nodes)

    def add_pdf_bytes(
        self, pdf_bytes: bytes, source: str = "unknown", **kwargs: Any
    ) -> dict:
        """Parse PDF bytes and add the result to the index.

        This is the top-level method called by the API's ingest endpoint.
        Returns a summary dict so the API can report what was indexed.
        """
        # Parse the PDF bytes → get TextBlocks + EmbeddedImages
        parsed = self._pdf_parser.parse_bytes(pdf_bytes, source=source)
        before_charts = len(self._chart_nodes)
        text_nodes = self.add_parsed_document(parsed, **kwargs)
        new_charts = self._chart_nodes[before_charts:]  # charts added in this call
        return {
            "text_nodes": text_nodes - len(new_charts),   # text-only nodes
            "chart_nodes": len(new_charts),                # chart caption nodes
            "chart_captions": [c.caption for c in new_charts if c.caption],
        }

    def add_pdf_file(self, path: str | Path, **kwargs: Any) -> int:
        """Parse a PDF file from disk and add it to the index.

        Convenience method — reads bytes from disk, then calls add_pdf_bytes().
        """
        parsed = self._pdf_parser.parse_file(path)
        return self.add_parsed_document(parsed, **kwargs)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> QueryResult:
        """Run a RAG query and return a structured result with answer + citations.

        Full RAG pipeline:
          1. Retrieve relevant chunks (hybrid BM25 + vector + reranker)
          2. Apply optional metadata filters (e.g. only from a specific PDF)
          3. Find chart images that share a page with the retrieved chunks
          4. Build a prompt: context + chart captions + question
          5. Call Bedrock Nova with the prompt (+ chart images if available)
          6. Return the answer + source nodes + chart nodes

        Parameters
        ----------
        question:
            Natural-language question from the user.
        top_k:
            Override the default retrieval top-k from settings.
        filters:
            Metadata filters e.g. {"source": "apple_10k.pdf", "page_number": 5}
        """
        # Make sure the index is loaded
        if self._index is None:
            self.load_or_build_index()

        effective_top_k = top_k or self.cfg.retriever_top_k

        # Step 1: Retrieve relevant chunks using hybrid search
        if self._retriever is not None:
            # Use our HybridRetriever (BM25 + vector + reranker) for best quality
            from llama_index.core import QueryBundle
            source_nodes = self._retriever.retrieve(QueryBundle(query_str=question))
        else:
            # Fallback: use pure vector search if no retriever is built yet
            retriever = self._index.as_retriever(similarity_top_k=effective_top_k)
            source_nodes = retriever.retrieve(question)

        # Step 2: Apply metadata filters if provided
        # e.g. only return chunks from a specific file or page
        if filters:
            source_nodes = _apply_filters(source_nodes, filters)

        # Step 3: Find chart images from the same pages as retrieved chunks
        # If we retrieved text from page 5, include any charts found on page 5
        relevant_charts = self._find_relevant_charts(source_nodes)

        # Step 4: Build the context string for the prompt
        # Each source chunk is numbered [Source 1], [Source 2], etc.
        # This lets the LLM reference specific sources in its answer
        context_parts = [f"[Source {i+1}]\n{n.node.get_content()}" for i, n in enumerate(source_nodes)]
        for chart in relevant_charts:
            context_parts.append(f"[Chart] {chart.caption}")  # include chart captions in context

        context = "\n\n---\n\n".join(context_parts)  # separate sources with a divider

        # System prompt — tells the LLM how to behave and what to do
        system_prompt = (
            "You are a financial document analyst. Answer the user's question using only "
            "the provided context from financial documents. Be precise and cite source numbers "
            "when referencing specific data. If the context does not contain enough information "
            "to answer the question, say so explicitly."
        )
        # Full prompt = system instruction + context + question
        full_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        # Step 5: Generate the answer with Bedrock Nova
        if relevant_charts:
            # Multimodal: include chart images so the LLM can "see" them
            from PIL import Image
            import io
            images = [Image.open(io.BytesIO(c.image_bytes)) for c in relevant_charts]
            answer = self._llm.complete_with_images(full_prompt, images=images)
        else:
            # Text-only: call complete() directly (bypasses LlamaIndex chat routing)
            answer = self._llm.complete(full_prompt).text
            if not answer:
                answer = "Unable to generate an answer from the provided context."

        return QueryResult(
            answer=answer,
            source_nodes=source_nodes,
            chart_nodes=relevant_charts,
            query=question,
            model_id=self.cfg.bedrock_model_id,
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _rebuild_retriever(self) -> None:
        """Rebuild the HybridRetriever after new documents are added.

        The BM25 index is built over all known nodes, so it must be
        rebuilt every time new nodes are added to the index.
        The vector retriever wraps the FAISS index (already updated).
        """
        if not self._all_nodes or self._index is None:
            return  # nothing to build yet
        # Vector retriever: similarity search via FAISS
        vector_retriever = self._index.as_retriever(
            similarity_top_k=self.cfg.retriever_top_k
        )
        # HybridRetriever: combines BM25 + vector + cross-encoder reranker
        self._retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            nodes=self._all_nodes,           # BM25 is built over all nodes
            top_k=self.cfg.retriever_top_k,
            reranker_top_n=self.cfg.reranker_top_n,
        )

    def _persist_index(self) -> None:
        """Save the current index state to disk.

        LlamaIndex writes JSON files (docstore.json, index_store.json, etc.)
        to index_persist_dir. These are loaded on startup to restore the index.
        """
        if self._index is not None:
            self._index.storage_context.persist(persist_dir=str(self.cfg.index_persist_dir))
            logger.info("Index persisted to %s", self.cfg.index_persist_dir)

    def _get_chart_extractor(self) -> ChartExtractor:
        """Return the ChartExtractor, creating it lazily on first use.

        CLIP and Bedrock client are only initialised when the first PDF with
        images is processed — avoids wasting resources if no charts exist.
        """
        if self._chart_extractor is None:
            self._chart_extractor = ChartExtractor(
                bedrock_model_id=self.cfg.bedrock_model_id,
                aws_region=self.cfg.aws_region,
                session_kwargs=self.cfg.boto3_session_kwargs,
            )
        return self._chart_extractor

    def _find_relevant_charts(
        self, source_nodes: list[NodeWithScore]
    ) -> list[ChartNode]:
        """Return chart nodes that share a page with any retrieved text node.

        When a text chunk from page 5 is retrieved, we also include any
        charts detected on page 5 — they're likely related to the same topic.

        Matching is done by (source_file, page_number) tuple to avoid
        false matches across different documents.
        """
        # Build a set of (source_file, page_number) tuples from retrieved nodes
        relevant_pages: set[tuple[str, int]] = set()
        for nws in source_nodes:
            meta = nws.node.metadata
            source = meta.get("source", "")
            page = meta.get("page_number", -1)
            if source and page >= 0:
                relevant_pages.add((source, page))

        # Return only charts whose (source, page) is in the relevant set
        return [
            c for c in self._chart_nodes
            if (c.source, c.page_number) in relevant_pages
        ]


# ── Utility Functions ─────────────────────────────────────────────────────────

def _apply_filters(
    nodes: list[NodeWithScore], filters: dict
) -> list[NodeWithScore]:
    """Filter retrieved nodes by metadata key=value pairs.

    Example: filters={"source": "apple_10k.pdf"} keeps only nodes
    from the Apple 10-K document.

    Like Filter DataTable in UiPath — keeps rows that match all conditions.
    """
    result = []
    for nws in nodes:
        meta = nws.node.metadata
        # Keep node only if ALL filter conditions match
        if all(meta.get(k) == v for k, v in filters.items()):
            result.append(nws)
    return result
