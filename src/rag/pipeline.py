"""LlamaIndex multimodal RAG pipeline for financial documents.

Orchestrates:
- Document ingestion (text + chart nodes)
- MultiModalVectorStoreIndex construction
- Hybrid retrieval (BM25 + vector + reranking)
- Bedrock Claude 3 generation with source citations
- Index persistence and incremental updates
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from llama_index.core import (
    Settings as LlamaSettings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import (
    Document,
    ImageDocument,
    NodeWithScore,
    TextNode,
)

from src.config import Settings, get_settings
from src.ingestion.chart_extractor import ChartExtractor, ChartNode
from src.ingestion.pdf_parser import ParsedDocument, PDFParser

from .bedrock_llm import BedrockLLM
from .embeddings import BedrockTitanEmbedding
from .retriever import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Structured result from the RAG pipeline."""

    answer: str
    source_nodes: list[NodeWithScore]
    chart_nodes: list[ChartNode]
    query: str
    model_id: str

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "query": self.query,
            "model_id": self.model_id,
            "sources": [
                {
                    "text": n.node.get_content()[:500],
                    "score": n.score,
                    "metadata": n.node.metadata,
                }
                for n in self.source_nodes
            ],
            "charts": [c.to_dict() for c in self.chart_nodes],
        }


class FinRAGPipeline:
    """End-to-end multimodal RAG pipeline for financial document intelligence.

    Parameters
    ----------
    settings:
        Application settings (uses cached singleton if None).
    pdf_parser:
        PDF parser instance (created from settings if None).
    chart_extractor:
        Chart extractor instance (created on demand if None; lazy to avoid
        loading CLIP at startup in environments that don't need it).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        pdf_parser: Optional[PDFParser] = None,
        chart_extractor: Optional[ChartExtractor] = None,
    ) -> None:
        self.cfg = settings or get_settings()
        self._pdf_parser = pdf_parser or PDFParser()
        self._chart_extractor = chart_extractor  # lazily initialised

        # Configure LlamaIndex global settings
        self._llm = BedrockLLM(
            model_id=self.cfg.bedrock_model_id,
            aws_region=self.cfg.aws_region,
            max_tokens=self.cfg.bedrock_max_tokens,
            temperature=self.cfg.bedrock_temperature,
        )
        self._embed_model = BedrockTitanEmbedding(
            model_id=self.cfg.bedrock_embed_model_id,
            aws_region=self.cfg.aws_region,
        )

        LlamaSettings.llm = self._llm
        LlamaSettings.embed_model = self._embed_model
        LlamaSettings.chunk_size = self.cfg.chunk_size
        LlamaSettings.chunk_overlap = self.cfg.chunk_overlap

        self._node_parser = SentenceSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )

        # Index state
        self._index: Optional[VectorStoreIndex] = None
        self._all_nodes: list[TextNode] = []
        self._chart_nodes: list[ChartNode] = []
        self._retriever: Optional[HybridRetriever] = None

    # ── Index management ──────────────────────────────────────────────────────

    def load_or_build_index(self, force_rebuild: bool = False) -> None:
        """Load index from disk if it exists, otherwise build a new one."""
        persist_dir = self.cfg.index_persist_dir
        index_file = persist_dir / "docstore.json"

        if index_file.exists() and not force_rebuild:
            logger.info("Loading existing index from %s", persist_dir)
            storage_ctx = StorageContext.from_defaults(persist_dir=str(persist_dir))
            self._index = load_index_from_storage(storage_ctx)
            self._rebuild_retriever()
        else:
            logger.info("Building new index (force_rebuild=%s)", force_rebuild)
            self._index = VectorStoreIndex(nodes=[], embed_model=self._embed_model)
            self._persist_index()

    def add_parsed_document(
        self, parsed_doc: ParsedDocument, generate_chart_captions: bool = True
    ) -> int:
        """Add a parsed document to the index.

        Parameters
        ----------
        parsed_doc:
            Result of ``PDFParser.parse_bytes()`` or ``PDFParser.parse_file()``.
        generate_chart_captions:
            If True, generate Bedrock captions for detected charts.

        Returns
        -------
        int
            Number of nodes added to the index.
        """
        if self._index is None:
            self.load_or_build_index()

        # Build LlamaIndex Documents from text blocks
        documents: list[Document] = []
        for block in parsed_doc.text_blocks:
            if not block.text.strip():
                continue
            documents.append(
                Document(
                    text=block.text,
                    metadata={
                        "source": parsed_doc.source,
                        "page_number": block.page_number,
                        "block_number": block.block_number,
                        "is_heading": block.is_heading,
                        **parsed_doc.metadata,
                    },
                )
            )

        # Parse into nodes
        nodes = self._node_parser.get_nodes_from_documents(documents)
        logger.info("Parsed %d documents → %d nodes for %s", len(documents), len(nodes), parsed_doc.source)

        # Extract and caption charts
        if parsed_doc.images:
            extractor = self._get_chart_extractor()
            new_charts = extractor.extract_charts(
                parsed_doc.images, generate_captions=generate_chart_captions
            )
            self._chart_nodes.extend(new_charts)

            # Add chart captions as text nodes so they're searchable
            for chart in new_charts:
                if chart.caption:
                    nodes.append(
                        TextNode(
                            text=f"[Chart on page {chart.page_number + 1}] {chart.caption}",
                            metadata={
                                "source": parsed_doc.source,
                                "page_number": chart.page_number,
                                "node_type": "chart_caption",
                                "chart_type": chart.chart_type,
                            },
                        )
                    )

        # Insert into index
        for node in nodes:
            self._index.insert_nodes([node])  # type: ignore[arg-type]

        self._all_nodes.extend(nodes)  # type: ignore[arg-type]
        self._rebuild_retriever()
        self._persist_index()

        return len(nodes)

    def add_pdf_bytes(
        self, pdf_bytes: bytes, source: str = "unknown", **kwargs: Any
    ) -> int:
        """Parse PDF bytes and add to the index."""
        parsed = self._pdf_parser.parse_bytes(pdf_bytes, source=source)
        return self.add_parsed_document(parsed, **kwargs)

    def add_pdf_file(self, path: str | Path, **kwargs: Any) -> int:
        """Parse a PDF file and add to the index."""
        parsed = self._pdf_parser.parse_file(path)
        return self.add_parsed_document(parsed, **kwargs)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> QueryResult:
        """Run a RAG query and return a structured result.

        Parameters
        ----------
        question:
            Natural-language question.
        top_k:
            Override the default retrieval top-k.
        filters:
            Metadata filters (currently applied post-retrieval).
        """
        if self._index is None:
            self.load_or_build_index()

        effective_top_k = top_k or self.cfg.retriever_top_k

        # Retrieve nodes
        if self._retriever is not None:
            from llama_index.core import QueryBundle
            source_nodes = self._retriever.retrieve(QueryBundle(query_str=question))
        else:
            retriever = self._index.as_retriever(similarity_top_k=effective_top_k)
            source_nodes = retriever.retrieve(question)

        # Apply metadata filters
        if filters:
            source_nodes = _apply_filters(source_nodes, filters)

        # Find charts referenced by retrieved text
        relevant_charts = self._find_relevant_charts(source_nodes)

        # Build augmented context
        context_parts = [f"[Source {i+1}]\n{n.node.get_content()}" for i, n in enumerate(source_nodes)]
        for chart in relevant_charts:
            context_parts.append(f"[Chart] {chart.caption}")

        context = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are a financial document analyst. Answer the user's question using only "
            "the provided context from financial documents. Be precise and cite source numbers "
            "when referencing specific data. If the context does not contain enough information "
            "to answer the question, say so explicitly."
        )
        full_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        # Generate answer with images if charts are available
        if relevant_charts:
            from PIL import Image
            import io
            images = [Image.open(io.BytesIO(c.image_bytes)) for c in relevant_charts]
            answer = self._llm.complete_with_images(full_prompt, images=images)
        else:
            answer = self._llm.complete(full_prompt).text

        return QueryResult(
            answer=answer,
            source_nodes=source_nodes,
            chart_nodes=relevant_charts,
            query=question,
            model_id=self.cfg.bedrock_model_id,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _rebuild_retriever(self) -> None:
        if not self._all_nodes or self._index is None:
            return
        vector_retriever = self._index.as_retriever(
            similarity_top_k=self.cfg.retriever_top_k
        )
        self._retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            nodes=self._all_nodes,
            top_k=self.cfg.retriever_top_k,
            reranker_top_n=self.cfg.reranker_top_n,
        )

    def _persist_index(self) -> None:
        if self._index is not None:
            self._index.storage_context.persist(persist_dir=str(self.cfg.index_persist_dir))
            logger.info("Index persisted to %s", self.cfg.index_persist_dir)

    def _get_chart_extractor(self) -> ChartExtractor:
        if self._chart_extractor is None:
            self._chart_extractor = ChartExtractor(
                bedrock_model_id=self.cfg.bedrock_model_id,
                aws_region=self.cfg.aws_region,
            )
        return self._chart_extractor

    def _find_relevant_charts(
        self, source_nodes: list[NodeWithScore]
    ) -> list[ChartNode]:
        """Return charts that share a page with any retrieved text node."""
        relevant_pages: set[tuple[str, int]] = set()
        for nws in source_nodes:
            meta = nws.node.metadata
            source = meta.get("source", "")
            page = meta.get("page_number", -1)
            if source and page >= 0:
                relevant_pages.add((source, page))

        return [
            c for c in self._chart_nodes
            if (c.source, c.page_number) in relevant_pages
        ]


# ── Utilities ─────────────────────────────────────────────────────────────────

def _apply_filters(
    nodes: list[NodeWithScore], filters: dict
) -> list[NodeWithScore]:
    """Filter nodes by metadata key=value pairs."""
    result = []
    for nws in nodes:
        meta = nws.node.metadata
        if all(meta.get(k) == v for k, v in filters.items()):
            result.append(nws)
    return result
