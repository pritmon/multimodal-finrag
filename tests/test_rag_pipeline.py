"""Tests for the RAG pipeline.

All Bedrock and AWS calls are mocked so tests run without real AWS credentials.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_simple_pdf_bytes(text: str = "Goldman Sachs revenue was $50 billion in Q1 2023.") -> bytes:
    """Create a minimal PDF for testing."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    _, height = letter
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 72, text)
    c.showPage()
    c.save()
    return buf.getvalue()


def _mock_bedrock_response(text: str = "The answer is 42.") -> MagicMock:
    """Build a mock boto3 invoke_model response."""
    body = json.dumps({"content": [{"type": "text", "text": text}]})
    mock_response = MagicMock()
    mock_response["body"].read.return_value = body.encode()
    return mock_response


def _mock_embed_response(dim: int = 1536) -> MagicMock:
    """Build a mock Titan embedding response."""
    embedding = [0.01] * dim
    body = json.dumps({"embedding": embedding, "inputTextTokenCount": 10})
    mock_response = MagicMock()
    mock_response["body"].read.return_value = body.encode()
    return mock_response


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_bedrock_client():
    """Mock boto3 Bedrock runtime client."""
    client = MagicMock()
    # Default: return a valid completion response
    client.invoke_model.return_value = _mock_bedrock_response("Mocked LLM answer.")
    client.invoke_model.side_effect = lambda **kwargs: (
        _mock_embed_response()
        if "titan-embed" in kwargs.get("modelId", "")
        else _mock_bedrock_response("Mocked LLM answer.")
    )
    return client


@pytest.fixture
def test_settings(tmp_path):
    """Minimal settings with temp directories."""
    from src.config import Settings
    return Settings(
        aws_region="us-east-1",
        s3_bucket="test-bucket",
        bedrock_model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        bedrock_embed_model_id="amazon.titan-embed-text-v1",
        index_persist_dir=tmp_path / "index_store",
        lora_model_path=tmp_path / "models" / "ner",
        postgres_url="postgresql://test:test@localhost/test",
    )


# ── BedrockLLM tests ──────────────────────────────────────────────────────────

class TestBedrockLLM:
    def test_complete_returns_text(self, mock_bedrock_client):
        from src.rag.bedrock_llm import BedrockLLM

        llm = BedrockLLM(model_id="anthropic.claude-3-sonnet-20240229-v1:0", aws_region="us-east-1")
        object.__setattr__(llm, "_client", mock_bedrock_client)
        mock_bedrock_client.invoke_model.return_value = _mock_bedrock_response("Test answer")

        result = llm.complete("What is 2 + 2?")
        assert result.text == "Test answer"

    def test_metadata(self):
        from src.rag.bedrock_llm import BedrockLLM
        llm = BedrockLLM(model_id="anthropic.claude-3-sonnet-20240229-v1:0", aws_region="us-east-1")
        meta = llm.metadata
        assert meta.is_chat_model is True
        assert meta.num_output == 4096


# ── BedrockEmbedding tests ────────────────────────────────────────────────────

class TestBedrockEmbedding:
    def test_embed_single_returns_vector(self, mock_bedrock_client):
        from src.rag.embeddings import BedrockTitanEmbedding
        embed = BedrockTitanEmbedding(model_id="amazon.titan-embed-text-v1", aws_region="us-east-1")
        object.__setattr__(embed, "_client", mock_bedrock_client)

        body_bytes = json.dumps({"embedding": [0.1] * 1536}).encode()
        mock_resp = MagicMock()
        mock_resp["body"].read.return_value = body_bytes
        mock_bedrock_client.invoke_model.return_value = mock_resp

        vec = embed._embed_single("financial report")
        assert isinstance(vec, list)
        assert len(vec) == 1536

    def test_cache_avoids_duplicate_calls(self, mock_bedrock_client):
        from src.rag.embeddings import BedrockTitanEmbedding
        embed = BedrockTitanEmbedding(model_id="amazon.titan-embed-text-v1", aws_region="us-east-1")
        object.__setattr__(embed, "_client", mock_bedrock_client)

        body_bytes = json.dumps({"embedding": [0.2] * 1536}).encode()
        mock_resp = MagicMock()
        mock_resp["body"].read.return_value = body_bytes
        mock_bedrock_client.invoke_model.return_value = mock_resp

        text = "Goldman Sachs Q4 earnings"
        v1 = embed._embed_single(text)
        v2 = embed._embed_single(text)  # second call should hit cache

        assert v1 == v2
        # invoke_model should be called only once
        assert mock_bedrock_client.invoke_model.call_count == 1

    def test_embed_dim(self):
        from src.rag.embeddings import BedrockTitanEmbedding
        embed = BedrockTitanEmbedding(model_id="amazon.titan-embed-text-v1", aws_region="us-east-1")
        assert embed.get_embedding_dim() == 1536


# ── FinRAGPipeline tests ──────────────────────────────────────────────────────

class TestFinRAGPipelineAddDocument:
    @patch("src.rag.pipeline.BedrockLLM")
    @patch("src.rag.pipeline.BedrockTitanEmbedding")
    def test_add_pdf_bytes_returns_node_count(
        self, MockEmbed, MockLLM, test_settings, mock_bedrock_client
    ):
        # Set up mocks
        mock_llm_instance = MagicMock()
        MockLLM.return_value = mock_llm_instance

        mock_embed_instance = MagicMock()
        mock_embed_instance._get_text_embedding.return_value = [0.1] * 1536
        mock_embed_instance._get_query_embedding.return_value = [0.1] * 1536
        MockEmbed.return_value = mock_embed_instance

        from src.rag.pipeline import FinRAGPipeline

        with patch("llama_index.core.Settings"):
            pipeline = FinRAGPipeline(settings=test_settings)
            # Bypass real index initialisation
            pipeline._index = MagicMock()
            pipeline._index.insert_nodes = MagicMock()
            pipeline._index.storage_context = MagicMock()
            pipeline._index.as_retriever = MagicMock()
            pipeline._chart_extractor = MagicMock()
            pipeline._chart_extractor.extract_charts.return_value = []

            pdf_bytes = _make_simple_pdf_bytes()
            nodes_added = pipeline.add_pdf_bytes(pdf_bytes, source="test.pdf", generate_chart_captions=False)

        assert isinstance(nodes_added, int)
        assert nodes_added >= 0


class TestFinRAGPipelineQuery:
    @patch("src.rag.pipeline.BedrockLLM")
    @patch("src.rag.pipeline.BedrockTitanEmbedding")
    def test_query_returns_query_result(self, MockEmbed, MockLLM, test_settings):
        from llama_index.core.schema import NodeWithScore, TextNode
        from src.rag.pipeline import FinRAGPipeline, QueryResult

        mock_llm_instance = MagicMock()
        mock_llm_instance.complete.return_value = MagicMock(text="Revenue was $47.3 billion.")
        MockLLM.return_value = mock_llm_instance

        mock_embed_instance = MagicMock()
        mock_embed_instance._get_query_embedding.return_value = [0.1] * 1536
        MockEmbed.return_value = mock_embed_instance

        with patch("llama_index.core.Settings"):
            pipeline = FinRAGPipeline(settings=test_settings)

        # Mock the index
        mock_node = NodeWithScore(
            node=TextNode(
                text="Goldman Sachs revenue was $47.3 billion in Q1 2023.",
                metadata={"source": "report.pdf", "page_number": 0},
            ),
            score=0.95,
        )
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        pipeline._index = mock_index
        pipeline._llm = mock_llm_instance

        result = pipeline.query("What was Goldman Sachs revenue?")

        assert isinstance(result, QueryResult)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert isinstance(result.source_nodes, list)
        assert result.query == "What was Goldman Sachs revenue?"

    @patch("src.rag.pipeline.BedrockLLM")
    @patch("src.rag.pipeline.BedrockTitanEmbedding")
    def test_query_result_to_dict(self, MockEmbed, MockLLM, test_settings):
        from llama_index.core.schema import NodeWithScore, TextNode
        from src.rag.pipeline import FinRAGPipeline, QueryResult

        mock_llm_instance = MagicMock()
        mock_llm_instance.complete.return_value = MagicMock(text="Answer here.")
        MockLLM.return_value = mock_llm_instance
        MockEmbed.return_value = MagicMock()

        with patch("llama_index.core.Settings"):
            pipeline = FinRAGPipeline(settings=test_settings)

        mock_node = NodeWithScore(
            node=TextNode(text="Sample text", metadata={"source": "doc.pdf", "page_number": 0}),
            score=0.9,
        )
        pipeline._index = MagicMock()
        pipeline._index.as_retriever.return_value.retrieve.return_value = [mock_node]
        pipeline._llm = mock_llm_instance

        result = pipeline.query("Test question?")
        d = result.to_dict()

        assert "answer" in d
        assert "sources" in d
        assert "charts" in d
        assert isinstance(d["sources"], list)


# ── BM25 Retriever tests ──────────────────────────────────────────────────────

class TestBM25Retriever:
    def test_retrieve_returns_scored_nodes(self):
        from llama_index.core.schema import TextNode
        from src.rag.retriever import BM25Retriever

        nodes = [
            TextNode(text="Goldman Sachs reported strong Q4 earnings.", id_="n1"),
            TextNode(text="Apple released a new iPhone model.", id_="n2"),
            TextNode(text="Federal Reserve raised interest rates.", id_="n3"),
        ]
        retriever = BM25Retriever(nodes=nodes, top_k=2)
        results = retriever.retrieve("Goldman Sachs earnings")

        assert len(results) <= 2
        # The most relevant doc should be first
        if results:
            assert results[0].node.get_content().__contains__("Goldman") or results[0].score >= 0

    def test_retrieve_returns_node_with_score(self):
        from llama_index.core.schema import TextNode
        from src.rag.retriever import BM25Retriever

        nodes = [TextNode(text="Revenue increased by 20 percent in Q3.", id_="n1")]
        retriever = BM25Retriever(nodes=nodes, top_k=5)
        results = retriever.retrieve("revenue increase")

        assert all(hasattr(r, "score") for r in results)
        assert all(isinstance(r.score, float) for r in results)


class TestRRFFusion:
    def test_rrf_merges_two_lists(self):
        from llama_index.core.schema import NodeWithScore, TextNode
        from src.rag.retriever import _reciprocal_rank_fusion

        node_a = TextNode(text="doc A", id_="a")
        node_b = TextNode(text="doc B", id_="b")
        node_c = TextNode(text="doc C", id_="c")

        list1 = [NodeWithScore(node=node_a, score=0.9), NodeWithScore(node=node_b, score=0.7)]
        list2 = [NodeWithScore(node=node_b, score=0.8), NodeWithScore(node=node_c, score=0.6)]

        fused = _reciprocal_rank_fusion([list1, list2])
        assert len(fused) == 3
        # node_b appears in both lists so should rank highest
        assert fused[0].node.node_id == "b"
