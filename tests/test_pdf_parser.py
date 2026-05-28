"""Tests for src/ingestion/pdf_parser.py.

Uses reportlab to create synthetic PDFs in fixtures so no real files are needed.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.pdf_parser import BoundingBox, PDFParser, ParsedDocument, TextBlock


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_simple_pdf(text_lines: list[str] = None) -> bytes:
    """Create a minimal single-page PDF using reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    if text_lines is None:
        text_lines = [
            "Annual Report 2023",
            "Goldman Sachs reported revenue of $47.3 billion.",
            "Net income increased 15% year-over-year.",
            "Operating margin improved to 32.4%.",
        ]

    y = height - 72  # 1 inch from top
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, y, text_lines[0])

    c.setFont("Helvetica", 12)
    for line in text_lines[1:]:
        y -= 24
        c.drawString(72, y, line)

    c.showPage()
    c.save()
    return buf.getvalue()


def _make_multipage_pdf(num_pages: int = 3) -> bytes:
    """Create a multi-page PDF with varying content."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    _, height = letter

    for page_num in range(num_pages):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 72, f"Page {page_num + 1} of {num_pages}")
        c.setFont("Helvetica", 12)
        c.drawString(72, height - 120, f"Financial data for Q{page_num + 1} 2023")
        c.drawString(72, height - 150, f"Revenue: ${(page_num + 1) * 12.5:.1f} billion")
        c.showPage()

    c.save()
    return buf.getvalue()


@pytest.fixture
def simple_pdf_bytes() -> bytes:
    return _make_simple_pdf()


@pytest.fixture
def multipage_pdf_bytes() -> bytes:
    return _make_multipage_pdf(num_pages=3)


@pytest.fixture
def pdf_parser() -> PDFParser:
    return PDFParser(dpi=72, min_image_size=(10, 10))


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_dimensions(self):
        bbox = BoundingBox(x0=10, y0=20, x1=110, y1=70)
        assert bbox.width == 100
        assert bbox.height == 50

    def test_to_dict(self):
        bbox = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        d = bbox.to_dict()
        assert set(d.keys()) == {"x0", "y0", "x1", "y1"}
        assert d["x1"] == 100


class TestPDFParserBasic:
    def test_parse_returns_parsed_document(self, pdf_parser, simple_pdf_bytes):
        result = pdf_parser.parse_bytes(simple_pdf_bytes, source="test.pdf")
        assert isinstance(result, ParsedDocument)

    def test_source_is_set(self, pdf_parser, simple_pdf_bytes):
        result = pdf_parser.parse_bytes(simple_pdf_bytes, source="annual_report.pdf")
        assert result.source == "annual_report.pdf"

    def test_num_pages(self, pdf_parser, simple_pdf_bytes):
        result = pdf_parser.parse_bytes(simple_pdf_bytes, source="test.pdf")
        assert result.num_pages == 1

    def test_text_blocks_extracted(self, pdf_parser, simple_pdf_bytes):
        result = pdf_parser.parse_bytes(simple_pdf_bytes, source="test.pdf")
        assert len(result.text_blocks) > 0

    def test_text_blocks_have_correct_types(self, pdf_parser, simple_pdf_bytes):
        result = pdf_parser.parse_bytes(simple_pdf_bytes, source="test.pdf")
        for block in result.text_blocks:
            assert isinstance(block, TextBlock)
            assert isinstance(block.text, str)
            assert block.text.strip()  # no empty blocks
            assert isinstance(block.page_number, int)
            assert block.page_number >= 0
            assert isinstance(block.bbox, BoundingBox)

    def test_financial_text_content(self, pdf_parser, simple_pdf_bytes):
        result = pdf_parser.parse_bytes(simple_pdf_bytes, source="test.pdf")
        full_text = result.full_text
        assert "Goldman Sachs" in full_text or "revenue" in full_text.lower()

    def test_metadata_populated(self, pdf_parser, simple_pdf_bytes):
        result = pdf_parser.parse_bytes(simple_pdf_bytes, source="test.pdf")
        assert "num_pages" in result.metadata
        assert result.metadata["num_pages"] == 1

    def test_full_text_concatenation(self, pdf_parser, simple_pdf_bytes):
        result = pdf_parser.parse_bytes(simple_pdf_bytes, source="test.pdf")
        full = result.full_text
        assert isinstance(full, str)
        assert len(full) > 10


class TestPDFParserMultipage:
    def test_multipage_page_count(self, pdf_parser, multipage_pdf_bytes):
        result = pdf_parser.parse_bytes(multipage_pdf_bytes, source="multi.pdf")
        assert result.num_pages == 3

    def test_blocks_span_multiple_pages(self, pdf_parser, multipage_pdf_bytes):
        result = pdf_parser.parse_bytes(multipage_pdf_bytes, source="multi.pdf")
        page_numbers = {b.page_number for b in result.text_blocks}
        # Should have blocks on at least 2 different pages
        assert len(page_numbers) >= 2

    def test_blocks_for_page_filter(self, pdf_parser, multipage_pdf_bytes):
        result = pdf_parser.parse_bytes(multipage_pdf_bytes, source="multi.pdf")
        page_0_blocks = result.blocks_for_page(0)
        page_1_blocks = result.blocks_for_page(1)
        # Every returned block must be on the requested page
        assert all(b.page_number == 0 for b in page_0_blocks)
        assert all(b.page_number == 1 for b in page_1_blocks)

    def test_correct_page_content(self, pdf_parser, multipage_pdf_bytes):
        result = pdf_parser.parse_bytes(multipage_pdf_bytes, source="multi.pdf")
        page_0_text = " ".join(b.text for b in result.blocks_for_page(0))
        assert "Page 1" in page_0_text or "Q1" in page_0_text or "Revenue" in page_0_text


class TestPDFParserFromFile:
    def test_parse_file(self, pdf_parser, tmp_path, simple_pdf_bytes):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(simple_pdf_bytes)
        result = pdf_parser.parse_file(pdf_file)
        assert result.num_pages == 1
        assert len(result.text_blocks) > 0
        assert result.source == str(pdf_file)

    def test_missing_file_raises(self, pdf_parser):
        with pytest.raises(FileNotFoundError):
            pdf_parser.parse_file("/nonexistent/path/file.pdf")


class TestHeadingDetection:
    def test_heading_detected_for_large_font(self, pdf_parser):
        """Create a PDF where the title has a noticeably larger font."""
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        _, height = letter

        c.setFont("Helvetica-Bold", 24)  # Large font → heading
        c.drawString(72, height - 72, "ANNUAL REPORT 2023")
        c.setFont("Helvetica", 10)       # Small font → body
        c.drawString(72, height - 120, "This is normal body text with a small font size.")
        c.showPage()
        c.save()
        pdf_bytes = buf.getvalue()

        result = pdf_parser.parse_bytes(pdf_bytes, source="heading_test.pdf")
        headings = [b for b in result.text_blocks if b.is_heading]
        body_blocks = [b for b in result.text_blocks if not b.is_heading]

        # At least one heading should be detected
        assert len(headings) >= 1
        # The heading text should contain the title
        heading_text = " ".join(b.text for b in headings)
        assert "ANNUAL" in heading_text or "REPORT" in heading_text
