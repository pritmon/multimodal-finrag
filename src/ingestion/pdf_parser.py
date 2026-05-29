"""PDF text and image extraction using PyMuPDF (fitz).

Extracts structured text blocks with page/bounding-box metadata and rasterises
embedded images for downstream multimodal processing.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in PDF points (72 pts per inch)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_fitz_rect(cls, rect: fitz.Rect) -> "BoundingBox":
        return cls(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class TextBlock:
    """A contiguous block of text extracted from a PDF page."""

    text: str
    page_number: int          # 0-indexed
    block_number: int
    bbox: BoundingBox
    source: str               # filename / S3 key
    block_type: str = "text"  # "text" or "image"
    font_sizes: list[float] = field(default_factory=list)
    is_heading: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "page_number": self.page_number,
            "block_number": self.block_number,
            "bbox": self.bbox.to_dict(),
            "source": self.source,
            "block_type": self.block_type,
            "is_heading": self.is_heading,
        }


@dataclass
class EmbeddedImage:
    """An image extracted from a PDF page."""

    image: Image.Image
    page_number: int
    image_index: int
    bbox: Optional[BoundingBox]
    source: str
    xref: int  # PyMuPDF cross-reference number

    def to_bytes(self, fmt: str = "PNG") -> bytes:
        buf = io.BytesIO()
        self.image.save(buf, format=fmt)
        return buf.getvalue()


@dataclass
class ParsedDocument:
    """Result of parsing a single PDF document."""

    source: str
    num_pages: int
    text_blocks: list[TextBlock]
    images: list[EmbeddedImage]
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenate all text blocks in reading order."""
        return "\n\n".join(block.text for block in self.text_blocks if block.text.strip())

    def blocks_for_page(self, page_number: int) -> list[TextBlock]:
        return [b for b in self.text_blocks if b.page_number == page_number]

    def images_for_page(self, page_number: int) -> list[EmbeddedImage]:
        return [img for img in self.images if img.page_number == page_number]


class PDFParser:
    """Extract structured text and images from PDF files using PyMuPDF.

    Parameters
    ----------
    dpi:
        Rasterisation resolution for embedded images (higher = better quality
        but larger memory footprint).
    min_image_size:
        Minimum width × height in pixels to keep an extracted image.
    heading_font_ratio:
        A text span is classified as a heading when its font size is at least
        this ratio above the median font size of the page.
    """

    def __init__(
        self,
        dpi: int = 150,
        min_image_size: tuple[int, int] = (50, 50),
        heading_font_ratio: float = 1.3,
    ) -> None:
        self.dpi = dpi
        self.min_image_size = min_image_size
        self.heading_font_ratio = heading_font_ratio

    # ── Public API ────────────────────────────────────────────────────────────

    def parse_file(self, path: str | Path) -> ParsedDocument:
        """Parse a PDF from disk."""
        path = Path(path)
        logger.info("Parsing PDF: %s", path)
        with open(path, "rb") as fh:
            raw = fh.read()
        return self.parse_bytes(raw, source=str(path))

    def parse_bytes(self, pdf_bytes: bytes, source: str = "unknown") -> ParsedDocument:
        """Parse a PDF from an in-memory byte string."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_blocks: list[TextBlock] = []
        images: list[EmbeddedImage] = []

        pdf_metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "num_pages": len(doc),
        }

        for page_idx, page in enumerate(doc):
            page_blocks = self._extract_text_blocks(page, page_idx, source)
            text_blocks.extend(page_blocks)

            # page_images = self._extract_images(doc, page, page_idx, source)
            # images.extend(page_images)

        doc.close()
        logger.info(
            "Parsed %s: %d pages, %d text blocks, %d images",
            source,
            pdf_metadata["num_pages"],
            len(text_blocks),
            len(images),
        )
        return ParsedDocument(
            source=source,
            num_pages=pdf_metadata["num_pages"],
            text_blocks=text_blocks,
            images=images,
            metadata=pdf_metadata,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_text_blocks(
        self, page: fitz.Page, page_idx: int, source: str
    ) -> list[TextBlock]:
        """Extract text blocks with layout information from a single page."""
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        result: list[TextBlock] = []

        # Compute median font size for heading detection
        all_sizes: list[float] = []
        for blk in blocks:
            if blk.get("type") != 0:
                continue
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    all_sizes.append(span.get("size", 12.0))

        median_size = _median(all_sizes) if all_sizes else 12.0

        for blk_idx, blk in enumerate(blocks):
            if blk.get("type") != 0:  # 0 = text, 1 = image
                continue

            lines_text: list[str] = []
            font_sizes: list[float] = []

            for line in blk.get("lines", []):
                line_parts: list[str] = []
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if span_text:
                        line_parts.append(span_text)
                        font_sizes.append(span.get("size", 12.0))
                if line_parts:
                    lines_text.append(" ".join(line_parts))

            combined = "\n".join(lines_text).strip()
            if not combined:
                continue

            is_heading = bool(
                font_sizes
                and max(font_sizes) >= median_size * self.heading_font_ratio
            )

            bbox = BoundingBox.from_fitz_rect(fitz.Rect(blk["bbox"]))
            result.append(
                TextBlock(
                    text=combined,
                    page_number=page_idx,
                    block_number=blk_idx,
                    bbox=bbox,
                    source=source,
                    block_type="text",
                    font_sizes=font_sizes,
                    is_heading=is_heading,
                )
            )

        return result

    def _extract_images(
        self, doc: fitz.Document, page: fitz.Page, page_idx: int, source: str
    ) -> list[EmbeddedImage]:
        """Rasterise and return images embedded in a PDF page."""
        result: list[EmbeddedImage] = []
        image_list = page.get_images(full=True)

        # Build a map from xref → bbox on this page
        xref_bbox: dict[int, fitz.Rect] = {}
        for img_info in page.get_image_info():
            xref_bbox[img_info.get("xref", -1)] = fitz.Rect(img_info.get("bbox", (0, 0, 0, 0)))

        for img_idx, img_data in enumerate(image_list):
            xref = img_data[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                w, h = pil_img.size
                if w < self.min_image_size[0] or h < self.min_image_size[1]:
                    logger.debug("Skipping tiny image xref=%d (%dx%d)", xref, w, h)
                    continue

                bbox = BoundingBox.from_fitz_rect(xref_bbox.get(xref, fitz.Rect()))
                result.append(
                    EmbeddedImage(
                        image=pil_img,
                        page_number=page_idx,
                        image_index=img_idx,
                        bbox=bbox,
                        source=source,
                        xref=xref,
                    )
                )
            except Exception as exc:
                logger.warning("Failed to extract image xref=%d on page %d: %s", xref, page_idx, exc)

        return result


# ── Utilities ─────────────────────────────────────────────────────────────────

def _median(values: list[float]) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]
