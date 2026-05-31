"""PDF text and image extraction using PyMuPDF (fitz).

Extracts structured text blocks with page/bounding-box metadata and rasterises
embedded images for downstream multimodal processing.

HOW IT WORKS (simple analogy):
  Think of this like a UiPath workflow that reads a PDF file and returns two things:
  1. All text paragraphs — with the page number and position on the page
  2. All images — with the page number and pixel data

  The final output (ParsedDocument) is like a DataTable containing every
  paragraph and every image from the whole PDF, neatly organised by page.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF — the library that reads PDF files
from PIL import Image  # Pillow — used to handle image pixel data

logger = logging.getLogger(__name__)


# ── Data Classes (like UiPath Custom Types / DataRow structures) ───────────────

@dataclass
class BoundingBox:
    """Stores the position of a text block or image on a PDF page.

    PDF coordinates use points (72 points = 1 inch).
    (x0, y0) = top-left corner
    (x1, y1) = bottom-right corner

    Think of it like X,Y coordinates in UiPath's Click activity.
    """

    x0: float  # left edge of the box
    y0: float  # top edge of the box
    x1: float  # right edge of the box
    y1: float  # bottom edge of the box

    @classmethod
    def from_fitz_rect(cls, rect: fitz.Rect) -> "BoundingBox":
        """Convert a PyMuPDF Rect object into our BoundingBox format."""
        return cls(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1)

    @property
    def width(self) -> float:
        """Width of the box in PDF points."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Height of the box in PDF points."""
        return self.y1 - self.y0

    def to_dict(self) -> dict:
        """Convert to a plain dictionary — useful for JSON serialisation."""
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class TextBlock:
    """Represents one paragraph (block) of text extracted from a PDF page.

    Like a single DataRow in UiPath — it holds all properties of one text chunk:
    - the actual text content
    - which page it came from
    - where on the page it sits (bounding box)
    - whether it looks like a heading (larger font than surrounding text)
    """

    text: str               # the actual paragraph text
    page_number: int        # 0-indexed page number (page 1 = index 0)
    block_number: int       # position of this block within the page
    bbox: BoundingBox       # where on the page this block appears
    source: str             # filename or S3 key this came from
    block_type: str = "text"           # "text" or "image"
    font_sizes: list[float] = field(default_factory=list)  # font sizes of all spans in this block
    is_heading: bool = False           # True if font is significantly larger than average

    def to_dict(self) -> dict:
        """Serialise to a plain dict (used in API responses and logging)."""
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
    """Represents one image extracted from a PDF page.

    Stores both the pixel data (PIL Image) and where the image was on the page.
    The xref is PyMuPDF's internal reference number — like a row ID in a database.
    """

    image: Image.Image     # the actual image pixels (PIL format)
    page_number: int       # which page the image appeared on
    image_index: int       # position within that page's image list
    bbox: Optional[BoundingBox]  # where on the page the image was placed
    source: str            # filename or S3 key
    xref: int              # PyMuPDF internal cross-reference number

    def to_bytes(self, fmt: str = "PNG") -> bytes:
        """Convert the PIL Image to raw bytes for storage or transmission."""
        buf = io.BytesIO()
        self.image.save(buf, format=fmt)
        return buf.getvalue()


@dataclass
class ParsedDocument:
    """The complete result of parsing one PDF file.

    Think of this as the final output variable of the whole UiPath workflow —
    it bundles together everything extracted from the PDF:
    - All text paragraphs (text_blocks)
    - All images (images)
    - Document metadata (title, author, page count)
    """

    source: str                         # filename or S3 key of the original PDF
    num_pages: int                      # total number of pages in the document
    text_blocks: list[TextBlock]        # all text paragraphs from all pages
    images: list[EmbeddedImage]         # all images from all pages
    metadata: dict = field(default_factory=dict)  # PDF metadata (title, author, etc.)

    @property
    def full_text(self) -> str:
        """Concatenate all text blocks in reading order into one big string.

        Useful for quick inspection or full-text search.
        Skips any blocks that are empty after stripping whitespace.
        """
        return "\n\n".join(block.text for block in self.text_blocks if block.text.strip())

    def blocks_for_page(self, page_number: int) -> list[TextBlock]:
        """Return only the text blocks that belong to a specific page.

        Example: result.blocks_for_page(0) → all paragraphs on page 1
        """
        return [b for b in self.text_blocks if b.page_number == page_number]

    def images_for_page(self, page_number: int) -> list[EmbeddedImage]:
        """Return only the images that belong to a specific page."""
        return [img for img in self.images if img.page_number == page_number]


# ── Main Parser Class ─────────────────────────────────────────────────────────

class PDFParser:
    """Extract structured text and images from PDF files using PyMuPDF.

    This is the main "worker" class — think of it as your UiPath Workflow file
    that does all the actual reading and processing.

    Parameters
    ----------
    dpi:
        Resolution for rasterising embedded images.
        Higher = better quality but more memory. Default 150 is a good balance.
    min_image_size:
        Skip images smaller than (width, height) in pixels.
        Filters out tiny logos, decorators, and noise. Default: 50x50 pixels.
    heading_font_ratio:
        A text block is marked as a heading if its largest font is at least
        this multiple of the page's median font size.
        Default 1.3 means "30% bigger than average = heading".
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

    # ── Public API — two entry points depending on where the PDF comes from ────

    def parse_file(self, path: str | Path) -> ParsedDocument:
        """Parse a PDF file stored on disk.

        Use this when the PDF is already saved locally.
        Reads raw bytes from disk and delegates to parse_bytes().
        """
        path = Path(path)
        logger.info("Parsing PDF: %s", path)
        with open(path, "rb") as fh:
            raw = fh.read()  # read the whole file as bytes
        return self.parse_bytes(raw, source=str(path))

    def parse_bytes(self, pdf_bytes: bytes, source: str = "unknown") -> ParsedDocument:
        """Parse a PDF from an in-memory byte string.

        Use this when the PDF arrives as bytes — e.g. from an HTTP upload
        or downloaded from S3. This is the core parsing method.

        Steps:
          1. Open the PDF bytes with PyMuPDF (fitz)
          2. Read metadata (title, author, page count)
          3. For each page: extract all text blocks + all images
          4. Return the combined ParsedDocument
        """
        # Open the PDF from raw bytes — fitz.open() works with both files and bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_blocks: list[TextBlock] = []
        images: list[EmbeddedImage] = []

        # Extract document-level metadata (title, author, etc.)
        pdf_metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "num_pages": len(doc),
        }

        # Loop through every page — like a For Each loop in UiPath
        for page_idx, page in enumerate(doc):
            # Extract all text paragraphs from this page
            page_blocks = self._extract_text_blocks(page, page_idx, source)
            text_blocks.extend(page_blocks)

            # Extract all images from this page
            page_images = self._extract_images(doc, page, page_idx, source)
            images.extend(page_images)

        doc.close()  # always close the document to free memory
        logger.info(
            "Parsed %s: %d pages, %d text blocks, %d images",
            source,
            pdf_metadata["num_pages"],
            len(text_blocks),
            len(images),
        )
        # Bundle everything into a ParsedDocument and return
        return ParsedDocument(
            source=source,
            num_pages=pdf_metadata["num_pages"],
            text_blocks=text_blocks,
            images=images,
            metadata=pdf_metadata,
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _extract_text_blocks(
        self, page: fitz.Page, page_idx: int, source: str
    ) -> list[TextBlock]:
        """Extract all text paragraphs from a single PDF page.

        PyMuPDF returns the page content as a nested dictionary:
          page → blocks → lines → spans
        A "span" is the smallest unit: a run of text with consistent font/size.
        We flatten spans → lines → blocks into simple TextBlock objects.

        Heading detection:
          Calculate the median font size of the whole page.
          Any block whose largest font is >= 1.3x the median is flagged as a heading.
          (Like flagging DataTable rows where Value > 1.3 * Average)
        """
        # Get the page content as a structured dictionary
        # "dict" format gives us blocks, lines, spans with font info
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        result: list[TextBlock] = []

        # Step 1: Collect all font sizes on this page to compute the median
        # This is needed for heading detection
        all_sizes: list[float] = []
        for blk in blocks:
            if blk.get("type") != 0:  # type 0 = text block, type 1 = image block
                continue
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    all_sizes.append(span.get("size", 12.0))

        # Compute the median font size — "what is the typical font size on this page?"
        median_size = _median(all_sizes) if all_sizes else 12.0

        # Step 2: Build a TextBlock for each text block on the page
        for blk_idx, blk in enumerate(blocks):
            # Skip image blocks (type 1) — we handle images separately
            if blk.get("type") != 0:
                continue

            lines_text: list[str] = []   # collect all lines of text in this block
            font_sizes: list[float] = [] # collect all font sizes in this block

            # Walk through lines → spans to extract the actual text
            for line in blk.get("lines", []):
                line_parts: list[str] = []
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if span_text:
                        line_parts.append(span_text)
                        font_sizes.append(span.get("size", 12.0))
                if line_parts:
                    lines_text.append(" ".join(line_parts))

            # Join all lines of this block into one string
            combined = "\n".join(lines_text).strip()
            if not combined:
                continue  # skip empty blocks (whitespace only)

            # Heading check: is the largest font in this block >= 1.3x median?
            is_heading = bool(
                font_sizes
                and max(font_sizes) >= median_size * self.heading_font_ratio
            )

            # Build the BoundingBox from PyMuPDF's bbox tuple
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
        """Extract and return all images embedded on a single PDF page.

        For each image:
          1. Get the raw image bytes using the xref (cross-reference number)
          2. Convert to a PIL Image (RGB format, easy to work with)
          3. Skip tiny images (smaller than min_image_size — likely logos/noise)
          4. Look up where the image sits on the page (bounding box)
          5. Return as an EmbeddedImage object

        'xref' is PyMuPDF's internal ID for each image — like a primary key.
        """
        result: list[EmbeddedImage] = []
        # Get a list of all images on this page (includes metadata like xref)
        image_list = page.get_images(full=True)

        # Build a mapping from xref → bounding box on this page
        # (where on the page is each image placed?)
        xref_bbox: dict[int, fitz.Rect] = {}
        for img_info in page.get_image_info():
            xref_bbox[img_info.get("xref", -1)] = fitz.Rect(img_info.get("bbox", (0, 0, 0, 0)))

        for img_idx, img_data in enumerate(image_list):
            xref = img_data[0]  # xref is the first element in the image tuple
            try:
                # Extract the raw image bytes from the PDF
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]

                # Convert raw bytes to a PIL Image in RGB mode
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                # Skip tiny images — they're usually decorative or noise
                w, h = pil_img.size
                if w < self.min_image_size[0] or h < self.min_image_size[1]:
                    logger.debug("Skipping tiny image xref=%d (%dx%d)", xref, w, h)
                    continue

                # Look up where this image sits on the page
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
                # Log a warning but continue — one bad image shouldn't stop the whole parse
                logger.warning("Failed to extract image xref=%d on page %d: %s", xref, page_idx, exc)

        return result


# ── Utility Functions ─────────────────────────────────────────────────────────

def _median(values: list[float]) -> float:
    """Compute the median of a list of floats.

    Used for heading detection — we compare each font size against the
    median font size of the page.

    Example: [10, 12, 14, 18] → median = 13.0
    """
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    # If even number of values, average the two middle ones
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]
