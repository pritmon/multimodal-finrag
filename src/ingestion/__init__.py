"""Document ingestion subpackage: PDF parsing, chart extraction, and S3 I/O."""

from .pdf_parser import ParsedDocument, PDFParser, TextBlock

__all__ = [
    "PDFParser",
    "ParsedDocument",
    "TextBlock",
]

try:
    from .chart_extractor import ChartExtractor, ChartNode
    from .s3_loader import S3Loader
    __all__ += ["ChartExtractor", "ChartNode", "S3Loader"]
except ImportError:
    pass
