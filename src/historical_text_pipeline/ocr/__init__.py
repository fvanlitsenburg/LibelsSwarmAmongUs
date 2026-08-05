"""OCR provider interfaces and implementations."""

from historical_text_pipeline.ocr.base import (
    OcrBackend,
    OcrPageResult,
)
from historical_text_pipeline.ocr.pdf_rendering import (
    RenderedPdfPage,
    render_pdf_page_as_jpeg,
)
from historical_text_pipeline.ocr.transkribus import (
    TranskribusOcrBackend,
)

__all__ = [
    "OcrBackend",
    "OcrPageResult",
    "RenderedPdfPage",
    "TranskribusOcrBackend",
    "render_pdf_page_as_jpeg",
]