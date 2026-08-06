"""OCR provider interfaces and implementations."""

from historical_text_pipeline.ocr.base import (
    OcrBackend,
    OcrPageResult,
)
from historical_text_pipeline.ocr.factory import (
    create_ocr_backend,
)
from historical_text_pipeline.ocr.mistral_ocr import (
    MistralOcrBackend,
    MistralOcrError,
)
from historical_text_pipeline.ocr.openai_vision import (
    OpenAiOcrBackend,
    OpenAiOcrError,
)
from historical_text_pipeline.ocr.pdf_rendering import (
    RenderedPdfPage,
    render_pdf_page_as_jpeg,
)

__all__ = [
    "MistralOcrBackend",
    "MistralOcrError",
    "OcrBackend",
    "OcrPageResult",
    "OpenAiOcrBackend",
    "OpenAiOcrError",
    "RenderedPdfPage",
    "create_ocr_backend",
    "render_pdf_page_as_jpeg",
]