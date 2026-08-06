"""OCR provider interfaces and implementations."""

from historical_text_pipeline.ocr.base import (
    OcrBackend,
    OcrEmbeddedImage,
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
    crop_knuttel_region,
    render_first_pdf_page_knuttel_region_as_jpeg,
    render_pdf_page_as_jpeg,
)

__all__ = [
    "MistralOcrBackend",
    "MistralOcrError",
    "OcrBackend",
    "OcrEmbeddedImage",
    "OcrPageResult",
    "OpenAiOcrBackend",
    "OpenAiOcrError",
    "RenderedPdfPage",
    "create_ocr_backend",
    "crop_knuttel_region",
    "render_first_pdf_page_knuttel_region_as_jpeg",
    "render_pdf_page_as_jpeg",
]