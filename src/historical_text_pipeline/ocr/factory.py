"""Construct the configured OCR provider."""

from historical_text_pipeline.config.settings import Settings
from historical_text_pipeline.ocr.base import OcrBackend
from historical_text_pipeline.ocr.mistral_ocr import (
    MistralOcrBackend,
)
from historical_text_pipeline.ocr.openai_vision import (
    OpenAiOcrBackend,
)


def create_ocr_backend(
    settings: Settings,
) -> OcrBackend:
    """Create the OCR provider selected in the settings."""

    if settings.ocr_provider == "mistral":
        return MistralOcrBackend.from_settings(settings)

    if settings.ocr_provider == "openai":
        return OpenAiOcrBackend.from_settings(settings)

    raise ValueError(
        f"Unsupported OCR provider: {settings.ocr_provider}"
    )