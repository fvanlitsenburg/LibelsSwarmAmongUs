"""Shared interfaces and result types for OCR providers."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class OcrPageResult:
    """Text returned after recognizing one page image."""

    provider: str
    model_id: str
    process_id: str
    text: str


class OcrBackend(Protocol):
    """Interface implemented by OCR providers."""

    def recognize_image(
        self,
        image_bytes: bytes,
    ) -> OcrPageResult:
        """Recognize the text contained in one image."""