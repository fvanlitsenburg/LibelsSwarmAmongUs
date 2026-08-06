"""Shared interfaces and result types for OCR providers."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class OcrEmbeddedImage:
    """An image region extracted by an OCR provider."""

    image_id: str
    mime_type: str
    image_bytes: bytes


@dataclass(frozen=True, slots=True)
class OcrPageResult:
    """Text and image regions returned after recognizing one page."""

    provider: str
    model: str
    text: str
    response_id: str | None = None
    embedded_images: tuple[OcrEmbeddedImage, ...] = ()


class OcrBackend(Protocol):
    """Interface implemented by OCR providers."""

    def recognize_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/jpeg",
        include_embedded_images: bool = False,
    ) -> OcrPageResult:
        """Recognize the text contained in one image."""

    def close(self) -> None:
        """Release network resources."""