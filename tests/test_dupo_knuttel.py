from pathlib import Path

import pytest

from historical_text_pipeline.ingest import dupo_knuttel
from historical_text_pipeline.ocr.base import (
    OcrEmbeddedImage,
    OcrPageResult,
)
from historical_text_pipeline.ocr.pdf_rendering import (
    RenderedPdfPage,
)


class FakeEmbeddedImageBackend:
    """Return an embedded image, then recognize its number."""

    def __init__(self) -> None:
        self.calls = 0

    def recognize_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/jpeg",
        include_embedded_images: bool = False,
    ) -> OcrPageResult:
        del image_bytes, mime_type

        self.calls += 1

        if self.calls == 1:
            assert include_embedded_images is True

            return OcrPageResult(
                provider="mistral",
                model="mistral-ocr-4-0",
                text="![img-0.jpeg](img-0.jpeg)",
                embedded_images=(
                    OcrEmbeddedImage(
                        image_id="img-0.jpeg",
                        mime_type="image/jpeg",
                        image_bytes=b"cropped number image",
                    ),
                ),
            )

        assert include_embedded_images is False

        return OcrPageResult(
            provider="mistral",
            model="mistral-ocr-4-0",
            text="418",
        )

    def close(self) -> None:
        """Match the OCR backend protocol."""


def test_uses_embedded_image_when_number_is_not_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    def fake_render(
        path: Path,
        *,
        dpi: int,
        jpeg_quality: int,
    ) -> RenderedPdfPage:
        del dpi, jpeg_quality

        return RenderedPdfPage(
            pdf_path=path,
            page_number=1,
            mime_type="image/jpeg",
            image_bytes=b"knuttel crop",
        )

    monkeypatch.setattr(
        dupo_knuttel,
        "render_first_pdf_page_knuttel_region_as_jpeg",
        fake_render,
    )

    backend = FakeEmbeddedImageBackend()

    result = (
        dupo_knuttel.extract_knuttel_number_from_first_page(
            pdf_path,
            backend=backend,
        )
    )

    assert result.knuttel_number == "418"
    assert result.candidates == ("418",)
    assert result.used_embedded_image_fallback is True
    assert result.embedded_images_checked == 1
    assert backend.calls == 2