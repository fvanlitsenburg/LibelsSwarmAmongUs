"""Render individual PDF pages as images for OCR."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium

MAX_TRANSKRIBUS_IMAGE_BYTES = 20 * 1024 * 1024


class PdfRenderingError(Exception):
    """Raised when a PDF page cannot be rendered."""


class RenderedImageTooLargeError(PdfRenderingError):
    """Raised when the rendered image exceeds the provider limit."""


@dataclass(frozen=True, slots=True)
class RenderedPdfPage:
    """A rendered page ready to send to an OCR provider."""

    pdf_path: Path
    page_number: int
    mime_type: str
    image_bytes: bytes


def render_pdf_page_as_jpeg(
    pdf_path: Path,
    page_number: int,
    *,
    dpi: int = 300,
    jpeg_quality: int = 90,
) -> RenderedPdfPage:
    """
    Render one one-based PDF page as a JPEG image.

    Page number 1 means the first page.
    """

    pdf_path = pdf_path.expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")

    if page_number < 1:
        raise ValueError("Page numbers begin at 1.")

    document = pdfium.PdfDocument(str(pdf_path))

    try:
        page_count = len(document)

        if page_number > page_count:
            raise ValueError(
                f"PDF has {page_count} pages; requested page "
                f"{page_number}."
            )

        page = document[page_number - 1]

        try:
            scale = dpi / 72
            bitmap = page.render(scale=scale)

            try:
                image = bitmap.to_pil().convert("RGB")
                output = BytesIO()

                image.save(
                    output,
                    format="JPEG",
                    quality=jpeg_quality,
                    optimize=True,
                )

                image_bytes = output.getvalue()

            finally:
                bitmap.close()

        finally:
            page.close()

    finally:
        document.close()

    if len(image_bytes) > MAX_TRANSKRIBUS_IMAGE_BYTES:
        size_mb = len(image_bytes) / (1024 * 1024)

        raise RenderedImageTooLargeError(
            f"Rendered page is {size_mb:.1f} MB, exceeding "
            "Transkribus's 20 MB image limit."
        )

    return RenderedPdfPage(
        pdf_path=pdf_path,
        page_number=page_number,
        mime_type="image/jpeg",
        image_bytes=image_bytes,
    )