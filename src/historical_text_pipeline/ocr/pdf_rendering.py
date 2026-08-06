"""Render individual PDF pages as images for OCR."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

MAX_OCR_IMAGE_BYTES = 20 * 1024 * 1024


class PdfRenderingError(Exception):
    """Raised when a PDF page cannot be rendered."""


class RenderedImageTooLargeError(PdfRenderingError):
    """Raised when a rendered image exceeds the configured limit."""


@dataclass(frozen=True, slots=True)
class RenderedPdfPage:
    """A rendered page ready to send to an OCR provider."""

    pdf_path: Path
    page_number: int
    mime_type: str
    image_bytes: bytes


DEFAULT_KNUTTEL_CROP_LEFT_FRACTION = 0.05
DEFAULT_KNUTTEL_CROP_RIGHT_FRACTION = 0.49


def crop_knuttel_region(
    image: Image.Image,
    *,
    left_fraction: float = DEFAULT_KNUTTEL_CROP_LEFT_FRACTION,
    right_fraction: float = DEFAULT_KNUTTEL_CROP_RIGHT_FRACTION,
) -> Image.Image:
    """
    Crop the Knuttel-number region from a DUPO first-page scan.

    The crop:
    - removes the ruler and outer margin on the far left;
    - excludes the right-hand printed page;
    - preserves the complete vertical extent of the scan.
    """

    if not 0 <= left_fraction < right_fraction <= 1:
        raise ValueError(
            "Crop fractions must satisfy "
            "0 <= left_fraction < right_fraction <= 1."
        )

    width, height = image.size

    left = round(width * left_fraction)
    right = round(width * right_fraction)

    return image.crop(
        (
            left,
            0,
            right,
            height,
        )
    )


def _render_pdf_page_as_image(
    pdf_path: Path,
    page_number: int,
    *,
    dpi: int,
) -> Image.Image:
    """Render one one-based PDF page as a Pillow image."""

    pdf_path = pdf_path.expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")

    if not pdf_path.is_file():
        raise PdfRenderingError(
            f"PDF path is not a file: {pdf_path}"
        )

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
            bitmap = page.render(scale=dpi / 72)

            try:
                # convert() creates an image independent of the PDF bitmap.
                return bitmap.to_pil().convert("RGB")

            finally:
                bitmap.close()

        finally:
            page.close()

    finally:
        document.close()


def _encode_image_as_jpeg(
    image: Image.Image,
    *,
    jpeg_quality: int,
) -> bytes:
    """Encode a Pillow image as JPEG."""

    output = BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=jpeg_quality,
        optimize=True,
    )

    image_bytes = output.getvalue()

    if len(image_bytes) > MAX_OCR_IMAGE_BYTES:
        size_mb = len(image_bytes) / (1024 * 1024)

        raise RenderedImageTooLargeError(
            f"Rendered image is {size_mb:.1f} MB, exceeding "
            "the configured 20 MB limit."
        )

    return image_bytes


def render_pdf_page_as_jpeg(
    pdf_path: Path,
    page_number: int,
    *,
    dpi: int = 300,
    jpeg_quality: int = 95,
) -> RenderedPdfPage:
    """Render one complete PDF page as a JPEG."""

    resolved_path = pdf_path.expanduser().resolve()

    image = _render_pdf_page_as_image(
        resolved_path,
        page_number,
        dpi=dpi,
    )

    try:
        image_bytes = _encode_image_as_jpeg(
            image,
            jpeg_quality=jpeg_quality,
        )
    finally:
        image.close()

    return RenderedPdfPage(
        pdf_path=resolved_path,
        page_number=page_number,
        mime_type="image/jpeg",
        image_bytes=image_bytes,
    )


def render_first_pdf_page_knuttel_region_as_jpeg(
    pdf_path: Path,
    *,
    dpi: int = 300,
    jpeg_quality: int = 95,
    crop_left_fraction: float = (
        DEFAULT_KNUTTEL_CROP_LEFT_FRACTION
    ),
    crop_right_fraction: float = (
        DEFAULT_KNUTTEL_CROP_RIGHT_FRACTION
    ),
) -> RenderedPdfPage:
    """
    Render the likely Knuttel-number region from PDF page 1.

    The full page height is retained. The ruler on the far left and the
    printed page on the right are removed.
    """

    resolved_path = pdf_path.expanduser().resolve()

    full_image = _render_pdf_page_as_image(
        resolved_path,
        page_number=1,
        dpi=dpi,
    )

    try:
        knuttel_image = crop_knuttel_region(
            full_image,
            left_fraction=crop_left_fraction,
            right_fraction=crop_right_fraction,
        )

        try:
            image_bytes = _encode_image_as_jpeg(
                knuttel_image,
                jpeg_quality=jpeg_quality,
            )
        finally:
            knuttel_image.close()

    finally:
        full_image.close()

    return RenderedPdfPage(
        pdf_path=resolved_path,
        page_number=1,
        mime_type="image/jpeg",
        image_bytes=image_bytes,
    )