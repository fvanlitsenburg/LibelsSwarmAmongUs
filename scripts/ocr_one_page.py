"""OCR one PDF page with Transkribus."""

import argparse
from pathlib import Path

from historical_text_pipeline.config.settings import get_settings
from historical_text_pipeline.ocr.pdf_rendering import (
    render_pdf_page_as_jpeg,
)
from historical_text_pipeline.ocr.transkribus import (
    TranskribusOcrBackend,
)


def parse_arguments() -> argparse.Namespace:
    """Read the PDF path and one-based page number."""

    parser = argparse.ArgumentParser(
        description="OCR one PDF page with Transkribus.",
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF.",
    )
    parser.add_argument(
        "page_number",
        type=int,
        help="One-based page number.",
    )

    return parser.parse_args()


def main() -> None:
    """Render and OCR one PDF page."""

    arguments = parse_arguments()
    settings = get_settings()

    rendered_page = render_pdf_page_as_jpeg(
        arguments.pdf_path,
        arguments.page_number,
        dpi=settings.pdf_render_dpi,
        jpeg_quality=settings.pdf_jpeg_quality,
    )

    print(
        f"Rendered page {rendered_page.page_number}: "
        f"{len(rendered_page.image_bytes) / 1024:.0f} KB"
    )
    print("Submitting to Transkribus...")

    backend = TranskribusOcrBackend.from_settings(settings)

    try:
        result = backend.recognize_image(
            rendered_page.image_bytes,
        )
    finally:
        backend.close()

    print()
    print(f"Process ID: {result.process_id}")
    print(f"Model ID:   {result.model_id}")
    print()
    print(result.text)


if __name__ == "__main__":
    main()