"""Transcribe one PDF page with OpenAI vision."""

import argparse
from pathlib import Path

from historical_text_pipeline.config.settings import get_settings
from historical_text_pipeline.ocr.openai_vision import (
    OpenAiOcrBackend,
)
from historical_text_pipeline.ocr.pdf_rendering import (
    render_pdf_page_as_jpeg,
)


def parse_arguments() -> argparse.Namespace:
    """Read the PDF path and one-based page number."""

    parser = argparse.ArgumentParser(
        description="Transcribe one PDF page with OpenAI vision.",
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
    """Render and transcribe one PDF page."""

    arguments = parse_arguments()
    settings = get_settings()

    rendered_page = render_pdf_page_as_jpeg(
        arguments.pdf_path,
        arguments.page_number,
        dpi=settings.pdf_render_dpi,
        jpeg_quality=settings.pdf_jpeg_quality,
    )

    image_size_kb = len(rendered_page.image_bytes) / 1024

    print(
        f"Rendered page {rendered_page.page_number}: "
        f"{image_size_kb:.0f} KB"
    )
    print(
        f"Submitting to OpenAI model "
        f"{settings.openai_ocr_model}..."
    )

    backend = OpenAiOcrBackend.from_settings(settings)

    try:
        result = backend.recognize_image(
            rendered_page.image_bytes,
            mime_type=rendered_page.mime_type,
        )
    finally:
        backend.close()

    print()
    print(f"Response ID: {result.response_id}")
    print(f"Model:       {result.model}")
    print()
    print(result.text)


if __name__ == "__main__":
    main()