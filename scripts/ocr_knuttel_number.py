"""OCR the left half of a DUPO first page for its Knuttel number."""

import argparse
from pathlib import Path

from historical_text_pipeline.config.settings import get_settings
from historical_text_pipeline.ingest.dupo_knuttel import (
    extract_knuttel_number_from_first_page,
)
from historical_text_pipeline.ocr.factory import (
    create_ocr_backend,
)


def parse_arguments() -> argparse.Namespace:
    """Read the PDF path."""

    parser = argparse.ArgumentParser(
        description=(
            "OCR the left half of page 1 and extract the "
            "Knuttel number."
        ),
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the DUPO PDF.",
    )

    return parser.parse_args()


def main() -> None:
    """Run cropped first-page OCR."""

    arguments = parse_arguments()
    settings = get_settings()

    backend = create_ocr_backend(settings)

    try:
        result = extract_knuttel_number_from_first_page(
            arguments.pdf_path,
            backend=backend,
            dpi=settings.pdf_render_dpi,
            jpeg_quality=settings.pdf_jpeg_quality,
        )
    finally:
        backend.close()

    print(f"Provider: {result.provider}")
    print(f"Model:    {result.model}")
    if result.used_embedded_image_fallback:
        print(
            "Fallback: "
            f"OCR'd {result.embedded_images_checked} "
            "embedded image(s)"
        )
    else:
        print("Fallback: not needed")

    if result.knuttel_number is not None:
        print(f"Knuttel:  {result.knuttel_number}")
    else:
        print("Knuttel:  not unambiguously identified")

        if result.candidates:
            print(
                "Candidates: "
                + ", ".join(result.candidates)
            )

    print()
    print("--- Cropped OCR text ---")
    print()
    print(result.ocr_text)


if __name__ == "__main__":
    main()