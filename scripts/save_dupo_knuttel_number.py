"""Extract and save the Knuttel number for a registered DUPO document."""

import argparse

from historical_text_pipeline.config.settings import get_settings
from historical_text_pipeline.db.session import get_session_factory
from historical_text_pipeline.ingest.dupo_knuttel import (
    extract_and_save_knuttel_number,
)
from historical_text_pipeline.ocr.factory import (
    create_ocr_backend,
)


def parse_arguments() -> argparse.Namespace:
    """Read the document ID and overwrite option."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract and save a registered DUPO document's "
            "Knuttel number."
        )
    )

    parser.add_argument(
        "document_id",
        type=int,
        help="Internal PostgreSQL document ID.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing Knuttel number.",
    )

    return parser.parse_args()


def main() -> None:
    """Extract and save Knuttel metadata."""

    arguments = parse_arguments()
    settings = get_settings()

    backend = create_ocr_backend(settings)
    session_factory = get_session_factory()

    try:
        with session_factory() as session:
            result = extract_and_save_knuttel_number(
                session,
                document_id=arguments.document_id,
                backend=backend,
                dpi=settings.pdf_render_dpi,
                jpeg_quality=settings.pdf_jpeg_quality,
                overwrite=arguments.overwrite,
            )

            session.commit()

    finally:
        backend.close()

    if result.skipped_existing:
        print(
            f"Document {result.document_id} already has "
            f"Knuttel number {result.knuttel_number}."
        )
        return

    if result.saved:
        print(
            f"Saved Knuttel number {result.knuttel_number} "
            f"for document {result.document_id}."
        )

        if result.used_embedded_image_fallback:
            print("The embedded-image OCR fallback was used.")

        return

    print(
        f"No unambiguous Knuttel number was found for "
        f"document {result.document_id}."
    )

    if result.candidates:
        print(
            "Candidates: "
            + ", ".join(result.candidates)
        )


if __name__ == "__main__":
    main()