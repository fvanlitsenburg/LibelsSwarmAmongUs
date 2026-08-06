"""OCR one registered DUPO page and save it to PostgreSQL."""

import argparse

from historical_text_pipeline.config.settings import get_settings
from historical_text_pipeline.db.session import get_session_factory
from historical_text_pipeline.ingest.dupo_ocr import (
    ocr_dupo_page,
)
from historical_text_pipeline.ocr.factory import (
    create_ocr_backend,
)


def parse_arguments() -> argparse.Namespace:
    """Read the document ID and page number."""

    parser = argparse.ArgumentParser(
        description=(
            "OCR one page of a registered DUPO document."
        )
    )

    parser.add_argument(
        "document_id",
        type=int,
        help="Internal PostgreSQL document ID.",
    )

    parser.add_argument(
        "page_number",
        type=int,
        help="One-based PDF page number.",
    )

    return parser.parse_args()


def main() -> None:
    """OCR and save one page."""

    arguments = parse_arguments()
    settings = get_settings()

    backend = create_ocr_backend(settings)
    session_factory = get_session_factory()

    try:
        with session_factory() as session:
            result = ocr_dupo_page(
                session,
                document_id=arguments.document_id,
                page_number=arguments.page_number,
                backend=backend,
                dpi=settings.pdf_render_dpi,
                jpeg_quality=settings.pdf_jpeg_quality,
            )

            session.commit()

    finally:
        backend.close()

    if result.skipped:
        print(
            f"Document {result.document_id}, page "
            f"{result.page_number} was already stored."
        )
        return

    print(
        f"Saved document {result.document_id}, "
        f"page {result.page_number}"
    )
    print(f"Provider:   {result.provider}")
    print(f"Model:      {result.model}")
    print(f"Characters: {result.character_count}")
    print(f"Words:      {result.word_count}")


if __name__ == "__main__":
    main()