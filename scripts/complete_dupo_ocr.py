"""Complete OCR for a relevant DUPO document."""

import argparse

from historical_text_pipeline.config.settings import (
    get_settings,
)
from historical_text_pipeline.db.models import Document
from historical_text_pipeline.db.session import (
    get_session_factory,
)
from historical_text_pipeline.domain import RelevanceStatus
from historical_text_pipeline.ingest.dupo_ocr import (
    get_missing_dupo_pages,
    ocr_dupo_page,
)
from historical_text_pipeline.ocr.factory import (
    create_ocr_backend,
)


def parse_arguments() -> argparse.Namespace:
    """Read the internal document ID."""

    parser = argparse.ArgumentParser(
        description=(
            "OCR every remaining page of a relevant DUPO "
            "document."
        )
    )

    parser.add_argument(
        "document_id",
        type=int,
        help="Internal PostgreSQL document ID.",
    )

    return parser.parse_args()


def record_failure(
    *,
    session_factory,
    document_id: int,
    page_number: int,
    error: Exception,
) -> None:
    """Store an OCR failure without losing completed pages."""

    with session_factory() as session:
        document = session.get(Document, document_id)

        if document is not None:
            document.processing_status = "ocr_error"
            document.error_message = (
                f"Page {page_number}: {error}"
            )

            session.commit()


def main() -> None:
    """OCR all pages that have not yet been stored."""

    arguments = parse_arguments()
    settings = get_settings()
    session_factory = get_session_factory()

    with session_factory() as session:
        document = session.get(
            Document,
            arguments.document_id,
        )

        if document is None:
            raise SystemExit(
                f"Document {arguments.document_id} "
                "does not exist."
            )

        if document.relevance_status != RelevanceStatus.RELEVANT:
            raise SystemExit(
                f"Document {document.id} is not marked relevant. "
                f"Current status: "
                f"{document.relevance_status.value}"
            )

        if document.total_units is None:
            raise SystemExit(
                f"Document {document.id} has not been inspected."
            )

        missing_pages = get_missing_dupo_pages(
            session,
            document_id=document.id,
        )

        total_pages = document.total_units
        stored_pages = total_pages - len(missing_pages)

    if not missing_pages:
        print(
            f"Document {arguments.document_id} already has "
            f"all {total_pages} pages stored."
        )
        return

    print(f"Document:        {arguments.document_id}")
    print(f"Total pages:     {total_pages}")
    print(f"Already stored:  {stored_pages}")
    print(f"Pages remaining: {len(missing_pages)}")
    print()

    backend = create_ocr_backend(settings)
    newly_saved = 0

    try:
        for page_number in missing_pages:
            print(
                f"OCR page {page_number}/{total_pages}...",
                flush=True,
            )

            try:
                with session_factory() as session:
                    result = ocr_dupo_page(
                        session,
                        document_id=arguments.document_id,
                        page_number=page_number,
                        backend=backend,
                        dpi=settings.pdf_render_dpi,
                        jpeg_quality=(
                            settings.pdf_jpeg_quality
                        ),
                    )

                    session.commit()

            except Exception as error:
                record_failure(
                    session_factory=session_factory,
                    document_id=arguments.document_id,
                    page_number=page_number,
                    error=error,
                )

                raise SystemExit(
                    f"OCR failed on page {page_number}: "
                    f"{error}\n"
                    "Previously completed pages were saved. "
                    "Rerun this command to resume."
                ) from error

            if result.created:
                newly_saved += 1

                print(
                    f"Saved page {page_number}: "
                    f"{result.character_count} characters"
                )
            else:
                print(
                    f"Page {page_number} was already stored."
                )

    finally:
        backend.close()

    with session_factory() as session:
        document = session.get(
            Document,
            arguments.document_id,
        )

        if document is None:
            raise SystemExit(
                "Document disappeared after OCR."
            )

        print()
        print(f"New pages saved: {newly_saved}")
        print(
            f"Stored pages:    "
            f"{document.units_processed}/"
            f"{document.total_units}"
        )
        print(
            f"Text complete:   "
            f"{document.text_complete}"
        )
        print(
            f"Status:          "
            f"{document.processing_status}"
        )


if __name__ == "__main__":
    main()