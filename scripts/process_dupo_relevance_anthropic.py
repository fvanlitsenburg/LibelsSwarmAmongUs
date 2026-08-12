"""Run one progressive Claude relevance batch for a DUPO document."""

import argparse
from pathlib import Path

from historical_text_pipeline.config.settings import get_settings
from historical_text_pipeline.db.models import Document
from historical_text_pipeline.db.session import get_session_factory
from historical_text_pipeline.domain import AnalysisProvider, Source
from historical_text_pipeline.ingest.dupo_ocr import ocr_dupo_page
from historical_text_pipeline.ocr.factory import create_ocr_backend
from historical_text_pipeline.relevance.anthropic_assessor import (
    AnthropicRelevanceAssessor,
)
from historical_text_pipeline.relevance.service import (
    assess_and_store_relevance,
    get_latest_assessment,
    get_next_batch_end_page,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Anthropic progressive relevance batch "
            "for a DUPO document."
        )
    )

    parser.add_argument(
        "document_id",
        type=int,
        help="Internal PostgreSQL document ID.",
    )

    return parser.parse_args()


def load_criteria(path: Path) -> str:
    resolved_path = path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Relevance criteria file does not exist: {resolved_path}"
        )

    criteria = resolved_path.read_text(
        encoding="utf-8"
    ).strip()

    if not criteria:
        raise ValueError(
            "The relevance criteria file is empty."
        )

    return criteria


def main() -> None:
    arguments = parse_arguments()
    settings = get_settings()
    session_factory = get_session_factory()

    criteria = load_criteria(
        settings.relevance_criteria_path
    )

    assessor = AnthropicRelevanceAssessor.from_settings(
        settings
    )

    try:
        with session_factory() as session:
            document = session.get(
                Document,
                arguments.document_id,
            )

            if document is None:
                raise SystemExit(
                    f"Document {arguments.document_id} does not exist."
                )

            if document.source != Source.DUPO:
                raise SystemExit(
                    f"Document {document.id} is not a DUPO document."
                )

            if document.total_units is None:
                raise SystemExit(
                    f"Document {document.id} has not been inspected."
                )

            latest = get_latest_assessment(
                session,
                document_id=document.id,
                provider=AnalysisProvider.ANTHROPIC,
            )

            next_end_page = get_next_batch_end_page(
                session,
                document_id=document.id,
                batch_size=settings.relevance_batch_size,
                provider=AnalysisProvider.ANTHROPIC,
            )

            if next_end_page is None:
                raise SystemExit(
                    "Claude has already assessed all available pages."
                )

            start_page = (
                latest.units_processed + 1
                if latest is not None
                else 1
            )

            assessment_number = (
                latest.sequence_number + 1
                if latest is not None
                else 1
            )

            total_pages = document.total_units

        print(f"Document:          {arguments.document_id}")
        print("Provider:          anthropic")
        print(
            f"Assessment:        {assessment_number}"
        )
        print(
            f"Assessment pages:  1-{next_end_page}"
        )
        print(
            f"New page range:    {start_page}-{next_end_page}"
        )
        print(f"Total PDF pages:   {total_pages}")
        print()

        # OCR is shared between providers. Existing pages are skipped.
        ocr_backend = create_ocr_backend(settings)

        try:
            for page_number in range(
                start_page,
                next_end_page + 1,
            ):
                print(
                    f"Checking OCR page "
                    f"{page_number}/{total_pages}...",
                    flush=True,
                )

                with session_factory() as session:
                    result = ocr_dupo_page(
                        session,
                        document_id=arguments.document_id,
                        page_number=page_number,
                        backend=ocr_backend,
                        dpi=settings.pdf_render_dpi,
                        jpeg_quality=settings.pdf_jpeg_quality,
                    )

                    session.commit()

                if result.created:
                    print(
                        f"  OCR saved: "
                        f"{result.character_count} characters"
                    )
                else:
                    print(
                        "  Already stored — no OCR request made."
                    )

        finally:
            ocr_backend.close()

        print()
        print("Running Claude relevance assessment...")

        with session_factory() as session:
            result = assess_and_store_relevance(
                session,
                document_id=arguments.document_id,
                through_page=next_end_page,
                criteria=criteria,
                assessor=assessor,
                stop_confidence_threshold=(
                    settings.relevance_stop_confidence_threshold
                ),
                max_assessments=(
                    settings.relevance_max_assessments
                ),
            )

            session.commit()

        print()
        print("=" * 60)
        print("CLAUDE RELEVANCE RESULT")
        print("=" * 60)
        print(
            f"Assessment: {result.assessment_number}"
        )
        print(
            f"Pages:      1-{result.pages_assessed}"
        )
        print(
            f"Decision:   {result.decision.value}"
        )
        print(
            f"Status:     {result.relevance_status.value}"
        )
        print(
            f"Score:      {result.relevance_score:.2f}"
        )
        print(
            f"Confidence: {result.confidence:.2f}"
        )
        print(
            f"Category:   {result.category}"
        )
        print(
            f"Topic:      {result.topic}"
        )
        print(
            f"STOP confirmed: {result.stop_confirmed}"
        )

        print()
        print("Reason")
        print("------")
        print(result.reason)

    finally:
        assessor.close()


if __name__ == "__main__":
    main()