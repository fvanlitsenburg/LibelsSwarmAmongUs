"""OCR the next DUPO page batch and assess its relevance."""

import sys
from pathlib import Path

from historical_text_pipeline.config.settings import get_settings
from historical_text_pipeline.db.models import Document
from historical_text_pipeline.db.session import get_session_factory
from historical_text_pipeline.domain import RelevanceStatus
from historical_text_pipeline.ingest.dupo_knuttel import (
    extract_and_save_knuttel_number,
)
from historical_text_pipeline.ingest.dupo_ocr import (
    ocr_dupo_page,
)
from historical_text_pipeline.ocr.factory import (
    create_ocr_backend,
)
from historical_text_pipeline.relevance.openai_assessor import (
    OpenAiRelevanceAssessor,
)
from historical_text_pipeline.relevance.service import (
    assess_and_store_relevance,
    get_latest_assessment,
    get_next_batch_end_page,
)


def load_criteria(path: Path) -> str:
    """Load and validate the research criteria."""

    resolved_path = path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Relevance criteria file does not exist: "
            f"{resolved_path}"
        )

    criteria = resolved_path.read_text(
        encoding="utf-8"
    ).strip()

    if not criteria:
        raise ValueError(
            "The relevance criteria file is empty."
        )

    if "[Replace" in criteria or "[Add " in criteria:
        raise ValueError(
            "The relevance criteria file still contains "
            "placeholder instructions."
        )

    return criteria


def main() -> None:
    """Process the next progressive page batch."""

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/process_dupo_relevance_batch.py "
            "DOCUMENT_ID"
        )

    document_id = int(sys.argv[1])

    settings = get_settings()
    criteria = load_criteria(
        settings.relevance_criteria_path
    )

    session_factory = get_session_factory()

    with session_factory() as session:
        document = session.get(Document, document_id)

        if document is None:
            raise SystemExit(
                f"Document {document_id} does not exist."
            )

        if document.relevance_status in {
            RelevanceStatus.RELEVANT,
            RelevanceStatus.IRRELEVANT,
        }:
            raise SystemExit(
                f"Document {document_id} already has final "
                f"relevance status "
                f"{document.relevance_status.value}."
            )

        latest = get_latest_assessment(
            session,
            document_id=document_id,
        )

        previous_end_page = (
            latest.units_processed
            if latest is not None
            else 0
        )

        end_page = get_next_batch_end_page(
            session,
            document_id=document_id,
            batch_size=settings.relevance_batch_size,
        )

    if end_page is None:
        raise SystemExit(
            "The complete document has already been assessed."
        )

    start_page = previous_end_page + 1

    print(
        f"Processing document {document_id}, "
        f"pages {start_page}–{end_page}"
    )
    print()

    ocr_backend = create_ocr_backend(settings)

    try:
        for page_number in range(
            start_page,
            end_page + 1,
        ):
            # Each page is committed separately. A later failure cannot
            # discard an earlier paid OCR result.
            with session_factory() as session:
                page_result = ocr_dupo_page(
                    session,
                    document_id=document_id,
                    page_number=page_number,
                    backend=ocr_backend,
                    dpi=settings.pdf_render_dpi,
                    jpeg_quality=settings.pdf_jpeg_quality,
                )

                session.commit()

            page_action = (
                "already stored"
                if page_result.skipped
                else "saved"
            )

            print(
                f"Page {page_number}: {page_action} "
                f"({page_result.character_count} characters)"
            )

        # Ensure Knuttel metadata is attempted during the first batch.
        # Existing metadata is skipped without another paid OCR request.
        if start_page == 1:
            with session_factory() as session:
                knuttel_result = extract_and_save_knuttel_number(
                    session,
                    document_id=document_id,
                    backend=ocr_backend,
                    dpi=settings.pdf_render_dpi,
                    jpeg_quality=settings.pdf_jpeg_quality,
                )

                session.commit()

            if knuttel_result.saved:
                print(
                    f"Knuttel number: "
                    f"{knuttel_result.knuttel_number}"
                )
            elif knuttel_result.skipped_existing:
                print(
                    f"Knuttel number: "
                    f"{knuttel_result.knuttel_number} "
                    "(already stored)"
                )
            else:
                print(
                    "Knuttel number: not unambiguously identified"
                )

    finally:
        ocr_backend.close()

    print()
    print("Assessing accumulated OCR text...")

    assessor = OpenAiRelevanceAssessor.from_settings(
        settings
    )

    try:
        with session_factory() as session:
            relevance_result = assess_and_store_relevance(
                session,
                document_id=document_id,
                through_page=end_page,
                criteria=criteria,
                assessor=assessor,
                stop_confidence_threshold=(
                    settings.relevance_stop_confidence_threshold
                ),
            )

            session.commit()

    finally:
        assessor.close()

    print()
    print(
        f"Assessment: "
        f"{relevance_result.assessment_number}"
    )
    print(
        f"Pages read: "
        f"{relevance_result.pages_assessed}"
    )
    print(
        f"Decision:   "
        f"{relevance_result.decision.value.upper()}"
    )
    print(
        f"Score:      "
        f"{relevance_result.relevance_score:.2f}"
    )
    print(
        f"Confidence: "
        f"{relevance_result.confidence:.2f}"
    )
    print(
        f"Category:   "
        f"{relevance_result.category}"
    )
    print(
        f"Topic:      "
        f"{relevance_result.topic}"
    )
    print(
        f"Reason:     "
        f"{relevance_result.reason}"
    )

    print()

    if relevance_result.decision.value == "continue":
        print(
            "The document is relevant and should receive "
            "complete OCR."
        )

    elif relevance_result.stop_confirmed:
        print(
            "Two confident STOP assessments have confirmed "
            "the document as irrelevant."
        )

    else:
        print(
            "Another page batch is required before making "
            "a final decision."
        )


if __name__ == "__main__":
    main()