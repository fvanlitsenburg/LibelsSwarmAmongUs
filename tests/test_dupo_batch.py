"""Tests for DUPO batch-queue selection."""

from pathlib import Path

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from historical_text_pipeline.batch import (
    DupoBatchStage,
    get_dupo_batch_document_ids,
)
from historical_text_pipeline.db.models import (
    Document,
    DupoMetadata,
)
from historical_text_pipeline.domain import (
    RelevanceStatus,
    Source,
)


def add_document(
    session: Session,
    tmp_path: Path,
    *,
    name: str,
    relevance_status: RelevanceStatus,
    text_complete: bool,
    summary: str | None = None,
    processing_status: str = "inspected",
) -> Document:
    """Create one document in a particular pipeline state."""

    pdf_path = tmp_path / f"{name}.pdf"
    pdf_path.write_bytes(name.encode())

    document = Document(
        source=Source.DUPO,
        source_path=str(pdf_path),
        source_filename=pdf_path.name,
        year=1660,
        language="nl",
        total_units=6,
        units_processed=(
            6
            if text_complete
            else 0
        ),
        text_complete=text_complete,
        text_method="ocr",
        relevance_status=relevance_status,
        processing_status=processing_status,
        summary=summary,
        dupo=DupoMetadata(),
    )

    session.add(document)
    session.flush()

    return document


def test_selects_documents_for_correct_batch_stage(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    with Session(db_engine) as session:
        pending = add_document(
            session,
            tmp_path,
            name="pending",
            relevance_status=(
                RelevanceStatus.NOT_ASSESSED
            ),
            text_complete=False,
        )

        uncertain = add_document(
            session,
            tmp_path,
            name="uncertain",
            relevance_status=(
                RelevanceStatus.UNCERTAIN
            ),
            text_complete=False,
        )

        relevant_partial = add_document(
            session,
            tmp_path,
            name="relevant-partial",
            relevance_status=(
                RelevanceStatus.RELEVANT
            ),
            text_complete=False,
        )

        relevant_complete = add_document(
            session,
            tmp_path,
            name="relevant-complete",
            relevance_status=(
                RelevanceStatus.RELEVANT
            ),
            text_complete=True,
        )

        uncertain_complete = add_document(
            session,
            tmp_path,
            name="uncertain-complete",
            relevance_status=(
                RelevanceStatus.UNCERTAIN
            ),
            text_complete=True,
        )

        add_document(
            session,
            tmp_path,
            name="irrelevant",
            relevance_status=(
                RelevanceStatus.IRRELEVANT
            ),
            text_complete=False,
        )

        add_document(
            session,
            tmp_path,
            name="already-final",
            relevance_status=(
                RelevanceStatus.RELEVANT
            ),
            text_complete=True,
            summary="Already analyzed.",
        )

        add_document(
            session,
            tmp_path,
            name="pdf-error",
            relevance_status=(
                RelevanceStatus.NOT_ASSESSED
            ),
            text_complete=False,
            processing_status="pdf_error",
        )

        session.commit()

        relevance_ids = get_dupo_batch_document_ids(
            session,
            stage=DupoBatchStage.RELEVANCE,
            limit=100,
        )

        ocr_ids = get_dupo_batch_document_ids(
            session,
            stage=DupoBatchStage.OCR,
            limit=100,
        )

        final_ids = get_dupo_batch_document_ids(
            session,
            stage=DupoBatchStage.FINAL,
            limit=100,
        )

        assert relevance_ids == [
            pending.id,
            uncertain.id,
        ]

        assert ocr_ids == [
            relevant_partial.id,
        ]

        assert final_ids == [
            relevant_complete.id,
            uncertain_complete.id,
        ]


def test_batch_limit_is_applied(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    with Session(db_engine) as session:
        first = add_document(
            session,
            tmp_path,
            name="first",
            relevance_status=(
                RelevanceStatus.NOT_ASSESSED
            ),
            text_complete=False,
        )

        add_document(
            session,
            tmp_path,
            name="second",
            relevance_status=(
                RelevanceStatus.NOT_ASSESSED
            ),
            text_complete=False,
        )

        session.commit()

        result = get_dupo_batch_document_ids(
            session,
            stage=DupoBatchStage.RELEVANCE,
            limit=1,
        )

        assert result == [first.id]