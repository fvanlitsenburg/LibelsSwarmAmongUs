"""Tests for progressive relevance decisions."""

from pathlib import Path

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import (
    Document,
    DocumentTextUnit,
    DupoMetadata,
)
from historical_text_pipeline.domain import (
    RelevanceStatus,
    Source,
)
from historical_text_pipeline.relevance.base import (
    RelevanceAssessmentOutput,
    RelevanceDecision,
)
from historical_text_pipeline.relevance.service import (
    assess_and_store_relevance,
)


class FakeAssessor:
    """Return predefined relevance outputs."""

    def __init__(
        self,
        outputs: list[RelevanceAssessmentOutput],
    ) -> None:
        self.outputs = outputs
        self.calls = 0

    def assess(
        self,
        *,
        text: str,
        criteria: str,
        assessment_number: int,
    ) -> RelevanceAssessmentOutput:
        assert text
        assert criteria
        assert assessment_number == self.calls + 1

        output = self.outputs[self.calls]
        self.calls += 1

        return output

    def close(self) -> None:
        """Match the relevance-assessor protocol."""


def make_output(
    decision: RelevanceDecision,
    *,
    confidence: float = 0.90,
) -> RelevanceAssessmentOutput:
    """Create a test assessment."""

    return RelevanceAssessmentOutput(
        decision=decision,
        relevance_score=(
            0.90
            if decision == RelevanceDecision.CONTINUE
            else 0.10
        ),
        confidence=confidence,
        reason="Evidence-based test decision.",
        category="test-category",
        topic="test topic",
        supporting_evidence=["Short test evidence."],
        missing_information=[],
    )


def add_document_with_pages(
    session: Session,
    pdf_path: Path,
    *,
    page_count: int = 6,
    stored_through: int = 3,
) -> Document:
    """Create a DUPO document with stored page text."""

    document = Document(
        source=Source.DUPO,
        source_path=str(pdf_path),
        source_filename=pdf_path.name,
        year=1660,
        language="nl",
        total_units=page_count,
        units_processed=stored_through,
        text_complete=False,
        text_method="ocr",
        processing_status="ocr_partial",
        dupo=DupoMetadata(),
    )

    session.add(document)
    session.flush()

    for page_number in range(1, stored_through + 1):
        text = f"Historical text from page {page_number}."

        session.add(
            DocumentTextUnit(
                document_id=document.id,
                unit_type="page",
                unit_number=page_number,
                text=text,
                character_count=len(text),
                word_count=len(text.split()),
                processing_method="ocr",
                ocr_provider="mistral",
                ocr_model="mistral-ocr-test",
            )
        )

    session.commit()

    return document


def add_more_pages(
    session: Session,
    *,
    document: Document,
    start_page: int,
    end_page: int,
) -> None:
    """Add another stored page batch."""

    for page_number in range(
        start_page,
        end_page + 1,
    ):
        text = f"Historical text from page {page_number}."

        session.add(
            DocumentTextUnit(
                document_id=document.id,
                unit_type="page",
                unit_number=page_number,
                text=text,
                character_count=len(text),
                word_count=len(text.split()),
                processing_method="ocr",
                ocr_provider="mistral",
                ocr_model="mistral-ocr-test",
            )
        )

    document.units_processed = end_page
    session.commit()


def test_continue_marks_document_relevant(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    assessor = FakeAssessor(
        [
            make_output(
                RelevanceDecision.CONTINUE
            )
        ]
    )

    with Session(db_engine) as session:
        document = add_document_with_pages(
            session,
            pdf_path,
        )

        result = assess_and_store_relevance(
            session,
            document_id=document.id,
            through_page=3,
            criteria="Test research criteria.",
            assessor=assessor,
        )

        session.commit()
        session.refresh(document)

        assert result.decision == RelevanceDecision.CONTINUE
        assert document.relevance_status == (
            RelevanceStatus.RELEVANT
        )
        assert document.primary_category == "test-category"
        assert document.topic == "test topic"


def test_first_stop_requires_another_batch(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    assessor = FakeAssessor(
        [
            make_output(RelevanceDecision.STOP),
        ]
    )

    with Session(db_engine) as session:
        document = add_document_with_pages(
            session,
            pdf_path,
        )

        result = assess_and_store_relevance(
            session,
            document_id=document.id,
            through_page=3,
            criteria="Test research criteria.",
            assessor=assessor,
        )

        session.commit()
        session.refresh(document)

        assert result.stop_confirmed is False
        assert document.relevance_status == (
            RelevanceStatus.UNCERTAIN
        )
        assert document.processing_status == (
            "relevance_stop_pending"
        )


def test_second_confident_stop_confirms_irrelevance(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    assessor = FakeAssessor(
        [
            make_output(RelevanceDecision.STOP),
            make_output(RelevanceDecision.STOP),
        ]
    )

    with Session(db_engine) as session:
        document = add_document_with_pages(
            session,
            pdf_path,
        )

        first_result = assess_and_store_relevance(
            session,
            document_id=document.id,
            through_page=3,
            criteria="Test research criteria.",
            assessor=assessor,
        )

        session.commit()

        add_more_pages(
            session,
            document=document,
            start_page=4,
            end_page=6,
        )

        second_result = assess_and_store_relevance(
            session,
            document_id=document.id,
            through_page=6,
            criteria="Test research criteria.",
            assessor=assessor,
        )

        session.commit()
        session.refresh(document)

        assert first_result.stop_confirmed is False
        assert second_result.stop_confirmed is True
        assert document.relevance_status == (
            RelevanceStatus.IRRELEVANT
        )
        assert document.processing_status == (
            "relevance_stopped"
        )