"""Tests for final full-text document assessment."""

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import (
    Document,
    DocumentAnalysis,
    DocumentTextUnit,
    DupoMetadata,
    RelevanceAssessment,
)
from historical_text_pipeline.domain import (
    AnalysisProvider,
    ClassificationStatus,
    RelevanceStatus,
    Source,
)
from historical_text_pipeline.relevance.final_assessment import (
    FinalAssessmentOutput,
    FinalAssessmentRun,
    FinalRelevanceDecision,
)
from historical_text_pipeline.relevance.final_service import (
    FinalAssessmentServiceError,
    assess_and_store_final_full_text,
)


class FakeFinalAssessor:
    """Return a predetermined final assessment."""

    def __init__(
        self,
        *,
        provider: AnalysisProvider = AnalysisProvider.OPENAI,
        category: str = "political polemic",
        topic: str = "Conflict over public authority",
        summary: str = "A test summary.",
    ) -> None:
        self._provider = provider
        self.category = category
        self.topic = topic
        self.summary = summary

    @property
    def provider(self) -> AnalysisProvider:
        """Return the provider represented by this fake assessor."""

        return self._provider

    def assess(
        self,
        *,
        text: str,
        criteria: str,
        document_context: str,
    ) -> FinalAssessmentRun:
        assert "PDF PAGE 1" in text
        assert criteria
        assert document_context

        return FinalAssessmentRun(
            output=FinalAssessmentOutput(
                decision=FinalRelevanceDecision.RELEVANT,
                relevance_score=0.94,
                confidence=0.91,
                relevance_explanation=(
                    "The complete document directly addresses "
                    "the research subject."
                ),
                category=self.category,
                topic=self.topic,
                summary=self.summary,
                supporting_evidence=[
                    "Test evidence.",
                ],
                caveats=[],
            ),
            provider=self.provider,
            model=(
                "gpt-test"
                if self.provider == AnalysisProvider.OPENAI
                else "claude-test"
            ),
            prompt_version="test-v1",
            response_id="response-test",
            input_tokens=1200,
            output_tokens=250,
        )

    def close(self) -> None:
        pass


def add_complete_document(
    session: Session,
    pdf_path: Path,
) -> Document:
    """Create a fully OCR'd, provisionally relevant document."""

    document = Document(
        source=Source.DUPO,
        source_path=str(pdf_path),
        source_filename=pdf_path.name,
        year=1660,
        language="nl",
        total_units=2,
        units_processed=2,
        text_complete=True,
        text_method="ocr",
        relevance_status=RelevanceStatus.RELEVANT,
        classification_status=ClassificationStatus.FULL_TEXT,
        processing_status="ocr_complete",
        dupo=DupoMetadata(
            knuttel_number="418",
        ),
    )

    session.add(document)
    session.flush()

    for page_number in (1, 2):
        text = f"Complete OCR text from page {page_number}."

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

    session.add(
        RelevanceAssessment(
            document_id=document.id,
            sequence_number=1,
            units_processed=2,
            decision=RelevanceStatus.RELEVANT,
            relevance_score=0.80,
            confidence=0.80,
            reason="Provisional relevance assessment.",
            primary_category="provisional",
            topic="provisional topic",
            classification_status=(
                ClassificationStatus.PARTIAL_TEXT
            ),
            supporting_evidence=[],
            missing_information=[],
        )
    )

    session.commit()

    return document


def test_final_assessment_updates_document(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    with Session(db_engine) as session:
        document = add_complete_document(
            session,
            pdf_path,
        )

        result = assess_and_store_final_full_text(
            session,
            document_id=document.id,
            criteria="Test research criteria and taxonomy.",
            assessor=FakeFinalAssessor(),
        )

        session.commit()
        session.refresh(document)

        assert result.assessment_number == 2
        assert result.decision == (
            FinalRelevanceDecision.RELEVANT
        )

        assert document.relevance_status == (
            RelevanceStatus.RELEVANT
        )
        assert document.relevance_score == pytest.approx(0.94)
        assert document.relevance_confidence == pytest.approx(0.91)
        assert document.primary_category == "political polemic"
        assert document.topic == "Conflict over public authority"
        assert document.summary is not None
        assert document.processing_status == "analysis_complete"

        final_assessment = session.scalar(
            select(RelevanceAssessment)
            .where(
                RelevanceAssessment.document_id == document.id,
                RelevanceAssessment.sequence_number == 2,
            )
        )

        assert final_assessment is not None
        assert final_assessment.primary_category == (
            "political polemic"
        )
        assert final_assessment.classification_status == (
            ClassificationStatus.FULL_TEXT
        )


def test_incomplete_document_is_rejected(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    with Session(db_engine) as session:
        document = Document(
            source=Source.DUPO,
            source_path=str(pdf_path),
            source_filename=pdf_path.name,
            year=1660,
            language="nl",
            total_units=2,
            units_processed=1,
            text_complete=False,
            relevance_status=RelevanceStatus.RELEVANT,
            processing_status="ocr_partial",
            dupo=DupoMetadata(),
        )

        session.add(document)
        session.commit()

        with pytest.raises(
            FinalAssessmentServiceError,
            match="does not have complete OCR",
        ):
            assess_and_store_final_full_text(
                session,
                document_id=document.id,
                criteria="Test criteria.",
                assessor=FakeFinalAssessor(),
            )
            
def test_openai_final_assessment_is_stored(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    with Session(db_engine) as session:
        document = add_complete_document(
            session,
            pdf_path,
        )

        result = assess_and_store_final_full_text(
            session,
            document_id=document.id,
            criteria="Test research criteria.",
            assessor=FakeFinalAssessor(
                provider=AnalysisProvider.OPENAI,
            ),
        )

        session.commit()
        
        session.refresh(document)

        assert document.primary_category == (
            "political polemic"
        )

        assert document.topic == (
            "Conflict over public authority"
        )

        assert document.summary == "A test summary."

        assert document.relevance_status == (
            RelevanceStatus.RELEVANT
        )

        analysis = session.scalar(
            select(DocumentAnalysis).where(
                DocumentAnalysis.document_id
                == document.id,
                DocumentAnalysis.provider
                == AnalysisProvider.OPENAI.value,
            )
        )

        assert analysis is not None

        assert analysis.model == "gpt-test"
        assert analysis.prompt_version == "test-v1"
        assert analysis.decision == "relevant"

        assert analysis.relevance_score == pytest.approx(
            0.94
        )

        assert analysis.confidence == pytest.approx(
            0.91
        )

        assert analysis.primary_category == (
            "political polemic"
        )

        assert analysis.topic == (
            "Conflict over public authority"
        )

        assert analysis.summary == "A test summary."

        assert result.provider == AnalysisProvider.OPENAI
        
def test_anthropic_analysis_does_not_replace_canonical_openai_result(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    with Session(db_engine) as session:
        document = add_complete_document(
            session,
            pdf_path,
        )

        assess_and_store_final_full_text(
            session,
            document_id=document.id,
            criteria="Test criteria.",
            assessor=FakeFinalAssessor(
                provider=AnalysisProvider.OPENAI,
                category="OpenAI category",
                topic="OpenAI topic",
                summary="OpenAI summary",
            ),
        )

        session.commit()
        
        assess_and_store_final_full_text(
            session,
            document_id=document.id,
            criteria="Test criteria.",
            assessor=FakeFinalAssessor(
                provider=AnalysisProvider.ANTHROPIC,
                category="Claude category",
                topic="Claude topic",
                summary="Claude summary",
            ),
            overwrite=True,
        )

        session.commit()
        session.refresh(document)
        
        analyses = list(
            session.scalars(
                select(DocumentAnalysis)
                .where(
                    DocumentAnalysis.document_id
                    == document.id
                )
                .order_by(DocumentAnalysis.id)
            )
        )

        assert len(analyses) == 2

        assert analyses[0].provider == "openai"
        assert analyses[0].summary == "OpenAI summary"

        assert analyses[1].provider == "anthropic"
        assert analyses[1].summary == "Claude summary"
        
        assert document.summary == "OpenAI summary"
        assert document.primary_category == (
            "OpenAI category"
        )
        assert document.topic == "OpenAI topic"