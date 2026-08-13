"""Tests for progressive relevance decisions."""

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import (
    Document,
    DocumentProviderState,
    DocumentTextUnit,
    DupoMetadata,
    RelevanceAssessment,
)
from historical_text_pipeline.domain import (
    AnalysisProvider,
    RelevanceStatus,
    Source,
)
from historical_text_pipeline.relevance.base import (
    RelevanceAssessmentOutput,
    RelevanceAssessmentRun,
    RelevanceDecision,
)
from historical_text_pipeline.relevance.service import (
    assess_and_store_relevance,
    get_next_batch_end_page,
)


class FakeAssessor:
    """Return predefined relevance outputs."""

    def __init__(
        self,
        outputs: list[RelevanceAssessmentOutput],
        *,
        provider: AnalysisProvider = AnalysisProvider.OPENAI,
    ) -> None:
        self.outputs = outputs
        self._provider = provider
        self.calls = 0

    @property
    def provider(self) -> AnalysisProvider:
        """Return the provider represented by this fake."""

        return self._provider

    def assess(
        self,
        *,
        text: str,
        criteria: str,
        assessment_number: int,
        final_progressive_assessment: bool = False,
    ) -> RelevanceAssessmentRun:
        """Return the next predefined result."""

        assert text
        assert criteria
        assert assessment_number == self.calls + 1

        output = self.outputs[self.calls]
        self.calls += 1

        return RelevanceAssessmentRun(
            output=output,
            provider=self._provider,
            model=(
                "gpt-test"
                if self._provider == AnalysisProvider.OPENAI
                else "claude-test"
            ),
            prompt_version="test-v1",
            response_id=(
                f"test-{self._provider.value}-"
                f"{assessment_number}"
            ),
            input_tokens=500,
            output_tokens=100,
        )

    def close(self) -> None:
        """Match the relevance-assessor protocol."""

def make_output(
    decision: RelevanceDecision,
    *,
    confidence: float = 0.90,
) -> RelevanceAssessmentOutput:
    """Create a test assessment."""

    if decision == RelevanceDecision.CONTINUE:
        relevance_score = 0.90
    elif decision == RelevanceDecision.STOP:
        relevance_score = 0.10
    else:
        relevance_score = 0.50

    return RelevanceAssessmentOutput(
        decision=decision,
        relevance_score=relevance_score,
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

        assess_and_store_relevance(
            session,
            document_id=document.id,
            through_page=3,
            criteria="Test research criteria.",
            assessor=assessor,
        )

        session.commit()
        state = session.scalar(
            select(DocumentProviderState).where(
                DocumentProviderState.document_id == document.id,
                DocumentProviderState.provider
                == AnalysisProvider.OPENAI.value,
            )
        )

        assert state is not None
        assert state.relevance_status == RelevanceStatus.RELEVANT.value
        assert state.primary_category == "test-category"
        assert state.topic == "test topic"


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

        assess_and_store_relevance(
            session,
            document_id=document.id,
            through_page=3,
            criteria="Test research criteria.",
            assessor=assessor,
        )

        session.commit()
        state = session.scalar(
            select(DocumentProviderState).where(
                DocumentProviderState.document_id == document.id,
                DocumentProviderState.provider
                == AnalysisProvider.OPENAI.value,
            )
        )

        assert state is not None
        assert state.relevance_status == RelevanceStatus.UNCERTAIN.value
        assert state.primary_category == "test-category"
        assert state.topic == "test topic"


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

        assess_and_store_relevance(
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

        assess_and_store_relevance(
            session,
            document_id=document.id,
            through_page=6,
            criteria="Test research criteria.",
            assessor=assessor,
        )

        session.commit()
        state = session.scalar(
            select(DocumentProviderState).where(
                DocumentProviderState.document_id == document.id,
                DocumentProviderState.provider
                == AnalysisProvider.OPENAI.value,
            )
        )

        assert state is not None
        assert state.relevance_status == RelevanceStatus.IRRELEVANT.value
        assert state.primary_category == "test-category"
        assert state.topic == "test topic"
        
def test_relevance_sequence_is_independent_per_provider(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    with Session(db_engine) as session:
        document = add_document_with_pages(
            session,
            pdf_path,
        )

        openai_assessor = FakeAssessor(
            [
                make_output(
                    RelevanceDecision.UNCERTAIN
                )
            ],
            provider=AnalysisProvider.OPENAI,
        )

        anthropic_assessor = FakeAssessor(
            [
                make_output(
                    RelevanceDecision.UNCERTAIN
                )
            ],
            provider=AnalysisProvider.ANTHROPIC,
        )

        openai_result = assess_and_store_relevance(
            session,
            document_id=document.id,
            through_page=3,
            criteria="Test criteria.",
            assessor=openai_assessor,
        )

        session.commit()

        anthropic_result = assess_and_store_relevance(
            session,
            document_id=document.id,
            through_page=3,
            criteria="Test criteria.",
            assessor=anthropic_assessor,
        )

        session.commit()

        assessments = list(
            session.scalars(
                select(RelevanceAssessment)
                .where(
                    RelevanceAssessment.document_id
                    == document.id
                )
                .order_by(
                    RelevanceAssessment.provider,
                    RelevanceAssessment.sequence_number,
                )
            )
        )

        assert len(assessments) == 2

        assert openai_result.assessment_number == 1
        assert anthropic_result.assessment_number == 1

        assert {
            (
                assessment.provider,
                assessment.sequence_number,
            )
            for assessment in assessments
        } == {
            ("openai", 1),
            ("anthropic", 1),
        }
        
def test_next_batch_is_independent_per_provider(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    with Session(db_engine) as session:
        document = add_document_with_pages(
            session,
            pdf_path,
            page_count=9,
            stored_through=3,
        )

        openai_assessor = FakeAssessor(
            [
                make_output(
                    RelevanceDecision.UNCERTAIN
                )
            ],
            provider=AnalysisProvider.OPENAI,
        )

        assess_and_store_relevance(
            session,
            document_id=document.id,
            through_page=3,
            criteria="Test criteria.",
            assessor=openai_assessor,
        )

        session.commit()

        openai_next = get_next_batch_end_page(
            session,
            document_id=document.id,
            batch_size=3,
            provider=AnalysisProvider.OPENAI,
        )

        anthropic_next = get_next_batch_end_page(
            session,
            document_id=document.id,
            batch_size=3,
            provider=AnalysisProvider.ANTHROPIC,
        )

        assert openai_next == 6
        assert anthropic_next == 3
        
def test_provider_states_are_independent(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    with Session(db_engine) as session:
        document = add_document_with_pages(
            session,
            pdf_path,
        )

        openai_assessor = FakeAssessor(
            [
                make_output(
                    RelevanceDecision.CONTINUE
                )
            ],
            provider=AnalysisProvider.OPENAI,
        )

        anthropic_assessor = FakeAssessor(
            [
                make_output(
                    RelevanceDecision.STOP
                )
            ],
            provider=AnalysisProvider.ANTHROPIC,
        )

        assess_and_store_relevance(
            session,
            document_id=document.id,
            through_page=3,
            criteria="Test criteria.",
            assessor=openai_assessor,
        )

        session.commit()

        assess_and_store_relevance(
            session,
            document_id=document.id,
            through_page=3,
            criteria="Test criteria.",
            assessor=anthropic_assessor,
        )

        session.commit()

        states = list(
            session.scalars(
                select(DocumentProviderState)
                .where(
                    DocumentProviderState.document_id
                    == document.id
                )
                .order_by(
                    DocumentProviderState.provider
                )
            )
        )

        assert len(states) == 2

        states_by_provider = {
            state.provider: state
            for state in states
        }

        openai_state = states_by_provider["openai"]
        anthropic_state = states_by_provider["anthropic"]

        assert openai_state.relevance_status == "relevant"

        # First STOP is still only uncertain.
        assert anthropic_state.relevance_status == "uncertain"

        assert openai_state.last_assessment_number == 1
        assert anthropic_state.last_assessment_number == 1

        assert openai_state.units_processed == 3
        assert anthropic_state.units_processed == 3
        
def test_provider_state_is_updated_not_duplicated(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    assessor = FakeAssessor(
        [
            make_output(
                RelevanceDecision.UNCERTAIN
            ),
            make_output(
                RelevanceDecision.CONTINUE
            ),
        ],
        provider=AnalysisProvider.ANTHROPIC,
    )

    with Session(db_engine) as session:
        document = add_document_with_pages(
            session,
            pdf_path,
            page_count=6,
            stored_through=3,
        )

        assess_and_store_relevance(
            session,
            document_id=document.id,
            through_page=3,
            criteria="Test criteria.",
            assessor=assessor,
        )

        session.commit()

        add_more_pages(
            session,
            document=document,
            start_page=4,
            end_page=6,
        )

        assess_and_store_relevance(
            session,
            document_id=document.id,
            through_page=6,
            criteria="Test criteria.",
            assessor=assessor,
        )

        session.commit()

        states = list(
            session.scalars(
                select(DocumentProviderState)
                .where(
                    DocumentProviderState.document_id
                    == document.id,
                    DocumentProviderState.provider
                    == "anthropic",
                )
            )
        )

        assert len(states) == 1

        state = states[0]

        assert state.relevance_status == "relevant"
        assert state.last_assessment_number == 2
        assert state.units_processed == 6
        
def test_uncertain_at_end_of_short_document_resolves_irrelevant(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    assessor = FakeAssessor(
        [
            make_output(
                RelevanceDecision.UNCERTAIN
            )
        ],
        provider=AnalysisProvider.ANTHROPIC,
    )

    with Session(db_engine) as session:
        document = add_document_with_pages(
            session,
            pdf_path,
            page_count=2,
            stored_through=2,
        )

        result = assess_and_store_relevance(
            session,
            document_id=document.id,
            through_page=2,
            criteria="Test criteria.",
            assessor=assessor,
            max_assessments=4,
        )

        session.commit()

        assert result.assessment_number == 1
        assert result.relevance_status == RelevanceStatus.IRRELEVANT

        state = session.scalar(
            select(DocumentProviderState).where(
                DocumentProviderState.document_id
                == document.id,
                DocumentProviderState.provider
                == AnalysisProvider.ANTHROPIC.value,
            )
        )

        assert state is not None
        assert state.relevance_status == "irrelevant"
        assert state.units_processed == 2