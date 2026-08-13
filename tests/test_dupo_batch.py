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
    DocumentAnalysis,
    DocumentProviderState,
    DupoMetadata,
)
from historical_text_pipeline.domain import (
    AnalysisProvider,
    RelevanceStatus,
    Source,
)


def add_document(
    session: Session,
    tmp_path: Path,
    *,
    name: str,
    relevance_status: RelevanceStatus | None,
    text_complete: bool,
    provider: AnalysisProvider = AnalysisProvider.OPENAI,
    has_final_analysis: bool = False,
    processing_status: str = "inspected",
) -> Document:
    """Create one document in a provider-specific pipeline state."""

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
        processing_status=processing_status,
        dupo=DupoMetadata(),
    )

    session.add(document)
    session.flush()

    # No provider-state row means this provider has never
    # assessed the document.
    if relevance_status is not None:
        session.add(
            DocumentProviderState(
                document_id=document.id,
                provider=provider.value,
                relevance_status=relevance_status.value,
                relevance_score=0.8,
                confidence=0.9,
                primary_category="test-category",
                topic="test topic",
                relevance_reason="Test relevance reason.",
                last_assessment_number=1,
                units_processed=3,
            )
        )

    if has_final_analysis:
        session.add(
            DocumentAnalysis(
                document_id=document.id,
                provider=provider.value,
                model="test-model",
                prompt_version="test-v1",
                decision=(
                    relevance_status.value
                    if relevance_status is not None
                    else RelevanceStatus.RELEVANT.value
                ),
                relevance_score=0.8,
                confidence=0.9,
                primary_category="test-category",
                topic="test topic",
                relevance_explanation=(
                    "Test final relevance explanation."
                ),
                summary="Already analyzed.",
                supporting_evidence=[],
                caveats=[],
            )
        )

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
            relevance_status=None,
            text_complete=False,
        )

        uncertain = add_document(
            session,
            tmp_path,
            name="uncertain",
            relevance_status=RelevanceStatus.UNCERTAIN,
            text_complete=False,
        )

        relevant_partial = add_document(
            session,
            tmp_path,
            name="relevant-partial",
            relevance_status=RelevanceStatus.RELEVANT,
            text_complete=False,
        )

        relevant_complete = add_document(
            session,
            tmp_path,
            name="relevant-complete",
            relevance_status=RelevanceStatus.RELEVANT,
            text_complete=True,
        )

        uncertain_complete = add_document(
            session,
            tmp_path,
            name="uncertain-complete",
            relevance_status=RelevanceStatus.UNCERTAIN,
            text_complete=True,
        )

        add_document(
            session,
            tmp_path,
            name="irrelevant",
            relevance_status=RelevanceStatus.IRRELEVANT,
            text_complete=False,
        )

        add_document(
            session,
            tmp_path,
            name="already-final",
            relevance_status=RelevanceStatus.RELEVANT,
            text_complete=True,
            has_final_analysis=True,
        )

        add_document(
            session,
            tmp_path,
            name="pdf-error",
            relevance_status=None,
            text_complete=False,
            processing_status="pdf_error",
        )

        session.commit()

        relevance_ids = get_dupo_batch_document_ids(
            session,
            stage=DupoBatchStage.RELEVANCE,
            provider=AnalysisProvider.OPENAI,
            limit=100,
        )

        ocr_ids = get_dupo_batch_document_ids(
            session,
            stage=DupoBatchStage.OCR,
            provider=AnalysisProvider.OPENAI,
            limit=100,
        )

        final_ids = get_dupo_batch_document_ids(
            session,
            stage=DupoBatchStage.FINAL,
            provider=AnalysisProvider.OPENAI,
            limit=100,
        )

        assert relevance_ids == [
            pending.id,
            uncertain.id,
            uncertain_complete.id,
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
            relevance_status=None,
            text_complete=False,
        )

        add_document(
            session,
            tmp_path,
            name="second",
            relevance_status=None,
            text_complete=False,
        )

        session.commit()

        result = get_dupo_batch_document_ids(
            session,
            stage=DupoBatchStage.RELEVANCE,
            provider=AnalysisProvider.OPENAI,
            limit=1,
        )

        assert result == [first.id]


def test_relevance_selection_is_provider_specific(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    with Session(db_engine) as session:
        document = add_document(
            session,
            tmp_path,
            name="provider-specific",
            relevance_status=RelevanceStatus.RELEVANT,
            text_complete=False,
            provider=AnalysisProvider.OPENAI,
        )

        session.commit()

        openai_ids = get_dupo_batch_document_ids(
            session,
            stage=DupoBatchStage.RELEVANCE,
            provider=AnalysisProvider.OPENAI,
            limit=100,
        )

        anthropic_ids = get_dupo_batch_document_ids(
            session,
            stage=DupoBatchStage.RELEVANCE,
            provider=AnalysisProvider.ANTHROPIC,
            limit=100,
        )

        assert document.id not in openai_ids
        assert document.id in anthropic_ids


def test_final_selection_is_provider_specific(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    with Session(db_engine) as session:
        document = add_document(
            session,
            tmp_path,
            name="final-provider-specific",
            relevance_status=RelevanceStatus.RELEVANT,
            text_complete=True,
            provider=AnalysisProvider.OPENAI,
            has_final_analysis=True,
        )

        session.commit()

        openai_ids = get_dupo_batch_document_ids(
            session,
            stage=DupoBatchStage.FINAL,
            provider=AnalysisProvider.OPENAI,
            limit=100,
        )

        anthropic_ids = get_dupo_batch_document_ids(
            session,
            stage=DupoBatchStage.FINAL,
            provider=AnalysisProvider.ANTHROPIC,
            limit=100,
        )

        assert document.id not in openai_ids
        assert document.id in anthropic_ids