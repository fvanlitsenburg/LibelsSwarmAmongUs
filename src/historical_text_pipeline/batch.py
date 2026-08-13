"""Select DUPO documents for batch processing."""

from enum import StrEnum

from sqlalchemy import exists, or_, select
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import (
    Document,
    DocumentAnalysis,
    DocumentProviderState,
)
from historical_text_pipeline.domain import (
    AnalysisProvider,
    RelevanceStatus,
    Source,
)


class DupoBatchStage(StrEnum):
    """Available DUPO batch-processing stages."""

    RELEVANCE = "relevance"
    OCR = "ocr"
    FINAL = "final"
    PIPELINE = "pipeline"


EXCLUDED_PROCESSING_STATUSES = (
    "pdf_error",
    "needs_manual_review",
)


def get_dupo_batch_document_ids(
    session: Session,
    *,
    stage: DupoBatchStage,
    provider: AnalysisProvider,
    limit: int,
    start_id: int | None = None,
) -> list[int]:
    """Return provider-specific eligible DUPO document IDs."""

    if limit < 1:
        raise ValueError("Batch limit must be at least one.")

    provider_state_exists = exists(
        select(DocumentProviderState.id).where(
            DocumentProviderState.document_id == Document.id,
            DocumentProviderState.provider == provider.value,
        )
    )

    provider_uncertain_exists = exists(
        select(DocumentProviderState.id).where(
            DocumentProviderState.document_id == Document.id,
            DocumentProviderState.provider == provider.value,
            DocumentProviderState.relevance_status
            == RelevanceStatus.UNCERTAIN.value,
        )
    )

    provider_relevant_exists = exists(
        select(DocumentProviderState.id).where(
            DocumentProviderState.document_id == Document.id,
            DocumentProviderState.provider == provider.value,
            DocumentProviderState.relevance_status
            == RelevanceStatus.RELEVANT.value,
        )
    )

    final_analysis_exists = exists(
        select(DocumentAnalysis.id).where(
            DocumentAnalysis.document_id == Document.id,
            DocumentAnalysis.provider == provider.value,
        )
    )

    conditions = [
        Document.source == Source.DUPO,
        Document.source_path.is_not(None),
        Document.total_units.is_not(None),
        or_(
            Document.processing_status.is_(None),
            Document.processing_status.notin_(
                EXCLUDED_PROCESSING_STATUSES
            ),
        ),
    ]

    if start_id is not None:
        conditions.append(
            Document.id >= start_id
        )

    if stage == DupoBatchStage.RELEVANCE:
        # This provider has not resolved progressive relevance yet.
        #
        # Do not require incomplete OCR here. A document may already
        # have full shared OCR because another provider caused it to
        # be completed, while this provider still needs its own
        # progressive relevance experiment.
        conditions.append(
            or_(
                ~provider_state_exists,
                provider_uncertain_exists,
            )
        )

    elif stage == DupoBatchStage.OCR:
        # OCR is shared, but completion is triggered by this
        # provider having judged the document relevant.
        conditions.extend(
            [
                Document.text_complete.is_(False),
                provider_relevant_exists,
            ]
        )

    elif stage == DupoBatchStage.FINAL:
        # Once full OCR exists, this provider may receive a final
        # assessment regardless of its earlier progressive verdict.
        conditions.extend(
            [
                Document.text_complete.is_(True),
                ~final_analysis_exists,
            ]
        )

    elif stage == DupoBatchStage.PIPELINE:
        conditions.append(
            or_(
                # Progressive relevance not resolved for provider.
                ~provider_state_exists,
                provider_uncertain_exists,

                # Provider found relevance, but shared OCR is
                # incomplete.
                (
                    provider_relevant_exists
                    & Document.text_complete.is_(False)
                ),

                # Shared OCR is complete, but this provider still
                # lacks a final full-text analysis.
                (
                    Document.text_complete.is_(True)
                    & ~final_analysis_exists
                ),
            )
        )

    else:
        raise ValueError(
            f"Unsupported DUPO batch stage: {stage}"
        )

    statement = (
        select(Document.id)
        .where(*conditions)
        .order_by(Document.id)
        .limit(limit)
    )

    return list(session.scalars(statement))

def get_anthropic_backfill_document_ids(
    session: Session,
    *,
    limit: int,
    start_id: int | None = None,
) -> list[int]:
    """Return DUPO documents still needing Anthropic processing."""

    if limit < 1:
        raise ValueError("Batch limit must be at least one.")

    final_analysis_exists = exists(
        select(DocumentAnalysis.id).where(
            DocumentAnalysis.document_id == Document.id,
            DocumentAnalysis.provider
            == AnalysisProvider.ANTHROPIC.value,
        )
    )

    anthropic_irrelevant_state_exists = exists(
        select(DocumentProviderState.id).where(
            DocumentProviderState.document_id == Document.id,
            DocumentProviderState.provider
            == AnalysisProvider.ANTHROPIC.value,
            DocumentProviderState.relevance_status
            == RelevanceStatus.IRRELEVANT.value,
        )
    )

    conditions = [
        Document.source == Source.DUPO,
        Document.source_path.is_not(None),
        Document.total_units.is_not(None),

        # A completed Claude full-text analysis means the
        # backfill for this document is finished.
        ~final_analysis_exists,

        # A partially OCR'd document that Claude has already
        # rejected is also finished. If full OCR already exists,
        # however, we still want Claude's final full-text analysis.
        or_(
            Document.text_complete.is_(True),
            ~anthropic_irrelevant_state_exists,
        ),

        # Completely unreadable PDFs cannot be processed.
        or_(
            Document.processing_status.is_(None),
            Document.processing_status != "pdf_error",
        ),
    ]

    if start_id is not None:
        conditions.append(
            Document.id >= start_id
        )

    statement = (
        select(Document.id)
        .where(*conditions)
        .order_by(Document.id)
        .limit(limit)
    )

    return list(session.scalars(statement))