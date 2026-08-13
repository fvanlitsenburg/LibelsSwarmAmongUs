"""Store final full-text document assessments."""

from dataclasses import dataclass
from math import ceil

from sqlalchemy import select
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import (
    Document,
    DocumentAnalysis,
    RelevanceAssessment,
)
from historical_text_pipeline.domain import (
    AnalysisProvider,
    ClassificationStatus,
    RelevanceStatus,
)
from historical_text_pipeline.relevance.final_assessment import (
    FinalAssessmentRun,
    FinalAssessor,
    FinalRelevanceDecision,
)
from historical_text_pipeline.relevance.service import (
    build_accumulated_page_text,
    get_latest_assessment,
    update_provider_state,
)


class FinalAssessmentServiceError(Exception):
    """Raised when a final assessment cannot be stored."""


@dataclass(frozen=True, slots=True)
class StoredFinalAssessmentResult:
    """Result of storing a complete-document assessment."""

    document_id: int
    assessment_number: int
    decision: FinalRelevanceDecision
    relevance_status: RelevanceStatus
    relevance_score: float
    confidence: float
    category: str
    topic: str
    summary: str
    relevance_explanation: str
    response_id: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    provider: AnalysisProvider
    prompt_version: str


def estimate_text_tokens(text: str) -> int:
    """
    Roughly estimate text tokens before making an API request.

    The estimate is intentionally conservative and is used only as a
    cost-safety warning.
    """

    return max(1, ceil(len(text) / 4))


def build_document_context(document: Document) -> str:
    """Create concise catalogue context for the final assessor."""

    details = [
        f"Internal document ID: {document.id}",
        f"Source: {document.source.value}",
        f"Year: {document.year or 'unknown'}",
        f"Title: {document.title or 'unknown'}",
        f"Author: {document.author or 'unknown'}",
        f"Filename: {document.source_filename or 'unknown'}",
    ]

    if document.dupo is not None:
        details.extend(
            [
                (
                    "DUPO ID: "
                    f"{document.dupo.dupo_id or 'unknown'}"
                ),
                (
                    "Knuttel number: "
                    f"{document.dupo.knuttel_number or 'unknown'}"
                ),
            ]
        )

    return "\n".join(details)


def _map_final_status(
    decision: FinalRelevanceDecision,
) -> RelevanceStatus:
    """Map the final structured decision to the database enum."""

    if decision == FinalRelevanceDecision.RELEVANT:
        return RelevanceStatus.RELEVANT

    if decision == FinalRelevanceDecision.IRRELEVANT:
        return RelevanceStatus.IRRELEVANT

    return RelevanceStatus.UNCERTAIN


def store_final_assessment_run(
    session: Session,
    *,
    document: Document,
    run: FinalAssessmentRun,
) -> StoredFinalAssessmentResult:
    """Save a completed API assessment and update the document."""

    latest = get_latest_assessment(
        session,
        document_id=document.id,
        provider=run.provider,
    )

    assessment_number = (
        latest.sequence_number + 1
        if latest is not None
        else 1
    )

    output = run.output
    relevance_status = _map_final_status(
        output.decision
    )

    analysis = DocumentAnalysis(
        document_id=document.id,
        provider=run.provider.value,
        model=run.model,
        prompt_version=run.prompt_version,
        decision=output.decision.value,
        relevance_score=output.relevance_score,
        confidence=output.confidence,
        primary_category=output.category,
        topic=output.topic,
        relevance_explanation=(
            output.relevance_explanation
        ),
        summary=output.summary,
        supporting_evidence=(
            output.supporting_evidence
        ),
        caveats=output.caveats,
        response_id=run.response_id,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
    )

    session.add(analysis)

    assessment = RelevanceAssessment(
        document_id=document.id,
        sequence_number=assessment_number,
        units_processed=document.total_units,
        decision=relevance_status,
        relevance_score=output.relevance_score,
        confidence=output.confidence,
        reason=output.relevance_explanation,
        primary_category=output.category,
        topic=output.topic,
        classification_status=ClassificationStatus.FULL_TEXT,
        supporting_evidence=output.supporting_evidence,
        missing_information=output.caveats,
        provider=run.provider.value,
        model=run.model,
        prompt_version=run.prompt_version,
        response_id=run.response_id,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
    )

    session.add(assessment)
    
    update_provider_state(
        session,
        document=document,
        provider=run.provider,
        relevance_status=relevance_status,
        relevance_score=output.relevance_score,
        confidence=output.confidence,
        category=output.category,
        topic=output.topic,
        reason=output.relevance_explanation,
        assessment_number=assessment_number,
        units_processed=document.total_units or 0,
    )

    if relevance_status == RelevanceStatus.RELEVANT:
        document.processing_status = "analysis_complete"

    elif relevance_status == RelevanceStatus.IRRELEVANT:
        document.processing_status = "final_irrelevant"

    else:
        document.processing_status = "needs_manual_review"

    session.flush()

    return StoredFinalAssessmentResult(
        document_id=document.id,
        assessment_number=assessment_number,
        provider=run.provider,
        model=run.model,
        prompt_version=run.prompt_version,
        decision=output.decision,
        relevance_status=relevance_status,
        relevance_score=output.relevance_score,
        confidence=output.confidence,
        category=output.category,
        topic=output.topic,
        summary=output.summary,
        relevance_explanation=(
            output.relevance_explanation
        ),
        response_id=run.response_id or "",
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
    )


def assess_and_store_final_full_text(
    session: Session,
    *,
    document_id: int,
    criteria: str,
    assessor: FinalAssessor,
    overwrite: bool = False,
) -> StoredFinalAssessmentResult:
    """Assess the complete transcription and store its final analysis."""

    document = session.get(Document, document_id)

    if document is None:
        raise FinalAssessmentServiceError(
            f"Document {document_id} does not exist."
        )

    if not document.text_complete:
        raise FinalAssessmentServiceError(
            f"Document {document_id} does not have complete OCR."
        )

    if document.total_units is None:
        raise FinalAssessmentServiceError(
            f"Document {document_id} has no recorded page count."
        )

    existing_analysis = session.scalar(
        select(DocumentAnalysis.id)
        .where(
            DocumentAnalysis.document_id == document.id,
            DocumentAnalysis.provider == assessor.provider.value,
        )
        .order_by(DocumentAnalysis.id.desc())
        .limit(1)
    )

    if existing_analysis is not None and not overwrite:
        raise FinalAssessmentServiceError(
            f"Document {document_id} already has a final "
            f"{assessor.provider.value} analysis. "
            "Use overwrite=True to reassess it."
        )

    transcription = build_accumulated_page_text(
        session,
        document_id=document.id,
        through_page=document.total_units,
    )

    context = build_document_context(document)

    run = assessor.assess(
        text=transcription,
        criteria=criteria,
        document_context=context,
    )

    return store_final_assessment_run(
        session,
        document=document,
        run=run,
    )