"""Store progressive relevance assessments in PostgreSQL."""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import (
    Document,
    DocumentProviderState,
    DocumentTextUnit,
    RelevanceAssessment,
)
from historical_text_pipeline.domain import (
    AnalysisProvider,
    ClassificationStatus,
    RelevanceStatus,
    Source,
)
from historical_text_pipeline.relevance.base import (
    RelevanceAssessmentOutput,
    RelevanceAssessor,
    RelevanceDecision,
)


class RelevanceServiceError(Exception):
    """Raised when a document cannot be assessed."""


@dataclass(frozen=True, slots=True)
class StoredRelevanceResult:
    """Result of one stored relevance assessment."""

    document_id: int
    assessment_number: int
    pages_assessed: int
    decision: RelevanceDecision
    relevance_status: RelevanceStatus
    relevance_score: float
    confidence: float
    category: str
    topic: str
    reason: str
    stop_confirmed: bool


def get_latest_assessment(
    session: Session,
    *,
    document_id: int,
    provider: AnalysisProvider = AnalysisProvider.OPENAI,
) -> RelevanceAssessment | None:
    """Return the most recent assessment for a document."""

    return session.scalar(
        select(RelevanceAssessment)
        .where(
            RelevanceAssessment.document_id == document_id,
            RelevanceAssessment.provider == provider.value,
        )
        .order_by(
            RelevanceAssessment.sequence_number.desc()
        )
        .limit(1)
    )


def get_next_batch_end_page(
    session: Session,
    *,
    document_id: int,
    batch_size: int,
    provider: AnalysisProvider = AnalysisProvider.OPENAI,
) -> int | None:
    """Return the last page number of the provider's next batch."""

    if batch_size < 1:
        raise ValueError("Batch size must be at least one.")

    document = session.get(Document, document_id)

    if document is None:
        raise RelevanceServiceError(
            f"Document {document_id} does not exist."
        )

    if document.total_units is None:
        raise RelevanceServiceError(
            f"Document {document_id} has not been inspected."
        )

    latest = get_latest_assessment(
        session,
        document_id=document_id,
        provider=provider,
    )

    previously_assessed = (
        latest.units_processed
        if latest is not None
        else 0
    )

    if previously_assessed >= document.total_units:
        return None

    return min(
        previously_assessed + batch_size,
        document.total_units,
    )
    
def _provider_relevance_status(
    *,
    output: RelevanceAssessmentOutput,
    stop_confirmed: bool,
    final_progressive_assessment: bool,
) -> RelevanceStatus:
    """Determine this provider's current document-level conclusion."""

    if output.decision == RelevanceDecision.CONTINUE:
        return RelevanceStatus.RELEVANT

    if stop_confirmed:
        return RelevanceStatus.IRRELEVANT

    if (
        final_progressive_assessment
        and output.decision
        in (
            RelevanceDecision.STOP,
            RelevanceDecision.UNCERTAIN,
        )
    ):
        return RelevanceStatus.IRRELEVANT

    return RelevanceStatus.UNCERTAIN

def build_accumulated_page_text(
    session: Session,
    *,
    document_id: int,
    through_page: int,
) -> str:
    """Combine stored page text through a specified page."""

    text_units = list(
        session.scalars(
            select(DocumentTextUnit)
            .where(
                DocumentTextUnit.document_id == document_id,
                DocumentTextUnit.unit_type == "page",
                DocumentTextUnit.unit_number <= through_page,
            )
            .order_by(DocumentTextUnit.unit_number)
        )
    )

    pages_by_number = {
        unit.unit_number: unit
        for unit in text_units
    }

    missing_pages = [
        page_number
        for page_number in range(1, through_page + 1)
        if page_number not in pages_by_number
    ]

    if missing_pages:
        missing = ", ".join(
            str(page_number)
            for page_number in missing_pages
        )

        raise RelevanceServiceError(
            "Cannot assess the document because these pages "
            f"have not been OCR'd: {missing}"
        )

    sections = [
        (
            f"--- PDF PAGE {page_number} ---\n"
            f"{pages_by_number[page_number].text}"
        )
        for page_number in range(1, through_page + 1)
    ]

    return "\n\n".join(sections)


def _assessment_status(
    decision: RelevanceDecision,
) -> RelevanceStatus:
    """Map an action decision to the existing database enum."""

    if decision == RelevanceDecision.CONTINUE:
        return RelevanceStatus.RELEVANT

    if decision == RelevanceDecision.STOP:
        return RelevanceStatus.IRRELEVANT

    return RelevanceStatus.UNCERTAIN


def _is_confirmed_stop(
    *,
    current: RelevanceAssessmentOutput,
    previous: RelevanceAssessment | None,
    through_page: int,
    confidence_threshold: float,
) -> bool:
    """Require two confident STOP decisions on different batches."""

    if current.decision != RelevanceDecision.STOP:
        return False

    if current.confidence < confidence_threshold:
        return False

    if previous is None:
        return False

    if previous.decision != RelevanceStatus.IRRELEVANT:
        return False

    if previous.confidence < confidence_threshold:
        return False

    return previous.units_processed < through_page


def assess_and_store_relevance(
    session: Session,
    *,
    document_id: int,
    through_page: int,
    criteria: str,
    assessor: RelevanceAssessor,
    stop_confidence_threshold: float = 0.80,
    max_assessments: int = 4,
) -> StoredRelevanceResult:
    """Assess accumulated OCR text and save the provider-specific result."""

    document = session.get(Document, document_id)

    if document is None:
        raise RelevanceServiceError(
            f"Document {document_id} does not exist."
        )

    if document.source != Source.DUPO:
        raise RelevanceServiceError(
            f"Document {document_id} is not a DUPO document."
        )

    # IMPORTANT: previous assessment from this provider only.
    previous = get_latest_assessment(
        session,
        document_id=document_id,
        provider=assessor.provider,
    )

    assessment_number = (
        previous.sequence_number + 1
        if previous is not None
        else 1
    )

    final_progressive_assessment = (
        assessment_number >= max_assessments
        or through_page >= document.total_units
    )

    text = build_accumulated_page_text(
        session,
        document_id=document_id,
        through_page=through_page,
    )

    run = assessor.assess(
        text=text,
        criteria=criteria,
        assessment_number=assessment_number,
        final_progressive_assessment=(
            final_progressive_assessment
        ),
    )

    if run.provider != assessor.provider:
        raise RelevanceServiceError(
            "Relevance assessor returned a mismatched provider."
        )

    output = run.output

    # This represents what the individual model call said.
    assessment_status = _assessment_status(
        output.decision
    )

    stop_confirmed = _is_confirmed_stop(
        current=output,
        previous=previous,
        through_page=through_page,
        confidence_threshold=stop_confidence_threshold,
    )

    # This represents this provider's current conclusion.
    provider_status = _provider_relevance_status(
        output=output,
        stop_confirmed=stop_confirmed,
        final_progressive_assessment=(
            final_progressive_assessment
        ),
    )

    classification_status = (
        ClassificationStatus.FULL_TEXT
        if document.text_complete
        else ClassificationStatus.PARTIAL_TEXT
    )

    stored_assessment = RelevanceAssessment(
        document_id=document.id,
        sequence_number=assessment_number,
        units_processed=through_page,
        decision=assessment_status,
        relevance_score=output.relevance_score,
        confidence=output.confidence,
        reason=output.reason,
        primary_category=output.category,
        topic=output.topic,
        classification_status=classification_status,
        supporting_evidence=output.supporting_evidence,
        missing_information=output.missing_information,
        provider=run.provider.value,
        model=run.model,
        prompt_version=run.prompt_version,
        response_id=run.response_id,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
    )

    session.add(stored_assessment)
    
    update_provider_state(
        session,
        document=document,
        provider=run.provider,
        relevance_status=provider_status,
        relevance_score=output.relevance_score,
        confidence=output.confidence,
        category=output.category,
        topic=output.topic,
        reason=output.reason,
        assessment_number=assessment_number,
        units_processed=through_page,
    )

    # Only OpenAI currently controls the canonical Document fields.
    if run.provider == AnalysisProvider.OPENAI:
        document.relevance_status = provider_status
        document.relevance_score = output.relevance_score
        document.relevance_confidence = output.confidence
        document.relevance_reason = output.reason
        document.primary_category = output.category
        document.topic = output.topic

        if provider_status == RelevanceStatus.RELEVANT:
            document.processing_status = (
                "relevance_confirmed"
            )

        elif provider_status == RelevanceStatus.IRRELEVANT:
            document.processing_status = (
                "relevance_stopped"
            )

        elif output.decision == RelevanceDecision.STOP:
            document.processing_status = (
                "relevance_stop_pending"
            )

        else:
            document.processing_status = (
                "relevance_uncertain"
            )

    session.flush()

    return StoredRelevanceResult(
        document_id=document.id,
        assessment_number=assessment_number,
        pages_assessed=through_page,
        decision=output.decision,
        relevance_status=provider_status,
        relevance_score=output.relevance_score,
        confidence=output.confidence,
        category=output.category,
        topic=output.topic,
        reason=output.reason,
        stop_confirmed=(
            stop_confirmed
            or (
                final_progressive_assessment
                and provider_status
                == RelevanceStatus.IRRELEVANT
            )
        ),
    )
    
def get_provider_state(
    session: Session,
    *,
    document_id: int,
    provider: AnalysisProvider,
) -> DocumentProviderState | None:
    """Return the current state for one provider."""

    return session.scalar(
        select(DocumentProviderState)
        .where(
            DocumentProviderState.document_id
            == document_id,
            DocumentProviderState.provider
            == provider.value,
        )
        .limit(1)
    )
    
def update_provider_state(
    session: Session,
    *,
    document: Document,
    provider: AnalysisProvider,
    relevance_status: RelevanceStatus,
    relevance_score: float,
    confidence: float,
    category: str,
    topic: str,
    reason: str,
    assessment_number: int,
    units_processed: int,
) -> DocumentProviderState:
    """Create or update the provider's current document state."""

    state = get_provider_state(
        session,
        document_id=document.id,
        provider=provider,
    )

    if state is None:
        state = DocumentProviderState(
            document_id=document.id,
            provider=provider.value,
            relevance_status=relevance_status.value,
            relevance_score=relevance_score,
            confidence=confidence,
            primary_category=category,
            topic=topic,
            relevance_reason=reason,
            last_assessment_number=assessment_number,
            units_processed=units_processed,
        )

        session.add(state)

    else:
        state.relevance_status = relevance_status.value
        state.relevance_score = relevance_score
        state.confidence = confidence
        state.primary_category = category
        state.topic = topic
        state.relevance_reason = reason
        state.last_assessment_number = assessment_number
        state.units_processed = units_processed

    return state