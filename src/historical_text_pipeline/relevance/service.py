"""Store progressive relevance assessments in PostgreSQL."""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import (
    Document,
    DocumentTextUnit,
    RelevanceAssessment,
)
from historical_text_pipeline.domain import (
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
) -> RelevanceAssessment | None:
    """Return the most recent assessment for a document."""

    return session.scalar(
        select(RelevanceAssessment)
        .where(
            RelevanceAssessment.document_id == document_id,
        )
        .order_by(
            RelevanceAssessment.sequence_number.desc(),
        )
        .limit(1)
    )


def get_next_batch_end_page(
    session: Session,
    *,
    document_id: int,
    batch_size: int,
) -> int | None:
    """Return the last page number of the next assessment batch."""

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
) -> StoredRelevanceResult:
    """Assess accumulated OCR text and save the result."""

    document = session.get(Document, document_id)

    if document is None:
        raise RelevanceServiceError(
            f"Document {document_id} does not exist."
        )

    if document.source != Source.DUPO:
        raise RelevanceServiceError(
            f"Document {document_id} is not a DUPO document."
        )

    previous = get_latest_assessment(
        session,
        document_id=document_id,
    )

    assessment_number = (
        previous.sequence_number + 1
        if previous is not None
        else 1
    )

    text = build_accumulated_page_text(
        session,
        document_id=document_id,
        through_page=through_page,
    )

    output = assessor.assess(
        text=text,
        criteria=criteria,
        assessment_number=assessment_number,
    )

    assessment_status = _assessment_status(
        output.decision
    )

    stop_confirmed = _is_confirmed_stop(
        current=output,
        previous=previous,
        through_page=through_page,
        confidence_threshold=stop_confidence_threshold,
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
    )

    session.add(stored_assessment)

    document.relevance_score = output.relevance_score
    document.relevance_confidence = output.confidence
    document.relevance_reason = output.reason
    document.primary_category = output.category
    document.topic = output.topic
    document.classification_status = classification_status

    if output.decision == RelevanceDecision.CONTINUE:
        document.relevance_status = RelevanceStatus.RELEVANT
        document.processing_status = "relevance_confirmed"

    elif stop_confirmed:
        document.relevance_status = RelevanceStatus.IRRELEVANT
        document.processing_status = "relevance_stopped"

    else:
        # A first STOP or an UNCERTAIN result requires another batch.
        document.relevance_status = RelevanceStatus.UNCERTAIN

        document.processing_status = (
            "relevance_stop_pending"
            if output.decision == RelevanceDecision.STOP
            else "relevance_uncertain"
        )

    session.flush()

    return StoredRelevanceResult(
        document_id=document.id,
        assessment_number=assessment_number,
        pages_assessed=through_page,
        decision=output.decision,
        relevance_status=document.relevance_status,
        relevance_score=output.relevance_score,
        confidence=output.confidence,
        category=output.category,
        topic=output.topic,
        reason=output.reason,
        stop_confirmed=stop_confirmed,
    )