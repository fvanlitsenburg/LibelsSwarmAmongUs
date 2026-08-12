"""Select DUPO documents for batch processing."""

from enum import StrEnum

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import Document
from historical_text_pipeline.domain import (
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
    limit: int,
    start_id: int | None = None,
) -> list[int]:
    """Return eligible document IDs for one batch stage."""

    if limit < 1:
        raise ValueError("Batch limit must be at least one.")

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
        conditions.append(Document.id >= start_id)

    if stage == DupoBatchStage.RELEVANCE:
        conditions.extend(
            [
                Document.text_complete.is_(False),
                Document.relevance_status.in_(
                    (
                        RelevanceStatus.NOT_ASSESSED,
                        RelevanceStatus.UNCERTAIN,
                    )
                ),
            ]
        )

    elif stage == DupoBatchStage.OCR:
        conditions.extend(
            [
                Document.text_complete.is_(False),
                Document.relevance_status
                == RelevanceStatus.RELEVANT,
            ]
        )

    elif stage == DupoBatchStage.FINAL:
        conditions.extend(
            [
                Document.text_complete.is_(True),
                or_(
                    Document.summary.is_(None),
                    Document.summary == "",
                ),
                Document.relevance_status.in_(
                    (
                        RelevanceStatus.NOT_ASSESSED,
                        RelevanceStatus.UNCERTAIN,
                        RelevanceStatus.RELEVANT,
                    )
                ),
            ]
        )

    elif stage == DupoBatchStage.PIPELINE:
        conditions.append(
            or_(
                # Not yet resolved for relevance.
                Document.relevance_status.in_(
                    (
                        RelevanceStatus.NOT_ASSESSED,
                        RelevanceStatus.UNCERTAIN,
                    )
                ),

                # Relevant, but OCR still incomplete.
                (
                    (Document.relevance_status == RelevanceStatus.RELEVANT)
                    & (Document.text_complete.is_(False))
                ),

                # Complete text but no final assessment yet.
                (
                    (Document.text_complete.is_(True))
                    & or_(
                        Document.summary.is_(None),
                        Document.summary == "",
                    )
                    & (
                        Document.relevance_status
                        != RelevanceStatus.IRRELEVANT
                    )
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