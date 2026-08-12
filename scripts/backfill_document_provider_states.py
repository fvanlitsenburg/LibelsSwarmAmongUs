"""Backfill current provider state from relevance assessment history."""

from sqlalchemy import select

from historical_text_pipeline.db.models import (
    DocumentProviderState,
    RelevanceAssessment,
)
from historical_text_pipeline.db.session import (
    get_session_factory,
)


def enum_value(value: object) -> str:
    """Return the stored string value of an enum or string."""

    candidate = getattr(value, "value", None)

    if isinstance(candidate, str):
        return candidate

    return str(value)


def main() -> None:
    """Create or update one current state per document/provider."""

    session_factory = get_session_factory()

    created = 0
    updated = 0
    skipped = 0

    with session_factory() as session:
        assessments = list(
            session.scalars(
                select(RelevanceAssessment)
                .order_by(
                    RelevanceAssessment.document_id,
                    RelevanceAssessment.provider,
                    RelevanceAssessment.sequence_number.desc(),
                )
            )
        )

        # Because the query is newest-first within each
        # document/provider pair, the first row we encounter is the
        # current one.
        latest_by_provider: dict[
            tuple[int, str],
            RelevanceAssessment,
        ] = {}

        for assessment in assessments:
            key = (
                assessment.document_id,
                assessment.provider,
            )

            if key not in latest_by_provider:
                latest_by_provider[key] = assessment

        for (
            document_id,
            provider,
        ), assessment in latest_by_provider.items():
            state = session.scalar(
                select(DocumentProviderState)
                .where(
                    DocumentProviderState.document_id
                    == document_id,
                    DocumentProviderState.provider
                    == provider,
                )
                .limit(1)
            )

            relevance_status = enum_value(
                assessment.decision
            )

            if state is None:
                state = DocumentProviderState(
                    document_id=document_id,
                    provider=provider,
                    relevance_status=relevance_status,
                    relevance_score=(
                        assessment.relevance_score
                    ),
                    confidence=assessment.confidence,
                    primary_category=(
                        assessment.primary_category
                    ),
                    topic=assessment.topic,
                    relevance_reason=assessment.reason,
                    last_assessment_number=(
                        assessment.sequence_number
                    ),
                    units_processed=(
                        assessment.units_processed
                    ),
                )

                session.add(state)
                created += 1

            elif (
                state.last_assessment_number
                < assessment.sequence_number
            ):
                state.relevance_status = relevance_status
                state.relevance_score = (
                    assessment.relevance_score
                )
                state.confidence = assessment.confidence
                state.primary_category = (
                    assessment.primary_category
                )
                state.topic = assessment.topic
                state.relevance_reason = assessment.reason
                state.last_assessment_number = (
                    assessment.sequence_number
                )
                state.units_processed = (
                    assessment.units_processed
                )

                updated += 1

            else:

                skipped += 1

        session.commit()

    print(
        f"Provider histories found: "
        f"{len(latest_by_provider)}"
    )
    print(f"Created: {created}")
    print(f"Updated: {updated}")
    print(f"Skipped newer states: {skipped}")


if __name__ == "__main__":
    main()