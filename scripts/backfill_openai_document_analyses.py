"""Backfill existing canonical final analyses as legacy OpenAI runs."""

from sqlalchemy import select

from historical_text_pipeline.db.models import (
    Document,
    DocumentAnalysis,
)
from historical_text_pipeline.db.session import (
    get_session_factory,
)
from historical_text_pipeline.domain import (
    AnalysisProvider,
)

LEGACY_MODEL = "legacy_unknown"
LEGACY_PROMPT_VERSION = "legacy"


def main() -> None:
    """Copy existing final document fields into analysis history."""

    session_factory = get_session_factory()

    created = 0
    skipped = 0

    with session_factory() as session:
        documents = list(
            session.scalars(
                select(Document)
                .where(
                    Document.text_complete.is_(True),
                    Document.summary.is_not(None),
                    Document.summary != "",
                )
                .order_by(Document.id)
            )
        )

        for document in documents:
            existing = session.scalar(
                select(DocumentAnalysis.id)
                .where(
                    DocumentAnalysis.document_id
                    == document.id,
                    DocumentAnalysis.provider
                    == AnalysisProvider.OPENAI.value,
                    DocumentAnalysis.prompt_version
                    == LEGACY_PROMPT_VERSION,
                )
                .limit(1)
            )

            if existing is not None:
                skipped += 1
                continue

            analysis = DocumentAnalysis(
                document_id=document.id,
                provider=(
                    AnalysisProvider.OPENAI.value
                ),
                model=LEGACY_MODEL,
                prompt_version=(
                    LEGACY_PROMPT_VERSION
                ),
                decision=(
                    document.relevance_status.value
                ),
                relevance_score=(
                    document.relevance_score
                    if document.relevance_score
                    is not None
                    else 0.0
                ),
                confidence=(
                    document.relevance_confidence
                    if document.relevance_confidence
                    is not None
                    else 0.0
                ),
                primary_category=(
                    document.primary_category
                    or "unknown"
                ),
                topic=(
                    document.topic
                    or "unknown"
                ),
                relevance_explanation=(
                    document.relevance_reason
                    or "Legacy analysis; explanation unavailable."
                ),
                summary=document.summary,
                supporting_evidence=[],
                caveats=[
                    (
                        "Backfilled from the canonical "
                        "document record; original model, "
                        "prompt version, token usage, and "
                        "response ID were not recorded."
                    )
                ],
                response_id=None,
                input_tokens=None,
                output_tokens=None,
            )

            session.add(analysis)
            created += 1

        session.commit()

    print(f"Created: {created}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()