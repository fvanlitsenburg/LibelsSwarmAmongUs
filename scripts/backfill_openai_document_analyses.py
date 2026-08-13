"""Backfill legacy OpenAI final results into document_analyses."""

import argparse

from sqlalchemy import exists, select

from historical_text_pipeline.db.models import (
    Document,
    DocumentAnalysis,
    RelevanceAssessment,
)
from historical_text_pipeline.db.session import get_session_factory
from historical_text_pipeline.domain import (
    AnalysisProvider,
    ClassificationStatus,
)


def parse_arguments() -> argparse.Namespace:
    """Read backfill options."""

    parser = argparse.ArgumentParser(
        description=(
            "Copy legacy OpenAI final analysis fields from "
            "documents into document_analyses."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be backfilled without writing anything.",
    )

    return parser.parse_args()


def enum_value(value: object | None) -> str | None:
    """Return the string value of an enum-like object."""

    if value is None:
        return None

    candidate = getattr(value, "value", None)

    if isinstance(candidate, str):
        return candidate

    return str(value)


def main() -> None:
    """Backfill legacy OpenAI final analyses."""

    arguments = parse_arguments()
    session_factory = get_session_factory()

    with session_factory() as session:
        openai_analysis_exists = exists(
            select(DocumentAnalysis.id).where(
                DocumentAnalysis.document_id == Document.id,
                DocumentAnalysis.provider
                == AnalysisProvider.OPENAI.value,
            )
        )

        documents = list(
            session.scalars(
                select(Document)
                .where(
                    # A legacy final assessment populated summary.
                    Document.summary.is_not(None),
                    Document.summary != "",

                    # Do not duplicate provider-aware analyses that
                    # have already been stored.
                    ~openai_analysis_exists,
                )
                .order_by(Document.id)
            )
        )

        print("OpenAI final-analysis backfill")
        print("============================")
        print(f"Candidates: {len(documents)}")

        if not documents:
            print("Nothing to backfill.")
            return

        created = 0

        for document in documents:
            # Older final-service versions also wrote a FULL_TEXT
            # relevance-assessment row. Use it where available to
            # preserve metadata and supporting evidence.
            final_assessment = session.scalar(
                select(RelevanceAssessment)
                .where(
                    RelevanceAssessment.document_id == document.id,
                    RelevanceAssessment.provider
                    == AnalysisProvider.OPENAI.value,
                    RelevanceAssessment.classification_status
                    == ClassificationStatus.FULL_TEXT,
                )
                .order_by(
                    RelevanceAssessment.sequence_number.desc()
                )
                .limit(1)
            )

            decision = enum_value(
                document.relevance_status
            )

            if decision is None:
                raise RuntimeError(
                    f"Document {document.id} has a summary but "
                    "no relevance status."
                )

            model = "unknown"
            prompt_version = "legacy"
            response_id = None
            input_tokens = None
            output_tokens = None
            supporting_evidence: list[str] = []

            if final_assessment is not None:
                if final_assessment.model:
                    model = final_assessment.model

                if final_assessment.prompt_version:
                    prompt_version = (
                        final_assessment.prompt_version
                    )

                response_id = (
                    final_assessment.response_id
                )
                input_tokens = (
                    final_assessment.input_tokens
                )
                output_tokens = (
                    final_assessment.output_tokens
                )

                supporting_evidence = list(
                    final_assessment.supporting_evidence
                    or []
                )

            print(
                f"document {document.id}: "
                f"{decision} "
                f"[{document.primary_category or '-'}]"
            )

            if arguments.dry_run:
                continue

            analysis = DocumentAnalysis(
                document_id=document.id,
                provider=AnalysisProvider.OPENAI.value,
                model=model,
                prompt_version=prompt_version,
                decision=decision,
                relevance_score=document.relevance_score,
                confidence=document.relevance_confidence,
                primary_category=document.primary_category,
                topic=document.topic,
                relevance_explanation=(
                    document.relevance_reason
                ),
                summary=document.summary,
                supporting_evidence=supporting_evidence,

                # The old schema did not preserve final-assessment
                # caveats separately.
                caveats=[],

                response_id=response_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            session.add(analysis)
            created += 1

        if arguments.dry_run:
            print()
            print(
                "Dry run only. "
                "No document_analyses rows were created."
            )
            return

        session.commit()

        print()
        print(f"Created: {created}")


if __name__ == "__main__":
    main()