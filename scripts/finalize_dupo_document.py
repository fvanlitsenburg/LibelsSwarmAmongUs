"""Run the final full-text assessment for a DUPO document."""

import argparse
from pathlib import Path

from historical_text_pipeline.config.settings import (
    get_settings,
)
from historical_text_pipeline.db.models import Document
from historical_text_pipeline.db.session import (
    get_session_factory,
)
from historical_text_pipeline.relevance.final_assessment import (
    OpenAiFinalAssessor,
)
from historical_text_pipeline.relevance.final_service import (
    assess_and_store_final_full_text,
    estimate_text_tokens,
)
from historical_text_pipeline.relevance.service import (
    build_accumulated_page_text,
)


def parse_arguments() -> argparse.Namespace:
    """Read the document ID and overwrite option."""

    parser = argparse.ArgumentParser(
        description=(
            "Assess the complete OCR transcription and save its "
            "final category, topic, relevance explanation, and summary."
        )
    )

    parser.add_argument(
        "document_id",
        type=int,
        help="Internal PostgreSQL document ID.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing final assessment.",
    )

    return parser.parse_args()


def load_criteria(path: Path) -> str:
    """Load the version-controlled relevance criteria."""

    resolved_path = path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Relevance criteria file does not exist: "
            f"{resolved_path}"
        )

    criteria = resolved_path.read_text(
        encoding="utf-8"
    ).strip()

    if not criteria:
        raise ValueError(
            "The relevance criteria file is empty."
        )

    return criteria


def main() -> None:
    """Run and save one final full-text assessment."""

    arguments = parse_arguments()
    settings = get_settings()
    session_factory = get_session_factory()

    criteria = load_criteria(
        settings.relevance_criteria_path
    )

    with session_factory() as session:
        document = session.get(
            Document,
            arguments.document_id,
        )

        if document is None:
            raise SystemExit(
                f"Document {arguments.document_id} does not exist."
            )

        if not document.text_complete:
            raise SystemExit(
                f"Document {document.id} does not have complete OCR."
            )

        if document.total_units is None:
            raise SystemExit(
                f"Document {document.id} has no page count."
            )

        transcription = build_accumulated_page_text(
            session,
            document_id=document.id,
            through_page=document.total_units,
        )

    estimated_input_tokens = estimate_text_tokens(
        transcription
    )

    print(f"Document:               {arguments.document_id}")
    print(f"Characters:             {len(transcription):,}")
    print(
        f"Estimated input tokens: "
        f"{estimated_input_tokens:,}"
    )
    print(f"Model:                  {settings.openai_final_model}")
    print()

    if (
        estimated_input_tokens
        > settings.final_assessment_max_estimated_input_tokens
    ):
        raise SystemExit(
            "The transcription exceeds LSAU's configured final-"
            "assessment safety limit.\n"
            "Increase "
            "LSAU_FINAL_ASSESSMENT_MAX_ESTIMATED_INPUT_TOKENS "
            "only after reviewing the document size."
        )

    print("Running final full-text assessment...")

    assessor = OpenAiFinalAssessor.from_settings(
        settings
    )

    try:
        with session_factory() as session:
            result = assess_and_store_final_full_text(
                session,
                document_id=arguments.document_id,
                criteria=criteria,
                assessor=assessor,
                overwrite=arguments.overwrite,
            )

            session.commit()

    finally:
        assessor.close()

    print()
    print(f"Assessment: {result.assessment_number}")
    print(f"Decision:   {result.decision.value}")
    print(f"Score:      {result.relevance_score:.2f}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Category:   {result.category}")
    print(f"Topic:      {result.topic}")
    print(f"Model:      {result.model}")

    if result.input_tokens is not None:
        print(f"Input tokens:  {result.input_tokens:,}")

    if result.output_tokens is not None:
        print(f"Output tokens: {result.output_tokens:,}")

    print()
    print("Relevance explanation")
    print("---------------------")
    print(result.relevance_explanation)

    print()
    print("Summary")
    print("-------")
    print(result.summary)


if __name__ == "__main__":
    main()