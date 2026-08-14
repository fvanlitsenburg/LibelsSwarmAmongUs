"""Process one DUPO document as far through the pipeline as possible."""

import argparse
import subprocess
import sys
from pathlib import Path

from sqlalchemy import exists, select

from historical_text_pipeline.db.models import (
    Document,
    DocumentAnalysis,
)
from historical_text_pipeline.db.session import (
    get_session_factory,
)
from historical_text_pipeline.domain import (
    AnalysisProvider,
    RelevanceStatus,
)
from historical_text_pipeline.relevance.service import (
    get_provider_state,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

RELEVANCE_SCRIPTS = {
    AnalysisProvider.OPENAI: (
        REPOSITORY_ROOT
        / "scripts"
        / "process_dupo_relevance_batch.py"
    ),
    AnalysisProvider.ANTHROPIC: (
        REPOSITORY_ROOT
        / "scripts"
        / "process_dupo_relevance_anthropic.py"
    ),
}

OCR_SCRIPT = (
    REPOSITORY_ROOT
    / "scripts"
    / "complete_dupo_ocr.py"
)

FINAL_SCRIPTS = {
    AnalysisProvider.OPENAI: (
        REPOSITORY_ROOT
        / "scripts"
        / "finalize_dupo_document.py"
    ),
    AnalysisProvider.ANTHROPIC: (
        REPOSITORY_ROOT
        / "scripts"
        / "finalize_dupo_document_anthropic.py"
    ),
}


def parse_arguments() -> argparse.Namespace:
    """Read the document ID and analysis provider."""

    parser = argparse.ArgumentParser(
        description=(
            "Process one DUPO document through all "
            "applicable pipeline stages."
        )
    )

    parser.add_argument(
        "document_id",
        type=int,
        help="Internal PostgreSQL document ID.",
    )

    parser.add_argument(
        "--provider",
        choices=[
            provider.value
            for provider in AnalysisProvider
        ],
        required=True,
        help="Analysis provider to use.",
    )

    return parser.parse_args()


def run_script(
    script: Path,
    document_id: int,
    *extra_arguments: str,
) -> None:
    """Run one existing processing script."""

    command = [
        sys.executable,
        str(script),
        str(document_id),
        *extra_arguments,
    ]

    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"{script.name} failed for document "
            f"{document_id} with exit code "
            f"{completed.returncode}."
        )


def load_document(document_id: int) -> Document:
    """Load the current shared state of a document."""

    session_factory = get_session_factory()

    with session_factory() as session:
        document = session.get(
            Document,
            document_id,
        )

        if document is None:
            raise RuntimeError(
                f"Document {document_id} does not exist."
            )

        session.expunge(document)

    return document


def load_relevance_status(
    document_id: int,
    provider: AnalysisProvider,
) -> RelevanceStatus | None:
    """Return the current relevance state for one provider."""

    session_factory = get_session_factory()

    with session_factory() as session:
        state = get_provider_state(
            session,
            document_id=document_id,
            provider=provider,
        )

        if state is None:
            return None

        return RelevanceStatus(
            state.relevance_status
        )


def has_final_analysis(
    document_id: int,
    provider: AnalysisProvider,
) -> bool:
    """Return whether this provider has a final analysis."""

    session_factory = get_session_factory()

    with session_factory() as session:
        statement = select(
            exists().where(
                DocumentAnalysis.document_id
                == document_id,
                DocumentAnalysis.provider
                == provider.value,
            )
        )

        return bool(
            session.scalar(statement)
        )


def run_relevance_until_resolved(
    document_id: int,
    provider: AnalysisProvider,
) -> RelevanceStatus:
    """Advance progressive relevance until it resolves."""

    relevance_script = RELEVANCE_SCRIPTS[provider]

    while True:
        relevance_status = load_relevance_status(
            document_id,
            provider,
        )

        if relevance_status in {
            RelevanceStatus.RELEVANT,
            RelevanceStatus.IRRELEVANT,
        }:
            return relevance_status

        run_script(
            relevance_script,
            document_id,
        )


def main() -> None:
    """Process one document through all applicable stages."""

    arguments = parse_arguments()

    document_id = arguments.document_id
    provider = AnalysisProvider(
        arguments.provider
    )

    print()
    print("=" * 72)
    print(
        f"FULL PIPELINE — document {document_id} "
        f"— {provider.value}"
    )
    print("=" * 72)

    document = load_document(document_id)

    # ---------------------------------------------------------
    # 1. Progressive relevance
    # ---------------------------------------------------------

    relevance_status = load_relevance_status(
        document_id,
        provider,
    )

    if relevance_status in {
        None,
        RelevanceStatus.UNCERTAIN,
    }:
        print()
        print(
            "Stage 1: progressive relevance "
            f"({provider.value})"
        )

        relevance_status = (
            run_relevance_until_resolved(
                document_id,
                provider,
            )
        )

        document = load_document(document_id)

    # ---------------------------------------------------------
    # 2. Stop on irrelevance only when OCR is incomplete
    # ---------------------------------------------------------

    if (
        relevance_status
        == RelevanceStatus.IRRELEVANT
        and not document.text_complete
    ):
        print()
        print(
            f"Document is irrelevant for "
            f"{provider.value}. "
            "Shared OCR is incomplete, so no further "
            "processing is required for this provider."
        )
        return

    # ---------------------------------------------------------
    # 3. Complete OCR when relevance is confirmed
    # ---------------------------------------------------------

    if (
        relevance_status
        == RelevanceStatus.RELEVANT
        and not document.text_complete
    ):
        print()
        print("Stage 2: completing OCR")

        run_script(
            OCR_SCRIPT,
            document_id,
            "--provider",
            provider.value,
        )

        document = load_document(document_id)

    # ---------------------------------------------------------
    # 4. Final full-text assessment
    # ---------------------------------------------------------

    final_complete = has_final_analysis(
        document_id,
        provider,
    )

    if (
        document.text_complete
        and not final_complete
    ):
        print()
        print(
            "Stage 3: final full-text assessment "
            f"({provider.value})"
        )

        run_script(
            FINAL_SCRIPTS[provider],
            document_id,
        )

        final_complete = has_final_analysis(
            document_id,
            provider,
        )

    print()
    print("=" * 72)
    print(f"DOCUMENT {document_id} COMPLETE")
    print("=" * 72)
    print(f"Provider:         {provider.value}")
    print(
        "Relevance:        "
        + (
            relevance_status.value
            if relevance_status is not None
            else "not assessed"
        )
    )
    print(
        f"Text complete:    "
        f"{document.text_complete}"
    )
    print(
        "Final assessment: "
        f"{'yes' if final_complete else 'no'}"
    )


if __name__ == "__main__":
    main()