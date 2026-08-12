"""Process one DUPO document as far through the pipeline as possible."""

import subprocess
import sys
from pathlib import Path

from historical_text_pipeline.db.models import Document
from historical_text_pipeline.db.session import (
    get_session_factory,
)
from historical_text_pipeline.domain import RelevanceStatus

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

RELEVANCE_SCRIPT = (
    REPOSITORY_ROOT
    / "scripts"
    / "process_dupo_relevance_batch.py"
)

OCR_SCRIPT = (
    REPOSITORY_ROOT
    / "scripts"
    / "complete_dupo_ocr.py"
)

FINAL_SCRIPT = (
    REPOSITORY_ROOT
    / "scripts"
    / "finalize_dupo_document.py"
)


def run_script(
    script: Path,
    document_id: int,
) -> None:
    """Run one existing processing script."""

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(document_id),
        ],
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
    """Load the current database state of a document."""

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

        # Detach it so its scalar fields remain available
        # outside the session.
        session.expunge(document)

    return document


def run_relevance_until_resolved(
    document_id: int,
) -> Document:
    """Keep advancing progressive relevance until resolved or complete."""

    while True:
        document = load_document(document_id)

        if document.relevance_status in (
            RelevanceStatus.RELEVANT,
            RelevanceStatus.IRRELEVANT,
        ):
            return document

        # If progressive OCR has reached the end of the document,
        # there is no further relevance batch to request.
        if document.text_complete:
            return document

        run_script(
            RELEVANCE_SCRIPT,
            document_id,
        )


def main() -> None:
    """Process one document through all applicable stages."""

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/process_dupo_document.py "
            "DOCUMENT_ID"
        )

    document_id = int(sys.argv[1])

    print()
    print("=" * 72)
    print(f"FULL PIPELINE — document {document_id}")
    print("=" * 72)

    document = load_document(document_id)

    # ---------------------------------------------------------
    # 1. Progressive relevance
    # ---------------------------------------------------------

    if document.relevance_status in (
        RelevanceStatus.NOT_ASSESSED,
        RelevanceStatus.UNCERTAIN,
    ):
        print()
        print("Stage 1: progressive relevance")

        document = run_relevance_until_resolved(
            document_id
        )

    # ---------------------------------------------------------
    # 2. Stop immediately on confirmed irrelevance
    # ---------------------------------------------------------

    if document.relevance_status == RelevanceStatus.IRRELEVANT:
        print()
        print(
            "Document is confirmed irrelevant. "
            "No further OCR or final assessment required."
        )
        return

    # ---------------------------------------------------------
    # 3. Complete OCR when relevance has been confirmed
    # ---------------------------------------------------------

    if (
        document.relevance_status == RelevanceStatus.RELEVANT
        and not document.text_complete
    ):
        print()
        print("Stage 2: completing OCR")

        run_script(
            OCR_SCRIPT,
            document_id,
        )

        document = load_document(document_id)

    # ---------------------------------------------------------
    # 4. Final full-text assessment
    # ---------------------------------------------------------

    if document.text_complete and not document.summary:
        print()
        print("Stage 3: final full-text assessment")

        run_script(
            FINAL_SCRIPT,
            document_id,
        )

        document = load_document(document_id)

    print()
    print("=" * 72)
    print(f"DOCUMENT {document_id} COMPLETE")
    print("=" * 72)
    print(
        f"Relevance: {document.relevance_status.value}"
    )
    print(
        f"Text complete: {document.text_complete}"
    )
    print(
        f"Final assessment: "
        f"{'yes' if document.summary else 'no'}"
    )


if __name__ == "__main__":
    main()