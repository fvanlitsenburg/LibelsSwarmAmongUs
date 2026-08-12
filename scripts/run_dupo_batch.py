"""Run one processing stage for a batch of DUPO documents."""

import argparse
import subprocess
import sys
from pathlib import Path

from historical_text_pipeline.batch import (
    DupoBatchStage,
    get_dupo_batch_document_ids,
)
from historical_text_pipeline.db.session import (
    get_session_factory,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

STAGE_SCRIPTS = {
    DupoBatchStage.RELEVANCE: (
        REPOSITORY_ROOT
        / "scripts"
        / "process_dupo_relevance_batch.py"
    ),
    DupoBatchStage.OCR: (
        REPOSITORY_ROOT
        / "scripts"
        / "complete_dupo_ocr.py"
    ),
    DupoBatchStage.FINAL: (
        REPOSITORY_ROOT
        / "scripts"
        / "finalize_dupo_document.py"
    ),
    DupoBatchStage.PIPELINE: (
        REPOSITORY_ROOT
        / "scripts"
        / "process_dupo_document.py"
    ),
}


def parse_arguments() -> argparse.Namespace:
    """Read the batch stage and safety limits."""

    parser = argparse.ArgumentParser(
        description=(
            "Process a sequential batch of registered DUPO "
            "documents."
        )
    )

    parser.add_argument(
        "stage",
        choices=[
            stage.value
            for stage in DupoBatchStage
        ],
        help=(
            "Stage to run: relevance, ocr, or final."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help=(
            "Maximum number of documents to process. "
            "This must be supplied explicitly."
        ),
    )

    parser.add_argument(
        "--start-id",
        type=int,
        default=None,
        help=(
            "Only select documents with this internal ID "
            "or higher."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show which documents would be processed without "
            "running any OCR or model calls."
        ),
    )

    return parser.parse_args()


def run_document(
    *,
    stage: DupoBatchStage,
    document_id: int,
) -> int:
    """Run the existing one-document script."""

    script_path = STAGE_SCRIPTS[stage]

    command = [
        sys.executable,
        str(script_path),
        str(document_id),
    ]

    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
    )

    return completed.returncode


def main() -> None:
    """Select and process one batch."""

    arguments = parse_arguments()

    if arguments.limit < 1:
        raise SystemExit("--limit must be at least 1.")

    stage = DupoBatchStage(arguments.stage)
    session_factory = get_session_factory()

    with session_factory() as session:
        document_ids = get_dupo_batch_document_ids(
            session,
            stage=stage,
            limit=arguments.limit,
            start_id=arguments.start_id,
        )

    print(f"Stage:              {stage.value}")
    print(f"Documents selected: {len(document_ids)}")

    if arguments.start_id is not None:
        print(f"Starting ID:        {arguments.start_id}")

    if not document_ids:
        print("No eligible documents were found.")
        return

    print(
        "Document IDs:      "
        + ", ".join(
            str(document_id)
            for document_id in document_ids
        )
    )

    if arguments.dry_run:
        print()
        print("Dry run: no processing was performed.")
        return

    successful: list[int] = []
    failed: list[int] = []

    for position, document_id in enumerate(
        document_ids,
        start=1,
    ):
        print()
        print("=" * 72)
        print(
            f"{stage.value.upper()} "
            f"{position}/{len(document_ids)} — "
            f"document {document_id}"
        )
        print("=" * 72)
        print()

        try:
            return_code = run_document(
                stage=stage,
                document_id=document_id,
            )
        except KeyboardInterrupt:
            print()
            print("Batch interrupted by user.")
            print(
                "Completed documents and pages remain stored."
            )
            raise SystemExit(130) from None

        if return_code == 0:
            successful.append(document_id)
        else:
            failed.append(document_id)

            print()
            print(
                f"Document {document_id} failed with "
                f"exit code {return_code}. Continuing."
            )

    print()
    print("=" * 72)
    print("BATCH SUMMARY")
    print("=" * 72)
    print(f"Selected:   {len(document_ids)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed:     {len(failed)}")

    if failed:
        print(
            "Failed IDs: "
            + ", ".join(
                str(document_id)
                for document_id in failed
            )
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()