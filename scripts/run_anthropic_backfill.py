"""Backfill Anthropic relevance and final analysis for DUPO documents."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import exists, select
from sqlalchemy.orm import Session, sessionmaker

from historical_text_pipeline.batch import (
    get_anthropic_backfill_document_ids,
)
from historical_text_pipeline.config.settings import (
    Settings,
    get_settings,
)
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
from historical_text_pipeline.ingest.dupo_ocr import (
    get_missing_dupo_pages,
    ocr_dupo_page,
)
from historical_text_pipeline.ocr.base import OcrBackend
from historical_text_pipeline.ocr.factory import (
    create_ocr_backend,
)
from historical_text_pipeline.relevance.anthropic_assessor import (
    AnthropicRelevanceAssessor,
)
from historical_text_pipeline.relevance.anthropic_final_assessor import (
    AnthropicFinalAssessor,
)
from historical_text_pipeline.relevance.final_service import (
    assess_and_store_final_full_text,
)
from historical_text_pipeline.relevance.service import (
    assess_and_store_relevance,
    get_next_batch_end_page,
    get_provider_state,
)


@dataclass(frozen=True, slots=True)
class AnthropicDocumentState:
    """Small snapshot used by the backfill orchestration."""

    document_id: int
    text_complete: bool
    total_units: int
    relevance_status: RelevanceStatus | None
    assessment_number: int


def parse_arguments() -> argparse.Namespace:
    """Read batch safety options."""

    parser = argparse.ArgumentParser(
        description=(
            "Backfill Claude relevance and final analyses "
            "for existing DUPO documents."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Maximum number of documents to process.",
    )

    parser.add_argument(
        "--start-id",
        type=int,
        default=None,
        help=(
            "Only process documents with this internal ID "
            "or higher."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show eligible documents without making OCR "
            "or Anthropic API calls."
        ),
    )

    return parser.parse_args()


def load_criteria(path: Path) -> str:
    """Load the shared relevance criteria."""

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


def load_anthropic_state(
    session: Session,
    *,
    document_id: int,
) -> AnthropicDocumentState:
    """Load the current Anthropic processing state."""

    document = session.get(
        Document,
        document_id,
    )

    if document is None:
        raise RuntimeError(
            f"Document {document_id} does not exist."
        )

    if document.total_units is None:
        raise RuntimeError(
            f"Document {document_id} has no recorded page count."
        )

    provider_state = get_provider_state(
        session,
        document_id=document_id,
        provider=AnalysisProvider.ANTHROPIC,
    )

    relevance_status: RelevanceStatus | None = None
    assessment_number = 0

    if provider_state is not None:
        relevance_status = RelevanceStatus(
            provider_state.relevance_status
        )
        assessment_number = (
            provider_state.last_assessment_number
        )

    return AnthropicDocumentState(
        document_id=document.id,
        text_complete=document.text_complete,
        total_units=document.total_units,
        relevance_status=relevance_status,
        assessment_number=assessment_number,
    )


def has_anthropic_final_analysis(
    session: Session,
    *,
    document_id: int,
) -> bool:
    """Return whether Claude final analysis already exists."""

    return bool(
        session.scalar(
            select(
                exists().where(
                    DocumentAnalysis.document_id
                    == document_id,
                    DocumentAnalysis.provider
                    == AnalysisProvider.ANTHROPIC.value,
                )
            )
        )
    )


def ensure_ocr_through_page(
    *,
    session_factory: sessionmaker[Session],
    backend: OcrBackend,
    settings: Settings,
    document_id: int,
    through_page: int,
) -> None:
    """OCR only missing pages needed by the next Claude batch."""

    with session_factory() as session:
        missing_pages = get_missing_dupo_pages(
            session,
            document_id=document_id,
        )

    required_pages = [
        page_number
        for page_number in missing_pages
        if page_number <= through_page
    ]

    if not required_pages:
        print(
            f"  OCR through page {through_page} "
            "already stored."
        )
        return

    for page_number in required_pages:
        print(
            f"  OCR page {page_number}...",
            flush=True,
        )

        with session_factory() as session:
            result = ocr_dupo_page(
                session,
                document_id=document_id,
                page_number=page_number,
                backend=backend,
                dpi=settings.pdf_render_dpi,
                jpeg_quality=settings.pdf_jpeg_quality,
            )

            session.commit()

        if result.created:
            print(
                f"    saved "
                f"{result.character_count:,} characters"
            )
        else:
            print("    already stored")


def complete_shared_ocr(
    *,
    session_factory: sessionmaker[Session],
    backend: OcrBackend,
    settings: Settings,
    document_id: int,
) -> None:
    """Complete all missing OCR pages for a Claude-relevant document."""

    with session_factory() as session:
        document = session.get(
            Document,
            document_id,
        )

        if document is None:
            raise RuntimeError(
                f"Document {document_id} does not exist."
            )

        if document.total_units is None:
            raise RuntimeError(
                f"Document {document_id} has no page count."
            )

        total_units = document.total_units

    print(
        f"Completing shared OCR through page "
        f"{total_units}..."
    )

    ensure_ocr_through_page(
        session_factory=session_factory,
        backend=backend,
        settings=settings,
        document_id=document_id,
        through_page=total_units,
    )


def run_progressive_anthropic_relevance(
    *,
    session_factory: sessionmaker[Session],
    backend: OcrBackend,
    settings: Settings,
    criteria: str,
    assessor: AnthropicRelevanceAssessor,
    document_id: int,
) -> RelevanceStatus:
    """Continue Claude progressive relevance until it resolves."""

    while True:
        with session_factory() as session:
            state = load_anthropic_state(
                session,
                document_id=document_id,
            )

        if state.relevance_status in {
            RelevanceStatus.RELEVANT,
            RelevanceStatus.IRRELEVANT,
        }:
            return state.relevance_status

        with session_factory() as session:
            next_end_page = get_next_batch_end_page(
                session,
                document_id=document_id,
                batch_size=settings.relevance_batch_size,
                provider=AnalysisProvider.ANTHROPIC,
            )

        if next_end_page is None:
            raise RuntimeError(
                "Claude has no additional page batch to assess "
                f"for document {document_id}, but its relevance "
                "state is still unresolved."
            )

        print(
            f"Claude relevance assessment "
            f"{state.assessment_number + 1}: "
            f"pages 1-{next_end_page}"
        )

        ensure_ocr_through_page(
            session_factory=session_factory,
            backend=backend,
            settings=settings,
            document_id=document_id,
            through_page=next_end_page,
        )

        with session_factory() as session:
            result = assess_and_store_relevance(
                session,
                document_id=document_id,
                through_page=next_end_page,
                criteria=criteria,
                assessor=assessor,
                stop_confidence_threshold=(
                    settings.relevance_stop_confidence_threshold
                ),
                max_assessments=(
                    settings.relevance_max_assessments
                ),
            )

            session.commit()

        print(
            f"  decision={result.decision.value} "
            f"status={result.relevance_status.value} "
            f"score={result.relevance_score:.2f} "
            f"confidence={result.confidence:.2f}"
        )

        if result.relevance_status in {
            RelevanceStatus.RELEVANT,
            RelevanceStatus.IRRELEVANT,
        }:
            return result.relevance_status


def run_anthropic_final_analysis(
    *,
    session_factory: sessionmaker[Session],
    criteria: str,
    assessor: AnthropicFinalAssessor,
    document_id: int,
) -> None:
    """Run Claude's complete-document analysis if needed."""

    with session_factory() as session:
        if has_anthropic_final_analysis(
            session,
            document_id=document_id,
        ):
            print(
                "Claude final analysis already exists."
            )
            return

    print("Running Claude final full-text analysis...")

    with session_factory() as session:
        result = assess_and_store_final_full_text(
            session,
            document_id=document_id,
            criteria=criteria,
            assessor=assessor,
        )

        session.commit()

    print(
        f"  final={result.decision.value} "
        f"score={result.relevance_score:.2f} "
        f"confidence={result.confidence:.2f}"
    )
    print(f"  category={result.category}")
    print(f"  topic={result.topic}")


def process_document(
    *,
    session_factory: sessionmaker[Session],
    backend: OcrBackend,
    settings: Settings,
    criteria: str,
    relevance_assessor: AnthropicRelevanceAssessor,
    final_assessor: AnthropicFinalAssessor,
    document_id: int,
) -> None:
    """Backfill one document as far as Claude requires."""

    with session_factory() as session:
        initial_state = load_anthropic_state(
            session,
            document_id=document_id,
        )

        already_full_text = (
            initial_state.text_complete
        )

    initial_relevance_status = (
        initial_state.relevance_status.value
        if initial_state.relevance_status is not None
        else "not_assessed"
    )

    print(
        f"Initial Claude state: {initial_relevance_status}"
    )
    print(
        f"Shared OCR complete: "
        f"{already_full_text}"
    )

    # First establish Claude's progressive relevance result.
    relevance_status = (
        run_progressive_anthropic_relevance(
            session_factory=session_factory,
            backend=backend,
            settings=settings,
            criteria=criteria,
            assessor=relevance_assessor,
            document_id=document_id,
        )
    )

    with session_factory() as session:
        current_state = load_anthropic_state(
            session,
            document_id=document_id,
        )

    if relevance_status == RelevanceStatus.RELEVANT:
        if not current_state.text_complete:
            complete_shared_ocr(
                session_factory=session_factory,
                backend=backend,
                settings=settings,
                document_id=document_id,
            )

        run_anthropic_final_analysis(
            session_factory=session_factory,
            criteria=criteria,
            assessor=final_assessor,
            document_id=document_id,
        )
        return

    # Claude rejected the document.
    #
    # If full OCR already existed before Claude began, running the
    # final assessment is still useful for comparison and costs no
    # additional OCR.
    if already_full_text:
        run_anthropic_final_analysis(
            session_factory=session_factory,
            criteria=criteria,
            assessor=final_assessor,
            document_id=document_id,
        )

        return

    print(
        "Claude concluded that the partially OCR'd "
        "document is irrelevant. Stopping without "
        "additional OCR."
    )


def print_dry_run(
    *,
    session_factory: sessionmaker[Session],
    document_ids: list[int],
) -> None:
    """Show documents selected for Anthropic processing."""

    print()
    print("Eligible documents")
    print("------------------")

    with session_factory() as session:
        for document_id in document_ids:
            document = session.get(
                Document,
                document_id,
            )

            if document is None:
                continue

            state = get_provider_state(
                session,
                document_id=document_id,
                provider=AnalysisProvider.ANTHROPIC,
            )

            status = (
                state.relevance_status
                if state is not None
                else "not_assessed"
            )

            print(
                f"{document.id:6}  "
                f"year={document.year or '-':4}  "
                f"ocr={'full' if document.text_complete else 'partial':7}  "
                f"claude={status}"
            )


def main() -> None:
    """Run the Anthropic backfill batch."""

    arguments = parse_arguments()

    if arguments.limit < 1:
        raise SystemExit(
            "--limit must be at least 1."
        )

    settings = get_settings()
    criteria = load_criteria(
        settings.relevance_criteria_path
    )
    session_factory = get_session_factory()

    with session_factory() as session:
        document_ids = (
            get_anthropic_backfill_document_ids(
                session,
                limit=arguments.limit,
                start_id=arguments.start_id,
            )
        )

    print("Anthropic backfill")
    print("==================")
    print(f"Selected: {len(document_ids)}")

    if not document_ids:
        print(
            "No documents currently require "
            "Anthropic processing."
        )
        return

    print(
        "IDs: "
        + ", ".join(
            str(document_id)
            for document_id in document_ids
        )
    )

    if arguments.dry_run:
        print_dry_run(
            session_factory=session_factory,
            document_ids=document_ids,
        )
        print()
        print(
            "Dry run only. No OCR or API calls were made."
        )
        return

    relevance_assessor = (
        AnthropicRelevanceAssessor.from_settings(
            settings
        )
    )

    final_assessor = (
        AnthropicFinalAssessor.from_settings(
            settings
        )
    )

    ocr_backend = create_ocr_backend(
        settings
    )

    successful: list[int] = []
    failed: list[int] = []

    try:
        for index, document_id in enumerate(
            document_ids,
            start=1,
        ):
            print()
            print("=" * 72)
            print(
                f"DOCUMENT {index}/{len(document_ids)} "
                f"— {document_id}"
            )
            print("=" * 72)

            try:
                process_document(
                    session_factory=session_factory,
                    backend=ocr_backend,
                    settings=settings,
                    criteria=criteria,
                    relevance_assessor=(
                        relevance_assessor
                    ),
                    final_assessor=final_assessor,
                    document_id=document_id,
                )

            except KeyboardInterrupt:
                print()
                print(
                    "Anthropic backfill interrupted. "
                    "Completed API results and OCR pages "
                    "remain stored."
                )
                raise SystemExit(130) from None

            # A failure in one document must not abort the
            # rest of the backfill batch.
            except Exception as error:  # noqa: BLE001
                failed.append(document_id)

                print()
                print(
                    f"Document {document_id} failed: "
                    f"{error}"
                )
                print(
                    "Continuing with the next document."
                )


            else:
                successful.append(document_id)

    finally:
        ocr_backend.close()
        relevance_assessor.close()
        final_assessor.close()

    print()
    print("=" * 72)
    print("ANTHROPIC BACKFILL SUMMARY")
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