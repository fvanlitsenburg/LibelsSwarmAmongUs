"""Register discovered DUPO PDFs in PostgreSQL."""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import Document, DupoMetadata
from historical_text_pipeline.domain import Source
from historical_text_pipeline.ingest.dupo import (
    DupoPdf,
    calculate_sha256,
)


@dataclass(frozen=True, slots=True)
class DupoRegistrationResult:
    """Summary of one DUPO registration run."""

    discovered: int
    registered: int
    skipped_by_path: int
    skipped_by_checksum: int


def find_existing_by_path(
    session: Session,
    path: Path,
) -> Document | None:
    """Find a DUPO document already registered at this path."""

    return session.scalar(
        select(Document).where(
            Document.source == Source.DUPO,
            Document.source_path == str(path),
        )
    )


def find_existing_by_checksum(
    session: Session,
    checksum: str,
) -> Document | None:
    """Find a DUPO document with the same file contents."""

    return session.scalar(
        select(Document).where(
            Document.source == Source.DUPO,
            Document.file_checksum == checksum,
        )
    )


def register_dupo_pdfs(
    session: Session,
    documents: list[DupoPdf],
) -> DupoRegistrationResult:
    """
    Register previously unseen DUPO PDFs.

    This function does not commit the transaction. The caller decides
    whether to commit or roll back.
    """

    registered = 0
    skipped_by_path = 0
    skipped_by_checksum = 0

    for discovered_document in documents:
        path = discovered_document.path.resolve()

        existing_path = find_existing_by_path(session, path)

        if existing_path is not None:
            skipped_by_path += 1
            continue

        checksum = calculate_sha256(path)

        existing_checksum = find_existing_by_checksum(
            session,
            checksum,
        )

        if existing_checksum is not None:
            skipped_by_checksum += 1
            continue

        database_document = Document(
            source=Source.DUPO,
            source_record_id=None,
            source_path=str(path),
            source_filename=path.name,
            file_checksum=checksum,
            language="nl",
            year=discovered_document.year,
            text_complete=False,
            text_method=None,
            processing_status="discovered",
            dupo=DupoMetadata(
                dupo_id=None,
                knuttel_number=None,
            ),
        )

        session.add(database_document)
        registered += 1

    session.flush()

    return DupoRegistrationResult(
        discovered=len(documents),
        registered=registered,
        skipped_by_path=skipped_by_path,
        skipped_by_checksum=skipped_by_checksum,
    )