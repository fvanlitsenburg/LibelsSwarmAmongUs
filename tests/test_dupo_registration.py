"""Tests for registering DUPO PDFs in PostgreSQL."""

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import Document
from historical_text_pipeline.domain import Source
from historical_text_pipeline.ingest.dupo import find_dupo_pdfs
from historical_text_pipeline.ingest.dupo_registration import (
    register_dupo_pdfs,
)


def create_file(path: Path, contents: bytes) -> None:
    """Create a test file and its parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)


def count_dupo_documents(session: Session) -> int:
    """Count registered DUPO documents."""

    return session.scalar(
        select(func.count(Document.id)).where(
            Document.source == Source.DUPO
        )
    ) or 0


def test_registers_new_dupo_pdfs(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "1650" / "first.pdf"
    second_path = tmp_path / "1651" / "second.pdf"

    create_file(first_path, b"first PDF contents")
    create_file(second_path, b"second PDF contents")

    discovered = find_dupo_pdfs(tmp_path)

    with Session(db_engine) as session:
        result = register_dupo_pdfs(session, discovered)
        session.commit()

        assert result.discovered == 2
        assert result.registered == 2
        assert result.skipped_by_path == 0
        assert result.skipped_by_checksum == 0
        assert count_dupo_documents(session) == 2

    with Session(db_engine) as session:
        saved = session.scalar(
            select(Document).where(
                Document.source_path == str(first_path.resolve())
            )
        )

        assert saved is not None
        assert saved.source == Source.DUPO
        assert saved.year == 1650
        assert saved.language == "nl"
        assert saved.source_filename == "first.pdf"
        assert saved.file_checksum is not None
        assert saved.processing_status == "discovered"
        assert saved.text_method == None
        assert saved.dupo is not None
        assert saved.dupo.dupo_id is None
        assert saved.dupo.knuttel_number is None


def test_second_registration_skips_known_paths(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    create_file(
        tmp_path / "1650" / "first.pdf",
        b"first PDF contents",
    )

    discovered = find_dupo_pdfs(tmp_path)

    with Session(db_engine) as session:
        first_result = register_dupo_pdfs(session, discovered)
        session.commit()

        assert first_result.registered == 1

    with Session(db_engine) as session:
        second_result = register_dupo_pdfs(session, discovered)
        session.commit()

        assert second_result.discovered == 1
        assert second_result.registered == 0
        assert second_result.skipped_by_path == 1
        assert second_result.skipped_by_checksum == 0
        assert count_dupo_documents(session) == 1


def test_duplicate_contents_are_not_registered_twice(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    identical_contents = b"the same PDF contents"

    first_root = tmp_path / "first-collection"
    second_root = tmp_path / "second-collection"

    create_file(
        first_root / "1650" / "first.pdf",
        identical_contents,
    )
    create_file(
        second_root / "1651" / "renamed.pdf",
        identical_contents,
    )

    with Session(db_engine) as session:
        first_result = register_dupo_pdfs(
            session,
            find_dupo_pdfs(first_root),
        )
        session.commit()

        assert first_result.registered == 1

    with Session(db_engine) as session:
        second_result = register_dupo_pdfs(
            session,
            find_dupo_pdfs(second_root),
        )
        session.commit()

        assert second_result.registered == 0
        assert second_result.skipped_by_path == 0
        assert second_result.skipped_by_checksum == 1
        assert count_dupo_documents(session) == 1