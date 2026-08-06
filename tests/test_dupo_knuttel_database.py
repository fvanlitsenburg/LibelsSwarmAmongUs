"""Tests for saving extracted Knuttel metadata."""

from pathlib import Path

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import (
    Document,
    DupoMetadata,
)
from historical_text_pipeline.domain import Source
from historical_text_pipeline.ingest import dupo_knuttel
from historical_text_pipeline.ingest.dupo_knuttel import (
    KnuttelExtraction,
    extract_and_save_knuttel_number,
)


class FakeBackend:
    """Minimal OCR backend for the database tests."""

    def recognize_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/jpeg",
        include_embedded_images: bool = False,
    ):
        raise AssertionError("OCR should have been mocked.")

    def close(self) -> None:
        """Match the OCR backend protocol."""


def add_document(
    session: Session,
    pdf_path: Path,
    *,
    knuttel_number: str | None = None,
) -> Document:
    """Create one DUPO database record."""

    document = Document(
        source=Source.DUPO,
        source_path=str(pdf_path),
        source_filename=pdf_path.name,
        year=1660,
        language="nl",
        processing_status="inspected",
        dupo=DupoMetadata(
            knuttel_number=knuttel_number,
        ),
    )

    session.add(document)
    session.commit()

    return document


def test_saves_unique_knuttel_number(
    db_engine: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    def fake_extraction(
        pdf_path: Path,
        *,
        backend: FakeBackend,
        dpi: int,
        jpeg_quality: int,
    ) -> KnuttelExtraction:
        del pdf_path, backend, dpi, jpeg_quality

        return KnuttelExtraction(
            knuttel_number="418",
            candidates=("418",),
            ocr_text="418",
            provider="mistral",
            model="mistral-ocr-test",
            response_id="request-123",
        )

    monkeypatch.setattr(
        dupo_knuttel,
        "extract_knuttel_number_from_first_page",
        fake_extraction,
    )

    with Session(db_engine) as session:
        document = add_document(session, pdf_path)

        result = extract_and_save_knuttel_number(
            session,
            document_id=document.id,
            backend=FakeBackend(),
        )

        session.commit()
        session.refresh(document)

        assert result.saved is True
        assert result.knuttel_number == "418"
        assert document.dupo is not None
        assert document.dupo.knuttel_number == "418"


def test_does_not_overwrite_existing_number(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    with Session(db_engine) as session:
        document = add_document(
            session,
            pdf_path,
            knuttel_number="418",
        )

        result = extract_and_save_knuttel_number(
            session,
            document_id=document.id,
            backend=FakeBackend(),
        )

        assert result.saved is False
        assert result.skipped_existing is True
        assert result.knuttel_number == "418"


def test_ambiguous_result_is_not_saved(
    db_engine: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    def fake_extraction(
        pdf_path: Path,
        *,
        backend: FakeBackend,
        dpi: int,
        jpeg_quality: int,
    ) -> KnuttelExtraction:
        del pdf_path, backend, dpi, jpeg_quality

        return KnuttelExtraction(
            knuttel_number=None,
            candidates=("418", "1660"),
            ocr_text="418\n1660",
            provider="mistral",
            model="mistral-ocr-test",
            response_id="request-123",
        )

    monkeypatch.setattr(
        dupo_knuttel,
        "extract_knuttel_number_from_first_page",
        fake_extraction,
    )

    with Session(db_engine) as session:
        document = add_document(session, pdf_path)

        result = extract_and_save_knuttel_number(
            session,
            document_id=document.id,
            backend=FakeBackend(),
        )

        session.commit()
        session.refresh(document)

        assert result.saved is False
        assert result.candidates == ("418", "1660")
        assert document.dupo is not None
        assert document.dupo.knuttel_number is None