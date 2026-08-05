"""Tests for updating DUPO records after PDF inspection."""

from pathlib import Path

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import Document, DupoMetadata
from historical_text_pipeline.domain import Source
from historical_text_pipeline.ingest import dupo_inspection
from historical_text_pipeline.ingest.dupo_inspection import (
    inspect_dupo_documents,
)
from historical_text_pipeline.ingest.pdf_inspection import PdfInspection


def add_dupo_document(
    session: Session,
    path: Path,
) -> Document:
    """Add one uninspected DUPO record."""

    document = Document(
        source=Source.DUPO,
        source_path=str(path),
        source_filename=path.name,
        year=1651,
        language="nl",
        processing_status="discovered",
        dupo=DupoMetadata(),
    )

    session.add(document)
    session.commit()

    return document


def test_document_with_no_text_layer_is_marked_for_ocr(
    db_engine: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    def fake_inspect_pdf(path: Path) -> PdfInspection:
        return PdfInspection(
            path=path,
            page_count=12,
            sample_pages=3,
            extracted_characters=0,
            alphabetic_ratio=0.0,
            usable_text_layer=False,
        )

    monkeypatch.setattr(
        dupo_inspection,
        "inspect_pdf",
        fake_inspect_pdf,
    )

    with Session(db_engine) as session:
        document = add_dupo_document(session, pdf_path)

        result = inspect_dupo_documents(session)
        session.commit()
        session.refresh(document)

        assert result.inspected == 1
        assert result.needs_ocr == 1
        assert document.total_units == 12
        assert document.text_method == "ocr"
        assert document.processing_status == "inspected"


def test_document_with_text_layer_skips_ocr(
    db_engine: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    def fake_inspect_pdf(path: Path) -> PdfInspection:
        return PdfInspection(
            path=path,
            page_count=8,
            sample_pages=3,
            extracted_characters=1200,
            alphabetic_ratio=0.80,
            usable_text_layer=True,
        )

    monkeypatch.setattr(
        dupo_inspection,
        "inspect_pdf",
        fake_inspect_pdf,
    )

    with Session(db_engine) as session:
        document = add_dupo_document(session, pdf_path)

        result = inspect_dupo_documents(session)
        session.commit()
        session.refresh(document)

        assert result.inspected == 1
        assert result.embedded_text == 1
        assert document.total_units == 8
        assert document.text_method == "embedded_text"
        assert document.processing_status == "inspected"