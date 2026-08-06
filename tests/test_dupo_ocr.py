"""Tests for storing DUPO page OCR."""

from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import (
    Document,
    DocumentTextUnit,
    DupoMetadata,
)
from historical_text_pipeline.domain import (
    ClassificationStatus,
    Source,
)
from historical_text_pipeline.ingest import dupo_ocr
from historical_text_pipeline.ingest.dupo_ocr import (
    get_missing_dupo_pages,
    ocr_dupo_page,
)
from historical_text_pipeline.ocr.base import OcrPageResult
from historical_text_pipeline.ocr.pdf_rendering import (
    RenderedPdfPage,
)


class FakeOcrBackend:
    """Return a fixed transcription and count requests."""

    def __init__(self) -> None:
        self.calls = 0

    def recognize_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/jpeg",
        include_embedded_images: bool = False,
    ) -> OcrPageResult:
        del image_bytes, mime_type, include_embedded_images

        self.calls += 1

        return OcrPageResult(
            provider="mistral",
            model="mistral-ocr-test",
            response_id="request-123",
            text="Dit is de tekst van de eerste pagina.",
        )

    def close(self) -> None:
        """Match the backend protocol."""


def add_dupo_document(
    session: Session,
    pdf_path: Path,
    *,
    page_count: int = 4,
) -> Document:
    """Create one registered and inspected DUPO document."""

    document = Document(
        source=Source.DUPO,
        source_path=str(pdf_path),
        source_filename=pdf_path.name,
        year=1660,
        language="nl",
        total_units=page_count,
        units_processed=0,
        text_complete=False,
        text_method="ocr",
        processing_status="inspected",
        dupo=DupoMetadata(),
    )

    session.add(document)
    session.commit()

    return document


def fake_render(
    pdf_path: Path,
    page_number: int,
    *,
    dpi: int,
    jpeg_quality: int,
) -> RenderedPdfPage:
    """Return a fake rendered image."""

    del dpi, jpeg_quality

    return RenderedPdfPage(
        pdf_path=pdf_path,
        page_number=page_number,
        mime_type="image/jpeg",
        image_bytes=b"fake rendered page",
    )


def test_saves_one_dupo_page(
    db_engine: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    monkeypatch.setattr(
        dupo_ocr,
        "render_pdf_page_as_jpeg",
        fake_render,
    )

    backend = FakeOcrBackend()

    with Session(db_engine) as session:
        document = add_dupo_document(
            session,
            pdf_path,
        )

        result = ocr_dupo_page(
            session,
            document_id=document.id,
            page_number=1,
            backend=backend,
        )

        session.commit()
        session.refresh(document)

        text_unit = session.scalar(
            select(DocumentTextUnit).where(
                DocumentTextUnit.document_id == document.id,
                DocumentTextUnit.unit_number == 1,
            )
        )

        assert result.created is True
        assert result.skipped is False
        assert backend.calls == 1

        assert text_unit is not None
        assert text_unit.unit_type == "page"
        assert text_unit.text == (
            "Dit is de tekst van de eerste pagina."
        )
        assert text_unit.processing_method == "ocr"
        assert text_unit.ocr_provider == "mistral"
        assert text_unit.ocr_model == "mistral-ocr-test"
        assert text_unit.provider_response_id == "request-123"

        assert document.units_processed == 1
        assert document.text_complete is False
        assert document.processing_status == "ocr_partial"


def test_existing_page_is_not_ocrd_again(
    db_engine: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    monkeypatch.setattr(
        dupo_ocr,
        "render_pdf_page_as_jpeg",
        fake_render,
    )

    backend = FakeOcrBackend()

    with Session(db_engine) as session:
        document = add_dupo_document(
            session,
            pdf_path,
        )

        first_result = ocr_dupo_page(
            session,
            document_id=document.id,
            page_number=1,
            backend=backend,
        )

        second_result = ocr_dupo_page(
            session,
            document_id=document.id,
            page_number=1,
            backend=backend,
        )

        session.commit()

        stored_count = session.scalar(
            select(func.count(DocumentTextUnit.id)).where(
                DocumentTextUnit.document_id == document.id,
                DocumentTextUnit.unit_type == "page",
                DocumentTextUnit.unit_number == 1,
            )
        )

        assert first_result.created is True
        assert second_result.skipped is True
        assert backend.calls == 1
        assert stored_count == 1


def test_final_page_marks_document_complete(
    db_engine: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "one-page.pdf"
    pdf_path.write_bytes(b"fake PDF")

    monkeypatch.setattr(
        dupo_ocr,
        "render_pdf_page_as_jpeg",
        fake_render,
    )

    backend = FakeOcrBackend()

    with Session(db_engine) as session:
        document = add_dupo_document(
            session,
            pdf_path,
            page_count=1,
        )

        ocr_dupo_page(
            session,
            document_id=document.id,
            page_number=1,
            backend=backend,
        )

        session.commit()
        session.refresh(document)

        assert document.units_processed == 1
        assert document.text_complete is True
        assert document.processing_status == "ocr_complete"
        assert document.classification_status == (
            ClassificationStatus.FULL_TEXT
        )
        
def test_returns_only_missing_page_numbers(
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    with Session(db_engine) as session:
        document = add_dupo_document(
            session,
            pdf_path,
            page_count=6,
        )

        for page_number in (1, 2, 3, 5):
            text = f"Text from page {page_number}."

            session.add(
                DocumentTextUnit(
                    document_id=document.id,
                    unit_type="page",
                    unit_number=page_number,
                    text=text,
                    character_count=len(text),
                    word_count=len(text.split()),
                    processing_method="ocr",
                    ocr_provider="mistral",
                    ocr_model="mistral-ocr-test",
                )
            )

        session.commit()

        missing_pages = get_missing_dupo_pages(
            session,
            document_id=document.id,
        )

        assert missing_pages == [4, 6]