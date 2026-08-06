"""OCR individual DUPO pages and store their text."""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import (
    Document,
    DocumentTextUnit,
)
from historical_text_pipeline.domain import (
    ClassificationStatus,
    Source,
)
from historical_text_pipeline.ocr.base import OcrBackend
from historical_text_pipeline.ocr.pdf_rendering import (
    render_pdf_page_as_jpeg,
)


class DupoPageOcrError(Exception):
    """Raised when a DUPO page cannot be processed."""


@dataclass(frozen=True, slots=True)
class DupoPageOcrResult:
    """Result of processing one DUPO page."""

    document_id: int
    page_number: int
    created: bool
    skipped: bool
    character_count: int
    word_count: int
    provider: str | None
    model: str | None


def _find_existing_page(
    session: Session,
    *,
    document_id: int,
    page_number: int,
) -> DocumentTextUnit | None:
    """Return an already stored page transcription."""

    return session.scalar(
        select(DocumentTextUnit).where(
            DocumentTextUnit.document_id == document_id,
            DocumentTextUnit.unit_type == "page",
            DocumentTextUnit.unit_number == page_number,
        )
    )


def _count_processed_pages(
    session: Session,
    *,
    document_id: int,
) -> int:
    """Count stored page transcriptions for one document."""

    count = session.scalar(
        select(func.count(DocumentTextUnit.id)).where(
            DocumentTextUnit.document_id == document_id,
            DocumentTextUnit.unit_type == "page",
        )
    )

    return int(count or 0)

def get_missing_dupo_pages(
    session: Session,
    *,
    document_id: int,
) -> list[int]:
    """Return page numbers that have not yet been stored."""

    document = session.get(Document, document_id)

    if document is None:
        raise DupoPageOcrError(
            f"Document {document_id} does not exist."
        )

    if document.source != Source.DUPO:
        raise DupoPageOcrError(
            f"Document {document_id} is not a DUPO document."
        )

    if document.total_units is None:
        raise DupoPageOcrError(
            f"Document {document_id} has not been inspected."
        )

    stored_page_numbers = set(
        session.scalars(
            select(DocumentTextUnit.unit_number).where(
                DocumentTextUnit.document_id == document_id,
                DocumentTextUnit.unit_type == "page",
            )
        )
    )

    return [
        page_number
        for page_number in range(
            1,
            document.total_units + 1,
        )
        if page_number not in stored_page_numbers
    ]

def ocr_dupo_page(
    session: Session,
    *,
    document_id: int,
    page_number: int,
    backend: OcrBackend,
    dpi: int = 300,
    jpeg_quality: int = 95,
) -> DupoPageOcrResult:
    """
    OCR one DUPO PDF page and store the result.

    An existing page transcription is returned without making another
    paid OCR request.
    """

    document = session.get(Document, document_id)

    if document is None:
        raise DupoPageOcrError(
            f"Document {document_id} does not exist."
        )

    if document.source != Source.DUPO:
        raise DupoPageOcrError(
            f"Document {document_id} is not a DUPO document."
        )

    if document.source_path is None:
        raise DupoPageOcrError(
            f"Document {document_id} has no source path."
        )

    if page_number < 1:
        raise ValueError("Page numbers begin at 1.")

    if (
        document.total_units is not None
        and page_number > document.total_units
    ):
        raise DupoPageOcrError(
            f"Document {document_id} has "
            f"{document.total_units} pages; requested page "
            f"{page_number}."
        )

    existing_page = _find_existing_page(
        session,
        document_id=document_id,
        page_number=page_number,
    )

    if existing_page is not None:
        return DupoPageOcrResult(
            document_id=document_id,
            page_number=page_number,
            created=False,
            skipped=True,
            character_count=existing_page.character_count,
            word_count=existing_page.word_count,
            provider=existing_page.ocr_provider,
            model=existing_page.ocr_model,
        )

    rendered_page = render_pdf_page_as_jpeg(
        Path(document.source_path),
        page_number,
        dpi=dpi,
        jpeg_quality=jpeg_quality,
    )

    ocr_result = backend.recognize_image(
        rendered_page.image_bytes,
        mime_type=rendered_page.mime_type,
    )

    transcription = ocr_result.text.strip()

    if not transcription:
        raise DupoPageOcrError(
            f"OCR returned no text for document {document_id}, "
            f"page {page_number}."
        )

    character_count = len(transcription)
    word_count = len(transcription.split())

    text_unit = DocumentTextUnit(
        document_id=document.id,
        unit_type="page",
        unit_number=page_number,
        text=transcription,
        character_count=character_count,
        word_count=word_count,
        processing_method="ocr",
        ocr_provider=ocr_result.provider,
        ocr_model=ocr_result.model,
        provider_response_id=ocr_result.response_id,
    )

    session.add(text_unit)
    session.flush()

    processed_pages = _count_processed_pages(
        session,
        document_id=document.id,
    )

    document.units_processed = processed_pages
    document.error_message = None

    document.text_complete = (
        document.total_units is not None
        and processed_pages >= document.total_units
    )

    if document.text_complete:
        document.processing_status = "ocr_complete"
        document.classification_status = (
            ClassificationStatus.FULL_TEXT
        )
    else:
        document.processing_status = "ocr_partial"
        document.classification_status = (
            ClassificationStatus.PARTIAL_TEXT
        )

    session.flush()

    return DupoPageOcrResult(
        document_id=document.id,
        page_number=page_number,
        created=True,
        skipped=False,
        character_count=character_count,
        word_count=word_count,
        provider=ocr_result.provider,
        model=ocr_result.model,
    )