"""Inspect registered DUPO PDFs and update their database records."""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import Document
from historical_text_pipeline.domain import Source
from historical_text_pipeline.ingest.pdf_inspection import (
    PdfInspectionError,
    inspect_pdf,
)


@dataclass(frozen=True, slots=True)
class DupoInspectionResult:
    """Summary of a DUPO inspection run."""

    pending: int
    inspected: int
    embedded_text: int
    needs_ocr: int
    failed: int


def get_pending_dupo_documents(session: Session) -> list[Document]:
    """Return registered DUPO documents awaiting PDF inspection."""

    return list(
        session.scalars(
            select(Document)
            .where(
                Document.source == Source.DUPO,
                Document.processing_status == "discovered",
            )
            .order_by(Document.id)
        )
    )


def inspect_dupo_documents(
    session: Session,
) -> DupoInspectionResult:
    """
    Inspect every registered DUPO PDF that is still marked as discovered.

    The caller controls the transaction and must commit or roll back.
    """

    pending_documents = get_pending_dupo_documents(session)

    inspected = 0
    embedded_text = 0
    needs_ocr = 0
    failed = 0

    for document in pending_documents:
        if document.source_path is None:
            document.processing_status = "pdf_error"
            document.error_message = "Document has no source path."
            failed += 1
            continue

        try:
            inspection = inspect_pdf(Path(document.source_path))

        except (OSError, PdfInspectionError) as error:
            document.processing_status = "pdf_error"
            document.error_message = str(error)
            failed += 1
            continue

        document.total_units = inspection.page_count
        document.units_processed = 0
        document.text_complete = False
        document.error_message = None
        document.processing_status = "inspected"

        if inspection.usable_text_layer:
            document.text_method = "embedded_text"
            embedded_text += 1
        else:
            document.text_method = "ocr"
            needs_ocr += 1

        inspected += 1

    session.flush()

    return DupoInspectionResult(
        pending=len(pending_documents),
        inspected=inspected,
        embedded_text=embedded_text,
        needs_ocr=needs_ocr,
        failed=failed,
    )