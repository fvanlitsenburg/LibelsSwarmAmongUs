"""Source-specific document ingestion."""

from historical_text_pipeline.ingest.dupo import (
    DupoPdf,
    calculate_sha256,
    find_dupo_pdfs,
)
from historical_text_pipeline.ingest.dupo_inspection import (
    DupoInspectionResult,
    inspect_dupo_documents,
)
from historical_text_pipeline.ingest.dupo_registration import (
    DupoRegistrationResult,
    register_dupo_pdfs,
)
from historical_text_pipeline.ingest.pdf_inspection import (
    PdfInspection,
    PdfInspectionError,
    inspect_pdf,
)

__all__ = [
    "DupoInspectionResult",
    "DupoPdf",
    "DupoRegistrationResult",
    "PdfInspection",
    "PdfInspectionError",
    "calculate_sha256",
    "find_dupo_pdfs",
    "inspect_dupo_documents",
    "inspect_pdf",
    "register_dupo_pdfs",
]