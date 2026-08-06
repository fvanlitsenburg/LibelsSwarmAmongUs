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
from historical_text_pipeline.ingest.dupo_knuttel import (
    DupoKnuttelError,
    KnuttelExtraction,
    KnuttelSaveResult,
    extract_and_save_knuttel_number,
    extract_knuttel_number_from_first_page,
    find_knuttel_candidates,
    find_knuttel_number,
)
from historical_text_pipeline.ingest.dupo_ocr import (
    DupoPageOcrError,
    DupoPageOcrResult,
    get_missing_dupo_pages,
    ocr_dupo_page,
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
    "DupoKnuttelError",
    "DupoPageOcrError",
    "DupoPageOcrResult",
    "DupoPdf",
    "DupoRegistrationResult",
    "KnuttelExtraction",
    "KnuttelSaveResult",
    "PdfInspection",
    "PdfInspectionError",
    "calculate_sha256",
    "extract_and_save_knuttel_number",
    "extract_knuttel_number_from_first_page",
    "find_dupo_pdfs",
    "find_knuttel_candidates",
    "find_knuttel_number",
    "get_missing_dupo_pages",
    "inspect_dupo_documents",
    "inspect_pdf",
    "ocr_dupo_page",
    "register_dupo_pdfs",
    
]