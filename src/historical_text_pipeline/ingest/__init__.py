"""Source-specific document ingestion."""

from historical_text_pipeline.ingest.dupo import (
    DupoPdf,
    find_dupo_pdfs,
)

__all__ = [
    "DupoPdf",
    "find_dupo_pdfs",
]