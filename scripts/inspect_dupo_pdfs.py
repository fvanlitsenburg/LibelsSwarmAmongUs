"""Inspect registered DUPO PDFs before OCR."""

from historical_text_pipeline.db.session import get_session_factory
from historical_text_pipeline.ingest.dupo_inspection import (
    inspect_dupo_documents,
)


def main() -> None:
    """Inspect pending DUPO PDFs and save the results."""

    session_factory = get_session_factory()

    with session_factory() as session:
        result = inspect_dupo_documents(session)
        session.commit()

    print(f"Awaiting inspection: {result.pending}")
    print(f"Successfully inspected: {result.inspected}")
    print(f"Usable text layer: {result.embedded_text}")
    print(f"OCR required: {result.needs_ocr}")
    print(f"Failed: {result.failed}")


if __name__ == "__main__":
    main()