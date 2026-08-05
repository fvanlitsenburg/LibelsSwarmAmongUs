"""List the DUPO PDFs found in the configured directory."""

from historical_text_pipeline.config.settings import get_settings
from historical_text_pipeline.ingest.dupo import find_dupo_pdfs


def main() -> None:
    """Print every discovered DUPO PDF."""

    settings = get_settings()
    documents = find_dupo_pdfs(settings.dupo_root)

    print(f"DUPO root: {settings.dupo_root}")
    print(f"PDFs found: {len(documents)}")
    print()

    for document in documents:
        print(f"{document.year}: {document.path}")


if __name__ == "__main__":
    main()