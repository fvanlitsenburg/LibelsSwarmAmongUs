"""Register PDFs from the configured DUPO directory."""

from historical_text_pipeline.config.settings import get_settings
from historical_text_pipeline.db.session import get_session_factory
from historical_text_pipeline.ingest.dupo import find_dupo_pdfs
from historical_text_pipeline.ingest.dupo_registration import (
    register_dupo_pdfs,
)


def main() -> None:
    """Discover and register DUPO PDFs."""

    settings = get_settings()
    documents = find_dupo_pdfs(settings.dupo_root)

    session_factory = get_session_factory()

    with session_factory() as session:
        result = register_dupo_pdfs(session, documents)
        session.commit()

    print(f"DUPO root:           {settings.dupo_root}")
    print(f"PDFs discovered:     {result.discovered}")
    print(f"Newly registered:    {result.registered}")
    print(f"Already known path:  {result.skipped_by_path}")
    print(f"Duplicate contents:  {result.skipped_by_checksum}")


if __name__ == "__main__":
    main()