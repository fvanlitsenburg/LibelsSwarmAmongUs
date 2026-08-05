"""Tests for the PostgreSQL test-database connection."""

from sqlalchemy import text

from historical_text_pipeline.config.settings import get_settings
from historical_text_pipeline.db.session import make_engine


def test_postgresql_test_database_is_reachable() -> None:
    settings = get_settings()
    engine = make_engine(settings.test_database_url)

    try:
        with engine.connect() as connection:
            database_name = connection.scalar(
                text("SELECT current_database()")
            )

        assert database_name == "lsau_test"

    finally:
        engine.dispose()