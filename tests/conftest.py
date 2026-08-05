"""Shared PostgreSQL test fixtures."""

from collections.abc import Iterator

import pytest
from sqlalchemy.engine import Engine

from historical_text_pipeline.config.settings import get_settings
from historical_text_pipeline.db.models import Document
from historical_text_pipeline.db.session import make_engine


@pytest.fixture
def db_engine() -> Iterator[Engine]:
    """
    Provide a clean PostgreSQL test schema.

    LSAU_TEST_DATABASE_URL must never point to the development database.
    """

    settings = get_settings()

    if settings.test_database_url == settings.database_url:
        raise RuntimeError(
            "The test and development database URLs must be different."
        )

    engine = make_engine(settings.test_database_url)
    metadata = Document.metadata

    metadata.drop_all(engine)
    metadata.create_all(engine)

    yield engine

    metadata.drop_all(engine)
    engine.dispose()