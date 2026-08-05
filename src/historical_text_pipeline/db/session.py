"""PostgreSQL engine and session configuration."""

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from historical_text_pipeline.config.settings import get_settings


def make_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine for a database URL."""

    return create_engine(
        database_url,
        pool_pre_ping=True,
    )


@lru_cache
def get_engine() -> Engine:
    """Return the main application database engine."""

    return make_engine(get_settings().database_url)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Return the main application session factory."""

    return sessionmaker(
        bind=get_engine(),
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )