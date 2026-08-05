"""Alembic migration environment for LSAU."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from historical_text_pipeline.config.settings import get_settings
from historical_text_pipeline.db.models import Document

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Load the real development URL from .env rather than storing it in Git.
database_url = get_settings().database_url.replace("%", "%%")
config.set_main_option("sqlalchemy.url", database_url)

# Importing Document loads the complete model module and its shared metadata.
target_metadata = Document.metadata


def run_migrations_offline() -> None:
    """Run migrations without creating a live connection."""

    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against the configured PostgreSQL database."""

    configuration = config.get_section(
        config.config_ini_section,
        {},
    )

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()