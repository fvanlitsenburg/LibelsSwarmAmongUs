"""Application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LSAU_",
        extra="ignore",
    )

    database_url: str
    test_database_url: str
    dupo_root: Path


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()