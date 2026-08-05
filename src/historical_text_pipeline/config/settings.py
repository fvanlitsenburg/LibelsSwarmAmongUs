"""Application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
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

    transkribus_api_base_url: str = (
        "https://transkribus.eu/processing/v1"
    )
    transkribus_token_url: str = (
        "https://account.readcoop.eu/auth/realms/readcoop/"
        "protocol/openid-connect/token"
    )
    
    transkribus_client_id: str = "processing-api-client"
    
    transkribus_model_id: int
    transkribus_username: str
    transkribus_password: SecretStr

    transkribus_poll_interval_seconds: float = 3.0
    transkribus_timeout_seconds: float = 300.0

    pdf_render_dpi: int = 300
    pdf_jpeg_quality: int = 90


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()