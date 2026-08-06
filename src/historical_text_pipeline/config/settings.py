"""Application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

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

    openai_api_key: SecretStr | None = None
    openai_ocr_model: str = "gpt-5.5"
    openai_ocr_max_output_tokens: int = 12_000
    openai_timeout_seconds: float = 300.0

    ocr_provider: Literal["mistral", "openai"] = "mistral"

    mistral_api_key: SecretStr | None = None
    mistral_api_base_url: str = "https://api.mistral.ai/v1"
    mistral_ocr_model: str = "mistral-ocr-4-0"
    mistral_timeout_seconds: float = 180.0

    pdf_render_dpi: int = 300
    pdf_jpeg_quality: int = 95
    
    


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()