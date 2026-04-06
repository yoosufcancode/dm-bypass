"""Application settings loaded from environment variables prefixed with DM_."""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Pydantic settings model; all fields overridable via DM_* env vars."""

    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")

    cors_origins: list[str] = ["*"]

    max_workers: int = 2

    class Config:
        env_prefix = "DM_"


settings = Settings()
