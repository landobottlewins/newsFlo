"""Configuration and logging setup for the recommender engine."""

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FeedSource(BaseModel):
    """Configured news feed source descriptor."""

    name: str
    url: str


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="Financial News Recommender", alias="APP_NAME")
    env: str = Field(default="development", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/newsflo",
        alias="DATABASE_URL",
    )
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    news_sources: str = Field(
        default="",
        alias="NEWS_SOURCES",
        description="Feed sources formatted as 'source_name,url;source_b,url_b'",
    )

    def get_feed_sources(self) -> list[FeedSource]:
        """Parse NEWS_SOURCES string into FeedSource objects."""
        sources: list[FeedSource] = []
        if not self.news_sources:
            return sources
        for item in self.news_sources.split(";"):
            item = item.strip()
            if not item:
                continue
            parts = item.split(",", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                sources.append(FeedSource(name=parts[0].strip(), url=parts[1].strip()))
        return sources


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


def setup_logging(level: str | None = None) -> None:
    """Configure basic standard logging format and level."""
    log_level = (level or get_settings().log_level).upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
