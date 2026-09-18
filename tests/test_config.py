"""Tests for configuration and logging initialization."""

import logging
from pathlib import Path

from recommender.config import Settings, get_settings, setup_logging


def test_default_settings():
    """Test default settings values."""
    settings = Settings()
    assert settings.app_name == "Financial News Recommender"
    assert settings.env == "development"
    assert settings.log_level == "INFO"
    assert "postgresql" in settings.database_url
    assert settings.data_dir == Path("data")


def test_settings_env_override(monkeypatch):
    """Test that environment variables override defaults."""
    monkeypatch.setenv("APP_NAME", "Custom Recommender")
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/testdb")
    monkeypatch.setenv("DATA_DIR", "/tmp/newsflo_data")

    settings = Settings()
    assert settings.app_name == "Custom Recommender"
    assert settings.env == "test"
    assert settings.log_level == "DEBUG"
    assert settings.database_url == "postgresql+psycopg://user:pass@localhost:5432/testdb"
    assert settings.data_dir == Path("/tmp/newsflo_data")


def test_get_settings_cached():
    """Test that get_settings returns a valid Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_setup_logging():
    """Test setup_logging configures root logger without errors."""
    setup_logging("DEBUG")
    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG
