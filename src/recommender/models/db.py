"""Database engine and session management utilities."""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from recommender.config import get_settings
from recommender.models.base import Base


def create_db_engine(database_url: str | None = None, **kwargs) -> Engine:
    """Create a SQLAlchemy engine configured for PostgreSQL or SQLite."""
    url = database_url or get_settings().database_url
    return create_engine(url, **kwargs)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a sessionmaker bound to the given engine."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create all database tables defined in the metadata."""
    Base.metadata.create_all(bind=engine)


def drop_db(engine: Engine) -> None:
    """Drop all database tables defined in the metadata."""
    Base.metadata.drop_all(bind=engine)
