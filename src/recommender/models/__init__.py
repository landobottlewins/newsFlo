"""Data models and database schemas."""

from recommender.models.article import Article
from recommender.models.base import Base
from recommender.models.db import create_db_engine, drop_db, get_session_factory, init_db
from recommender.models.interaction import Interaction, InteractionType
from recommender.models.user import User, UserInterest

__all__ = [
    "Article",
    "Base",
    "Interaction",
    "InteractionType",
    "User",
    "UserInterest",
    "create_db_engine",
    "drop_db",
    "get_session_factory",
    "init_db",
]
