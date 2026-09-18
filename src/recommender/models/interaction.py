"""SQLAlchemy model and enums for tracking user article interactions."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from recommender.models.base import Base

if TYPE_CHECKING:
    from recommender.models.user import User


class InteractionType(StrEnum):
    """Supported user interaction event types."""

    IMPRESSION = "IMPRESSION"
    OPEN = "OPEN"
    SKIP = "SKIP"
    READ_30S = "READ_30S"
    READ_2M = "READ_2M"
    DEEP_DIVE = "DEEP_DIVE"
    BOOKMARK = "BOOKMARK"
    FOLLOW_TOPIC = "FOLLOW_TOPIC"
    RELATED_STORY = "RELATED_STORY"


VALID_INTERACTION_TYPES = {e.value for e in InteractionType}


class Interaction(Base):
    """User interaction event log entry."""

    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    article_id: Mapped[int | None] = mapped_column(
        ForeignKey("articles.id", ondelete="SET NULL"), nullable=True, index=True
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=lambda: datetime.now(UTC),
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="interactions")

    def __init__(
        self,
        user_id: int,
        event_type: str | InteractionType,
        article_id: int | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ):
        raw_event = event_type.value if isinstance(event_type, InteractionType) else str(event_type)
        if raw_event not in VALID_INTERACTION_TYPES:
            valid_list = sorted(VALID_INTERACTION_TYPES)
            raise ValueError(f"Invalid event_type {event_type!r}. Must be one of {valid_list}")

        super().__init__(
            user_id=user_id,
            article_id=article_id,
            event_type=raw_event,
            timestamp=timestamp or datetime.now(UTC),
            metadata_json=metadata,
            **kwargs,
        )

    @property
    def event_metadata(self) -> dict[str, Any] | None:
        """Alias for interaction metadata."""
        return self.metadata_json

    @event_metadata.setter
    def event_metadata(self, value: dict[str, Any] | None) -> None:
        self.metadata_json = value

    def __repr__(self) -> str:
        return (
            f"<Interaction(id={self.id}, user_id={self.user_id}, "
            f"article_id={self.article_id}, event_type={self.event_type!r})>"
        )
