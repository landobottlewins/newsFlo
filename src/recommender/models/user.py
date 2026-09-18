"""SQLAlchemy models for users and user interests."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from recommender.models.base import Base

if TYPE_CHECKING:
    from recommender.models.interaction import Interaction


class User(Base):
    """User account entity representing a reader on the platform."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=lambda: datetime.now(UTC),
    )

    interests: Mapped[list["UserInterest"]] = relationship(
        "UserInterest",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="UserInterest.score.desc()",
    )
    interactions: Mapped[list["Interaction"]] = relationship(
        "Interaction",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="Interaction.timestamp.asc()",
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username!r})>"


class UserInterest(Base):
    """Normalized interest score for a user in a specific financial topic."""

    __tablename__ = "user_interests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    topic: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=lambda: datetime.now(UTC),
    )

    user: Mapped["User"] = relationship("User", back_populates="interests")

    __table_args__ = (UniqueConstraint("user_id", "topic", name="uq_user_interest_user_topic"),)

    def __repr__(self) -> str:
        return f"<UserInterest(user_id={self.user_id}, topic={self.topic!r}, score={self.score})>"
