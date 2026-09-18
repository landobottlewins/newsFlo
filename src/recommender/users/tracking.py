"""User interaction event logging and retrieval."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from recommender.models.interaction import Interaction, InteractionType
from recommender.models.user import User


def record_interaction(
    user_id: int,
    event_type: str | InteractionType,
    article_id: int | None = None,
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    session: Session | None = None,
) -> Interaction:
    """Record a user interaction event.

    Storage is strictly decoupled from recommendation logic and interest updates.

    Args:
        user_id: ID of the user performing the action.
        event_type: Valid InteractionType or string name.
        article_id: Optional ID of the interacted article.
        timestamp: Optional interaction timestamp (defaults to UTC now).
        metadata: Optional dictionary containing event details.
        session: Active database session.

    Returns:
        The created Interaction instance.

    Raises:
        ValueError: If event_type is not a supported interaction type.
    """
    if session is None:
        raise ValueError("A database session must be provided to record_interaction")

    # Ensure user exists
    user = session.get(User, user_id)
    if user is None:
        user = User(id=user_id, username=f"user_{user_id}")
        session.add(user)
        session.flush()

    interaction = Interaction(
        user_id=user_id,
        event_type=event_type,
        article_id=article_id,
        timestamp=timestamp or datetime.now(UTC),
        metadata=metadata,
    )
    session.add(interaction)
    session.flush()
    return interaction


def get_user_interactions(
    user_id: int,
    session: Session,
    limit: int = 100,
) -> list[Interaction]:
    """Retrieve chronologically ordered interaction history for a user."""
    stmt = (
        select(Interaction)
        .where(Interaction.user_id == user_id)
        .order_by(Interaction.timestamp.asc())
        .limit(limit)
    )
    return list(session.scalars(stmt).all())

