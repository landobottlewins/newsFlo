"""User interest profile management."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from recommender.models.user import User, UserInterest


def normalize_interest_score(score: float) -> float:
    """Normalize interest score to sensible [0.0, 1.0] boundary."""
    return round(max(0.0, min(1.0, float(score))), 4)


def get_user_interests(user_id: int, session: Session) -> dict[str, float]:
    """Retrieve all weighted topic interests for a given user.

    Args:
        user_id: Target user identifier.
        session: Active database session.

    Returns:
        Mapping of topic name to normalized interest score.
    """
    stmt = (
        select(UserInterest.topic, UserInterest.score)
        .where(UserInterest.user_id == user_id)
        .order_by(UserInterest.score.desc())
    )
    rows = session.execute(stmt).all()
    return {topic: score for topic, score in rows}


def set_interest(
    user_id: int,
    topic: str,
    score: float,
    session: Session,
) -> UserInterest:
    """Set or update a normalized interest score for a user in a topic.

    Args:
        user_id: Target user identifier.
        topic: Topic name from taxonomy.
        score: Desired interest score (automatically normalized to [0.0, 1.0]).
        session: Active database session.

    Returns:
        Persisted UserInterest record.
    """
    normalized_score = normalize_interest_score(score)

    # Ensure user exists
    user = session.get(User, user_id)
    if user is None:
        user = User(id=user_id, username=f"user_{user_id}")
        session.add(user)
        session.flush()

    # Check for existing interest record
    stmt = select(UserInterest).where(
        UserInterest.user_id == user_id,
        UserInterest.topic == topic,
    )
    interest = session.scalars(stmt).first()

    now = datetime.now(UTC)
    if interest is not None:
        interest.score = normalized_score
        interest.last_updated = now
    else:
        interest = UserInterest(
            user_id=user_id,
            topic=topic,
            score=normalized_score,
            last_updated=now,
        )
        session.add(interest)

    session.flush()
    return interest

