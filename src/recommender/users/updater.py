"""Behavioral interest updater and time decay algorithms."""

import math
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from recommender.models.article import Article
from recommender.models.interaction import Interaction, InteractionType
from recommender.models.user import UserInterest
from recommender.processing.topics import classify_topics
from recommender.users.profiles import get_user_interests, set_interest

# Signal weights specified in Task 010
INTERACTION_SIGNALS: dict[str, float] = {
    InteractionType.IMPRESSION.value: 0.0,
    InteractionType.OPEN.value: 1.0,
    InteractionType.READ_30S.value: 2.0,
    InteractionType.READ_2M.value: 4.0,
    InteractionType.DEEP_DIVE.value: 6.0,
    InteractionType.BOOKMARK.value: 5.0,
    InteractionType.FOLLOW_TOPIC.value: 7.0,
    InteractionType.RELATED_STORY.value: 5.0,
    InteractionType.SKIP.value: -2.0,
}

# Scaling factor to convert raw integer interaction points into [0.0, 1.0] delta range
SIGNAL_SCALE = 0.05  # e.g., +4 points -> +0.20 delta; -2 points -> -0.10 delta


def apply_decay(
    score: float,
    elapsed_time: timedelta | float,
    half_life_days: float = 14.0,
) -> float:
    """Apply time decay to an interest score.

    Older interactions gradually have less influence.

    Args:
        score: Initial interest score.
        elapsed_time: Elapsed time as a timedelta or duration in seconds.
        half_life_days: Time in days after which score halves.

    Returns:
        Decayed score bounded in [0.0, 1.0].
    """
    if isinstance(elapsed_time, timedelta):
        days = elapsed_time.total_seconds() / 86400.0
    else:
        # If passed as numeric seconds or days
        days = float(elapsed_time) / 86400.0 if elapsed_time > 1000 else float(elapsed_time)

    if days <= 0 or score <= 0:
        return max(0.0, min(1.0, score))

    decay_factor = math.pow(0.5, days / half_life_days)
    decayed = score * decay_factor
    return round(max(0.0, min(1.0, decayed)), 4)


def update_user_interests_from_interaction(
    interaction: Interaction,
    session: Session,
    decay: bool = True,
    current_time: datetime | None = None,
) -> dict[str, float]:
    """Update a user's topic interests based on a recorded interaction event.

    Algorithm:
    1. Retrieve article topics or event topic metadata
    2. Retrieve user's existing topic interests
    3. Apply time decay to relevant existing topics
    4. Apply interaction signal weight
    5. Normalize scores into [0.0, 1.0]
    6. Update database records and timestamps

    Args:
        interaction: The user Interaction instance.
        session: Active database session.
        decay: Whether to apply time decay since the topic's last update.
        current_time: Optional reference time (defaults to interaction timestamp).

    Returns:
        Mapping of updated topic names to new normalized scores.
    """
    now = current_time or interaction.timestamp or datetime.now(UTC)
    user_id = interaction.user_id
    event_type = interaction.event_type

    signal_weight = INTERACTION_SIGNALS.get(event_type, 0.0)

    # 1. Determine relevant topics and their weights
    topic_weights: dict[str, float] = {}

    # Check for direct topic interaction (e.g. FOLLOW_TOPIC)
    meta = interaction.event_metadata or {}
    if "topic" in meta and meta["topic"]:
        topic_weights[meta["topic"]] = 1.0

    # Retrieve topics from associated article if present
    if interaction.article_id:
        article = session.get(Article, interaction.article_id)
        if article is not None:
            art_topics = article.topics
            if not art_topics:
                art_topics = classify_topics(article)
                session.flush()
            for topic, relevance in (art_topics or {}).items():
                # Blend with any existing weight
                topic_weights[topic] = max(topic_weights.get(topic, 0.0), relevance)

    if not topic_weights:
        return get_user_interests(user_id, session)

    # 2 & 3. Retrieve user interests and apply decay + delta
    for topic, relevance in topic_weights.items():
        # Get existing interest record if present
        existing_interest = (
            session.query(UserInterest)
            .filter(UserInterest.user_id == user_id, UserInterest.topic == topic)
            .first()
        )

        current_score = existing_interest.score if existing_interest else 0.0

        # Apply decay if existing interest and time elapsed
        if decay and existing_interest and existing_interest.last_updated:
            last_ts = existing_interest.last_updated
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=UTC)
            ref_now = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
            elapsed = ref_now - last_ts
            current_score = apply_decay(current_score, elapsed)

        # Calculate score adjustment: signal_weight * relevance * SIGNAL_SCALE
        delta = signal_weight * relevance * SIGNAL_SCALE
        new_score = current_score + delta

        # Normalize score to sensible [0.0, 1.0] boundary
        set_interest(user_id, topic, new_score, session)

    session.flush()
    return get_user_interests(user_id, session)
