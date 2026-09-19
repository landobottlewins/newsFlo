"""Transient session-interest profiles and long-term interest blending.

Long-term interests remain in ``UserInterest`` records.  A ``SessionProfile``
is deliberately in-memory: ending or resetting it cannot overwrite the
persisted profile.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from recommender.models.article import Article
from recommender.models.interaction import Interaction
from recommender.processing.topics import classify_topics
from recommender.users.profiles import get_user_interests, normalize_interest_score
from recommender.users.updater import INTERACTION_SIGNALS


@dataclass(frozen=True)
class SessionPersonalizationConfig:
    """Tunable parameters for short-term personalization.

    The default blend follows Task 014 exactly.  Session scores decay with a
    short half-life so a topic read earlier in the visit loses influence as the
    reader explores something else.
    """

    long_term_weight: float = 0.7
    session_weight: float = 0.3
    session_half_life_minutes: float = 30.0
    interaction_scale: float = 0.05

    def __post_init__(self) -> None:
        if self.long_term_weight < 0 or self.session_weight < 0:
            raise ValueError("Interest blend weights must be non-negative")
        if not math.isclose(self.long_term_weight + self.session_weight, 1.0):
            raise ValueError("Interest blend weights must add up to 1.0")
        if self.session_half_life_minutes <= 0:
            raise ValueError("session_half_life_minutes must be positive")
        if self.interaction_scale < 0:
            raise ValueError("interaction_scale must be non-negative")


@dataclass
class SessionProfile:
    """A user's temporary interest state during one browsing session."""

    user_id: int
    session_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    interests: dict[str, float] = field(default_factory=dict)
    last_updated: dict[str, datetime] = field(default_factory=dict)
    config: SessionPersonalizationConfig = field(default_factory=SessionPersonalizationConfig)

    def record_topics(
        self,
        topic_weights: Mapping[str, float],
        *,
        signal_weight: float = 1.0,
        occurred_at: datetime | None = None,
    ) -> dict[str, float]:
        """Apply one session interaction and return the current session profile.

        Each topic is decayed from its own most recent interaction before the
        new signal is added.  Consequently, a fresh repeated interaction has
        more effect than an equally weighted, older one.
        """
        when = _as_utc(occurred_at or datetime.now(UTC))
        for topic, relevance in topic_weights.items():
            if not topic or not isinstance(relevance, (int, float)):
                continue

            previous = self.interests.get(topic, 0.0)
            previous_at = self.last_updated.get(topic)
            if previous_at is not None:
                previous = _decay_session_score(
                    previous,
                    when - _as_utc(previous_at),
                    self.config.session_half_life_minutes,
                )

            delta = signal_weight * float(relevance) * self.config.interaction_scale
            self.interests[topic] = normalize_interest_score(previous + delta)
            self.last_updated[topic] = when

        return dict(self.interests)

    def reset(self, *, started_at: datetime | None = None) -> None:
        """Clear only transient state, readying this profile for a new session."""
        self.interests.clear()
        self.last_updated.clear()
        self.started_at = _as_utc(started_at or datetime.now(UTC))

    def combined_interests(
        self,
        long_term_interests: Mapping[str, float],
        *,
        config: SessionPersonalizationConfig | None = None,
    ) -> dict[str, float]:
        """Return the blend used as a recommendation input for this session."""
        return combine_interests(long_term_interests, self.interests, config=config or self.config)


def create_session(
    user_id: int,
    *,
    session_id: str | None = None,
    started_at: datetime | None = None,
    config: SessionPersonalizationConfig | None = None,
) -> SessionProfile:
    """Create an empty, isolated profile for a new user session."""
    return SessionProfile(
        user_id=user_id,
        session_id=session_id or uuid4().hex,
        started_at=_as_utc(started_at or datetime.now(UTC)),
        config=config or SessionPersonalizationConfig(),
    )


def reset_session(profile: SessionProfile, *, started_at: datetime | None = None) -> SessionProfile:
    """End the current transient state without changing long-term interests."""
    profile.reset(started_at=started_at)
    return profile


def update_session_from_interaction(
    profile: SessionProfile,
    interaction: Interaction,
    db_session: Session,
) -> dict[str, float]:
    """Update a session profile from a stored interaction without touching the DB profile."""
    if interaction.user_id != profile.user_id:
        raise ValueError("Interaction user_id must match the session profile user_id")

    topic_weights: dict[str, float] = {}
    metadata = interaction.event_metadata or {}
    topic = metadata.get("topic")
    if isinstance(topic, str) and topic:
        topic_weights[topic] = 1.0

    if interaction.article_id is not None:
        article = db_session.get(Article, interaction.article_id)
        if article is not None:
            article_topics = article.topics or classify_topics(article)
            for article_topic, relevance in (article_topics or {}).items():
                topic_weights[article_topic] = max(topic_weights.get(article_topic, 0.0), relevance)

    signal_weight = INTERACTION_SIGNALS.get(interaction.event_type, 0.0)
    return profile.record_topics(
        topic_weights,
        signal_weight=signal_weight,
        occurred_at=interaction.timestamp,
    )


def combine_interests(
    long_term_interests: Mapping[str, float],
    session_interests: Mapping[str, float],
    *,
    config: SessionPersonalizationConfig | None = None,
) -> dict[str, float]:
    """Blend persisted and session interests using configurable profile weights."""
    active_config = config or SessionPersonalizationConfig()
    topics = set(long_term_interests) | set(session_interests)
    return {
        topic: normalize_interest_score(
            (active_config.long_term_weight * float(long_term_interests.get(topic, 0.0)))
            + (active_config.session_weight * float(session_interests.get(topic, 0.0)))
        )
        for topic in topics
    }


def get_combined_interests(
    user_id: int,
    profile: SessionProfile,
    db_session: Session,
    *,
    config: SessionPersonalizationConfig | None = None,
) -> dict[str, float]:
    """Fetch persisted interests and blend them with a matching session profile."""
    if profile.user_id != user_id:
        raise ValueError("profile user_id must match user_id")
    return combine_interests(
        get_user_interests(user_id, db_session),
        profile.interests,
        config=config or profile.config,
    )


def _decay_session_score(score: float, elapsed, half_life_minutes: float) -> float:
    """Decay a transient score over a timedelta, tolerating out-of-order events."""
    elapsed_minutes = max(0.0, elapsed.total_seconds() / 60.0)
    return score * math.pow(0.5, elapsed_minutes / half_life_minutes)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
