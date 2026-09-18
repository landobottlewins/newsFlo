"""Explainable multi-signal feed ranking engine."""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from recommender.models.interaction import Interaction
from recommender.recommendation.engine import score_article
from recommender.users.profiles import get_user_interests


@dataclass
class ScoredCandidate:
    """Container storing individual scoring components and final rank for an article."""

    article: Any
    interest: float
    recency: float
    quality: float
    novelty: float
    popularity: float
    exploration: float
    final_score: float

    def explain(self) -> str:
        """Format an explainable breakdown of the score components for debugging."""
        if isinstance(self.article, dict):
            art_id = self.article.get("id", "Unknown")
        else:
            art_id = getattr(self.article, "id", "Unknown")
        return (
            f"Article #{art_id}\n\n"
            f"interest:     {self.interest:.2f}\n"
            f"recency:      {self.recency:.2f}\n"
            f"quality:      {self.quality:.2f}\n"
            f"novelty:      {self.novelty:.2f}\n"
            f"popularity:   {self.popularity:.2f}\n"
            f"exploration:  {self.exploration:.2f}\n\n"
            f"FINAL:        {self.final_score:.2f}"
        )


def calculate_interest_score(user: Any, article: Any, session: Session | None = None) -> float:
    """Calculate user interest alignment using topic relevance."""
    return score_article(user, article, session=session)


def calculate_recency_score(
    article: Any,
    current_time: datetime | None = None,
    half_life_hours: float = 24.0,
) -> float:
    """Calculate publication recency score using exponential decay over time."""
    pub_time = getattr(article, "published_at", None)
    if not pub_time and isinstance(article, dict):
        pub_time = article.get("published_at")

    if not pub_time:
        return 0.50

    now = current_time or datetime.now(UTC)
    if pub_time.tzinfo is None:
        pub_time = pub_time.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    diff = now - pub_time
    hours = max(0.0, diff.total_seconds() / 3600.0)

    # Exponential decay based on half-life hours
    decay = math.pow(0.5, hours / half_life_hours)
    return round(max(0.0, min(1.0, decay)), 4)


def calculate_quality_score(article: Any) -> float:
    """Calculate article quality heuristic based on text length and completeness."""
    title_raw = getattr(article, "title", None) or (
        article.get("title") if isinstance(article, dict) else ""
    )
    title = (title_raw or "").strip()

    desc_raw = getattr(article, "description", None) or (
        article.get("description") if isinstance(article, dict) else ""
    )
    desc = (desc_raw or "").strip()

    raw_val = getattr(article, "raw_text", None) or (
        article.get("raw_text") if isinstance(article, dict) else ""
    )
    raw = (raw_val or "").strip()

    if not title:
        return 0.10

    score = 0.50  # Baseline
    total_len = len(desc) + len(raw)

    if total_len > 300:
        score += 0.25
    elif total_len > 100:
        score += 0.15

    # Check for presence of description
    if desc:
        score += 0.15

    # Source credibility heuristic
    source_val = getattr(article, "source", "") or (
        article.get("source") if isinstance(article, dict) else ""
    )
    source = (source_val or "").lower()
    if source in {"reuters", "bloomberg", "wsj", "ft", "financial times"}:
        score += 0.10

    return round(max(0.0, min(1.0, score)), 4)


def calculate_novelty_score(
    user: Any,
    article: Any,
    session: Session | None = None,
) -> float:
    """Calculate novelty score, penalizing stories the user has already interacted with."""
    user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
    art_id = getattr(article, "id", None) or (
        article.get("id") if isinstance(article, dict) else None
    )

    if user_id is None or art_id is None or session is None:
        return 0.90

    # Check if user previously interacted with this article
    stmt = select(func.count(Interaction.id)).where(
        Interaction.user_id == user_id,
        Interaction.article_id == art_id,
    )
    interaction_count = session.scalar(stmt) or 0

    if interaction_count > 0:
        return 0.10

    return 0.90


def calculate_popularity_score(
    article: Any,
    session: Session | None = None,
) -> float:
    """Calculate popularity score based on aggregate interaction frequency."""
    art_id = getattr(article, "id", None) or (
        article.get("id") if isinstance(article, dict) else None
    )
    if art_id is None or session is None:
        return 0.50

    stmt = select(func.count(Interaction.id)).where(Interaction.article_id == art_id)
    count = session.scalar(stmt) or 0

    # Scale count logarithmically to [0.2, 1.0]
    scaled = min(1.0, 0.2 + (math.log1p(count) * 0.25))
    return round(scaled, 4)


def calculate_exploration_score(
    user: Any,
    article: Any,
    session: Session | None = None,
) -> float:
    """Reward articles covering topics that are novel or unexplored by the user."""
    user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
    interests: dict[str, float] = {}

    if user_id is not None and session is not None:
        interests = get_user_interests(user_id, session)
    elif isinstance(user, dict):
        interests = user.get("interests", {})

    art_topics = (
        getattr(article, "topics", None)
        or (article.get("topics") if isinstance(article, dict) else {})
        or {}
    )
    if not art_topics:
        return 0.50

    # If article contains topics user has low or zero interest in, score higher
    unfamiliar_scores = [
        1.0 - interests.get(t, 0.0)
        for t in art_topics
    ]
    if unfamiliar_scores:
        avg_unfamiliar = sum(unfamiliar_scores) / len(unfamiliar_scores)
        return round(max(0.0, min(1.0, avg_unfamiliar)), 4)

    return 0.50


def rank_candidates(
    user: Any,
    candidates: list[Any],
    session: Session | None = None,
    current_time: datetime | None = None,
) -> list[ScoredCandidate]:
    """Rank candidate articles using the weighted multi-signal formula.

    Formula:
        score =
            0.40 × interest
          + 0.20 × recency
          + 0.15 × quality
          + 0.10 × novelty
          + 0.10 × popularity
          + 0.05 × exploration

    Args:
        user: User instance, dict, or user ID.
        candidates: List of candidate articles.
        session: Optional database session.
        current_time: Reference timestamp for recency calculations.

    Returns:
        List of ScoredCandidate instances sorted by final_score descending.
    """
    ranked: list[ScoredCandidate] = []

    for art in candidates:
        interest = calculate_interest_score(user, art, session=session)
        recency = calculate_recency_score(art, current_time=current_time)
        quality = calculate_quality_score(art)
        novelty = calculate_novelty_score(user, art, session=session)
        popularity = calculate_popularity_score(art, session=session)
        exploration = calculate_exploration_score(user, art, session=session)

        final_score = (
            (0.40 * interest)
            + (0.20 * recency)
            + (0.15 * quality)
            + (0.10 * novelty)
            + (0.10 * popularity)
            + (0.05 * exploration)
        )
        final_score = round(max(0.0, min(1.0, final_score)), 4)

        ranked.append(
            ScoredCandidate(
                article=art,
                interest=interest,
                recency=recency,
                quality=quality,
                novelty=novelty,
                popularity=popularity,
                exploration=exploration,
                final_score=final_score,
            )
        )

    # Sort descending by final score
    ranked.sort(key=lambda item: item.final_score, reverse=True)
    return ranked
