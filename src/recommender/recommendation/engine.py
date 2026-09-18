"""Transparent baseline recommendation engine using topic matching."""

import math
from typing import Any

from sqlalchemy.orm import Session

from recommender.processing.topics import classify_topics
from recommender.users.profiles import get_user_interests


def _extract_user_topics(user: Any, session: Session | None = None) -> dict[str, float]:
    """Extract topic interest weights from a user dict, object, or database profile."""
    if isinstance(user, dict):
        if "interests" in user and isinstance(user["interests"], dict):
            return user["interests"]
        return {k: float(v) for k, v in user.items() if isinstance(v, (int, float))}

    # If User ORM model
    user_id = getattr(user, "id", None)
    if user_id is not None and session is not None:
        db_interests = get_user_interests(user_id, session)
        if db_interests:
            return db_interests

    # Check for direct attribute
    interests_attr = getattr(user, "interests", None)
    if isinstance(interests_attr, dict):
        return interests_attr
    elif isinstance(interests_attr, list):
        return {item.topic: float(item.score) for item in interests_attr if hasattr(item, "topic")}

    return {}


def _extract_article_topics(article: Any) -> dict[str, float]:
    """Extract topic relevance weights from an article."""
    if isinstance(article, dict):
        topics = article.get("topics")
        if topics and isinstance(topics, dict):
            return topics
        return classify_topics(article)

    topics = getattr(article, "topics", None)
    if topics and isinstance(topics, dict):
        return topics

    return classify_topics(article)


def score_article(
    user: Any,
    article: Any,
    session: Session | None = None,
) -> float:
    """Calculate transparent relevance score between user interests and article topics.

    Uses normalized weighted cosine dot-product between user interest vector
    and article topic relevance vector.

    Args:
        user: User instance or interest dictionary.
        article: Article instance or dictionary with topics.
        session: Optional active database session.

    Returns:
        Relevance score in [0.0, 1.0].
    """
    user_interests = _extract_user_topics(user, session)
    article_topics = _extract_article_topics(article)

    if not user_interests or not article_topics:
        return 0.0

    common_topics = set(user_interests.keys()) & set(article_topics.keys())
    if not common_topics:
        return 0.0

    dot_product = sum(user_interests[t] * article_topics[t] for t in common_topics)

    # Calculate L2 norms for cosine normalization
    norm_user = math.sqrt(sum(v * v for v in user_interests.values()))
    norm_article = math.sqrt(sum(v * v for v in article_topics.values()))

    if norm_user == 0.0 or norm_article == 0.0:
        return 0.0

    cosine = dot_product / (norm_user * norm_article)
    return round(max(0.0, min(1.0, cosine)), 4)


def recommend(
    user: Any,
    articles: list[Any],
    k: int = 20,
    session: Session | None = None,
) -> list[Any]:
    """Recommend top-k articles for a user based on topic relevance.

    Args:
        user: User profile or interest mapping.
        articles: Pool of articles to rank.
        k: Maximum number of recommendations to return.
        session: Optional database session.

    Returns:
        List of up to k articles, ordered by descending relevance.
    """
    if not articles:
        return []

    scored: list[tuple[float, Any]] = []
    for art in articles:
        rel_score = score_article(user, art, session=session)
        scored.append((rel_score, art))

    # Sort descending by relevance score
    scored.sort(key=lambda item: item[0], reverse=True)
    return [art for _, art in scored[:k]]

