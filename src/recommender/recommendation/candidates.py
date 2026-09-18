"""Candidate article generation across multiple diverse sources."""

from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from recommender.models.article import Article
from recommender.models.interaction import Interaction
from recommender.processing.topics import TOPIC_TAXONOMY
from recommender.users.profiles import get_user_interests


def generate_candidates(
    user: Any,
    session: Session,
    limit: int = 50,
) -> list[Article]:
    """Retrieve a balanced, deduplicated pool of candidate articles for ranking.

    Combines multiple candidate sources:
    - User interests (articles matching user's top topics)
    - Recency (newest published articles)
    - Popularity (most interacted stories in the system)
    - Exploration (articles from unfamiliar topics)

    Args:
        user: User instance or user_id integer or dict.
        session: Active database session.
        limit: Total maximum candidates to return.

    Returns:
        Deduplicated list of Article instances.
    """
    user_id = user.id if hasattr(user, "id") else (user if isinstance(user, int) else None)
    interests: dict[str, float] = {}

    if user_id is not None:
        interests = get_user_interests(user_id, session)
    elif isinstance(user, dict):
        interests = user.get("interests", user)

    candidate_ids: set[int] = set()
    candidates: list[Article] = []

    def _add_article(art: Article | None) -> None:
        if art is not None and art.id not in candidate_ids and len(candidates) < limit:
            candidate_ids.add(art.id)
            candidates.append(art)

    # 1. User Interest Candidates (~40% of limit)
    if interests:
        top_topics = sorted(interests.keys(), key=lambda t: interests[t], reverse=True)[:5]
        # Query recent articles matching top topics
        stmt_arts = select(Article).order_by(desc(Article.published_at)).limit(limit * 2)
        all_recent = session.scalars(stmt_arts).all()
        for art in all_recent:
            if art.topics and any(t in art.topics for t in top_topics):
                _add_article(art)
                if len(candidates) >= int(limit * 0.40):
                    break

    # 2. Popular / Trending Candidates (~25% of limit)
    # Count interactions per article
    stmt_pop = (
        select(Interaction.article_id)
        .where(Interaction.article_id.isnot(None))
        .group_by(Interaction.article_id)
        .order_by(desc(Interaction.id))
        .limit(limit)
    )
    popular_ids = session.scalars(stmt_pop).all()
    for art_id in popular_ids:
        if art_id is not None and art_id not in candidate_ids:
            art = session.get(Article, art_id)
            _add_article(art)
            if len(candidates) >= int(limit * 0.65):
                break

    # 3. Recency Candidates (~25% of limit)
    stmt_recent = (
        select(Article)
        .order_by(desc(Article.published_at))
        .limit(limit)
    )
    for art in session.scalars(stmt_recent).all():
        _add_article(art)
        if len(candidates) >= int(limit * 0.90):
            break

    # 4. Exploration Candidates (~10% of limit)
    # Find topics outside the user's top interests
    all_taxonomy_topics = {topic for sub in TOPIC_TAXONOMY.values() for topic in sub}
    unexplored_topics = list(all_taxonomy_topics - set(interests.keys()))
    if unexplored_topics:
        stmt_explore = select(Article).order_by(desc(Article.published_at)).limit(limit)
        for art in session.scalars(stmt_explore).all():
            if art.topics and any(t in art.topics for t in unexplored_topics):
                _add_article(art)
                if len(candidates) >= limit:
                    break

    # Backfill if pool is still below limit
    if len(candidates) < limit:
        stmt_fill = select(Article).order_by(desc(Article.published_at)).limit(limit)
        for art in session.scalars(stmt_fill).all():
            _add_article(art)
            if len(candidates) >= limit:
                break

    return candidates

