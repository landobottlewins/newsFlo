"""Tests for Feed Ranking Engine (Task 013)."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from recommender.models import (
    Article,
    Interaction,
    InteractionType,
    User,
    create_db_engine,
    drop_db,
    get_session_factory,
    init_db,
)
from recommender.recommendation import (
    ScoredCandidate,
    calculate_exploration_score,
    calculate_interest_score,
    calculate_novelty_score,
    calculate_popularity_score,
    calculate_quality_score,
    calculate_recency_score,
    rank_candidates,
)
from recommender.users import set_interest


@pytest.fixture
def session():
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    factory = get_session_factory(engine)
    with factory() as sess:
        yield sess
    drop_db(engine)
    engine.dispose()


def test_individual_score_components(session: Session):
    """Verify each ranking score component can be calculated in isolation."""
    now = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)
    art = Article(
        id=42,
        source="Bloomberg",
        title="NVIDIA Reports Record Earnings with AI Acceleration",
        description="Comprehensive quarterly financial statements and guidance breakdown.",
        raw_text="Detailed analysis of data center segment growth and foundry partnership.",
        url="https://example.com/nvda-42",
        published_at=now - timedelta(hours=2),
        topics={"AI": 0.90, "Semiconductors": 0.85},
    )
    user = {"interests": {"AI": 0.90, "Semiconductors": 0.85}}

    # Interest score
    interest = calculate_interest_score(user, art, session=session)
    assert interest > 0.80

    # Recency score (2 hours old is very fresh)
    recency = calculate_recency_score(art, current_time=now)
    assert recency > 0.90

    # Quality score (reputable source, description, raw_text > 100)
    quality = calculate_quality_score(art)
    assert quality >= 0.80

    # Novelty score (no prior interaction)
    novelty = calculate_novelty_score(user, art, session=session)
    assert novelty == 0.90

    # Popularity score (baseline with 0 interactions)
    popularity = calculate_popularity_score(art, session=session)
    assert 0.0 <= popularity <= 1.0

    # Exploration score (already interested in AI, so exploration is low)
    exploration = calculate_exploration_score(user, art, session=session)
    assert exploration < 0.30


def test_explainable_score_formatting():
    """Verify ScoredCandidate.explain() produces readable debugging output."""
    art = {"id": 42, "title": "Headline"}
    candidate = ScoredCandidate(
        article=art,
        interest=0.88,
        recency=0.94,
        quality=0.90,
        novelty=0.52,
        popularity=0.61,
        exploration=0.20,
        final_score=0.79,
    )
    explanation = candidate.explain()
    assert "Article #42" in explanation
    assert "interest:     0.88" in explanation
    assert "recency:      0.94" in explanation
    assert "FINAL:        0.79" in explanation


def test_ranking_deterministic_and_weighted(session: Session):
    """Acceptance criteria: rank_candidates produces deterministic, properly weighted ranks."""
    now = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)
    user = {"interests": {"AI": 0.95}}

    # Highly relevant and fresh article
    top_art = Article(
        id=1,
        source="Reuters",
        title="NVIDIA Announces Groundbreaking AI Architecture",
        description="Full report on next generation GPU performance.",
        published_at=now - timedelta(minutes=30),
        topics={"AI": 0.95},
    )

    # Completely unrelated and older article
    low_art = Article(
        id=2,
        source="RandomBlog",
        title="Local Livestock Auction",
        description="",
        published_at=now - timedelta(days=5),
        topics={"Commodities": 0.20},
    )

    candidates = [low_art, top_art]
    ranked = rank_candidates(user, candidates, session=session, current_time=now)

    assert len(ranked) == 2
    assert ranked[0].article.id == 1
    assert ranked[1].article.id == 2
    assert ranked[0].final_score > ranked[1].final_score

    # Determinism: running again with identical inputs yields identical scores
    ranked_2 = rank_candidates(user, candidates, session=session, current_time=now)
    assert ranked[0].final_score == ranked_2[0].final_score
    assert ranked[1].final_score == ranked_2[1].final_score

