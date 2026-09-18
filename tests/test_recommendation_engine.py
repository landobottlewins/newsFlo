"""Tests for Basic Recommendation Engine (Task 011)."""

from recommender.recommendation.engine import recommend, score_article


def test_score_article_relevance():
    """Verify calculating relevance score between user interests and article topics."""
    user = {"AI": 0.9, "Markets": 0.6, "Crypto": 0.1}
    art_ai = {"title": "AI Boom", "topics": {"AI": 0.8, "Markets": 0.7, "Crypto": 0.0}}
    art_unrelated = {"title": "Gold Mining", "topics": {"Commodities": 0.9}}

    score_ai = score_article(user, art_ai)
    score_unrelated = score_article(user, art_unrelated)

    assert 0.0 < score_ai <= 1.0
    assert score_unrelated == 0.0


def test_different_users_receive_different_rankings():
    """Acceptance criteria: two users with distinct interests receive different rankings."""
    tech_user = {"interests": {"AI": 0.95, "Semiconductors": 0.90}}
    macro_user = {"interests": {"Inflation": 0.95, "Interest Rates": 0.90, "Bonds": 0.85}}

    art_nvda = {"id": 1, "title": "NVIDIA New Chip", "topics": {"AI": 0.9, "Semiconductors": 0.9}}
    art_fed = {
        "id": 2,
        "title": "Fed Rate Decision",
        "topics": {"Interest Rates": 0.9, "Inflation": 0.8},
    }
    art_apple = {"id": 3, "title": "Apple Hardware", "topics": {"AI": 0.5, "Hardware": 0.8}}

    pool = [art_nvda, art_fed, art_apple]

    tech_feed = recommend(tech_user, pool, k=3)
    macro_feed = recommend(macro_user, pool, k=3)

    # Tech user should have NVIDIA first
    assert tech_feed[0]["id"] == 1

    # Macro user should have Fed Rate Decision first
    assert macro_feed[0]["id"] == 2

    # Rankings must be different
    assert [a["id"] for a in tech_feed] != [a["id"] for a in macro_feed]


def test_empty_pool_and_empty_interests():
    """Verify empty article list or empty user interests return gracefully."""
    assert recommend({"AI": 0.9}, [], k=10) == []
    pool = [{"id": 1, "topics": {"AI": 0.5}}]
    # User with no interests scores 0 for all
    recs = recommend({}, pool, k=10)
    assert len(recs) == 1
