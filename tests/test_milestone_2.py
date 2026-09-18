"""End-to-end integration test for Milestone 2: Personalized Recommender (Tasks 006–013).

Verifies that:
articles + user behavior -> personalized feed
and different users receive different rankings.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from recommender.models import (
    Article,
    InteractionType,
    User,
    create_db_engine,
    drop_db,
    get_session_factory,
    init_db,
)
from recommender.processing import classify_topics, embed_article
from recommender.recommendation import generate_candidates, rank_candidates
from recommender.users import record_interaction, update_user_interests_from_interaction


@pytest.fixture
def session():
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    factory = get_session_factory(engine)
    with factory() as sess:
        yield sess
    drop_db(engine)
    engine.dispose()


def test_milestone_2_personalized_recommender(session: Session):
    """Verify end-to-end flow: articles + user behavior -> personalized feeds.

    Ensures distinct rankings for users with different interests.
    """
    now = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)

    # 1. Populate pool of financial articles across diverse topics
    raw_articles = [
        Article(
            id=101,
            source="Bloomberg",
            title="NVIDIA Unveils New Blackwell AI Chip Architecture",
            description="Jensen Huang details massive GPU data center infrastructure demand.",
            url="https://example.com/nvda-ai",
            published_at=now - timedelta(hours=1),
        ),
        Article(
            id=102,
            source="Reuters",
            title="TSMC Reports Soaring AI Semiconductor Foundry Bookings",
            description="Taiwan chipmaker expands advanced wafer packaging capacity.",
            url="https://example.com/tsmc-chips",
            published_at=now - timedelta(hours=2),
        ),
        Article(
            id=103,
            source="WSJ",
            title="Fed Prepares 25 Bps Rate Cut as Inflation Cools",
            description="Federal Reserve policymakers signal monetary easing after CPI drops.",
            url="https://example.com/fed-cut",
            published_at=now - timedelta(hours=3),
        ),
        Article(
            id=104,
            source="FT",
            title="Treasury Yields Fall Globally Following Central Bank Signals",
            description="Government bond yields slide as economic growth moderates.",
            url="https://example.com/bonds-yields",
            published_at=now - timedelta(hours=4),
        ),
        Article(
            id=105,
            source="CNBC",
            title="Crude Oil Prices Stabilize After Middle East Supply Report",
            description="Commodities futures settle as inventories rise.",
            url="https://example.com/oil-commodities",
            published_at=now - timedelta(hours=5),
        ),
    ]

    # Process articles: classify topics and embed
    for art in raw_articles:
        classify_topics(art)
        embed_article(art)
        session.add(art)
    session.commit()

    # 2. Create User 1 (Tech/AI Enthusiast) and User 2 (Macroeconomic Trader)
    user_tech = User(id=1, username="ai_engineer")
    user_macro = User(id=2, username="macro_trader")
    session.add_all([user_tech, user_macro])
    session.commit()

    # 3. Simulate User Behavior for User 1: Reads AI/Chip articles for 2 minutes and bookmarks
    int_1a = record_interaction(
        user_id=user_tech.id,
        event_type=InteractionType.READ_2M,
        article_id=101,
        session=session,
    )
    int_1b = record_interaction(
        user_id=user_tech.id,
        event_type=InteractionType.BOOKMARK,
        article_id=102,
        session=session,
    )
    update_user_interests_from_interaction(int_1a, session=session)
    update_user_interests_from_interaction(int_1b, session=session)

    # 4. Simulate User Behavior for User 2: Reads Fed/Bond articles and follows interest rates
    int_2a = record_interaction(
        user_id=user_macro.id,
        event_type=InteractionType.READ_2M,
        article_id=103,
        session=session,
    )
    int_2b = record_interaction(
        user_id=user_macro.id,
        event_type=InteractionType.FOLLOW_TOPIC,
        metadata={"topic": "Interest Rates"},
        session=session,
    )
    int_2c = record_interaction(
        user_id=user_macro.id,
        event_type=InteractionType.SKIP,
        article_id=101,  # Skips AI
        session=session,
    )
    update_user_interests_from_interaction(int_2a, session=session)
    update_user_interests_from_interaction(int_2b, session=session)
    update_user_interests_from_interaction(int_2c, session=session)
    session.commit()

    # 5. Candidate Generation
    cands_tech = generate_candidates(user_tech, session=session, limit=10)
    cands_macro = generate_candidates(user_macro, session=session, limit=10)

    assert len(cands_tech) > 0
    assert len(cands_macro) > 0

    # 6. Feed Ranking
    feed_tech = rank_candidates(user_tech, cands_tech, session=session, current_time=now)
    feed_macro = rank_candidates(user_macro, cands_macro, session=session, current_time=now)

    # User 1 top recommendation must be Tech/AI (101 or 102)
    top_tech_id = feed_tech[0].article.id
    assert top_tech_id in {101, 102}

    # User 2 top recommendation must be Macro/Rates/Bonds (103 or 104)
    top_macro_id = feed_macro[0].article.id
    assert top_macro_id in {103, 104}

    # Verify feeds produce distinctly different rankings
    ranked_tech_ids = [c.article.id for c in feed_tech]
    ranked_macro_ids = [c.article.id for c in feed_macro]
    assert ranked_tech_ids != ranked_macro_ids

    # Explainability check: each candidate provides a human-readable explanation
    explanation = feed_tech[0].explain()
    assert "interest:" in explanation
    assert "recency:" in explanation
    assert "FINAL:" in explanation
