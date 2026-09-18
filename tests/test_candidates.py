"""Tests for Candidate Generation (Task 012)."""

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
from recommender.recommendation import generate_candidates
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


@pytest.fixture
def populated_articles(session: Session):
    now = datetime.now(UTC)
    articles = [
        Article(
            id=1,
            source="Bloomberg",
            title="NVIDIA AI Revolution",
            url="https://example.com/1",
            published_at=now - timedelta(hours=1),
            topics={"AI": 0.95, "Semiconductors": 0.90},
        ),
        Article(
            id=2,
            source="Reuters",
            title="Fed Rate Cuts Coming Soon",
            url="https://example.com/2",
            published_at=now - timedelta(hours=2),
            topics={"Interest Rates": 0.90, "Inflation": 0.80},
        ),
        Article(
            id=3,
            source="WSJ",
            title="Oil Prices Spike on Middle East News",
            url="https://example.com/3",
            published_at=now - timedelta(hours=3),
            topics={"Commodities": 0.85, "Forex": 0.40},
        ),
        Article(
            id=4,
            source="FT",
            title="Banking Giants Report Quarterly Incomes",
            url="https://example.com/4",
            published_at=now - timedelta(hours=4),
            topics={"Banking": 0.90, "Finance": 0.80},
        ),
        Article(
            id=5,
            source="CNBC",
            title="Tesla Cybertruck Deliveries Surge",
            url="https://example.com/5",
            published_at=now - timedelta(hours=5),
            topics={"Tesla": 0.90, "Companies": 0.80},
        ),
    ]
    session.add_all(articles)
    session.commit()
    return articles


def test_candidates_match_user_interests(session: Session, populated_articles):
    """Verify candidate generation surfaces articles corresponding to user interests."""
    user = User(id=10, username="tech_investor")
    session.add(user)
    session.commit()

    set_interest(user_id=10, topic="AI", score=0.9, session=session)
    session.commit()

    candidates = generate_candidates(user, session=session, limit=3)
    candidate_ids = [c.id for c in candidates]

    # Article 1 (AI) should definitely be included
    assert 1 in candidate_ids
    assert len(candidates) <= 3


def test_cold_start_user_candidate_generation(session: Session, populated_articles):
    """Verify users with no established interests still receive candidate articles."""
    new_user = User(id=99, username="newbie")
    session.add(new_user)
    session.commit()

    candidates = generate_candidates(new_user, session=session, limit=4)
    assert len(candidates) == 4
    # Unique articles
    assert len(set(c.id for c in candidates)) == 4


def test_candidate_pool_deduplication(session: Session, populated_articles):
    """Verify that candidates retrieved from different sources do not contain duplicate articles."""
    user = User(id=20, username="trader")
    session.add(user)
    session.commit()

    set_interest(user_id=20, topic="Interest Rates", score=0.9, session=session)
    # Add interaction on article 2 to make it popular too
    interaction = Interaction(
        user_id=20,
        event_type=InteractionType.OPEN,
        article_id=2,
    )
    session.add(interaction)
    session.commit()

    candidates = generate_candidates(user, session=session, limit=10)
    candidate_ids = [c.id for c in candidates]

    # Verify all returned candidates are strictly unique
    assert len(candidate_ids) == len(set(candidate_ids))

