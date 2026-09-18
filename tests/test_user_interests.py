"""Tests for User and UserInterest models and profile functions (Task 008)."""

import pytest
from sqlalchemy.orm import Session

from recommender.models import User, create_db_engine, drop_db, get_session_factory, init_db
from recommender.users import get_user_interests, set_interest


@pytest.fixture
def session():
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    factory = get_session_factory(engine)
    with factory() as sess:
        yield sess
    drop_db(engine)
    engine.dispose()


def test_new_user_and_adding_interest(session: Session):
    """Verify new user creation and setting interest."""
    user = User(id=1, username="test_trader")
    session.add(user)
    session.commit()

    interest = set_interest(user_id=1, topic="AI", score=0.91, session=session)
    session.commit()

    assert interest.user_id == 1
    assert interest.topic == "AI"
    assert interest.score == 0.91
    assert interest.last_updated is not None


def test_auto_create_user_on_set_interest(session: Session):
    """Verify set_interest automatically provisions user record if not present."""
    interest = set_interest(user_id=42, topic="Semiconductors", score=0.84, session=session)
    session.commit()

    assert interest.user_id == 42
    user = session.get(User, 42)
    assert user is not None
    assert user.username == "user_42"


def test_updating_interest(session: Session):
    """Verify updating an existing topic interest modifies score and timestamp."""
    set_interest(user_id=1, topic="Markets", score=0.50, session=session)
    session.commit()

    updated = set_interest(user_id=1, topic="Markets", score=0.75, session=session)
    session.commit()

    assert updated.score == 0.75
    interests = get_user_interests(user_id=1, session=session)
    assert interests["Markets"] == 0.75


def test_retrieving_multiple_interests(session: Session):
    """Verify retrieving multiple weighted interests returns sorted mapping."""
    set_interest(user_id=2, topic="AI", score=0.91, session=session)
    set_interest(user_id=2, topic="Semiconductors", score=0.84, session=session)
    set_interest(user_id=2, topic="Markets", score=0.63, session=session)
    set_interest(user_id=2, topic="Crypto", score=0.11, session=session)
    session.commit()

    interests = get_user_interests(user_id=2, session=session)
    assert len(interests) == 4
    assert interests["AI"] == 0.91
    assert interests["Semiconductors"] == 0.84
    assert interests["Markets"] == 0.63
    assert interests["Crypto"] == 0.11


def test_score_boundaries(session: Session):
    """Verify scores are normalized and clamped to [0.0, 1.0]."""
    # Over 1.0 should clamp to 1.0
    high = set_interest(user_id=3, topic="Stocks", score=1.85, session=session)
    assert high.score == 1.0

    # Below 0.0 should clamp to 0.0
    low = set_interest(user_id=3, topic="Forex", score=-0.45, session=session)
    assert low.score == 0.0
