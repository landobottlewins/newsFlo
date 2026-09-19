"""Tests for Task 014 session-level personalization."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from recommender.models import (
    Article,
    InteractionType,
    create_db_engine,
    drop_db,
    get_session_factory,
    init_db,
)
from recommender.users import (
    SessionPersonalizationConfig,
    combine_interests,
    create_session,
    get_combined_interests,
    record_interaction,
    reset_session,
    set_interest,
    update_session_from_interaction,
)


@pytest.fixture
def session():
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    factory = get_session_factory(engine)
    with factory() as db_session:
        yield db_session
    drop_db(engine)
    engine.dispose()


def test_new_session_starts_with_no_transient_interests():
    profile = create_session(1, session_id="morning")

    assert profile.user_id == 1
    assert profile.session_id == "morning"
    assert profile.interests == {}


def test_repeated_topic_interactions_build_session_interest_more_when_recent():
    profile = create_session(1)
    first = datetime(2026, 9, 19, 9, 0, tzinfo=UTC)
    profile.record_topics({"Oil": 1.0}, signal_weight=4, occurred_at=first)
    profile.record_topics({"Oil": 1.0}, signal_weight=4, occurred_at=first + timedelta(minutes=1))

    # Two recent READ_2M-like signals compound beyond one signal (0.20).
    assert profile.interests["Oil"] > 0.20

    stale = create_session(1)
    stale.record_topics({"Oil": 1.0}, signal_weight=4, occurred_at=first)
    stale.record_topics({"Oil": 1.0}, signal_weight=4, occurred_at=first + timedelta(hours=2))
    assert stale.interests["Oil"] < profile.interests["Oil"]


def test_session_reset_does_not_change_persisted_long_term_interests(session: Session):
    set_interest(1, "AI", 0.8, session)
    session.commit()
    profile = create_session(1)
    profile.record_topics({"Oil": 1.0}, signal_weight=6)

    reset_session(profile)

    assert profile.interests == {}
    assert get_combined_interests(1, profile, session) == {"AI": 0.56}


def test_interaction_updates_session_but_not_long_term_profile(session: Session):
    article = Article(
        source="Reuters",
        title="Oil prices climb",
        url="https://example.com/oil",
        published_at=datetime.now(UTC),
        topics={"Oil": 0.9, "Energy": 0.8},
    )
    session.add(article)
    session.commit()
    profile = create_session(3)
    interaction = record_interaction(3, InteractionType.DEEP_DIVE, article.id, session=session)

    updated = update_session_from_interaction(profile, interaction, session)

    assert updated["Oil"] > 0
    # Task 014 keeps the transient session model separate from UserInterest.
    assert get_combined_interests(3, profile, session)["Oil"] == pytest.approx(updated["Oil"] * 0.3)


def test_combined_score_uses_default_and_configurable_weights():
    assert combine_interests({"AI": 0.8}, {"AI": 0.2, "Oil": 0.9}) == {
        "AI": 0.62,
        "Oil": 0.27,
    }

    config = SessionPersonalizationConfig(long_term_weight=0.4, session_weight=0.6)
    assert combine_interests({"AI": 0.8}, {"AI": 0.2}, config=config) == {"AI": 0.44}
