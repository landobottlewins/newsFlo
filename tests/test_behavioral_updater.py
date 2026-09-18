"""Tests for Behavioral Interest Updater and Time Decay (Task 010)."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from recommender.models import (
    Article,
    Interaction,
    InteractionType,
    create_db_engine,
    drop_db,
    get_session_factory,
    init_db,
)
from recommender.users import (
    apply_decay,
    get_user_interests,
    record_interaction,
    set_interest,
    update_user_interests_from_interaction,
)


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
def sample_articles(session: Session) -> dict[str, Article]:
    """Create sample articles with established topics."""
    now = datetime.now(UTC)
    ai_art = Article(
        source="Bloomberg",
        title="NVIDIA Unveils New AI Architecture",
        url="https://example.com/ai-1",
        published_at=now,
        topics={"AI": 0.90, "Semiconductors": 0.80, "Technology": 0.85},
    )
    crypto_art = Article(
        source="CoinDesk",
        title="Bitcoin Plunges After Regulatory Action",
        url="https://example.com/crypto-1",
        published_at=now,
        topics={"Crypto": 0.95, "Markets": 0.70},
    )
    bonds_art = Article(
        source="FT",
        title="Treasury Yields Rise to 4.5%",
        url="https://example.com/bonds-1",
        published_at=now,
        topics={"Bonds": 0.90, "Economy": 0.75},
    )
    session.add_all([ai_art, crypto_art, bonds_art])
    session.commit()
    return {"ai": ai_art, "crypto": crypto_art, "bonds": bonds_art}


def test_positive_interaction(session: Session, sample_articles):
    """Verify reading an AI article increases user interest in AI."""
    user_id = 1
    set_interest(user_id=user_id, topic="AI", score=0.50, session=session)
    session.commit()

    interaction = record_interaction(
        user_id=user_id,
        event_type=InteractionType.READ_2M,  # +4 weight
        article_id=sample_articles["ai"].id,
        session=session,
    )

    updated = update_user_interests_from_interaction(interaction, session=session, decay=False)
    session.commit()

    # Initial 0.50 + (+4 * 0.90 * 0.05) = 0.50 + 0.18 = 0.68
    assert updated["AI"] > 0.50
    assert updated["AI"] == pytest.approx(0.68, abs=0.01)


def test_negative_interaction(session: Session, sample_articles):
    """Verify skipping an article decreases interest in its topics."""
    user_id = 2
    set_interest(user_id=user_id, topic="Crypto", score=0.60, session=session)
    session.commit()

    interaction = record_interaction(
        user_id=user_id,
        event_type=InteractionType.SKIP,  # -2 weight
        article_id=sample_articles["crypto"].id,
        session=session,
    )

    updated = update_user_interests_from_interaction(interaction, session=session, decay=False)
    session.commit()

    # Initial 0.60 + (-2 * 0.95 * 0.05) = 0.60 - 0.095 = 0.505
    assert updated["Crypto"] < 0.60
    assert updated["Crypto"] == pytest.approx(0.505, abs=0.01)


def test_repeated_interactions(session: Session, sample_articles):
    """Verify repeated positive interactions accumulate interest."""
    user_id = 3
    set_interest(user_id=user_id, topic="AI", score=0.20, session=session)
    session.commit()

    # 3 consecutive reads
    for _ in range(3):
        interaction = record_interaction(
            user_id=user_id,
            event_type=InteractionType.READ_30S,  # +2 weight
            article_id=sample_articles["ai"].id,
            session=session,
        )
        update_user_interests_from_interaction(interaction, session=session, decay=False)
    session.commit()

    interests = get_user_interests(user_id=user_id, session=session)
    # 0.20 + 3 * (+2 * 0.90 * 0.05) = 0.20 + 0.27 = 0.47
    assert interests["AI"] == pytest.approx(0.47, abs=0.01)


def test_time_decay_isolated():
    """Verify apply_decay reduces score based on elapsed time."""
    # Zero elapsed time -> no decay
    assert apply_decay(0.80, timedelta(days=0)) == 0.80

    # 14 days elapsed (one half-life) -> score halves
    half_life_decay = apply_decay(0.80, timedelta(days=14), half_life_days=14.0)
    assert half_life_decay == pytest.approx(0.40, abs=0.01)

    # 28 days elapsed (two half-lives) -> score drops to quarter
    two_half_lives = apply_decay(0.80, timedelta(days=28), half_life_days=14.0)
    assert two_half_lives == pytest.approx(0.20, abs=0.01)


def test_score_normalization(session: Session, sample_articles):
    """Verify updated scores remain bounded in [0.0, 1.0]."""
    user_id = 4

    # Pushing score above 1.0
    set_interest(user_id=user_id, topic="AI", score=0.98, session=session)
    session.commit()
    interaction_follow = record_interaction(
        user_id=user_id,
        event_type=InteractionType.FOLLOW_TOPIC,
        metadata={"topic": "AI"},
        session=session,
    )
    res_high = update_user_interests_from_interaction(interaction_follow, session=session)
    assert res_high["AI"] == 1.0

    # Pushing score below 0.0
    set_interest(user_id=user_id, topic="Crypto", score=0.05, session=session)
    session.commit()
    interaction_skip = record_interaction(
        user_id=user_id,
        event_type=InteractionType.SKIP,
        article_id=sample_articles["crypto"].id,
        session=session,
    )
    res_low = update_user_interests_from_interaction(interaction_skip, session=session)
    assert res_low["Crypto"] == 0.0


def test_unrelated_topics_unaffected(session: Session, sample_articles):
    """Verify interacting with an AI article does not alter unrelated topics like Bonds."""
    user_id = 5
    set_interest(user_id=user_id, topic="AI", score=0.50, session=session)
    set_interest(user_id=user_id, topic="Bonds", score=0.75, session=session)
    session.commit()

    interaction = record_interaction(
        user_id=user_id,
        event_type=InteractionType.DEEP_DIVE,
        article_id=sample_articles["ai"].id,
        session=session,
    )
    update_user_interests_from_interaction(interaction, session=session, decay=False)
    session.commit()

    interests = get_user_interests(user_id=user_id, session=session)
    assert interests["AI"] > 0.50
    assert interests["Bonds"] == 0.75  # Unrelated topic remains exactly 0.75

