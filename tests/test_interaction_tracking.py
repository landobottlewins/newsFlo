"""Tests for User Interaction Tracking (Task 009)."""

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
from recommender.users import get_user_interactions, record_interaction


@pytest.fixture
def session():
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    factory = get_session_factory(engine)
    with factory() as sess:
        yield sess
    drop_db(engine)
    engine.dispose()


def test_interaction_creation(session: Session):
    """Verify recording a basic interaction event."""
    interaction = record_interaction(
        user_id=1,
        event_type=InteractionType.OPEN,
        article_id=10,
        metadata={"source": "feed_recs"},
        session=session,
    )
    session.commit()

    assert interaction.id is not None
    assert interaction.user_id == 1
    assert interaction.event_type == "OPEN"
    assert interaction.article_id == 10
    assert interaction.event_metadata == {"source": "feed_recs"}
    assert interaction.timestamp is not None


def test_invalid_event_type_rejected(session: Session):
    """Verify invalid event types are strictly rejected."""
    with pytest.raises(ValueError) as exc_info:
        record_interaction(user_id=1, event_type="UNKNOWN_CLICK", session=session)
    assert "Invalid event_type" in str(exc_info.value)

    with pytest.raises(ValueError):
        Interaction(user_id=1, event_type="BOGUS")


def test_realistic_interaction_sequence(session: Session):
    """Verify recording a realistic chronological sequence of user actions."""
    base_time = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)

    # User views impression, opens article, reads for 30s, then bookmarks
    seq = [
        (InteractionType.IMPRESSION, 101, base_time),
        (InteractionType.OPEN, 101, base_time + timedelta(seconds=2)),
        (InteractionType.READ_30S, 101, base_time + timedelta(seconds=32)),
        (InteractionType.READ_2M, 101, base_time + timedelta(seconds=122)),
        (InteractionType.BOOKMARK, 101, base_time + timedelta(seconds=130)),
        (InteractionType.SKIP, 102, base_time + timedelta(seconds=140)),
        (InteractionType.FOLLOW_TOPIC, None, base_time + timedelta(seconds=150)),
    ]

    for event, art_id, ts in seq:
        meta = {"topic": "AI"} if event == InteractionType.FOLLOW_TOPIC else None
        record_interaction(
            user_id=5,
            event_type=event,
            article_id=art_id,
            timestamp=ts,
            metadata=meta,
            session=session,
        )
    session.commit()

    history = get_user_interactions(user_id=5, session=session)
    assert len(history) == len(seq)
    assert [h.event_type for h in history] == [s[0].value for s in seq]


def test_event_storage_separated_from_user_interests(session: Session):
    """Verify recording interactions does not automatically alter user interests directly."""
    user = User(id=7, username="passive_user")
    art = Article(
        id=55,
        source="Reuters",
        title="AI Boom Continues",
        url="https://example.com/ai-boom",
        published_at=datetime.now(UTC),
        topics={"AI": 0.9},
    )
    session.add_all([user, art])
    session.commit()

    # Record strong positive interaction
    record_interaction(
        user_id=7,
        event_type=InteractionType.DEEP_DIVE,
        article_id=55,
        session=session,
    )
    session.commit()

    # Model separation: interests must remain untouched until updater is explicitly run
    user_interests = user.interests
    assert len(user_interests) == 0
