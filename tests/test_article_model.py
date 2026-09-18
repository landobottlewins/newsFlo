"""Tests for Article data model and database persistence."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from recommender.models import (
    Article,
    create_db_engine,
    drop_db,
    get_session_factory,
    init_db,
)


@pytest.fixture
def engine():
    """Create an isolated in-memory SQLite database engine."""
    eng = create_db_engine("sqlite:///:memory:")
    init_db(eng)
    yield eng
    drop_db(eng)
    eng.dispose()


@pytest.fixture
def session(engine):
    """Provide a transactional database session for tests."""
    factory = get_session_factory(engine)
    with factory() as sess:
        yield sess


def test_create_article(session):
    """1. Test creating an article and persisting to database."""
    now = datetime.now(UTC)
    article = Article(
        source="Bloomberg",
        source_article_id="BBG-12345",
        title="Fed Signals Rate Decision Ahead",
        description="Federal Reserve policymakers gathered to discuss monetary policy.",
        url="https://example.com/bloomberg/fed-signals-rate-decision",
        published_at=now,
        raw_text="Full article text about the Federal Reserve decision.",
        language="en",
    )
    session.add(article)
    session.commit()

    assert article.id is not None
    assert article.id > 0
    assert article.title == "Fed Signals Rate Decision Ahead"
    assert article.source == "Bloomberg"
    assert article.language == "en"
    assert article.created_at is not None
    assert article.updated_at is not None
    assert article.retrieved_at is not None
    expected_repr = (
        f"<Article(id={article.id}, source='Bloomberg', title='Fed Signals Rate Decision Ahead')>"
    )
    assert repr(article) == expected_repr


def test_retrieve_article(session):
    """2. Test retrieving an article by id, URL, and source identifiers."""
    now = datetime.now(UTC)
    article = Article(
        source="Reuters",
        source_article_id="REUT-999",
        title="Tech Rally Boosts Markets",
        description="Semiconductor shares led global market gains.",
        url="https://example.com/reuters/tech-rally",
        published_at=now,
    )
    session.add(article)
    session.commit()

    # Query by ID
    by_id = session.get(Article, article.id)
    assert by_id is not None
    assert by_id.title == "Tech Rally Boosts Markets"

    # Query by URL
    stmt_url = select(Article).where(Article.url == "https://example.com/reuters/tech-rally")
    by_url = session.scalars(stmt_url).one_or_none()
    assert by_url is not None
    assert by_url.id == article.id

    # Query by source and source_article_id
    stmt_source = select(Article).where(
        Article.source == "Reuters",
        Article.source_article_id == "REUT-999",
    )
    by_source = session.scalars(stmt_source).one_or_none()
    assert by_source is not None
    assert by_source.id == article.id


def test_duplicate_url_handling(session):
    """3. Test duplicate URL enforcement raises IntegrityError."""
    now = datetime.now(UTC)
    article1 = Article(
        source="FT",
        title="First Article",
        url="https://example.com/shared-url",
        published_at=now,
    )
    article2 = Article(
        source="WSJ",
        title="Second Article with same URL",
        url="https://example.com/shared-url",
        published_at=now,
    )

    session.add(article1)
    session.commit()

    session.add(article2)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_duplicate_source_and_article_id_handling(session):
    """Test uniqueness constraint on (source, source_article_id)."""
    now = datetime.now(UTC)
    article1 = Article(
        source="CNBC",
        source_article_id="CNBC-100",
        title="Market Open",
        url="https://example.com/cnbc/open",
        published_at=now,
    )
    article2 = Article(
        source="CNBC",
        source_article_id="CNBC-100",
        title="Market Open Update",
        url="https://example.com/cnbc/open-update",
        published_at=now,
    )

    session.add(article1)
    session.commit()

    session.add(article2)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    # Verify that multiple articles with NULL source_article_id from the same source are permitted
    article3 = Article(
        source="CNBC",
        source_article_id=None,
        title="CNBC Brief 1",
        url="https://example.com/cnbc/brief-1",
        published_at=now,
    )
    article4 = Article(
        source="CNBC",
        source_article_id=None,
        title="CNBC Brief 2",
        url="https://example.com/cnbc/brief-2",
        published_at=now,
    )
    session.add(article3)
    session.add(article4)
    session.commit()
    assert article3.id is not None
    assert article4.id is not None


def test_required_fields(session):
    """4. Test missing required fields violate not-null constraints."""
    now = datetime.now(UTC)

    # Missing title
    with pytest.raises(IntegrityError):
        session.add(
            Article(source="Src", url="https://example.com/1", published_at=now, title=None)
        )
        session.commit()
    session.rollback()

    # Missing source
    with pytest.raises(IntegrityError):
        session.add(
            Article(title="Title", url="https://example.com/2", published_at=now, source=None)
        )
        session.commit()
    session.rollback()

    # Missing url
    with pytest.raises(IntegrityError):
        session.add(Article(source="Src", title="Title", published_at=now, url=None))
        session.commit()
    session.rollback()

    # Missing published_at
    with pytest.raises(IntegrityError):
        session.add(
            Article(source="Src", title="Title", url="https://example.com/4", published_at=None)
        )
        session.commit()
    session.rollback()


def test_publication_timestamps(session):
    """5. Test publication timestamps ordering and timezone persistence."""
    t1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 1, 2, 12, 0, 0, tzinfo=UTC)
    t3 = datetime(2026, 1, 3, 12, 0, 0, tzinfo=UTC)

    a1 = Article(source="S", title="Article 1", url="https://example.com/a1", published_at=t2)
    a2 = Article(source="S", title="Article 2", url="https://example.com/a2", published_at=t1)
    a3 = Article(source="S", title="Article 3", url="https://example.com/a3", published_at=t3)

    session.add_all([a1, a2, a3])
    session.commit()

    # Query ordered by published_at ascending
    stmt = select(Article).order_by(Article.published_at.asc())
    ordered = session.scalars(stmt).all()

    assert [a.title for a in ordered] == ["Article 2", "Article 1", "Article 3"]
    assert ordered[0].published_at == t1
    assert ordered[1].published_at == t2
    assert ordered[2].published_at == t3


def test_reproducible_schema(engine):
    """Verify reproducible schema initialization and teardown."""
    # Drop and re-initialize cleanly
    drop_db(engine)
    init_db(engine)

    factory = get_session_factory(engine)
    with factory() as sess:
        art = Article(
            source="Test",
            title="Post Re-init",
            url="https://example.com/reinit",
            published_at=datetime.now(UTC),
        )
        sess.add(art)
        sess.commit()
        assert art.id is not None
