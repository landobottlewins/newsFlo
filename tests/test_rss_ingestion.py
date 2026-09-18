"""Tests for RSS news feed ingestion pipeline."""

import urllib.error
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from recommender.config import Settings
from recommender.ingestion import (
    FeedFetchError,
    fetch_feed,
    ingest_all_sources,
    ingest_feed,
    normalize_entry,
)
from recommender.models import (
    Article,
    create_db_engine,
    drop_db,
    get_session_factory,
    init_db,
)

SAMPLE_VALID_FEED = """<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
<channel>
  <title>Financial Market News</title>
  <link>https://example.com/finance</link>
  <description>Latest financial news</description>
  <language>en</language>
  <item>
    <title>Global Stocks Rise on Central Bank Optimism</title>
    <link>https://example.com/news/101</link>
    <description>Markets responded positively to policy statements.</description>
    <guid>ITEM-101</guid>
    <pubDate>Fri, 18 Sep 2026 10:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Oil Prices Stabilize After Supply Report</title>
    <link>https://example.com/news/102</link>
    <description>Crude futures steady following inventory figures.</description>
    <guid>ITEM-102</guid>
    <pubDate>Fri, 18 Sep 2026 11:30:00 GMT</pubDate>
  </item>
</channel>
</rss>
"""

SAMPLE_EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
<channel>
  <title>Empty Feed</title>
  <link>https://example.com/empty</link>
  <description>No items currently available</description>
</channel>
</rss>
"""

SAMPLE_MALFORMED_ENTRY_FEED = """<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
<channel>
  <title>Malformed Items</title>
  <link>https://example.com/malformed</link>
  <item>
    <!-- Missing title and valid link -->
    <description>Just a floating description without link or title</description>
  </item>
  <item>
    <title>Valid Item Following Malformed</title>
    <link>https://example.com/news/valid-201</link>
    <description>This item should be processed normally.</description>
    <guid>ITEM-201</guid>
  </item>
</channel>
</rss>
"""

SAMPLE_FEED_WITH_DUPLICATES = """<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
<channel>
  <title>Duplicate Feed</title>
  <link>https://example.com/dups</link>
  <item>
    <title>Breaking Headline</title>
    <link>https://example.com/news/duplicate-url</link>
    <guid>DUP-1</guid>
  </item>
  <item>
    <title>Breaking Headline Duplicate Entry</title>
    <link>https://example.com/news/duplicate-url</link>
    <guid>DUP-2</guid>
  </item>
</channel>
</rss>
"""

SAMPLE_MINIMAL_FEED = """<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
<channel>
  <title>Minimal Feed</title>
  <item>
    <title>Headline with Minimal Fields</title>
    <link>https://example.com/news/minimal-301</link>
  </item>
</channel>
</rss>
"""


@pytest.fixture
def engine():
    eng = create_db_engine("sqlite:///:memory:")
    init_db(eng)
    yield eng
    drop_db(eng)
    eng.dispose()


@pytest.fixture
def session(engine):
    factory = get_session_factory(engine)
    with factory() as sess:
        yield sess


def test_source_configuration():
    """Verify NEWS_SOURCES parses correctly into structured feed sources."""
    settings = Settings(
        news_sources="Reuters,https://reuters.com/rss; Bloomberg, https://bloomberg.com/feed "
    )
    sources = settings.get_feed_sources()
    assert len(sources) == 2
    assert sources[0].name == "Reuters"
    assert sources[0].url == "https://reuters.com/rss"
    assert sources[1].name == "Bloomberg"
    assert sources[1].url == "https://bloomberg.com/feed"


def test_fetch_feed_with_xml_string():
    """Verify fetch_feed parses direct XML string."""
    feed = fetch_feed(SAMPLE_VALID_FEED)
    assert len(feed.entries) == 2
    assert feed.entries[0].title == "Global Stocks Rise on Central Bank Optimism"


def test_fetch_feed_network_error():
    """Verify fetch_feed raises FeedFetchError on network failure."""
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        with pytest.raises(FeedFetchError) as exc_info:
            fetch_feed("https://unreachable.example.com/rss")
        assert "Network error fetching" in str(exc_info.value)


def test_ingest_valid_feed(session):
    """1. Valid feed: produces persisted Article records with correct fields."""
    articles = ingest_feed("MarketWatch", SAMPLE_VALID_FEED, session=session)
    session.commit()

    assert len(articles) == 2
    assert articles[0].source == "MarketWatch"
    assert articles[0].source_article_id == "ITEM-101"
    assert articles[0].title == "Global Stocks Rise on Central Bank Optimism"
    assert articles[0].url == "https://example.com/news/101"
    assert articles[0].published_at == datetime(2026, 9, 18, 10, 0, tzinfo=UTC)

    # Verify database persistence
    db_count = session.query(Article).count()
    assert db_count == 2


def test_ingest_empty_feed(session):
    """2. Empty feed: returns empty list without error."""
    articles = ingest_feed("EmptySource", SAMPLE_EMPTY_FEED, session=session)
    session.commit()

    assert articles == []
    assert session.query(Article).count() == 0


def test_ingest_malformed_entry(session):
    """3. Malformed entry: skips invalid entry and ingests valid entry."""
    articles = ingest_feed("MalformedSource", SAMPLE_MALFORMED_ENTRY_FEED, session=session)
    session.commit()

    assert len(articles) == 1
    assert articles[0].title == "Valid Item Following Malformed"
    assert articles[0].url == "https://example.com/news/valid-201"
    assert session.query(Article).count() == 1


def test_duplicate_article_handling(session):
    """4. Duplicate article: repeated ingestion does not create duplicates."""
    # First ingestion
    first_run = ingest_feed("SourceA", SAMPLE_VALID_FEED, session=session)
    session.commit()
    assert len(first_run) == 2
    assert session.query(Article).count() == 2

    # Second ingestion of the same feed (repeated ingestion)
    second_run = ingest_feed("SourceA", SAMPLE_VALID_FEED, session=session)
    session.commit()
    assert len(second_run) == 0
    assert session.query(Article).count() == 2

    # Intra-feed duplicate handling
    dup_run = ingest_feed("SourceB", SAMPLE_FEED_WITH_DUPLICATES, session=session)
    session.commit()
    assert len(dup_run) == 1
    assert dup_run[0].url == "https://example.com/news/duplicate-url"


def test_unavailable_feed_does_not_terminate(session, caplog):
    """5. Unavailable feed: logs source, error, timestamp and returns empty."""
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(
            "https://down.example.com/rss", 404, "Not Found", {}, None
        ),
    ):
        articles = ingest_feed("BrokenFeed", "https://down.example.com/rss", session=session)

    assert articles == []
    # Check that error was logged with source and error
    assert any("Failed to ingest feed" in r.message for r in caplog.records)
    assert any("BrokenFeed" in r.message for r in caplog.records)


def test_missing_optional_fields(session):
    """6. Missing optional fields: successfully creates Article with defaults."""
    articles = ingest_feed("MinimalSource", SAMPLE_MINIMAL_FEED, session=session)
    session.commit()

    assert len(articles) == 1
    art = articles[0]
    assert art.title == "Headline with Minimal Fields"
    assert art.url == "https://example.com/news/minimal-301"
    assert art.description is None
    assert art.source_article_id is None
    assert art.language == "en"
    assert art.published_at is not None
    assert art.retrieved_at is not None


def test_one_broken_source_does_not_stop_others(session):
    """Verify multi-source ingestion continues when one source fails."""
    sources = [
        ("BrokenSource", "https://broken.example.com/rss"),
        ("WorkingSourceA", SAMPLE_VALID_FEED),
        ("WorkingSourceB", SAMPLE_MINIMAL_FEED),
    ]

    with patch(
        "recommender.ingestion.rss.fetch_feed",
        side_effect=[
            FeedFetchError("Network unreachable"),
            fetch_feed(SAMPLE_VALID_FEED),
            fetch_feed(SAMPLE_MINIMAL_FEED),
        ],
    ):
        results = ingest_all_sources(sources=sources, session=session)
        session.commit()

    assert len(results["BrokenSource"]) == 0
    assert len(results["WorkingSourceA"]) == 2
    assert len(results["WorkingSourceB"]) == 1
    assert session.query(Article).count() == 3


def test_normalize_entry_validation():
    """Verify normalize_entry returns None when required fields are missing."""
    # Missing link
    assert normalize_entry({"title": "Only Title"}, "Source") is None
    # Missing title
    assert normalize_entry({"link": "https://example.com"}, "Source") is None
    # Empty title and link
    assert normalize_entry({"title": "", "link": ""}, "Source") is None


def test_ingest_cli_script(engine, monkeypatch):
    """Verify running the ingestion script against test feeds produces Article records."""
    import scripts.ingest

    test_settings = Settings(
        database_url="sqlite:///:memory:",
        news_sources="TestFeed,https://example.com/rss",
    )
    monkeypatch.setattr("recommender.config.get_settings", lambda: test_settings)
    monkeypatch.setattr("scripts.ingest.get_settings", lambda: test_settings)
    monkeypatch.setattr(
        "sys.argv",
        ["ingest.py", "--init-db", "--source", f"CliSource,{SAMPLE_VALID_FEED}"],
    )

    with patch("scripts.ingest.create_db_engine", return_value=engine):
        exit_code = scripts.ingest.main()
        assert exit_code == 0

    factory = get_session_factory(engine)
    with factory() as sess:
        count = sess.query(Article).filter(Article.source == "CliSource").count()
        assert count == 2
