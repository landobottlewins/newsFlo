"""RSS news feed fetching, normalization, and ingestion pipeline."""

import logging
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any

import feedparser
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from recommender.config import FeedSource, get_settings
from recommender.models import Article, create_db_engine, get_session_factory

logger = logging.getLogger(__name__)


class FeedFetchError(Exception):
    """Raised when an RSS feed cannot be fetched or parsed."""

    pass


def fetch_feed(url: str, timeout: float = 10.0) -> feedparser.FeedParserDict:
    """Fetch and parse an RSS/Atom feed from a URL or raw content string.

    Args:
        url: HTTP/HTTPS URL or XML content string.
        timeout: Request timeout in seconds.

    Returns:
        Parsed feed dictionary.

    Raises:
        FeedFetchError: If the feed cannot be retrieved or network fails.
    """
    if url.startswith(("http://", "https://")):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "NewsFlo-Recommender/0.1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as err:
            raise FeedFetchError(f"Network error fetching {url}: {err}") from err
    else:
        content = url

    try:
        feed = feedparser.parse(content)
    except Exception as err:
        raise FeedFetchError(f"Error parsing feed content: {err}") from err

    return feed


def _parse_entry_published_at(entry: Any) -> datetime:
    """Extract and normalize publication timestamp to timezone-aware UTC datetime."""
    parsed_time = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed_time:
        try:
            return datetime(*parsed_time[:6], tzinfo=UTC)
        except Exception:
            pass

    return datetime.now(UTC)


def normalize_entry(
    entry: Any, source_name: str, default_language: str = "en"
) -> dict[str, Any] | None:
    """Normalize a raw RSS entry into fields matching the Article model.

    Returns None if essential required fields (title, url) are missing.
    """
    title = (entry.get("title") or "").strip()
    url = (entry.get("link") or "").strip()

    if not title or not url:
        return None

    source_article_id = entry.get("id") or entry.get("guid") or None
    description = entry.get("summary") or entry.get("description") or None

    raw_text = None
    if "content" in entry and isinstance(entry["content"], list) and entry["content"]:
        raw_text = entry["content"][0].get("value")
    elif description:
        raw_text = description

    language = entry.get("language") or default_language
    published_at = _parse_entry_published_at(entry)
    retrieved_at = datetime.now(UTC)

    return {
        "source": source_name,
        "source_article_id": str(source_article_id) if source_article_id else None,
        "title": title,
        "description": description,
        "url": url,
        "published_at": published_at,
        "retrieved_at": retrieved_at,
        "raw_text": raw_text,
        "language": language,
    }


def ingest_feed(
    source_name: str,
    url: str,
    session: Session | None = None,
    timeout: float = 10.0,
) -> list[Article]:
    """Ingest articles from a single RSS feed into the database.

    Avoids duplicate articles based on URL and (source, source_article_id).
    Errors are logged without terminating the process.

    Args:
        source_name: Name identifier for the feed source.
        url: Feed URL or raw XML feed string.
        session: Optional existing SQLAlchemy session. If None, one is created.
        timeout: Feed fetch timeout.

    Returns:
        List of newly created and persisted Article objects.
    """
    try:
        feed = fetch_feed(url, timeout=timeout)
    except Exception as exc:
        logger.error(
            "Failed to ingest feed [source=%s] [error=%s] [timestamp=%s]",
            source_name,
            str(exc),
            datetime.now(UTC).isoformat(),
        )
        return []

    if session is not None:
        return _process_entries(source_name, feed, session)

    engine = create_db_engine()
    factory = get_session_factory(engine)
    with factory() as new_session:
        articles = _process_entries(source_name, feed, new_session)
        new_session.commit()
        return articles


def _process_entries(
    source_name: str,
    feed: feedparser.FeedParserDict,
    session: Session,
) -> list[Article]:
    """Process feed entries, filter duplicates, and persist new Article instances."""
    new_articles: list[Article] = []
    seen_urls: set[str] = set()
    seen_source_ids: set[str] = set()

    feed_lang = feed.feed.get("language", "en") if getattr(feed, "feed", None) else "en"

    for entry in feed.entries:
        normalized = normalize_entry(entry, source_name=source_name, default_language=feed_lang)
        if not normalized:
            logger.warning(
                "Skipping malformed entry in feed [source=%s] [timestamp=%s]",
                source_name,
                datetime.now(UTC).isoformat(),
            )
            continue

        entry_url = normalized["url"]
        entry_source_id = normalized["source_article_id"]

        # Check intra-batch duplicates
        if entry_url in seen_urls:
            continue
        if entry_source_id and entry_source_id in seen_source_ids:
            continue

        # Check database for existing duplicates
        conditions = [Article.url == entry_url]
        if entry_source_id:
            conditions.append(
                (Article.source == source_name) & (Article.source_article_id == entry_source_id)
            )

        existing = session.scalars(select(Article).where(or_(*conditions))).first()
        if existing is not None:
            continue

        # Mark as seen in this run
        seen_urls.add(entry_url)
        if entry_source_id:
            seen_source_ids.add(entry_source_id)

        article = Article(**normalized)
        session.add(article)
        new_articles.append(article)

    session.flush()
    logger.info(
        "Successfully ingested %d new articles [source=%s] [timestamp=%s]",
        len(new_articles),
        source_name,
        datetime.now(UTC).isoformat(),
    )
    return new_articles


def ingest_all_sources(
    sources: list[tuple[str, str] | FeedSource] | None = None,
    session: Session | None = None,
) -> dict[str, list[Article]]:
    """Ingest articles from all configured news sources.

    Failure in one feed does not interrupt the ingestion of other feeds.
    """
    if sources is None:
        configured = get_settings().get_feed_sources()
        feed_list = [(s.name, s.url) for s in configured]
    else:
        feed_list = [(s.name, s.url) if isinstance(s, FeedSource) else s for s in sources]

    results: dict[str, list[Article]] = {}
    for name, url in feed_list:
        try:
            articles = ingest_feed(source_name=name, url=url, session=session)
            results[name] = articles
        except Exception as exc:
            logger.error(
                "Failed to ingest source [source=%s] [error=%s] [timestamp=%s]",
                name,
                str(exc),
                datetime.now(UTC).isoformat(),
            )
            results[name] = []

    return results
