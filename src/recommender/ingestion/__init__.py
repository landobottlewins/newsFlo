"""News ingestion and feed collection modules."""

from recommender.ingestion.rss import (
    FeedFetchError,
    fetch_feed,
    ingest_all_sources,
    ingest_feed,
    normalize_entry,
)

__all__ = [
    "FeedFetchError",
    "fetch_feed",
    "ingest_all_sources",
    "ingest_feed",
    "normalize_entry",
]
