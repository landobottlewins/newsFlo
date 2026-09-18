"""Tests for news article deduplication and clustering pipeline."""

from datetime import UTC, datetime

from recommender.models import Article
from recommender.processing import (
    are_duplicates,
    clean_title_for_comparison,
    compute_title_similarity,
    group_duplicate_articles,
)


def test_identical_urls():
    """1. Test identical URLs (including tracking parameter differences)."""
    art1 = Article(
        source="SourceA",
        title="Stock Market Opens Higher",
        url="https://example.com/markets/stocks-open?utm_source=twitter",
        published_at=datetime.now(UTC),
    )
    art2 = Article(
        source="SourceB",
        title="Different Title Heading Here",
        url="https://example.com/markets/stocks-open?utm_medium=email#chart",
        published_at=datetime.now(UTC),
    )
    assert are_duplicates(art1, art2) is True


def test_identical_titles():
    """2. Test identical titles across different URLs and sources."""
    art1 = {
        "source": "Reuters",
        "title": "ECB Signals Further Rate Cuts Ahead",
        "url": "https://reuters.com/ecb-1",
    }
    art2 = {
        "source": "Bloomberg",
        "title": "  ecb signals further rate cuts ahead  ",
        "url": "https://bloomberg.com/ecb-cuts",
    }
    assert are_duplicates(art1, art2) is True


def test_slightly_modified_titles():
    """3. Test slightly modified titles representing the same event."""
    # Singular vs plural and phrasing
    art1 = {
        "source": "FT",
        "title": "Fed raises interest rates by 25 bps",
        "url": "https://ft.com/fed-25bps",
    }
    art2 = {
        "source": "WSJ",
        "title": "Fed raises interest rate by 25 bps",
        "url": "https://wsj.com/fed-rate-25bps",
    }
    assert are_duplicates(art1, art2) is True

    # Synonyms / near phrasing
    art3 = {
        "source": "CNBC",
        "title": "Apple reaches $3 trillion market cap",
        "url": "https://cnbc.com/aapl-3t",
    }
    art4 = {
        "source": "MarketWatch",
        "title": "Apple hits $3 trillion market cap",
        "url": "https://marketwatch.com/apple-3-trillion",
    }
    assert are_duplicates(art3, art4) is True


def test_completely_different_stories():
    """4. Test completely different stories return False."""
    art1 = {
        "source": "Reuters",
        "title": "Tesla opens new factory in Texas",
        "url": "https://reuters.com/tesla-texas",
    }
    art2 = {
        "source": "Bloomberg",
        "title": "Boeing delays 777 deliveries",
        "url": "https://bloomberg.com/boeing-777",
    }
    assert are_duplicates(art1, art2) is False

    # Same template but different company entities
    art3 = {
        "source": "WSJ",
        "title": "Apple reports quarterly earnings",
        "url": "https://wsj.com/aapl-earnings",
    }
    art4 = {
        "source": "CNBC",
        "title": "Amazon reports quarterly earnings",
        "url": "https://cnbc.com/amzn-earnings",
    }
    assert are_duplicates(art3, art4) is False


def test_same_event_from_different_sources():
    """5. Test the task specification example: same underlying event across multiple sources."""
    reuters_art = Article(
        source="Reuters",
        title="Reuters: NVIDIA reports record revenue",
        url="https://reuters.com/nvidia-revenue",
        published_at=datetime.now(UTC),
    )
    cnbc_art = Article(
        source="CNBC",
        title="CNBC: NVIDIA revenue hits record",
        url="https://cnbc.com/nvidia-record-revenue",
        published_at=datetime.now(UTC),
    )
    assert are_duplicates(reuters_art, cnbc_art) is True


def test_source_article_id_duplicate():
    """Test exact match on composite (source, source_article_id)."""
    art1 = {
        "source": "Bloomberg",
        "source_article_id": "BBG-999",
        "title": "Title Version 1",
        "url": "https://bloomberg.com/story-1",
    }
    art2 = {
        "source": "Bloomberg",
        "source_article_id": "BBG-999",
        "title": "Title Version 2 Updated",
        "url": "https://bloomberg.com/story-2",
    }
    assert are_duplicates(art1, art2) is True


def test_clean_title_for_comparison():
    """Test source prefix and suffix stripping."""
    assert (
        clean_title_for_comparison("Reuters: Stock rally continues", "Reuters")
        == "Stock rally continues"
    )
    assert (
        clean_title_for_comparison("Stock rally continues - Bloomberg", "Bloomberg")
        == "Stock rally continues"
    )
    assert clean_title_for_comparison("[CNBC]: Oil rises 2%", "CNBC") == "Oil rises 2%"


def test_compute_title_similarity_bounds():
    """Verify similarity values remain bounded in [0.0, 1.0]."""
    assert compute_title_similarity("", "") == 0.0
    assert compute_title_similarity("Same Title", "Same Title") == 1.0
    score = compute_title_similarity("Oil jumps 5%", "Crude gains 5%")
    assert 0.0 <= score <= 1.0


def test_group_duplicate_articles():
    """Test grouping collections of articles into duplicate clusters while preserving sources."""
    now = datetime.now(UTC)

    # 3 articles about NVIDIA
    nv1 = Article(
        source="Reuters",
        title="NVIDIA reports record revenue",
        url="https://reuters.com/nv-1",
        published_at=now,
    )
    nv2 = Article(
        source="CNBC",
        title="NVIDIA revenue hits record",
        url="https://cnbc.com/nv-2",
        published_at=now,
    )

    # 2 articles about Fed rates
    fed1 = Article(
        source="FT",
        title="Fed raises interest rates by 25 bps",
        url="https://ft.com/fed-1",
        published_at=now,
    )
    fed2 = Article(
        source="WSJ",
        title="Fed raises interest rate by 25 bps",
        url="https://wsj.com/fed-2",
        published_at=now,
    )

    # 2 independent stories
    tesla = Article(
        source="Bloomberg",
        title="Tesla opens new factory in Texas",
        url="https://bloomberg.com/tsla-1",
        published_at=now,
    )
    boeing = Article(
        source="AP",
        title="Boeing delays 777 deliveries",
        url="https://apnews.com/ba-1",
        published_at=now,
    )

    articles = [nv1, fed1, tesla, nv2, boeing, fed2]
    groups = group_duplicate_articles(articles)

    assert len(groups) == 4

    # Group counts
    group_sizes = sorted([len(g) for g in groups], reverse=True)
    assert group_sizes == [2, 2, 1, 1]

    # Find the NVIDIA group
    nv_group = next(g for g in groups if nv1 in g)
    assert nv2 in nv_group
    # Verify original source attribution is preserved
    sources_in_nv = {a.source for a in nv_group}
    assert sources_in_nv == {"Reuters", "CNBC"}

    # Find the Fed group
    fed_group = next(g for g in groups if fed1 in g)
    assert fed2 in fed_group
    sources_in_fed = {a.source for a in fed_group}
    assert sources_in_fed == {"FT", "WSJ"}


def test_group_duplicate_articles_empty_and_single():
    """Test group_duplicate_articles on empty and single item lists."""
    assert group_duplicate_articles([]) == []

    single = {"source": "S", "title": "Headline", "url": "https://example.com/1"}
    groups = group_duplicate_articles([single])
    assert len(groups) == 1
    assert groups[0] == [single]
