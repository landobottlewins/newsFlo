"""Tests for financial topic classification and taxonomy."""

from datetime import UTC, datetime

import pytest

from recommender.models import (
    Article,
    create_db_engine,
    drop_db,
    get_session_factory,
    init_db,
)
from recommender.processing import (
    TopicClassifier,
    classify_topics,
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


def test_single_topic_classification():
    """Verify an article focusing on semiconductors classifies correctly."""
    art = {
        "title": "TSMC ramps up advanced 2nm chip foundry production",
        "description": "The semiconductor wafer maker is seeing surge in silicon orders.",
    }
    topics = classify_topics(art)
    assert "Semiconductors" in topics
    assert "Technology" in topics  # Inherited category
    assert topics["Semiconductors"] >= 0.5


def test_multi_topic_classification():
    """Verify an article spanning multiple domains assigns multiple topic scores."""
    art = {
        "title": "Fed rate hike hits Wall Street stocks and corporate bond yields",
        "description": "Central bank monetary policy tightens as inflation remains high.",
    }
    topics = classify_topics(art)
    assert "Interest Rates" in topics
    assert "Stocks" in topics
    assert "Bonds" in topics
    assert "Inflation" in topics
    assert "Economy" in topics
    assert "Markets" in topics


def test_scores_bounded():
    """Verify all scores produced by the classifier stay strictly in [0.0, 1.0]."""
    art = {
        "title": "NVIDIA reveals generative AI chips for cloud data center servers",
        "description": "Jensen Huang announced new GPUs beating street expectations.",
    }
    topics = classify_topics(art)
    assert len(topics) > 0
    for _topic, score in topics.items():
        assert 0.0 <= score <= 1.0


def test_empty_and_unmatched_article():
    """Verify empty or non-financial content returns empty topic mapping."""
    empty_art = {"title": "", "description": ""}
    assert classify_topics(empty_art) == {}

    unrelated_art = {
        "title": "Local zoo welcomes baby panda cub",
        "description": "Visitors gathered to see the newborn cub play in the grass.",
    }
    assert classify_topics(unrelated_art) == {}


def test_custom_classifier_interface():
    """Verify custom classifier implementations can be plugged in."""

    class MockClassifier(TopicClassifier):
        def classify(self, text: str, title: str | None = None) -> dict[str, float]:
            return {"Forex": 0.88, "World": 0.75}

    art = {"title": "Generic title"}
    topics = classify_topics(art, classifier=MockClassifier())
    assert topics == {"Forex": 0.88, "World": 0.75}


def test_orm_article_topics_persistence(session):
    """Verify Article.topics JSON field persists and retrieves correctly."""
    article = Article(
        source="Bloomberg",
        title="Fed Signals Interest Rate Cut Ahead",
        description="Federal Reserve policymakers prepare for economic easing.",
        url="https://bloomberg.com/fed-cut-2026",
        published_at=datetime.now(UTC),
    )
    topics = classify_topics(article)
    session.add(article)
    session.commit()

    retrieved = session.get(Article, article.id)
    assert retrieved is not None
    assert retrieved.topics is not None
    assert retrieved.topics["Interest Rates"] == topics["Interest Rates"]
    assert "Economy" in retrieved.topics
