"""Tests for article embeddings, provider abstraction, and vector similarity."""

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
    DEFAULT_EMBEDDING_DIM,
    DeterministicHashEmbeddingProvider,
    EmbeddingError,
    EmbeddingProvider,
    cosine_similarity,
    embed_article,
)


class MockEmbeddingProvider(EmbeddingProvider):
    """Configurable mock provider for testing."""

    def __init__(self, dimension: int = 4, should_fail: bool = False):
        self._dim = dimension
        self.should_fail = should_fail

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        if self.should_fail:
            raise EmbeddingError("Mock embedding provider network failure")
        return [0.5] * self._dim


@pytest.fixture
def session():
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    factory = get_session_factory(engine)
    with factory() as sess:
        yield sess
    drop_db(engine)
    engine.dispose()


def test_successful_embedding():
    """Verify embedding generation on an article returns a valid vector."""
    art = {
        "title": "NVIDIA expands AI infrastructure",
        "description": "Massive GPU clusters deployed for large scale training.",
    }
    vector = embed_article(art)
    assert vector is not None
    assert len(vector) == DEFAULT_EMBEDDING_DIM
    assert art["embedding"] == vector


def test_failed_embedding():
    """Verify provider failures raise EmbeddingError."""
    failing_provider = MockEmbeddingProvider(should_fail=True)
    art = {"title": "Valid Headline", "description": "Some article content"}

    with pytest.raises(EmbeddingError) as exc_info:
        embed_article(art, provider=failing_provider)
    assert "Mock embedding provider network failure" in str(exc_info.value)


def test_empty_article():
    """Verify empty article returns None without calling provider."""
    mock = MockEmbeddingProvider()
    assert embed_article(None, provider=mock) is None
    assert embed_article({}, provider=mock) is None
    assert embed_article({"title": "", "description": ""}, provider=mock) is None


def test_correct_vector_dimensions():
    """Verify vector matches configured provider dimensions."""
    provider_128 = DeterministicHashEmbeddingProvider(dimension=128)
    art = {"title": "Global Market Update"}
    vec = embed_article(art, provider=provider_128)
    assert vec is not None
    assert len(vec) == 128


def test_storage_and_retrieval(session):
    """Verify storing and retrieving vectors on Article model."""
    now = datetime.now(UTC)
    art = Article(
        source="Reuters",
        title="Fed Interest Rate Decision",
        url="https://reuters.com/fed-decision-2026",
        published_at=now,
    )
    vec = embed_article(art)
    assert vec is not None
    session.add(art)
    session.commit()

    retrieved = session.get(Article, art.id)
    assert retrieved is not None
    assert retrieved.embedding is not None
    assert len(retrieved.embedding) == DEFAULT_EMBEDDING_DIM
    assert retrieved.embedding[0] == pytest.approx(vec[0], abs=1e-5)


def test_acceptance_criteria_cosine_similarity():
    """Acceptance criteria: two articles converted to vectors and compared via cosine similarity."""
    art1 = {
        "title": "NVIDIA expands AI infrastructure",
        "description": "Chipmaker scales advanced AI hardware for data centers.",
    }
    art2 = {
        "title": "TSMC expects huge demand for advanced AI chips",
        "description": "Semiconductor foundry forecasts growth driven by AI acceleration.",
    }
    unrelated = {
        "title": "Severe winter blizzard causes airport delays across Midwest",
        "description": "Flight cancellations mount as heavy snowfall blankets region.",
    }

    v1 = embed_article(art1)
    v2 = embed_article(art2)
    v_unrelated = embed_article(unrelated)

    assert v1 is not None and v2 is not None and v_unrelated is not None

    related_sim = cosine_similarity(v1, v2)
    unrelated_sim = cosine_similarity(v1, v_unrelated)

    # Both are normalized vectors
    assert -1.0 <= related_sim <= 1.0
    assert -1.0 <= unrelated_sim <= 1.0

    # The AI/chip articles should be significantly more similar to each other
    # than to winter blizzard
    assert related_sim > unrelated_sim
