"""Article embedding generation, provider abstraction, and vector comparison."""

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Any

# Default dimension matching standard lightweight text embedding models
DEFAULT_EMBEDDING_DIM = 384


class EmbeddingError(Exception):
    """Raised when an embedding provider fails to generate a vector."""

    pass


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension produced by this provider."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Convert a single text string into a semantic embedding vector."""
        pass

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Convert a batch of texts into semantic embedding vectors."""
        return [self.embed_text(t) for t in texts]


class DeterministicHashEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashing-based embedding provider for testing and zero-dependency environments.

    Projects token n-grams into a unit-normalized vector space where texts sharing
    vocabulary and concepts yield positive cosine similarity.
    """

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIM):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            raise EmbeddingError("Text contains no valid tokens to embed")

        vector = [0.0] * self._dim
        for token in tokens:
            # Map token deterministically to bucket indices and sign
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vector[idx] += sign

        # Add 2-gram representations for phrase awareness
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i + 1]}"
            h = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vector[idx] += 1.5 * sign

        # L2 normalization to unit sphere
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return [0.0] * self._dim

        return [round(v / norm, 6) for v in vector]


_default_provider: EmbeddingProvider = DeterministicHashEmbeddingProvider()


def get_default_embedding_provider() -> EmbeddingProvider:
    """Return default embedding provider instance."""
    return _default_provider


def set_default_embedding_provider(provider: EmbeddingProvider) -> None:
    """Set global default embedding provider."""
    global _default_provider
    _default_provider = provider


def cosine_similarity(v1: list[float] | None, v2: list[float] | None) -> float:
    """Calculate cosine similarity between two vector lists in [-1.0, 1.0]."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    dot = sum(a * b for a, b in zip(v1, v2, strict=True))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return max(-1.0, min(1.0, dot / (norm1 * norm2)))


def prepare_article_text_for_embedding(article: Any) -> str:
    """Extract and format meaningful article content for embedding generation."""
    if article is None:
        return ""

    if isinstance(article, dict):
        title = (article.get("title") or "").strip()
        desc = (article.get("description") or "").strip()
        raw = (article.get("raw_text") or "").strip()
    else:
        title = (getattr(article, "title", "") or "").strip()
        desc = (getattr(article, "description", "") or "").strip()
        raw = (getattr(article, "raw_text", "") or "").strip()

    # Prioritize title and description; append beginning of raw_text if helpful
    parts: list[str] = []
    if title:
        parts.append(title)
    if desc and desc != title:
        parts.append(desc)
    elif raw and raw != title:
        # Include first 500 chars of body if description is absent
        parts.append(raw[:500])

    return ". ".join(parts).strip()


def embed_article(
    article: Any,
    provider: EmbeddingProvider | None = None,
) -> list[float] | None:
    """Generate a vector embedding from an article's content and attach it.

    Args:
        article: Article model instance, dictionary, or article-like object.
        provider: Optional EmbeddingProvider (defaults to configured provider).

    Returns:
        The generated float vector, or None if article content is empty.

    Raises:
        EmbeddingError: If the provider fails to generate an embedding.
    """
    if article is None:
        return None

    text = prepare_article_text_for_embedding(article)
    if not text:
        return None

    active_provider = provider or get_default_embedding_provider()
    vector = active_provider.embed_text(text)

    if isinstance(article, dict):
        article["embedding"] = vector
    elif hasattr(article, "embedding"):
        article.embedding = vector

    return vector
