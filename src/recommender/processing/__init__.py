"""Article cleaning, normalization, deduplication, topic classification, and embedding modules."""

from recommender.processing.cleaning import (
    clean_article,
    normalize_text,
    normalize_title,
    normalize_url,
    normalize_whitespace,
    remove_html,
    remove_tracking_parameters,
)
from recommender.processing.deduplication import (
    are_duplicates,
    clean_title_for_comparison,
    compute_title_similarity,
    group_duplicate_articles,
)
from recommender.processing.embeddings import (
    DEFAULT_EMBEDDING_DIM,
    DeterministicHashEmbeddingProvider,
    EmbeddingError,
    EmbeddingProvider,
    cosine_similarity,
    embed_article,
    get_default_embedding_provider,
    set_default_embedding_provider,
)
from recommender.processing.topics import (
    TOPIC_KEYWORDS,
    TOPIC_TAXONOMY,
    RuleBasedTopicClassifier,
    TopicClassifier,
    classify_topics,
)

__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "DeterministicHashEmbeddingProvider",
    "EmbeddingError",
    "EmbeddingProvider",
    "RuleBasedTopicClassifier",
    "TOPIC_KEYWORDS",
    "TOPIC_TAXONOMY",
    "TopicClassifier",
    "are_duplicates",
    "clean_article",
    "clean_title_for_comparison",
    "classify_topics",
    "compute_title_similarity",
    "cosine_similarity",
    "embed_article",
    "get_default_embedding_provider",
    "group_duplicate_articles",
    "normalize_text",
    "normalize_title",
    "normalize_url",
    "normalize_whitespace",
    "remove_html",
    "remove_tracking_parameters",
    "set_default_embedding_provider",
]
