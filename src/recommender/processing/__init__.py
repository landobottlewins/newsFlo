"""Article cleaning, normalization, and deduplication modules."""

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

__all__ = [
    "are_duplicates",
    "clean_article",
    "clean_title_for_comparison",
    "compute_title_similarity",
    "group_duplicate_articles",
    "normalize_text",
    "normalize_title",
    "normalize_url",
    "normalize_whitespace",
    "remove_html",
    "remove_tracking_parameters",
]
