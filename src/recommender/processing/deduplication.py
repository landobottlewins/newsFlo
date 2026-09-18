"""Deterministic deduplication and grouping for financial news articles."""

from difflib import SequenceMatcher
import re
from typing import Any

from recommender.processing.cleaning import normalize_title, normalize_url

RE_WORDS = re.compile(r"[\$\w\%\.]+")
RE_PREFIX = re.compile(r"^\[?[A-Za-z0-9\s]+\]?:\s*")
RE_SUFFIX = re.compile(r"\s*[-|–—]\s*[A-Za-z0-9\s]+$")


def _get_field(obj: Any, field_name: str) -> Any:
    """Retrieve field value from either dictionary or object attribute."""
    if isinstance(obj, dict):
        return obj.get(field_name)
    return getattr(obj, field_name, None)


def clean_title_for_comparison(title: str | None, source: str | None = None) -> str:
    """Strip common news source prefixes and suffixes for cleaner comparison."""
    if not title:
        return ""

    t = title.strip()
    t = RE_PREFIX.sub("", t)
    t = RE_SUFFIX.sub("", t)

    if source:
        escaped = re.escape(source.strip())
        t = re.sub(rf"^{escaped}\s*[:\-]?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(rf"\s*[:\-]?\s*{escaped}$", "", t, flags=re.IGNORECASE)

    return t.strip()


def compute_title_similarity(
    title_a: str | None,
    title_b: str | None,
    source_a: str | None = None,
    source_b: str | None = None,
) -> float:
    """Compute semantic text similarity between two titles using deterministic string metrics.

    Combines token Jaccard similarity, sequence ratio, and token sort ratio while
    differentiating distinct capitalized entities (e.g. company names).
    """
    clean_a = clean_title_for_comparison(title_a, source_a)
    clean_b = clean_title_for_comparison(title_b, source_b)

    if not clean_a or not clean_b:
        return 0.0

    if clean_a.lower() == clean_b.lower():
        return 1.0

    w1_orig = RE_WORDS.findall(clean_a)
    w2_orig = RE_WORDS.findall(clean_b)
    w1 = [w.lower() for w in w1_orig]
    w2 = [w.lower() for w in w2_orig]
    s1, s2 = set(w1), set(w2)

    if not s1 or not s2:
        return 0.0

    if s1 == s2:
        return 1.0

    common = s1 & s2
    if not common:
        return 0.0

    # Differentiate distinct capitalized entities (e.g. Apple vs Amazon)
    caps1 = {w for w in w1_orig if w[0].isupper() and len(w) > 1 and w.lower() not in common}
    caps2 = {w for w in w2_orig if w[0].isupper() and len(w) > 1 and w.lower() not in common}
    if caps1 and caps2 and caps1 != caps2:
        return 0.0

    jaccard = len(common) / len(s1 | s2)
    overlap = len(common) / min(len(s1), len(s2))
    seq_ratio = SequenceMatcher(None, clean_a.lower(), clean_b.lower()).ratio()

    sorted_1 = " ".join(sorted(w1))
    sorted_2 = " ".join(sorted(w2))
    sort_ratio = SequenceMatcher(None, sorted_1, sorted_2).ratio()

    return max(jaccard, 0.5 * (jaccard + overlap), seq_ratio, sort_ratio)


def are_duplicates(
    article_a: Any,
    article_b: Any,
    similarity_threshold: float = 0.65,
) -> bool:
    """Determine whether two articles represent the same story or duplicate content.

    Stage 1: Checks exact matches (URL, source_article_id, or normalized title).
    Stage 2: Checks fuzzy title similarity without machine learning embeddings.

    Args:
        article_a: First article (dict or Article model instance).
        article_b: Second article (dict or Article model instance).
        similarity_threshold: Minimum similarity score required for near-duplicate match.

    Returns:
        True if the articles are duplicates, False otherwise.
    """
    if article_a is None or article_b is None:
        return False

    if article_a is article_b:
        return True

    url_a = _get_field(article_a, "url")
    url_b = _get_field(article_b, "url")
    if url_a and url_b:
        norm_url_a = normalize_url(str(url_a))
        norm_url_b = normalize_url(str(url_b))
        if norm_url_a and norm_url_a == norm_url_b:
            return True

    source_a = _get_field(article_a, "source")
    source_b = _get_field(article_b, "source")
    id_a = _get_field(article_a, "source_article_id")
    id_b = _get_field(article_b, "source_article_id")
    if source_a and source_b and source_a == source_b and id_a and id_b and id_a == id_b:
        return True

    title_a = _get_field(article_a, "title")
    title_b = _get_field(article_b, "title")
    if title_a and title_b:
        norm_title_a = normalize_title(str(title_a)).lower()
        norm_title_b = normalize_title(str(title_b)).lower()
        if norm_title_a and norm_title_a == norm_title_b:
            return True

        similarity = compute_title_similarity(
            str(title_a),
            str(title_b),
            source_a=str(source_a) if source_a else None,
            source_b=str(source_b) if source_b else None,
        )
        if similarity >= similarity_threshold:
            return True

    return False


def group_duplicate_articles(
    articles: list[Any],
    similarity_threshold: float = 0.65,
) -> list[list[Any]]:
    """Group a collection of articles into clusters representing the same news story.

    Preserves original article objects and their source attributions.

    Args:
        articles: List of article models or dictionaries.
        similarity_threshold: Threshold score for title similarity clustering.

    Returns:
        List of article groups, where each group is a list of duplicate articles.
    """
    groups: list[list[Any]] = []

    for article in articles:
        matched_group: list[Any] | None = None
        for group in groups:
            # Check if this article matches any existing member in the group
            if any(
                are_duplicates(article, member, similarity_threshold=similarity_threshold)
                for member in group
            ):
                matched_group = group
                break

        if matched_group is not None:
            matched_group.append(article)
        else:
            groups.append([article])

    return groups

