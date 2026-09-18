"""Deterministic cleaning and normalization pipeline for news articles."""

import html
import re
from typing import Any
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Query parameters commonly used for campaign and analytics tracking
TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "gclsrc",
    "dclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "yclid",
    "_hsenc",
    "_hsmi",
    "igshid",
    "ref_src",
    "ref_url",
    "spref",
}

# Regex to match HTML tags
RE_HTML_TAGS = re.compile(r"<[^>]+>")

# Regex to convert block/line break tags into newlines before stripping
RE_HTML_BREAKS = re.compile(r"<(?:br|br\s*/|p|/p|div|/div|li|/li|tr|/tr)[^>]*>", re.IGNORECASE)

# Regex to remove scripts and style blocks entirely
RE_SCRIPT_STYLE = re.compile(r"<(?:script|style)[^>]*>[\s\S]*?</(?:script|style)>", re.IGNORECASE)

# Regex for multiple horizontal whitespace characters
RE_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\r\n]+")

# Regex for collapsing 3 or more consecutive newlines into 2
RE_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")


def remove_html(text: str | None) -> str:
    """Strip HTML tags while preserving line breaks and unescaping entities.

    Args:
        text: Input string potentially containing HTML markup.

    Returns:
        Clean plain text.
    """
    if not text:
        return ""

    # Strip script and style blocks entirely
    cleaned = RE_SCRIPT_STYLE.sub(" ", text)

    # Convert block elements and breaks into newlines so adjacent text does not merge
    cleaned = RE_HTML_BREAKS.sub("\n", cleaned)

    # Strip remaining HTML tags
    cleaned = RE_HTML_TAGS.sub("", cleaned)

    # Decode HTML entities (&amp;, &nbsp;, &#39;, &lt;, &gt;, etc.)
    cleaned = html.unescape(cleaned)

    return cleaned


def normalize_whitespace(text: str | None) -> str:
    """Normalize horizontal spaces, non-breaking spaces, and excessive newlines.

    Preserves single newlines or paragraph breaks without collapsing words.
    """
    if not text:
        return ""

    # Normalize unicode spaces (e.g. non-breaking space \u00a0, thin space, etc.)
    cleaned = text.replace("\u00a0", " ").replace("\u200b", "")

    # Split lines and normalize horizontal whitespace per line
    lines = [RE_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in cleaned.splitlines()]

    # Rejoin with newlines and collapse more than two consecutive newlines
    result = "\n".join(lines)
    result = RE_EXCESSIVE_NEWLINES.sub("\n\n", result)

    return result.strip()


def normalize_text(text: str | None) -> str:
    """Clean and normalize general text (descriptions, body text).

    Applies HTML removal, unicode normalization, and whitespace cleanup.
    Preserves all financial symbols ($100, 5.4%, P/E, S&P 500, etc.).
    """
    if not text:
        return ""

    # Remove HTML markup and decode entities
    cleaned = remove_html(text)

    # Normalize unicode to NFKC (standardizes characters without stripping financial symbols)
    cleaned = unicodedata.normalize("NFKC", cleaned)

    # Clean whitespace
    return normalize_whitespace(cleaned)


def normalize_title(title: str | None) -> str:
    """Normalize an article headline or title.

    Collapses any embedded newlines into single spaces and trims whitespace.
    """
    if not title:
        return ""

    # Remove HTML and decode entities
    cleaned = remove_html(title)

    # Normalize unicode
    cleaned = unicodedata.normalize("NFKC", cleaned)

    # Replace newlines/tabs with spaces for single-line titles
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)

    # Collapse multiple spaces into one
    cleaned = RE_HORIZONTAL_WHITESPACE.sub(" ", cleaned)

    return cleaned.strip()


def remove_tracking_parameters(url: str | None) -> str:
    """Remove common marketing and campaign tracking query parameters from a URL."""
    if not url:
        return ""

    parsed = urlsplit(url.strip())
    if not parsed.query:
        return url.strip()

    # Filter query parameters
    query_params = parse_qsl(parsed.query, keep_blank_values=True)
    filtered_params = [
        (k, v)
        for k, v in query_params
        if not (k.lower().startswith("utm_") or k.lower() in TRACKING_PARAMS)
    ]

    new_query = urlencode(filtered_params)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, new_query, parsed.fragment))


def normalize_url(url: str | None) -> str:
    """Normalize a URL by removing tracking params, stripping fragments, and
    standardizing casing.
    """
    if not url:
        return ""

    # Strip whitespace
    cleaned = url.strip()

    # Remove tracking query parameters
    cleaned = remove_tracking_parameters(cleaned)

    # Parse URL components
    parsed = urlsplit(cleaned)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Strip default ports (:80 for http, :443 for https)
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    path = parsed.path or "/"

    # Strip fragment (#section) for canonical URL uniqueness
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def clean_article(article: Any) -> Any:
    """Clean and normalize fields of an article instance or dictionary.

    Works on SQLAlchemy Article ORM instances, dicts, or objects with article fields.
    Modifies in-place where applicable and returns the cleaned object.
    """
    if article is None:
        return None

    if isinstance(article, dict):
        cleaned_dict = dict(article)
        if "title" in cleaned_dict and cleaned_dict["title"]:
            cleaned_dict["title"] = normalize_title(cleaned_dict["title"])
        if "url" in cleaned_dict and cleaned_dict["url"]:
            cleaned_dict["url"] = normalize_url(cleaned_dict["url"])
        if "description" in cleaned_dict:
            desc = normalize_text(cleaned_dict["description"])
            cleaned_dict["description"] = desc if desc else None
        if "raw_text" in cleaned_dict:
            raw = normalize_text(cleaned_dict["raw_text"])
            cleaned_dict["raw_text"] = raw if raw else None
        return cleaned_dict

    # Handle object / SQLAlchemy model instance
    if hasattr(article, "title") and article.title:
        article.title = normalize_title(article.title)

    if hasattr(article, "url") and article.url:
        article.url = normalize_url(article.url)

    if hasattr(article, "description"):
        desc = normalize_text(article.description)
        article.description = desc if desc else None

    if hasattr(article, "raw_text"):
        raw = normalize_text(article.raw_text)
        article.raw_text = raw if raw else None

    return article
