"""Controlled financial topic taxonomy and classification pipeline."""

from abc import ABC, abstractmethod
import re
from typing import Any

# Controlled taxonomy of financial categories and topics
TOPIC_TAXONOMY: dict[str, list[str]] = {
    "Technology": ["AI", "Semiconductors", "Hardware"],
    "Markets": ["Stocks", "Bonds", "Commodities", "Forex"],
    "Economy": ["Inflation", "Interest Rates", "GDP", "Employment"],
    "Companies": ["NVIDIA", "Apple", "Tesla", "Reliance", "TSMC"],
    "Finance": ["Banking", "Fintech", "Payments", "Credit"],
    "World": ["India", "USA", "China", "Europe"],
}

# Mapping of keywords for deterministic rule-based classification
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "AI": [
        "ai",
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "generative ai",
        "llm",
        "openai",
    ],
    "Semiconductors": [
        "semiconductor",
        "semiconductors",
        "chip",
        "chips",
        "wafer",
        "foundry",
        "gpu",
        "tsmc",
        "nvidia",
        "intel",
        "amd",
    ],
    "Hardware": ["hardware", "datacenter", "data center", "server", "servers", "devices"],
    "Stocks": [
        "stock",
        "stocks",
        "equity",
        "equities",
        "shares",
        "share price",
        "nasdaq",
        "s&p",
        "s&p 500",
        "dow jones",
        "wall street",
    ],
    "Bonds": ["bond", "bonds", "treasury", "treasuries", "yield", "yields", "fixed income"],
    "Commodities": [
        "commodity",
        "commodities",
        "oil",
        "crude",
        "brent",
        "gold",
        "silver",
        "gas",
        "metals",
    ],
    "Forex": ["forex", "fx", "currency", "currencies", "dollar", "euro", "yen", "yuan", "rupee"],
    "Inflation": ["inflation", "cpi", "consumer prices", "cost of living", "deflation"],
    "Interest Rates": [
        "interest rate",
        "interest rates",
        "rate cut",
        "rate hike",
        "rate pause",
        "fed",
        "federal reserve",
        "central bank",
        "monetary policy",
    ],
    "GDP": ["gdp", "gross domestic product", "economic growth", "recession"],
    "Employment": [
        "employment",
        "unemployment",
        "jobs",
        "jobless",
        "payroll",
        "payrolls",
        "hiring",
        "layoffs",
    ],
    "NVIDIA": ["nvidia", "nvda", "jensen huang"],
    "Apple": ["apple", "aapl", "iphone", "ipad", "macbook", "tim cook"],
    "Tesla": ["tesla", "tsla", "elon musk", "cybertruck", "gigafactory"],
    "Reliance": ["reliance", "jio", "mukesh ambani", "reliance industries"],
    "TSMC": ["tsmc", "taiwan semiconductor"],
    "Banking": [
        "bank",
        "banks",
        "banking",
        "jpmorgan",
        "goldman sachs",
        "morgan stanley",
        "citigroup",
        "wells fargo",
    ],
    "Fintech": ["fintech", "digital banking", "neobank", "robo-advisor"],
    "Payments": ["payment", "payments", "visa", "mastercard", "stripe", "paypal"],
    "Credit": ["credit", "debt", "lending", "loan", "loans", "default", "borrowing"],
    "India": ["india", "indian", "rbi", "nifty", "sensex", "mumbai", "rupee"],
    "USA": ["usa", "united states", "us economy", "federal reserve", "wall street", "washington"],
    "China": ["china", "chinese", "beijing", "pboc", "shanghai", "yuan"],
    "Europe": ["europe", "european", "ecb", "eurozone", "eu", "euro"],
}

# Precompile regex word patterns for fast keyword boundary matching
_COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {
    topic: [
        re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
        for kw in keywords
    ]
    for topic, keywords in TOPIC_KEYWORDS.items()
}


class TopicClassifier(ABC):
    """Abstract interface for topic classification engines."""

    @abstractmethod
    def classify(self, text: str, title: str | None = None) -> dict[str, float]:
        """Classify given text into topics with relevance scores in [0.0, 1.0]."""
        pass


class RuleBasedTopicClassifier(TopicClassifier):
    """Deterministic, rule-based keyword classifier for financial topics."""

    def __init__(self, min_score: float = 0.25):
        self.min_score = min_score

    def classify(self, text: str, title: str | None = None) -> dict[str, float]:
        scores: dict[str, float] = {}
        combined_text = f"{title or ''} {text or ''}".strip()
        if not combined_text:
            return {}

        title_lower = (title or "").lower()
        body_lower = (text or "").lower()

        # Score specific leaf topics
        for topic, patterns in _COMPILED_PATTERNS.items():
            title_hits = sum(1 for p in patterns if p.search(title_lower))
            body_hits = sum(1 for p in patterns if p.search(body_lower))

            if title_hits == 0 and body_hits == 0:
                continue

            # Title matches carry higher weight than body text matches
            raw_score = (title_hits * 0.5) + (body_hits * 0.2)
            # Bound and scale to [0.3, 0.95]
            normalized = min(0.95, max(0.3, 0.3 + (raw_score * 0.2)))
            scores[topic] = round(normalized, 2)

        # Propagate child topic confidence to parent category if present
        for category, subtopics in TOPIC_TAXONOMY.items():
            child_scores = [scores[t] for t in subtopics if t in scores]
            if child_scores:
                # Parent relevance inherits a weighted proportion of its strongest child topic
                parent_score = round(min(0.95, max(child_scores) * 0.9), 2)
                scores[category] = parent_score

        # Filter by minimum score threshold
        return {topic: s for topic, s in scores.items() if s >= self.min_score}


_default_classifier = RuleBasedTopicClassifier()


def classify_topics(
    article: Any,
    classifier: TopicClassifier | None = None,
) -> dict[str, float]:
    """Classify an article into topics with relevance scores in [0.0, 1.0].

    Args:
        article: Article ORM model, dict, or object with title/description/raw_text.
        classifier: Optional custom TopicClassifier instance.

    Returns:
        Dictionary mapping topic name to relevance score.
    """
    if article is None:
        return {}

    clf = classifier or _default_classifier

    if isinstance(article, dict):
        title = article.get("title") or ""
        text = f"{article.get('description') or ''} {article.get('raw_text') or ''}".strip()
        topics = clf.classify(text=text, title=title)
        article["topics"] = topics
        return topics

    title = getattr(article, "title", "") or ""
    desc = getattr(article, "description", "") or ""
    raw = getattr(article, "raw_text", "") or ""
    text = f"{desc} {raw}".strip()

    topics = clf.classify(text=text, title=title)
    if hasattr(article, "topics"):
        article.topics = topics

    return topics

