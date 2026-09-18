"""Recommendation engines, candidate generation, and ranking modules."""

from recommender.recommendation.candidates import generate_candidates
from recommender.recommendation.engine import recommend, score_article
from recommender.recommendation.ranking import (
    ScoredCandidate,
    calculate_exploration_score,
    calculate_interest_score,
    calculate_novelty_score,
    calculate_popularity_score,
    calculate_quality_score,
    calculate_recency_score,
    rank_candidates,
)

__all__ = [
    "ScoredCandidate",
    "calculate_exploration_score",
    "calculate_interest_score",
    "calculate_novelty_score",
    "calculate_popularity_score",
    "calculate_quality_score",
    "calculate_recency_score",
    "generate_candidates",
    "rank_candidates",
    "recommend",
    "score_article",
]
