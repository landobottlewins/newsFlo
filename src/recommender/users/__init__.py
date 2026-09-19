"""User profiling, interaction tracking, and behavioral updates."""

from recommender.users.profiles import (
    get_user_interests,
    normalize_interest_score,
    set_interest,
)
from recommender.users.sessions import (
    SessionPersonalizationConfig,
    SessionProfile,
    combine_interests,
    create_session,
    get_combined_interests,
    reset_session,
    update_session_from_interaction,
)
from recommender.users.tracking import (
    get_user_interactions,
    record_interaction,
)
from recommender.users.updater import (
    INTERACTION_SIGNALS,
    apply_decay,
    update_user_interests_from_interaction,
)

__all__ = [
    "INTERACTION_SIGNALS",
    "SessionPersonalizationConfig",
    "SessionProfile",
    "apply_decay",
    "combine_interests",
    "create_session",
    "get_combined_interests",
    "get_user_interests",
    "get_user_interactions",
    "normalize_interest_score",
    "record_interaction",
    "reset_session",
    "update_session_from_interaction",
    "set_interest",
    "update_user_interests_from_interaction",
]
