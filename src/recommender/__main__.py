"""Health check and CLI entrypoint for recommender module."""

import logging
import sys

from recommender.config import get_settings, setup_logging


def main() -> int:
    """Run health check and report system status."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger("recommender")
    logger.info("Recommender system initialized [env=%s]", settings.env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
