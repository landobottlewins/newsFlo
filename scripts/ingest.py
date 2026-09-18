"""CLI script to trigger RSS news ingestion."""

import argparse
import logging
import sys

from recommender.config import get_settings, setup_logging
from recommender.ingestion import ingest_all_sources, ingest_feed
from recommender.models import create_db_engine, get_session_factory, init_db

logger = logging.getLogger("recommender.scripts.ingest")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest financial news RSS feeds.")
    parser.add_argument("--source", help="Single source formatted as 'name,url'")
    parser.add_argument(
        "--init-db", action="store_true", help="Initialize database tables before ingestion"
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    engine = create_db_engine()
    if args.init_db:
        logger.info("Initializing database tables...")
        init_db(engine)

    factory = get_session_factory(engine)
    with factory() as session:
        if args.source:
            parts = args.source.split(",", 1)
            if len(parts) != 2:
                logger.error("Invalid source format. Expected 'name,url'")
                return 1
            name, url = parts[0].strip(), parts[1].strip()
            articles = ingest_feed(name, url, session=session)
            session.commit()
            logger.info("Ingested %d articles from %s", len(articles), name)
        else:
            results = ingest_all_sources(session=session)
            session.commit()
            total = sum(len(arts) for arts in results.values())
            logger.info(
                "Ingestion finished: %d total new articles ingested across %d sources",
                total,
                len(results),
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
