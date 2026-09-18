# Financial News Recommendation Engine

A modular recommendation and content-understanding backend designed to power personalized financial-news feeds.

> **Note**: This repository focuses on the backend recommendation and content pipeline. It is not a frontend application or web server.

---

## Architecture Overview

The system is structured as a pipeline:

```text
                 NEWS SOURCES
                      ↓
                  INGESTION
                      ↓
                   CLEANING
                      ↓
               DEDUPLICATION
                      ↓
            CONTENT UNDERSTANDING
                ↙             ↘
             TOPICS        EMBEDDINGS
                ↘             ↙
                 ARTICLE STORE
                      ↓
               CANDIDATE GENERATION
                      ↓
                  RANKING ENGINE
                      ↓
                    USER FEED
                      ↓
                  INTERACTIONS
                      ↓
                 USER MODEL UPDATE
                      ↓
                NEXT RECOMMENDATION
```

---

## Directory Structure

```text
src/
    recommender/
        __init__.py          # Package root
        __main__.py          # Health-check entrypoint
        config.py            # Environment & logging configuration
        models/              # Data schemas and SQLAlchemy models
        ingestion/           # Feed collectors (RSS, financial sources)
        processing/          # Cleaning, normalization, deduplication
        recommendation/      # Candidate generation and ranking logic
        users/               # User profiles and interaction tracking
        evaluation/          # Offline/online metrics & experiments

tests/                       # Test suites

scripts/                     # Utility and operational scripts

data/
    raw/                     # Raw incoming feed data
    processed/               # Cleaned and normalized data
```

---

## Quickstart

### Requirements
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`

### Setup

Using `uv`:
```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Using standard `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Environment Configuration

The application reads configuration from environment variables or an optional `.env` file in the project root:

| Variable | Type | Default | Description |
|---|---|---|---|
| `APP_NAME` | string | `Financial News Recommender` | Application name |
| `ENV` | string | `development` | Environment (`development`, `test`, `production`) |
| `LOG_LEVEL` | string | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `DATABASE_URL` | string | `postgresql+psycopg://postgres:postgres@localhost:5432/newsflo` | PostgreSQL database connection URL |
| `DATA_DIR` | string | `data` | Base data directory path |

### Health Check

Run the entrypoint module to verify initialization:

```bash
python -m recommender
```

Expected output:
```text
YYYY-MM-DD HH:MM:SS [INFO] recommender: Recommender system initialized [env=development]
```

### Running Tests

```bash
pytest
```

### Code Formatting & Linting

```bash
ruff check .
ruff format --check .
```

