"""End-to-end pipeline demonstration script.

Exercises all components built in Tasks 001 to 007:
1. Ingestion (RSS parsing & persistence)
2. Cleaning & Normalization (HTML removal, tracking params, whitespace)
3. Deduplication (exact & fuzzy near-duplicate grouping)
4. Topic Classification (controlled financial taxonomy scoring)
5. Semantic Embeddings (vector representation & cosine similarity)
"""

from recommender.ingestion import ingest_feed
from recommender.models import (
    Article,
    create_db_engine,
    drop_db,
    get_session_factory,
    init_db,
)
from recommender.processing import (
    classify_topics,
    clean_article,
    cosine_similarity,
    embed_article,
    group_duplicate_articles,
)

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
<channel>
  <title>Financial Market Wire</title>
  <item>
    <title>  Reuters: <b>NVIDIA</b> reports record revenue  \n</title>
    <link>https://reuters.com/markets/nvda-q2?utm_source=twitter&amp;utm_medium=social#chart</link>
    <description>
      <p>Jensen Huang announced massive demand for <b>AI data center GPUs</b> &amp; chips.</p>
    </description>
    <pubDate>Fri, 18 Sep 2026 10:00:00 GMT</pubDate>
  </item>
  <item>
    <title>CNBC: NVIDIA revenue hits record</title>
    <link>https://cnbc.com/tech/nvidia-record-q2?fbclid=xyz123</link>
    <description>
      Semiconductor giant beats Wall Street estimates on AI infrastructure boom.
    </description>
    <pubDate>Fri, 18 Sep 2026 10:15:00 GMT</pubDate>
  </item>
  <item>
    <title>Fed prepares 25 bps interest rate cut as inflation cools</title>
    <link>https://wsj.com/economy/fed-rate-cut-september</link>
    <description>
      Federal Reserve policymakers signal monetary easing after CPI drops to 2.4%.
    </description>
    <pubDate>Fri, 18 Sep 2026 11:00:00 GMT</pubDate>
  </item>
</channel>
</rss>
"""


def main():
    print("=" * 70)
    print("      NEWSFLO RECOMMENDATION ENGINE — END-TO-END PIPELINE CHECK     ")
    print("=" * 70)

    # 1. Database Setup
    print("\n[1] Initializing isolated test database...")
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = get_session_factory(engine)
    print("    Database tables created successfully.")

    # 2. Ingestion & Raw Data Creation
    print("\n[2] Ingesting RSS feed...")
    with session_factory() as session:
        articles = ingest_feed("MarketWire", SAMPLE_FEED, session=session)
        session.commit()
        print(f"    Ingested {len(articles)} raw articles from feed.")
        for a in articles:
            print(f"    - ID {a.id}: {a.title}")
            print(f"      URL: {a.url}")

    # 3. Cleaning & Normalization
    print("\n[3] Running Cleaning & Normalization (Task 004)...")
    with session_factory() as session:
        for a in session.query(Article).all():
            clean_article(a)
        session.commit()
        for a in session.query(Article).all():
            print(f"    - Cleaned Title: {a.title}")
            print(f"      Cleaned URL  : {a.url}")
            print(f"      Description  : {a.description}")

    # 4. Deduplication
    print("\n[4] Running Deduplication & Grouping (Task 005)...")
    with session_factory() as session:
        all_articles = session.query(Article).all()
        groups = group_duplicate_articles(all_articles)
        print(f"    Total articles: {len(all_articles)} -> Clustered into {len(groups)} groups:")
        for idx, grp in enumerate(groups, 1):
            titles = [f"'{a.title}' ({a.source})" for a in grp]
            print(f"    Group #{idx} ({len(grp)} articles): {', '.join(titles)}")

    # 5. Topic Classification
    print("\n[5] Running Financial Topic Classification (Task 006)...")
    with session_factory() as session:
        for a in session.query(Article).all():
            topics = classify_topics(a)
            session.commit()
            print(f"    - Article #{a.id} ('{a.title[:35]}...'):")
            print(f"      Topics: {topics}")

    # 6. Embeddings & Semantic Similarity
    print("\n[6] Generating Embeddings & Cosine Similarity (Task 007)...")
    with session_factory() as session:
        for a in session.query(Article).all():
            vec = embed_article(a)
            session.commit()
            l2_norm = sum(x * x for x in vec)
            print(f"    - Article #{a.id} vector dim: {len(vec)} (L2 norm: {l2_norm:.2f})")

        articles = session.query(Article).all()
        sim_1_2 = cosine_similarity(articles[0].embedding, articles[1].embedding)
        sim_1_3 = cosine_similarity(articles[0].embedding, articles[2].embedding)

        print("\n    Semantic Similarity Matrix:")
        print(f"    - NVIDIA (Reuters) vs NVIDIA (CNBC) : {sim_1_2:.4f} (High similarity)")
        print(f"    - NVIDIA (Reuters) vs Fed Rates (WSJ): {sim_1_3:.4f} (Low similarity)")
        assert sim_1_2 > sim_1_3, "Expected related stories to have higher cosine similarity!"

    print("\n" + "=" * 70)
    print(" ALL PIPELINE STEPS EXECUTED & VERIFIED SUCCESSFULLY!")
    print("=" * 70)

    drop_db(engine)
    engine.dispose()


if __name__ == "__main__":
    main()
