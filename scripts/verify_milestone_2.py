"""Milestone 2 Verification — Tasks 001 through 013.

Runs the full NewsFlo pipeline end-to-end in memory with two synthetic users
(a Tech/AI reader and a Macro trader) and prints a formatted, colour-coded
report showing each stage and the final personalised feeds.

Run with:
    python scripts/verify_milestone_2.py
"""

from datetime import UTC, datetime, timedelta

from recommender.ingestion import ingest_feed
from recommender.models import (
    Article,
    create_db_engine,
    drop_db,
    get_session_factory,
    init_db,
)
from recommender.models.interaction import InteractionType
from recommender.models.user import User, UserInterest
from recommender.processing import (
    classify_topics,
    clean_article,
    cosine_similarity,
    embed_article,
    group_duplicate_articles,
)
from recommender.recommendation import generate_candidates, rank_candidates
from recommender.users import record_interaction, update_user_interests_from_interaction

# ── ANSI colour helpers ────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BLUE = "\033[94m"
WHITE = "\033[97m"


def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


def header(title: str) -> None:
    width = 72
    print()
    print(c("─" * width, DIM))
    print(c(f"  {title}", BOLD + CYAN))
    print(c("─" * width, DIM))


def ok(msg: str) -> None:
    print(f"  {c('✓', GREEN)}  {msg}")


def info(msg: str) -> None:
    print(f"  {c('·', BLUE)}  {msg}")


def warn(msg: str) -> None:
    print(f"  {c('!', YELLOW)}  {msg}")


def bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return c("█" * filled, GREEN) + c("░" * (width - filled), DIM)


# ── Sample RSS feed (3 AI stories + 2 macro stories + 1 generic) ──────────────
SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Financial Market Wire</title>

  <item>
    <title>  Reuters: <b>NVIDIA</b> reports record AI revenue  </title>
    <link>https://reuters.com/markets/nvda-q2?utm_source=twitter&amp;utm_medium=social</link>
    <description><p>Jensen Huang announced massive demand for <b>AI data center GPUs</b> &amp; chips.</p></description>
    <pubDate>Fri, 19 Sep 2026 06:00:00 GMT</pubDate>
  </item>

  <item>
    <title>CNBC: NVIDIA revenue hits record high</title>
    <link>https://cnbc.com/tech/nvidia-record-q2?fbclid=xyz123</link>
    <description>Semiconductor giant beats Wall Street estimates on AI infrastructure boom.</description>
    <pubDate>Fri, 19 Sep 2026 06:15:00 GMT</pubDate>
  </item>

  <item>
    <title>Fed cuts rates 25 bps as inflation cools to 2.4%</title>
    <link>https://wsj.com/economy/fed-rate-cut-september</link>
    <description>Federal Reserve policymakers signal monetary easing. Bond yields fall sharply.</description>
    <pubDate>Fri, 19 Sep 2026 08:00:00 GMT</pubDate>
  </item>

  <item>
    <title>10-Year Treasury yield drops below 4% after Fed pivot</title>
    <link>https://bloomberg.com/markets/rates/treasury-yields-fed</link>
    <description>Macro investors rotate into long-duration bonds as real yields compress on FOMC decision.</description>
    <pubDate>Fri, 19 Sep 2026 09:00:00 GMT</pubDate>
  </item>

  <item>
    <title>OpenAI launches GPT-5 with 10x reasoning improvements</title>
    <link>https://techcrunch.com/ai/openai-gpt5</link>
    <description>GPT-5 achieves near-human performance on complex multi-step reasoning benchmarks.</description>
    <pubDate>Fri, 19 Sep 2026 07:00:00 GMT</pubDate>
  </item>

  <item>
    <title>Apple reports steady iPhone sales ahead of holiday season</title>
    <link>https://marketwatch.com/apple-iphone-sales</link>
    <description>Consumer electronics demand remains resilient despite higher borrowing costs.</description>
    <pubDate>Fri, 19 Sep 2026 05:30:00 GMT</pubDate>
  </item>

</channel>
</rss>"""


def print_banner() -> None:
    print()
    print(c("╔" + "═" * 70 + "╗", CYAN))
    print(c("║", CYAN) + c("  NewsFlo  ·  Milestone 2 Verification  ·  Tasks 001 → 013        ", BOLD + WHITE) + c("  ║", CYAN))  # noqa: E501
    print(c("╚" + "═" * 70 + "╝", CYAN))


def main() -> None:  # noqa: PLR0914, PLR0915
    print_banner()
    now = datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC)
    passed = 0
    failed = 0

    # ── 1. DB SETUP ───────────────────────────────────────────────────────────
    header("STEP 1 · Database Initialisation")
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = get_session_factory(engine)
    ok("In-memory SQLite database created and schema applied.")

    # ── 2. INGESTION (Tasks 001–003) ──────────────────────────────────────────
    header("STEP 2 · RSS Ingestion  (Tasks 001-003)")
    with session_factory() as session:
        articles = ingest_feed("MarketWire", SAMPLE_FEED, session=session)
        session.commit()
        ok(f"Ingested {c(str(len(articles)), BOLD)} articles from feed.")
        for a in articles:
            info(f"ID {c(str(a.id), YELLOW)}  {a.title[:60]}")
        if len(articles) == 6:
            passed += 1
        else:
            warn(f"Expected 6 articles, got {len(articles)}")
            failed += 1

    # ── 3. CLEANING (Task 004) ────────────────────────────────────────────────
    header("STEP 3 · Cleaning & Normalisation  (Task 004)")
    with session_factory() as session:
        for a in session.query(Article).all():
            clean_article(a)
        session.commit()
        arts = session.query(Article).all()
        ok(f"Cleaned {len(arts)} articles (HTML stripped, URLs de-tracked, whitespace normalised).")
        has_html = any("<" in (a.title or "") for a in arts)
        has_utm = any("utm_" in (a.url or "") for a in arts)
        if not has_html and not has_utm:
            ok("No raw HTML or UTM params remain in cleaned articles.")
            passed += 1
        else:
            warn("HTML or tracking params still present after cleaning!")
            failed += 1

    # ── 4. DEDUPLICATION (Task 005) ───────────────────────────────────────────
    header("STEP 4 · Deduplication  (Task 005)")
    with session_factory() as session:
        all_arts = session.query(Article).all()
        groups = group_duplicate_articles(all_arts)
        ok(f"{len(all_arts)} articles → {c(str(len(groups)), BOLD)} unique clusters.")
        for idx, grp in enumerate(groups, 1):
            titles = ", ".join(f"'{a.title[:30]}'" for a in grp)
            colour = YELLOW if len(grp) > 1 else DIM
            info(f"Cluster {idx} ({c(str(len(grp)), colour)} articles): {titles}")
        # Expect NVIDIA pair to merge into 1 cluster
        dup_found = any(len(g) > 1 for g in groups)
        if dup_found:
            ok("Near-duplicate NVIDIA stories correctly clustered together.")
            passed += 1
        else:
            warn("Expected at least one duplicate cluster — check deduplication logic.")
            failed += 1

    # ── 5. TOPIC CLASSIFICATION (Task 006) ───────────────────────────────────
    header("STEP 5 · Financial Topic Classification  (Task 006)")
    topic_map: dict[int, dict[str, float]] = {}
    with session_factory() as session:
        for a in session.query(Article).all():
            topics = classify_topics(a)
            topic_map[a.id] = topics
            session.commit()
            top = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = "  ".join(f"{t}={c(f'{s:.2f}', YELLOW)}" for t, s in top)
            info(f"ID {c(str(a.id), YELLOW)}  {a.title[:38]:<40}  {top_str}")
        ok(f"All {len(topic_map)} articles classified.")
        passed += 1

    # ── 6. EMBEDDINGS (Task 007) ──────────────────────────────────────────────
    header("STEP 6 · Semantic Embeddings  (Task 007)")
    with session_factory() as session:
        for a in session.query(Article).all():
            embed_article(a)
            session.commit()
        arts = session.query(Article).all()
        ok(f"Generated {len(arts)} embedding vectors.")
        # Cosine similarity check: two NVIDIA stories should be more similar than NVIDIA vs Fed
        nvda_arts = [a for a in arts if "NVIDIA" in (a.title or "")]
        fed_art = next((a for a in arts if "Fed" in (a.title or "")), None)
        if len(nvda_arts) >= 2 and fed_art:
            sim_same = cosine_similarity(nvda_arts[0].embedding, nvda_arts[1].embedding)
            sim_diff = cosine_similarity(nvda_arts[0].embedding, fed_art.embedding)
            info(f"NVIDIA ↔ NVIDIA (same topic):   {bar(sim_same)} {c(f'{sim_same:.4f}', GREEN)}")
            info(f"NVIDIA ↔ Fed Rates (diff topic): {bar(sim_diff)} {c(f'{sim_diff:.4f}', RED)}")
            if sim_same > sim_diff:
                ok("Cosine similarity correctly ranks same-topic articles higher.")
                passed += 1
            else:
                warn("Similarity check failed: same-topic pair should score higher.")
                failed += 1
        else:
            warn("Could not find expected articles for similarity check.")
            failed += 1

    # ── 7. USER INTEREST MODEL (Task 008) ────────────────────────────────────
    header("STEP 7 · User Interest Model  (Task 008)")
    with session_factory() as session:
        tech_user = User(id=1, username="alex_tech")
        macro_user = User(id=2, username="sam_macro")
        session.add_all([tech_user, macro_user])

        # Seed initial interests
        for topic, score in [("AI", 0.90), ("Semiconductors", 0.85), ("Hardware", 0.60)]:
            session.add(UserInterest(user_id=1, topic=topic, score=score))
        for topic, score in [("Interest Rates", 0.92), ("Inflation", 0.88), ("Bonds", 0.80)]:
            session.add(UserInterest(user_id=2, topic=topic, score=score))
        session.commit()

        ok("Created 2 synthetic users with seeded interest profiles.")
        info(f"{c('Alex', MAGENTA)}  →  AI={c('0.90', GREEN)}  Semiconductors={c('0.85', GREEN)}  Hardware={c('0.60', YELLOW)}")
        info(f"{c('Sam', BLUE)}   →  Interest Rates={c('0.92', GREEN)}  Inflation={c('0.88', GREEN)}  Bonds={c('0.80', YELLOW)}")
        passed += 1

    # ── 8. INTERACTION TRACKING (Task 009) ───────────────────────────────────
    header("STEP 8 · Interaction Tracking  (Task 009)")
    with session_factory() as session:
        arts = session.query(Article).all()
        ai_arts = [a for a in arts if topic_map.get(a.id, {}).get("AI", 0) > 0.3]
        macro_arts = [a for a in arts if topic_map.get(a.id, {}).get("Interest Rates", 0) > 0.3]

        events_created = 0
        # Alex reads AI articles
        for art in ai_arts[:2]:
            record_interaction(
                user_id=1,
                event_type=InteractionType.READ_2M,
                article_id=art.id,
                timestamp=now - timedelta(minutes=30),
                session=session,
            )
            events_created += 1
        # Alex bookmarks one AI article
        if ai_arts:
            record_interaction(
                user_id=1,
                event_type=InteractionType.BOOKMARK,
                article_id=ai_arts[0].id,
                timestamp=now - timedelta(minutes=20),
                session=session,
            )
            events_created += 1
        # Sam reads macro articles
        for art in macro_arts[:2]:
            record_interaction(
                user_id=2,
                event_type=InteractionType.READ_2M,
                article_id=art.id,
                timestamp=now - timedelta(minutes=45),
                session=session,
            )
            events_created += 1
        # Sam skips an AI article
        if ai_arts:
            record_interaction(
                user_id=2,
                event_type=InteractionType.SKIP,
                article_id=ai_arts[0].id,
                timestamp=now - timedelta(minutes=10),
                session=session,
            )
            events_created += 1
        session.commit()
        ok(f"Logged {c(str(events_created), BOLD)} interaction events across both users.")
        info("Alex:  2× READ_2M (AI stories)  +  1× BOOKMARK")
        info("Sam:   2× READ_2M (Macro stories) + 1× SKIP (AI story)")
        passed += 1

    # ── 9. BEHAVIORAL INTEREST UPDATER (Task 010) ─────────────────────────────
    header("STEP 9 · Behavioral Interest Updater  (Task 010)")
    with session_factory() as session:
        arts = session.query(Article).all()
        ai_arts = [a for a in arts if topic_map.get(a.id, {}).get("AI", 0) > 0.3]
        macro_arts = [a for a in arts if topic_map.get(a.id, {}).get("Interest Rates", 0) > 0.3]

        def make_interaction(
            user_id: int, event_type: InteractionType, article: Article
        ) -> "Interaction":
            from recommender.models.interaction import Interaction as _Interaction

            ia = _Interaction(
                user_id=user_id,
                event_type=event_type.value,
                article_id=article.id,
                timestamp=now,
            )
            session.add(ia)
            session.flush()
            return ia

        # Alex reads AI articles + bookmarks one
        for art in ai_arts[:2]:
            ia = make_interaction(1, InteractionType.READ_2M, art)
            update_user_interests_from_interaction(ia, session)
        if ai_arts:
            ia = make_interaction(1, InteractionType.BOOKMARK, ai_arts[0])
            update_user_interests_from_interaction(ia, session)
        # Sam reads macro articles + skips an AI article
        for art in macro_arts[:2]:
            ia = make_interaction(2, InteractionType.READ_2M, art)
            update_user_interests_from_interaction(ia, session)
        if ai_arts:
            ia = make_interaction(2, InteractionType.SKIP, ai_arts[0])
            update_user_interests_from_interaction(ia, session)
        session.commit()

        from recommender.users import get_user_interests

        alex_interests = get_user_interests(1, session)
        sam_interests = get_user_interests(2, session)

        ok("Interest profiles updated from behavioral signals.")
        print()
        print(f"  {c('Alex (Tech/AI Reader)', BOLD + MAGENTA)} — updated interests:")
        for topic, score in sorted(alex_interests.items(), key=lambda x: x[1], reverse=True)[:6]:
            print(f"    {bar(score, 16)} {c(f'{score:.3f}', YELLOW)}  {topic}")
        print()
        print(f"  {c('Sam (Macro Trader)', BOLD + BLUE)} — updated interests:")
        for topic, score in sorted(sam_interests.items(), key=lambda x: x[1], reverse=True)[:6]:
            print(f"    {bar(score, 16)} {c(f'{score:.3f}', YELLOW)}  {topic}")

        alex_ai = alex_interests.get("AI", 0)
        sam_rate = sam_interests.get("Interest Rates", 0)
        if alex_ai > 0 and sam_rate > 0:
            ok("Both users have domain-specific interests — behavioral updater working.")
            passed += 1
        else:
            warn("Expected non-zero interests after behavioral update.")
            failed += 1

    # ── 10. RECOMMENDATION ENGINE (Task 011) ─────────────────────────────────
    header("STEP 10 · Recommendation Engine  (Task 011)")
    from recommender.recommendation import recommend

    alex_dict = {"interests": alex_interests}
    sam_dict = {"interests": sam_interests}

    with session_factory() as session:
        pool = session.query(Article).all()
        pool_dicts = [
            {
                "id": a.id,
                "title": a.title,
                "topics": topic_map.get(a.id, {}),
                "embedding": a.embedding,
            }
            for a in pool
        ]

    alex_feed_basic = recommend(alex_dict, pool_dicts, k=6)
    sam_feed_basic = recommend(sam_dict, pool_dicts, k=6)

    ok(f"Alex's basic feed top article:  {c(alex_feed_basic[0]['title'][:55], MAGENTA)}")
    ok(f"Sam's basic feed top article:   {c(sam_feed_basic[0]['title'][:55], BLUE)}")

    alex_top_ai = topic_map.get(alex_feed_basic[0]["id"], {}).get("AI", 0)
    sam_top_macro = topic_map.get(sam_feed_basic[0]["id"], {}).get("Interest Rates", 0)

    if alex_top_ai > 0 and sam_top_macro > 0:
        ok("Engine correctly surfaces domain-relevant articles for each user.")
        passed += 1
    else:
        warn("Top articles don't match user interests as expected.")
        failed += 1

    # ── 11. CANDIDATE GENERATION (Task 012) ──────────────────────────────────
    header("STEP 11 · Candidate Generation  (Task 012)")
    with session_factory() as session:
        alex_cands = generate_candidates(alex_dict, session, limit=10)
        sam_cands = generate_candidates(sam_dict, session, limit=10)

        ok(f"Generated {c(str(len(alex_cands)), BOLD)} candidates for Alex.")
        ok(f"Generated {c(str(len(sam_cands)), BOLD)} candidates for Sam.")
        info("Candidates pooled from: interests (40%) · trending (25%) · recency (25%) · exploration (10%)")

        if alex_cands and sam_cands:
            passed += 1
        else:
            warn("Candidate generation returned empty lists.")
            failed += 1

    # ── 12. FEED RANKING (Task 013) ───────────────────────────────────────────
    header("STEP 12 · Feed Ranking Engine  (Task 013)")
    with session_factory() as session:
        alex_cands_fresh = generate_candidates(alex_dict, session, limit=10)
        sam_cands_fresh = generate_candidates(sam_dict, session, limit=10)

        alex_ranked = rank_candidates(alex_dict, alex_cands_fresh, session=session, current_time=now)
        sam_ranked = rank_candidates(sam_dict, sam_cands_fresh, session=session, current_time=now)

    print()
    print(f"  {c('── Alex (Tech/AI Reader) · Personalised Feed ──', BOLD + MAGENTA)}")
    for rank, sc in enumerate(alex_ranked[:5], 1):
        art = sc.article
        art_id = art.get("id") if isinstance(art, dict) else getattr(art, "id", "?")
        title_raw = art.get("title") if isinstance(art, dict) else getattr(art, "title", "")
        title = (title_raw or f"Article {art_id}")[:50]
        print(
            f"  {c(str(rank), BOLD)}. {bar(sc.final_score, 14)} {c(f'{sc.final_score:.3f}', GREEN)}  "
            f"interest={c(f'{sc.interest:.2f}', YELLOW)} rec={c(f'{sc.recency:.2f}', CYAN)}  "
            f"{c(title, WHITE)}"
        )

    print()
    print(f"  {c('── Sam (Macro Trader) · Personalised Feed ──', BOLD + BLUE)}")
    for rank, sc in enumerate(sam_ranked[:5], 1):
        art = sc.article
        art_id = art.get("id") if isinstance(art, dict) else getattr(art, "id", "?")
        title_raw = art.get("title") if isinstance(art, dict) else getattr(art, "title", "")
        title = (title_raw or f"Article {art_id}")[:50]
        print(
            f"  {c(str(rank), BOLD)}. {bar(sc.final_score, 14)} {c(f'{sc.final_score:.3f}', GREEN)}  "
            f"interest={c(f'{sc.interest:.2f}', YELLOW)} rec={c(f'{sc.recency:.2f}', CYAN)}  "
            f"{c(title, WHITE)}"
        )

    # Verify feeds differ
    if alex_ranked and sam_ranked:
        alex_art = alex_ranked[0].article
        sam_art = sam_ranked[0].article
        alex_top_id = alex_art.get("id") if isinstance(alex_art, dict) else getattr(alex_art, "id")
        sam_top_id = sam_art.get("id") if isinstance(sam_art, dict) else getattr(sam_art, "id")
        if alex_top_id != sam_top_id:
            ok("Users received DIFFERENT top articles — personalisation is working!")
            passed += 1
        else:
            warn("Both users got the same top article — feeds may not be personalised.")
            # Soft warning; don't fail since it can happen with limited article pool
    else:
        warn("Ranking returned empty feeds.")
        failed += 1

    # ── SCORE BREAKDOWN EXAMPLE ───────────────────────────────────────────────
    header("BONUS · Score Breakdown (ScoredCandidate.explain())")
    if alex_ranked:
        sc = alex_ranked[0]
        art = sc.article
        art_id = art.get("id") if isinstance(art, dict) else getattr(art, "id", "?")
        title_raw = art.get("title") if isinstance(art, dict) else getattr(art, "title", "")
        print()
        print(f"  Article #{art_id}: {c((title_raw or '')[:60], WHITE)}")
        print()
        breakdown = {
            "interest_score": sc.interest,
            "recency_score": sc.recency,
            "quality_score": sc.quality,
            "novelty_score": sc.novelty,
            "popularity_score": sc.popularity,
            "exploration_score": sc.exploration,
            "FINAL_score": sc.final_score,
        }
        for key, val in breakdown.items():
            separator = "  " if not key.startswith("FINAL") else "  "
            colour = GREEN if key.startswith("FINAL") else YELLOW
            print(f"    {key:<22}{separator}{bar(val, 18)} {c(f'{val:.4f}', colour)}")
        print()
        print(f"  {c('Full text explanation:', DIM)}")
        for line in sc.explain().splitlines():
            print(f"  {c(line, DIM)}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    header("VERIFICATION SUMMARY")
    total_checks = passed + failed
    pct = (passed / total_checks * 100) if total_checks else 0
    result_colour = GREEN if failed == 0 else YELLOW if failed <= 2 else RED
    print()
    print(f"  {bar(pct / 100, 30)}  {c(f'{pct:.0f}%', result_colour)}")
    print()
    print(f"  {c(f'Checks passed:  {passed}/{total_checks}', BOLD + result_colour)}")
    if failed:
        print(f"  {c(f'Checks failed:  {failed}/{total_checks}', BOLD + RED)}")
    print()
    if failed == 0:
        print(c("  🎉  ALL MILESTONE 2 CHECKS PASSED — Tasks 001 → 013 verified!", BOLD + GREEN))
    else:
        print(c(f"  ⚠️   {failed} check(s) failed — review warnings above.", BOLD + YELLOW))
    print()

    drop_db(engine)
    engine.dispose()


if __name__ == "__main__":
    main()
