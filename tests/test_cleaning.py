"""Tests for article cleaning and normalization pipeline."""

from datetime import UTC, datetime

from recommender.models import Article
from recommender.processing import (
    clean_article,
    normalize_text,
    normalize_title,
    normalize_url,
    normalize_whitespace,
    remove_html,
    remove_tracking_parameters,
)


def test_remove_html_basic_and_nested():
    """Verify HTML tags are removed and entities decoded."""
    html_input = "<p>Revenue increased by <b>$10 billion</b> in Q3 &amp; beat expectations.</p>"
    expected = "Revenue increased by $10 billion in Q3 & beat expectations."
    assert remove_html(html_input).strip() == expected


def test_remove_html_preserves_spacing_between_blocks():
    """Verify block elements don't cause text from adjacent tags to merge."""
    html_input = "<p>First paragraph.</p><p>Second paragraph with 5.4% yield.</p>"
    result = remove_html(html_input)
    assert "First paragraph." in result
    assert "Second paragraph with 5.4% yield." in result
    # Text must not be merged together
    assert "paragraph.Second" not in result


def test_remove_html_removes_scripts_and_styles():
    """Verify script and style blocks are stripped completely."""
    html_input = (
        "<style>.ad { display: none; }</style>"
        "<div>Core financial text.</div>"
        "<script>window.analytics.track();</script>"
    )
    result = remove_html(html_input)
    assert "Core financial text." in result
    assert "display: none" not in result
    assert "window.analytics" not in result


def test_normalize_whitespace_horizontal_and_vertical():
    """Verify horizontal spaces, tabs, and excessive newlines are normalized."""
    text = "Fed    signals   rate   pause  for   S&P 500.\n\n\n\nNext paragraph here."
    result = normalize_whitespace(text)
    assert result == "Fed signals rate pause for S&P 500.\n\nNext paragraph here."


def test_normalize_whitespace_unicode_spaces():
    """Verify non-breaking and zero-width spaces are handled."""
    text = "Inflation\u00a0rate\u200b is 3.2%."
    assert normalize_whitespace(text) == "Inflation rate is 3.2%."


def test_remove_tracking_parameters():
    """Verify marketing tracking query params are stripped while keeping valid params."""
    url = (
        "https://example.com/article"
        "?ticker=AAPL"
        "&utm_source=twitter"
        "&utm_medium=social"
        "&fbclid=IwAR123"
        "&gclid=Cj0K456"
        "&page=2"
    )
    result = remove_tracking_parameters(url)
    assert "utm_source" not in result
    assert "utm_medium" not in result
    assert "fbclid" not in result
    assert "gclid" not in result
    assert "ticker=AAPL" in result
    assert "page=2" in result


def test_normalize_url():
    """Verify URL scheme, host casing, default ports, and fragments are cleaned."""
    messy_url = (
        "  HTTPS://WWW.Bloomberg.COM:443/news/markets/fed-rate-cut"
        "?utm_campaign=daily_brief#summary-chart  "
    )
    result = normalize_url(messy_url)
    expected = "https://www.bloomberg.com/news/markets/fed-rate-cut"
    assert result == expected


def test_financial_symbols_preserved():
    """Verify financial symbols, acronyms, and formats are strictly preserved."""
    financial_text = (
        "The S&P 500 jumped 1.8% to $5,200.50 after the Fed chairman's speech.\n"
        "The P/E ratio stands at 24.5x, while GDP expanded by 2.8%.\n"
        "European stocks traded at €45.20, and UK bonds yielded 4.15%."
    )
    cleaned = normalize_text(financial_text)

    # Assert all key financial symbols remain intact
    assert "$5,200.50" in cleaned
    assert "1.8%" in cleaned
    assert "2.8%" in cleaned
    assert "4.15%" in cleaned
    assert "P/E" in cleaned
    assert "GDP" in cleaned
    assert "S&P 500" in cleaned
    assert "€45.20" in cleaned


def test_normalize_title():
    """Verify title normalization removes HTML, handles entities, and flattens newlines."""
    messy_title = "<h1>  Fed Signals <b>$100</b> Oil &amp; 5.4% Rate Peak \n\n Ahead  </h1>"
    result = normalize_title(messy_title)
    assert result == "Fed Signals $100 Oil & 5.4% Rate Peak Ahead"


def test_empty_and_null_inputs():
    """Verify empty or None inputs are gracefully handled."""
    assert remove_html(None) == ""
    assert remove_html("   ") == "   "
    assert normalize_whitespace(None) == ""
    assert normalize_whitespace("   \n\t  ") == ""
    assert normalize_text(None) == ""
    assert normalize_text("<p>  </p>") == ""
    assert normalize_title(None) == ""
    assert normalize_url(None) == ""
    assert clean_article(None) is None


def test_clean_article_orm_instance():
    """Verify clean_article operates on SQLAlchemy Article model instances."""
    art = Article(
        source="Reuters",
        title="  Fed   signals   <b>$100</b> Oil Ahead \n  ",
        description="<p>Crude rose <b>2.5%</b> today.<br/>Gas prices steady.</p>",
        raw_text="<div>Full report: P/E multiples at 18x &amp; GDP steady.</div>",
        url="https://Reuters.com:443/article/oil?utm_source=rss#chart",
        published_at=datetime.now(UTC),
    )

    cleaned = clean_article(art)
    assert cleaned is art
    assert cleaned.title == "Fed signals $100 Oil Ahead"
    assert cleaned.url == "https://reuters.com/article/oil"
    assert "Crude rose 2.5% today." in cleaned.description
    assert "Gas prices steady." in cleaned.description
    assert "P/E multiples at 18x & GDP steady." in cleaned.raw_text


def test_clean_article_dict():
    """Verify clean_article operates on dictionaries."""
    data = {
        "title": "  <b>Apple</b> P/E hits 30x \n\t ",
        "description": "<p>Shares rose 1.5% to $180.00.</p>",
        "raw_text": "   ",
        "url": "https://example.com/aapl?utm_medium=feed",
    }
    cleaned = clean_article(data)
    assert cleaned["title"] == "Apple P/E hits 30x"
    assert cleaned["description"] == "Shares rose 1.5% to $180.00."
    assert cleaned["raw_text"] is None  # empty whitespace converted to None
    assert cleaned["url"] == "https://example.com/aapl"


def test_cleaning_determinism_and_idempotency():
    """Acceptance criteria: same input always produces same output and is idempotent."""
    sample = (
        "<p>  The <b>S&amp;P 500</b> surged 3.4% to $5,120.\n\n\n"
        "Meanwhile P/E ratios and GDP estimates remained unchanged.  </p>"
    )

    run1 = normalize_text(sample)
    run2 = normalize_text(sample)
    assert run1 == run2

    # Idempotent: cleaning already cleaned text does not alter it further
    run3 = normalize_text(run1)
    assert run3 == run1
