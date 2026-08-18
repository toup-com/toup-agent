"""app/websearch/freshness.py — recency classifier, Brave param builder, page
dates, the 18-month filter and web/news merge.

Every example below is either one of the incident's real queries (2026-08-18,
docs/web-search/freshness-incident.md) or a false-positive guard for a query
shape the classifier must leave alone.

Run: cd backend && python3 -m pytest tests/test_websearch_freshness.py -q
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.websearch import freshness as F

TODAY = date(2026, 8, 18)
NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


# ── Classifier ───────────────────────────────────────────────────────

@pytest.mark.parametrize("query", [
    # the incident's queries, verbatim from the container log
    "site:openai.com latest GPT model announcement August 2026",
    "site:anthropic.com/news newest Claude model August 2026",
    "site:openai.com latest GPT flagship model official",
    "site:anthropic.com news latest Claude flagship model official",
    "Artificial Analysis intelligence index latest AI models ranking",
    "LMSYS Chatbot Arena leaderboard latest models",
    # the user's own phrasing and the neutral rewrites the prompt steers to
    "Search what is the new model for gpt",
    "anthropic newest model",
    "What is the most capable model",
    "openai new gpt model",
    # the spec's pattern list
    "who is the ceo of openai",
    "bitcoin price",
    "iphone 17 release date",
    "latest version of python",
    "what happened in ukraine today",
    "best laptops 2026",
    "claude opus 5 pricing",
    "weather in paris",
])
def test_recency_queries_are_classed_recent(query):
    v = F.classify(query, today=TODAY)
    assert v.is_recent, (query, v.reasons)
    assert v.reasons


@pytest.mark.parametrize("query", [
    "who is ada lovelace",
    "python list comprehension",
    "how do i reverse a string in python",
    "how to update a row in sql",
    "z scores statistics",
    "best pizza recipe",
    "history of rome",
    "tide times",
    "new york pizza",
    "",
    "   ",
])
def test_evergreen_queries_are_left_alone(query):
    v = F.classify(query, today=TODAY)
    assert not v.is_recent, (query, v.reasons)
    assert v.ladder == (None,)


def test_reasons_name_the_signal():
    v = F.classify("site:anthropic.com/news newest Claude model August 2026", today=TODAY)
    assert "latest" in v.reasons and "year" in v.reasons


def test_year_signal_is_current_or_previous_year_only():
    assert F.classify("best laptops 2026", today=TODAY).is_recent
    assert F.classify("best laptops 2025", today=TODAY).is_recent
    assert not F.classify("world cup 2018 final", today=TODAY).is_recent


def test_ladders_by_shape():
    # event → month first
    assert F.classify("anthropic newest model", today=TODAY).ladder == (F.FRESH_MONTH, F.FRESH_YEAR, None)
    # strong "now" → week first
    assert F.classify("what happened today", today=TODAY).ladder == (F.FRESH_WEEK, F.FRESH_MONTH, F.FRESH_YEAR, None)
    # pure state (no event word) → year first: a state is set by an event that
    # may be months old (measured: pm excluded the June-9 Fable 5 launch)
    assert F.classify("most capable AI model", today=TODAY).ladder == (F.FRESH_YEAR, None)
    assert F.classify("who is the ceo of openai", today=TODAY).ladder == (F.FRESH_YEAR, None)
    # state + event word → event ladder
    assert F.classify("latest version of python", today=TODAY).ladder == (F.FRESH_MONTH, F.FRESH_YEAR, None)


def test_classifier_never_raises():
    class Weird:
        def __str__(self):
            raise RuntimeError("boom")
    v = F.classify(Weird())  # type: ignore[arg-type]
    assert v.freshness_class in F.FRESHNESS_CLASSES


def test_normalize_class():
    assert F.normalize_class("recent") == "recent"
    assert F.normalize_class(" Evergreen ") == "evergreen"
    assert F.normalize_class("fresh") is None
    assert F.normalize_class(None) is None


# ── Query surgery / params ───────────────────────────────────────────

def test_split_site_operator():
    neutral, op = F.split_site_operator("site:anthropic.com/news newest Claude model August 2026")
    assert neutral == "newest Claude model August 2026"
    assert op == "site:anthropic.com/news"
    assert F.split_site_operator("plain query") == ("plain query", None)
    assert F.split_site_operator("site:x.com") == ("", "site:x.com")
    assert F.split_site_operator("a inurl:blog b") == ("a b", "inurl:blog")


def test_with_year_only_when_absent():
    assert F.with_year("anthropic newest model", today=TODAY) == "anthropic newest model 2026"
    assert F.with_year("best laptops 2025", today=TODAY) == "best laptops 2025"


def test_brave_params_shape():
    p = F.brave_params("q", 8, "pm", country="gb")
    assert p == {"q": "q", "count": 8, "extra_snippets": "true", "freshness": "pm", "country": "gb"}
    # evergreen: exactly the pre-incident call — no freshness key at all
    assert "freshness" not in F.brave_params("q", 8, None)
    assert "extra_snippets" not in F.brave_params("q", 5, "pm", extra_snippets=False)


# ── Dates ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("2026-07-24T00:00:00", date(2026, 7, 24)),
    ("2026-07-24", date(2026, 7, 24)),
    ("July 9, 2026", date(2026, 7, 9)),
    ("May 25, 2023", date(2023, 5, 25)),
    ("3 weeks ago", date(2026, 7, 28)),
    ("1 month ago", date(2026, 7, 19)),
    ("14 hours ago", date(2026, 8, 18)),
    ("2 years ago", date(2024, 8, 18)),
    (None, None),
    ("", None),
    ("garbage", None),
    ("2026-13-45T00:00:00", None),
    ("Smarch 5, 2026", None),
])
def test_parse_page_date(raw, expected):
    assert F.parse_page_date(raw, now=NOW) == expected


def test_result_date_prefers_iso_page_age():
    r = {"page_age": "2026-07-24T00:00:00", "age": "3 weeks ago"}
    assert F.result_date(r, now=NOW) == date(2026, 7, 24)
    assert F.age_days(r, now=NOW) == 25
    assert F.age_days({"age": None}, now=NOW) is None


def test_human_age():
    assert F.human_age(None) == "date unknown"
    assert F.human_age(0) == "today"
    assert F.human_age(1) == "1 day ago"
    assert F.human_age(5) == "5 days ago"
    assert F.human_age(25) == "3 weeks ago"
    assert F.human_age(100) == "3 months ago"
    assert F.human_age(1200) == "3 years ago"


# ── Stale filter ─────────────────────────────────────────────────────

def test_filter_stale_drops_old_keeps_undated():
    results = [
        {"url": "https://a/1", "page_age": "2026-07-24T00:00:00"},   # 25 days
        {"url": "https://a/2", "age": "May 25, 2023"},                # the LMSYS blog
        {"url": "https://a/3"},                                       # undated → kept
        {"url": "https://a/4", "page_age": "2025-01-01T00:00:00"},    # 19 months → dropped
    ]
    kept, dropped = F.filter_stale(results, max_age_days=548, now=NOW)
    assert [r["url"] for r in kept] == ["https://a/1", "https://a/3"]
    assert dropped == 2


def test_filter_stale_boundary_is_inclusive():
    r = [{"url": "u", "page_age": (TODAY.replace(year=2025, month=2, day=17)).isoformat()}]  # exactly 547 days
    kept, dropped = F.filter_stale(r, max_age_days=548, now=NOW)
    assert kept and not dropped


# ── Merge / canonical URLs ───────────────────────────────────────────

def test_canonical_url_is_scheme_www_slash_tracking_insensitive():
    a = F.canonical_url("https://www.anthropic.com/news/claude-opus-5/?utm_source=x#top")
    b = F.canonical_url("http://anthropic.com/news/claude-opus-5")
    assert a == b
    assert F.canonical_url("https://x.com/a?b=1&a=2") == F.canonical_url("https://x.com/a?a=2&b=1")
    assert F.canonical_url("https://x.com/a?id=1") != F.canonical_url("https://x.com/a?id=2")


def test_merge_interleaves_and_dedupes():
    web = [{"url": "https://w/1"}, {"url": "https://w/2"}, {"url": "https://w/3"}]
    news = [{"url": "https://n/1"}, {"url": "https://www.w/2/"}]     # second is a dup of web #2
    disc = [{"url": "https://d/1"}]
    out = F.merge_results(web, disc, news, limit=10, interleave=True)
    assert [r["url"] for r in out] == [
        "https://w/1", "https://d/1", "https://n/1",
        "https://w/2", "https://w/3",
    ]


def test_merge_respects_limit_and_sequential_mode():
    web = [{"url": f"https://w/{i}"} for i in range(5)]
    news = [{"url": f"https://n/{i}"} for i in range(5)]
    assert len(F.merge_results(web, news, limit=4)) == 4
    seq = F.merge_results(web, news, limit=7, interleave=False)
    assert [r["url"] for r in seq[:5]] == [r["url"] for r in web]


def test_date_span():
    rs = [{"page_age": "2026-07-24T00:00:00"}, {"age": "May 25, 2023"}, {}]
    assert F.date_span(rs, now=NOW) == ("2023-05-25", "2026-07-24", 1)
    assert F.date_span([{}], now=NOW) == (None, None, 1)
