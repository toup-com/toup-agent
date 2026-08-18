"""app/websearch/render.py + app/websearch/citations.py.

Renderer: the numbered block the model reads — verbatim, dated, budget-safe,
and byte-compatible with the web client's WebSearchResultsCard parser (a
faithful Python port of that parser is used below so the contract is tested
here, not discovered in the browser).

Citations: every http(s) URL in an answer must be grounded in this turn's
tool output; ungrounded ones are marked or stripped and reported.

Run: cd backend && python3 -m pytest tests/test_websearch_render_and_citations.py -q
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from app.agent.smart_fetch._budget import estimate_tokens
from app.websearch import render as R
from app.websearch.citations import CitationGate, MODE_STRIP, extract_urls

NOW = datetime(2026, 8, 18, 12, 55, tzinfo=timezone.utc)


def _brave(n=8, *, snippets=3, desc_len=280, snip_len=350, dated=True):
    return {
        "web": {"results": [
            {
                "title": f"Result {i} — a fairly long title to exercise the width of the header line",
                "url": f"https://example.com/articles/{i}",
                "description": ("d" * desc_len),
                "extra_snippets": [("s" * snip_len) for _ in range(snippets)],
                "age": ("3 weeks ago" if dated else None),
                "page_age": ("2026-07-24T00:00:00" if dated else None),
            }
            for i in range(1, n + 1)
        ]}
    }


# ── A faithful port of frontend/src/modules/chat/WebSearchResultsCard.tsx ─

def parse_like_web_card(summary: str):
    """Python port of `parseSearchResults`: `N. title` starts a result, the
    first `http` line is the URL, non-URL lines BEFORE the URL are ignored,
    lines AFTER it are folded into the snippet."""
    out, cur = [], None
    for raw in summary.split("\n"):
        line = raw.rstrip()
        m = re.match(r"^\s*(\d+)\.\s+(.+)$", line)
        if m:
            if cur:
                out.append(cur)
            cur = {"title": m.group(2).strip(), "url": "", "snippet": ""}
        elif cur:
            t = line.strip()
            if not t:
                continue
            if not cur["url"] and re.match(r"^https?://", t, re.I):
                cur["url"] = t
            elif cur["url"]:
                cur["snippet"] = f"{cur['snippet']} {t}".strip()
    if cur:
        out.append(cur)
    return [r for r in out if re.match(r"^https?://", r["url"], re.I)]


# ── Renderer ─────────────────────────────────────────────────────────

def test_block_shape_title_date_url_snippet():
    res = R.brave_web_to_dicts(_brave(1))
    block = R.render_block(res, query="q", freshness_class="recent", freshness_applied="pm", now=NOW)
    lines = block.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("1. "))
    assert lines[i + 1] == "   published: 2026-07-24 (3 weeks ago)"
    assert lines[i + 2] == "   https://example.com/articles/1"
    assert lines[i + 3].startswith("   ddd")


def test_web_card_parser_contract_holds():
    """The date line and the header must be invisible to the web card and the
    URL / title / snippet must come out exactly as before."""
    res = R.brave_web_to_dicts(_brave(3, snippets=1, desc_len=40, snip_len=30))
    block = R.render_block(res, query="q", freshness_class="recent", freshness_applied="pm", now=NOW)
    parsed = parse_like_web_card("cache: miss\n" + block)
    assert [p["url"] for p in parsed] == [f"https://example.com/articles/{i}" for i in (1, 2, 3)]
    assert parsed[0]["title"].startswith("Result 1")
    assert "published" not in parsed[0]["snippet"] and "cache:" not in parsed[0]["snippet"]
    assert parsed[0]["snippet"].startswith("d" * 40)


def test_undated_results_say_so():
    res = R.brave_web_to_dicts(_brave(1, dated=False))
    block = R.render_block(res, query="q", freshness_class="recent", now=NOW)
    assert "   published: date unknown" in block


def test_news_results_are_flagged():
    res = R.brave_news_to_dicts({"results": [{"title": "N", "url": "https://n/1", "description": "x", "age": "2 days ago"}]})
    block = R.render_block(res, query="q", freshness_class="recent", now=NOW)
    assert "published: 2026-08-16 (2 days ago) · news" in block


def test_verbatim_with_word_boundary_clip():
    text = "alpha beta gamma delta " * 40   # 920 chars
    c = R.clip(text, 300)
    assert c.endswith("…") and len(c) <= 301
    assert c[:-1] in text                       # verbatim prefix — nothing invented
    assert not c[:-1].endswith(" ")
    assert R.clip("short", 300) == "short"


def test_ten_results_fit_the_raised_budget_and_are_all_visible():
    """The incident: 8-result blocks were 8.9–11.4k chars against an 8k budget
    and 2 results were silently cut. Ten worst-case results must fit 3000
    tokens with every rank line present."""
    from app.config import Settings
    budget = Settings.model_fields["search_token_budget"].default
    assert budget >= 3000, "budget must have been raised alongside the renderer"
    res = R.brave_web_to_dicts(_brave(10, snippets=4, desc_len=600, snip_len=600))
    block = R.render_block(res, query="most capable AI model", freshness_class="recent",
                           freshness_applied="pm", dropped_stale=1, stale_days=548, news_count=3, now=NOW)
    assert estimate_tokens(block) <= budget, (len(block), estimate_tokens(block))
    assert all(f"\n{i}. " in block for i in range(1, 11))


def test_extra_snippets_capped_at_two_and_deduped_against_description():
    r = {"title": "t", "url": "https://x/1", "description": "same", "extra_snippets": ["same", "a", "b", "c"]}
    lines = R.render_result_lines([r], now=NOW, show_dates=False)
    body = [l for l in lines if l.startswith("   ") and not l.startswith("   http")]
    assert body == ["   same", "   a", "   b"]


def test_header_carries_freshness_and_stale_drops():
    res = R.brave_web_to_dicts(_brave(2))
    block = R.render_block(res, query="  anthropic   newest model ", freshness_class="recent",
                           freshness_applied="pm", dropped_stale=2, stale_days=548, news_count=1,
                           tier="search gateway", now=NOW)
    head = block.splitlines()[0]
    assert head.startswith('Web results for "anthropic newest model" — 2 results')
    assert "freshness: recent (Brave freshness=pm)" in head
    assert "2 results older than 18 months dropped" in head
    assert "1 from news index" in head
    assert "via search gateway" in head
    assert "retrieved 2026-08-18 12:55 UTC" in head
    assert "prefer the NEWEST dated result" in block.splitlines()[1]


def test_evergreen_header_has_no_recency_guidance():
    res = R.brave_web_to_dicts(_brave(1))
    block = R.render_block(res, query="q", freshness_class="evergreen", now=NOW)
    assert "freshness: evergreen" in block.splitlines()[0]
    assert "prefer the NEWEST" not in block


def test_empty_is_the_exact_sentinel_the_agent_matches():
    assert R.render_block([], query="q", freshness_class="recent") == "No results found."


def test_brave_dict_normalisers_keep_fields_verbatim_and_drop_urlless():
    web = R.brave_web_to_dicts({"web": {"results": [
        {"title": " T ", "url": " https://a/1 ", "description": " D ", "extra_snippets": ["", "s"], "age": "1 day ago"},
        {"title": "no url"},
    ]}})
    assert web == [{"title": "T", "url": "https://a/1", "description": "D", "extra_snippets": ["s"],
                    "age": "1 day ago", "page_age": None, "source": "web"}]
    news = R.brave_news_to_dicts({"results": [{"title": "N", "url": "https://n/1"}]})
    assert news[0]["source"] == "news"


# ── Citation gate ────────────────────────────────────────────────────

TOOL_OUTPUT = (
    "1. Introducing Claude Opus 5\n   published: 2026-07-24 (3 weeks ago)\n"
    "   https://www.anthropic.com/news/claude-opus-5\n   snippet\n\n"
    "2. TechCrunch\n   https://techcrunch.com/2026/07/24/anthropic-launches-opus-5/\n   snippet\n"
)


def test_extract_urls_handles_prose_markdown_and_parens():
    urls = extract_urls("see https://x.com/a, and (https://y.org/b). <https://z.net/c> [l](https://q.io/p) "
                        "wiki https://en.wikipedia.org/wiki/GPT-5_(model).")
    assert urls == ["https://x.com/a", "https://y.org/b", "https://z.net/c", "https://q.io/p",
                    "https://en.wikipedia.org/wiki/GPT-5_(model)"]


def test_grounded_urls_are_untouched_byte_for_byte():
    g = CitationGate(); g.add_text(TOOL_OUTPUT)
    ans = ("Opus 5 is the newest ([Anthropic](https://www.anthropic.com/news/claude-opus-5), "
           "https://techcrunch.com/2026/07/24/anthropic-launches-opus-5/). Front door https://anthropic.com/.")
    r = g.apply(ans)
    assert r.clean and r.text == ans and r.checked == 3


def test_incident_turn3_fabricated_citation_is_marked():
    """Turn 3 cited anthropic.com URLs that appeared in none of its four
    searches. The gate marks them and reports exactly those."""
    g = CitationGate(); g.add_text(TOOL_OUTPUT)
    ans = ("Most capable is [Claude Fable 5](https://www.anthropic.com/news/claude-fable-5); "
           "see also https://www.anthropic.com/news/claude-opus-4-8 and "
           "[Opus 5](https://www.anthropic.com/news/claude-opus-5).")
    r = g.apply(ans)
    assert r.violations == ["https://www.anthropic.com/news/claude-fable-5",
                            "https://www.anthropic.com/news/claude-opus-4-8"]
    assert "Claude Fable 5 (unverified: https://www.anthropic.com/news/claude-fable-5)" in r.text
    assert "https://www.anthropic.com/news/claude-opus-4-8 (unverified)" in r.text
    assert "[Opus 5](https://www.anthropic.com/news/claude-opus-5)" in r.text   # grounded link intact
    assert "(unverified: https://www.anthropic.com/news/claude-fable-5 (unverified))" not in r.text  # no double-marking


def test_strip_mode_removes_the_link_but_keeps_the_label():
    g = CitationGate(); g.add_text(TOOL_OUTPUT)
    r = g.apply("See [Fable 5](https://www.anthropic.com/news/claude-fable-5) and https://x.com/y.", mode=MODE_STRIP)
    assert r.text == "See Fable 5 (unverified) and (unverified link removed)."


def test_grounding_is_canonical_not_literal():
    g = CitationGate(); g.add_text("https://www.anthropic.com/news/claude-opus-5/")
    assert g.is_grounded("http://anthropic.com/news/claude-opus-5?utm_source=chat")
    assert not g.is_grounded("https://anthropic.com/news/claude-opus-4-8")


def test_user_message_and_fetch_input_ground_urls():
    g = CitationGate(trusted_text=["please summarise https://mysite.example/post/1"])
    g.add_url("https://fetched.example/page")
    r = g.apply("Summary of https://mysite.example/post/1 and https://fetched.example/page")
    assert r.clean


def test_non_http_links_are_ignored():
    g = CitationGate()
    ans = "Report ready: [Open report](toup://report?path=out.md) — mailto:x@y.z"
    r = g.apply(ans)
    assert r.clean and r.text == ans and r.checked == 0


def test_gate_never_raises_on_odd_input():
    g = CitationGate()
    assert g.apply("").text == ""
    assert g.apply("no urls here").clean
    assert g.apply("https://").checked >= 0
