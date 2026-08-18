"""Render search results as the numbered block the model reads.

Shape (one result):

    1. <title>
       published: 2026-07-24 (3 weeks ago) · news
       https://example.com/page
       <description, verbatim, ≤ ~300 chars>
       <extra snippet, verbatim, ≤ ~200 chars>

The order of the three lines under the title is load-bearing for the web
client: ``WebSearchResultsCard.parseSearchResults`` takes the first line
starting with ``http`` as the URL, ignores non-URL lines BEFORE it, and folds
lines AFTER it into the snippet. So the date line goes between title and URL
(invisible to the card, visible to the model) and stays out of the snippet.

Everything is verbatim extraction — title/description/snippets are the
engine's own strings, only cut at a word boundary. Nothing here is generated.

Budget: the incident's blocks were 8.9–11.4k chars against an 8k budget and
the tail results were silently truncated away (6 of 8 visible). Per-result caps
below hold ten results to ≈9k chars, and the caller raises the budget so the
last result is never the one that gets cut.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .freshness import RECENT, age_days, human_age, result_date

DESCRIPTION_CHARS = 300
EXTRA_SNIPPET_CHARS = 200
EXTRA_SNIPPETS_PER_RESULT = 2
TITLE_CHARS = 160

# Sentinel the agent's ladder matches EXACTLY (see tool_executor) — keep.
NO_RESULTS = "No results found."


def clip(text: str, limit: int) -> str:
    """Cut at a word boundary near ``limit`` and mark the cut with an ellipsis.
    Never invents text; the ellipsis is the only non-verbatim character."""
    t = " ".join((text or "").split())
    if len(t) <= limit:
        return t
    cut = t[:limit]
    sp = cut.rfind(" ")
    if sp > limit - 60:
        cut = cut[:sp]
    return cut.rstrip(" ,;:-") + "…"


def _flag(r: Dict[str, Any]) -> str:
    src = (r.get("source") or "").strip().lower()
    return " · news" if src == "news" else ""


def render_result_lines(
    results: Sequence[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    show_dates: bool = True,
) -> List[str]:
    lines: List[str] = []
    for i, r in enumerate(results, 1):
        title = clip(r.get("title") or "", TITLE_CHARS) or "(untitled)"
        lines.append(f"{i}. {title}")
        if show_dates:
            d = result_date(r, now=now)
            days = age_days(r, now=now)
            if d is not None:
                lines.append(f"   published: {d.isoformat()} ({human_age(days)}){_flag(r)}")
            else:
                lines.append(f"   published: date unknown{_flag(r)}")
        lines.append(f"   {(r.get('url') or '').strip()}")
        desc = clip(r.get("description") or r.get("snippet") or "", DESCRIPTION_CHARS)
        if desc:
            lines.append(f"   {desc}")
        emitted = 0
        seen = {desc}
        for snip in (r.get("extra_snippets") or []):
            if emitted >= EXTRA_SNIPPETS_PER_RESULT:
                break
            s = clip(snip or "", EXTRA_SNIPPET_CHARS)
            # A passage that repeats the description (Brave often does) is
            # budget spent on nothing — skip it rather than let it use a slot.
            if s and s not in seen:
                seen.add(s)
                lines.append(f"   {s}")
                emitted += 1
        lines.append("")
    return lines


def render_header(
    *,
    query: str,
    n_results: int,
    freshness_class: Optional[str],
    freshness_applied: Optional[str] = None,
    dropped_stale: int = 0,
    stale_days: Optional[int] = None,
    news_count: int = 0,
    retrieved_at: Optional[datetime] = None,
    cache: Optional[str] = None,        # "hit" | "miss" | None
    cache_stored_at: Optional[datetime] = None,
    tier: Optional[str] = None,
) -> List[str]:
    """Metadata the model needs to weigh the block: what was searched, how
    fresh the page is, whether it came from cache and when. Lines before the
    first ``N.`` line are ignored by the web card parser."""
    ts = (retrieved_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    bits = [f"{n_results} result{'s' if n_results != 1 else ''}"]
    if freshness_class == RECENT:
        fr = f"Brave freshness={freshness_applied}" if freshness_applied else "no date filter"
        bits.append(f"freshness: recent ({fr})")
    else:
        bits.append("freshness: evergreen")
    if dropped_stale:
        months = f"{stale_days // 30} months" if stale_days else "the cutoff"
        bits.append(f"{dropped_stale} result{'s' if dropped_stale != 1 else ''} older than {months} dropped")
    if news_count:
        bits.append(f"{news_count} from news index")
    if tier:
        bits.append(f"via {tier}")
    bits.append(f"retrieved {ts.strftime('%Y-%m-%d %H:%M')} UTC")
    if cache == "hit":
        when = f" (stored {cache_stored_at.astimezone(timezone.utc).strftime('%H:%M')} UTC)" if cache_stored_at else ""
        bits.append(f"cache: hit{when}")
    elif cache == "miss":
        bits.append("cache: miss")
    head = [f'Web results for "{" ".join((query or "").split())}" — ' + " · ".join(bits)]
    if freshness_class == RECENT:
        head.append(
            "Dates are the page's published/updated date as reported by the search "
            "engine. For any latest/newest/current/most-capable claim prefer the "
            "NEWEST dated result from an official source, and cite only URLs listed here."
        )
    head.append("")
    return head


def render_block(
    results: Sequence[Dict[str, Any]],
    *,
    query: str,
    freshness_class: Optional[str],
    now: Optional[datetime] = None,
    **header_kwargs: Any,
) -> str:
    """Header + results. Returns ``NO_RESULTS`` for an empty page so the
    agent's exact-match sentinel keeps working."""
    if not results:
        return NO_RESULTS
    lines = render_header(
        query=query, n_results=len(results), freshness_class=freshness_class,
        retrieved_at=now, **header_kwargs,
    )
    lines.extend(render_result_lines(results, now=now))
    return "\n".join(lines).rstrip() + "\n"


def brave_web_to_dicts(payload: Dict[str, Any], *, source: str = "web") -> List[Dict[str, Any]]:
    """Normalise one Brave web/search page to the dict shape the renderer,
    filter and merger share. Keeps every field verbatim."""
    out: List[Dict[str, Any]] = []
    for r in ((payload.get("web") or {}).get("results") or []):
        if not r.get("url"):
            continue
        out.append({
            "title": (r.get("title") or "").strip(),
            "url": (r.get("url") or "").strip(),
            "description": (r.get("description") or "").strip(),
            "extra_snippets": [x for x in (r.get("extra_snippets") or []) if x],
            "age": r.get("age"),
            "page_age": r.get("page_age"),
            "source": source,
        })
    return out


def brave_news_to_dicts(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalise one Brave news/search page. The news endpoint returns a flat
    ``results`` list with the same field names (title/url/description/age/
    page_age); ``extra_snippets`` is absent there."""
    out: List[Dict[str, Any]] = []
    for r in (payload.get("results") or []):
        if not r.get("url"):
            continue
        out.append({
            "title": (r.get("title") or "").strip(),
            "url": (r.get("url") or "").strip(),
            "description": (r.get("description") or "").strip(),
            "extra_snippets": [x for x in (r.get("extra_snippets") or []) if x],
            "age": r.get("age"),
            "page_age": r.get("page_age"),
            "source": "news",
        })
    return out
