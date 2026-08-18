"""Recency intent → Brave freshness policy → page-age hygiene.

Background (incident 2026-08-18, ``docs/web-search/freshness-incident.md``):
the gateway sent Brave ``{q, count, extra_snippets}`` and nothing else. For
"newest Claude model" Brave ranked by relevance and returned Sonnet 5 / Opus
4.8 / Claude 3 pages while the July-24 Opus 5 announcement was absent; the
same query with ``freshness=pm`` returned Opus 5 at #1. A "LMSYS leaderboard
latest models" query put a 2023-05-25 blog post at #1. Every result also
arrived undated because ``age``/``page_age`` were dropped on the wire, so the
model had no way to tell 2023 from last week.

This module is the single place that decides:

  * ``classify`` — is this query asking about the PRESENT (recency intent) or
    about something that does not go stale (evergreen)?
  * ``freshness_ladder`` — which Brave ``freshness`` values to try, in order,
    for that class (start narrow, widen only when the page comes back thin).
  * ``brave_params`` — the exact upstream query string for one attempt.
  * ``parse_page_date`` / ``age_days`` — one parser for Brave's three date
    shapes so a result's age is a number, not a string.
  * ``filter_stale`` — for recency intent, drop pages older than the cutoff.
  * ``merge_results`` — web + news + discovery into one de-duplicated list.
  * ``split_site_operator`` — so a ``site:``-anchored recency query can also
    run as a neutral discovery query (the incident's queries were 4/6 ``site:``).

Everything is pure and stdlib-only; both the platform gateway and the agent
import it, and it must never raise on odd input — a classifier exception must
not become a failed search.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# ── Vocabulary ───────────────────────────────────────────────────────

RECENT = "recent"
EVERGREEN = "evergreen"
FRESHNESS_CLASSES = (RECENT, EVERGREEN)

# Brave `freshness` values. `pd`=24h, `pw`=7d, `pm`=31d, `py`=365d.
FRESH_DAY, FRESH_WEEK, FRESH_MONTH, FRESH_YEAR = "pd", "pw", "pm", "py"

# Ladders: which Brave `freshness` values to try, in order, widening only
# while the page is thin.
#   EVENT  (latest / newest / release / news / price …): a month first — the
#          thing being asked about is an event and recent ones outrank.
#   NOW    (today / breaking / this week): a week first.
#   STATE  (most capable / who is the CEO / version / year, with NO event
#          word): a year first. A state is set by an event that may be 2-3
#          months old — measured 2026-08-18: `pm` excluded the June-9 Fable 5
#          launch article for "anthropic most capable model" while `py` kept
#          it at #3 — and every result is date-annotated, so the model still
#          prefers the newest. Stale (>18 mo) pages are filtered either way.
LADDER_DEFAULT: Tuple[Optional[str], ...] = (FRESH_MONTH, FRESH_YEAR, None)
LADDER_NOW: Tuple[Optional[str], ...] = (FRESH_WEEK, FRESH_MONTH, FRESH_YEAR, None)
LADDER_STATE: Tuple[Optional[str], ...] = (FRESH_YEAR, None)
STATE_REASONS = frozenset({"sota", "who_is", "version", "year"})

# Pages older than this are not evidence for a "newest / current / latest"
# claim. 18 months, per the incident spec; configurable by the callers.
DEFAULT_STALE_DAYS = 548


# ── Classifier ───────────────────────────────────────────────────────

# Each entry: (reason, compiled regex). Order does not matter — every match is
# recorded so the log line can say WHY a query was classed recent.
_W = r"(?<![\w-])"   # word-ish left boundary that also stops at a hyphen
_E = r"(?![\w-])"    # right boundary


def _rx(pattern: str) -> "re.Pattern[str]":
    return re.compile(_W + pattern + _E, re.IGNORECASE)


_NOW_SIGNALS = [
    ("today", _rx(r"today|tonight|yesterday|this (?:morning|afternoon|evening|week|weekend)|right now|just now|breaking|last (?:24|48) hours|past (?:24|48) hours")),
]

_RECENT_SIGNALS = [
    ("latest", _rx(r"latest|newest|most recent|recent(?:ly)?|current(?:ly)?|now|nowadays|as of|so far|up[- ]to[- ]date|still|these days|this (?:month|year|quarter)|last (?:week|month|year)|next (?:week|month|year)|upcoming|coming soon")),
    ("release", _rx(r"releases?d?|launch(?:es|ed)?|announce(?:s|d|ment)?|unveil(?:s|ed)?|debuts?|rolls? out|rollout|changelog|release notes|what'?s new|new (?:[\w.-]+ ){0,2}(?:model|models|version|release|releases|update|feature|features|generation|flagship|phone|chip|gpu|car|ev)|next[- ]gen(?:eration)?")),
    ("version", _rx(r"version|v\d+(?:\.\d+)+|\d+\.\d+ (?:release|update)|latest version|newest version")),
    ("price", _rx(r"prices?|pricing|costs?|how much|cheapest|deals?|discounts?|exchange rate|stock price|share price|market cap|net worth")),
    ("news", _rx(r"news|headlines?|happening|happened|what'?s going on|who won|winner|election|forecast|weather")),
    ("sota", _rx(r"most (?:capable|powerful|advanced|intelligent|accurate)|strongest|smartest|state[- ]of[- ]the[- ]art|sota|frontier|flagship|leaderboard|rankings?|ranked|benchmarks?|top[- ]\d+|best (?:\w+ ){0,3}(?:model|models|llm|llms|ai|gpu|gpus|phone|phones|laptop|laptops|car|cars|ev|evs|tool|tools|app|apps)")),
    ("who_is", re.compile(r"\bwho (?:is|are|'s)\b.*\b(?:the )?(?:current |new |acting |interim )?(?:ceo|cto|cfo|coo|president|prime minister|chancellor|governor|mayor|head|chief|chair(?:man|woman|person)?|coach|manager|owner|leader|director|secretary|minister)\b", re.IGNORECASE)),
]

# A mention of the current or previous calendar year is a recency signal; a
# mention of an older year is not (it is more often historical: "in 2019").
_YEAR = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


@dataclass
class RecencyVerdict:
    freshness_class: str
    reasons: List[str] = field(default_factory=list)
    strong_now: bool = False

    @property
    def is_recent(self) -> bool:
        return self.freshness_class == RECENT

    @property
    def ladder(self) -> Tuple[Optional[str], ...]:
        if not self.is_recent:
            return (None,)
        if self.strong_now:
            return LADDER_NOW
        if self.reasons and set(self.reasons) <= STATE_REASONS:
            return LADDER_STATE
        return LADDER_DEFAULT


def classify(query: str, *, today: Optional[date] = None) -> RecencyVerdict:
    """Decide whether ``query`` asks about the present.

    Deliberately biased toward RECENT: a false positive costs one narrower
    Brave call (widened automatically when thin) and a shorter cache TTL; a
    false negative reproduces the incident. Never raises.
    """
    try:
        q = " ".join((query or "").split())
        if not q:
            return RecencyVerdict(EVERGREEN)
        reasons: List[str] = []
        strong = False
        for reason, rx in _NOW_SIGNALS:
            if rx.search(q):
                reasons.append(reason)
                strong = True
        for reason, rx in _RECENT_SIGNALS:
            if rx.search(q):
                reasons.append(reason)
        today = today or datetime.now(timezone.utc).date()
        for m in _YEAR.finditer(q):
            year = int(m.group(1))
            if year in (today.year, today.year - 1):
                reasons.append("year")
                break
        if reasons:
            return RecencyVerdict(RECENT, reasons, strong_now=strong)
        return RecencyVerdict(EVERGREEN)
    except Exception:  # pragma: no cover — must never break a search
        return RecencyVerdict(EVERGREEN, ["classifier_error"])


def normalize_class(value: Optional[str]) -> Optional[str]:
    """Accept a caller-supplied class only if it is one of ours."""
    if not value:
        return None
    v = str(value).strip().lower()
    return v if v in FRESHNESS_CLASSES else None


# ── Query surgery ────────────────────────────────────────────────────

_SITE_OP = re.compile(r"(?:^|\s)(?:site|inurl|intitle):\S+", re.IGNORECASE)


def split_site_operator(query: str) -> Tuple[str, Optional[str]]:
    """Return ``(query without site:/inurl:/intitle: operators, first operator)``.

    A ``site:``-anchored query hands ranking to the site's internal relevance
    (the incident's ``site:anthropic.com/news newest Claude model`` returned
    Sonnet 5, Claude 4, Claude 3 …), so for recency intent the gateway also
    runs the neutral form. Whitespace is normalised; if the query is nothing
    but operators, the neutral form is empty and callers skip the extra call.
    """
    ops = _SITE_OP.findall(query or "")
    neutral = " ".join(_SITE_OP.sub(" ", query or "").split())
    return neutral, (ops[0].strip() if ops else None)


def with_year(query: str, *, today: Optional[date] = None) -> str:
    """Append the current year unless a 20xx year is already present."""
    if _YEAR.search(query or ""):
        return query
    today = today or datetime.now(timezone.utc).date()
    return f"{query} {today.year}".strip()


def brave_params(
    query: str,
    count: int,
    freshness: Optional[str],
    *,
    extra_snippets: bool = True,
    country: Optional[str] = None,
) -> Dict[str, Any]:
    """The upstream query string for ONE Brave web/news call."""
    params: Dict[str, Any] = {"q": query, "count": int(count)}
    if extra_snippets:
        params["extra_snippets"] = "true"
    if freshness:
        params["freshness"] = freshness
    if country:
        params["country"] = country
    return params


# ── Dates ────────────────────────────────────────────────────────────

_MONTHS = {
    m: i for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"], 1)
}
_ABS_DATE = re.compile(r"^\s*([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\s*$")
_REL_AGE = re.compile(r"^\s*(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago\s*$", re.IGNORECASE)
_ISO_PREFIX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def parse_page_date(value: Any, *, now: Optional[datetime] = None) -> Optional[date]:
    """Parse Brave's ``page_age`` (ISO) or ``age`` (absolute or relative).

    Returns a ``date`` or None. Relative ages ("3 weeks ago") are resolved
    against ``now`` (UTC default) and are approximate by nature — a month is
    taken as 30 days — which is far more precision than the caller needs to
    tell 2023 from last week.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    m = _ISO_PREFIX.match(s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = _ABS_DATE.match(s)
    if m:
        mon = _MONTHS.get(m.group(1).lower())
        if mon:
            try:
                return date(int(m.group(3)), mon, int(m.group(2)))
            except ValueError:
                return None
        return None
    m = _REL_AGE.match(s)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        days = {"second": 0, "minute": 0, "hour": 0, "day": 1, "week": 7, "month": 30, "year": 365}[unit]
        base = (now or datetime.now(timezone.utc)).date()
        return base - timedelta(days=n * days)
    return None


def result_date(result: Dict[str, Any], *, now: Optional[datetime] = None) -> Optional[date]:
    """Best available date for one Brave result: ``page_age`` (ISO, precise)
    first, then the human ``age`` string. Both are absent for undated pages."""
    return (
        parse_page_date(result.get("page_age"), now=now)
        or parse_page_date(result.get("age"), now=now)
    )


def age_days(result: Dict[str, Any], *, now: Optional[datetime] = None) -> Optional[int]:
    d = result_date(result, now=now)
    if d is None:
        return None
    base = (now or datetime.now(timezone.utc)).date()
    return max(0, (base - d).days)


def human_age(days: Optional[int]) -> str:
    if days is None:
        return "date unknown"
    if days < 1:
        return "today"
    if days == 1:
        return "1 day ago"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    if days < 730:
        return f"{days // 30} months ago"
    return f"{days // 365} years ago"


# ── Filtering / merging ─────────────────────────────────────────────


def filter_stale(
    results: Sequence[Dict[str, Any]],
    *,
    max_age_days: int = DEFAULT_STALE_DAYS,
    now: Optional[datetime] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop results dated older than ``max_age_days``. Undated results are
    KEPT — an unknown date is not evidence of staleness — and are annotated by
    the renderer as "date unknown" so the model treats them with care.
    Returns ``(kept, dropped_count)``."""
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for r in results:
        d = age_days(r, now=now)
        if d is not None and d > max_age_days:
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


_TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "ref", "ref_src", "mc_cid", "mc_eid")


def canonical_url(url: str) -> str:
    """Scheme-, www-, fragment-, tracking-param- and trailing-slash-insensitive
    key. Used for de-duplication only; the original URL is what gets shown."""
    try:
        parts = urlsplit((url or "").strip())
        host = (parts.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = (parts.path or "").rstrip("/")
        qs = [
            (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith(_TRACKING_PARAMS)
        ]
        return urlunsplit(("", host, path.lower(), urlencode(sorted(qs)), "")).lstrip("/")
    except Exception:
        return (url or "").strip().lower()


def merge_results(
    *sources: Iterable[Dict[str, Any]],
    limit: int,
    interleave: bool = True,
) -> List[Dict[str, Any]]:
    """Merge several ranked result lists into one, de-duplicated by canonical
    URL, capped at ``limit``.

    ``interleave=True`` round-robins across sources so each source's top hits
    survive the cap (source A #1, source B #1, source A #2 …). That is what a
    "newest X" question needs: the top of the official-site list AND the top
    of the news list AND the top of the neutral discovery list, rather than
    eight results from whichever source ran first. Ties keep source order.
    """
    lists = [list(s) for s in sources]
    seen: set = set()
    out: List[Dict[str, Any]] = []

    def _take(r: Dict[str, Any]) -> None:
        key = canonical_url(r.get("url") or "")
        if not key or key in seen:
            return
        seen.add(key)
        out.append(r)

    if interleave:
        idx = 0
        while len(out) < limit and any(idx < len(l) for l in lists):
            for l in lists:
                if idx < len(l):
                    _take(l[idx])
                    if len(out) >= limit:
                        break
            idx += 1
    else:
        for l in lists:
            for r in l:
                if len(out) >= limit:
                    break
                _take(r)
    return out[:limit]


def date_span(results: Sequence[Dict[str, Any]], *, now: Optional[datetime] = None) -> Tuple[Optional[str], Optional[str], int]:
    """(oldest_iso, newest_iso, undated_count) for one result page — the two
    numbers the F7 log line needs to say what the model was actually shown."""
    dates = [result_date(r, now=now) for r in results]
    known = [d for d in dates if d is not None]
    if not known:
        return None, None, len(dates)
    return min(known).isoformat(), max(known).isoformat(), len(dates) - len(known)
