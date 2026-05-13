"""
Page suggestions — the "wow moment" Haiku call.

When a user opens the Chrome sidepanel on any page, the extension hits
POST /api/extension/suggest with the current page_context. We ask Haiku
for 2–3 short contextual chips the user can tap to start a conversation
without typing anything.

The call is intentionally fast and cheap:
  - PINNED to Haiku via `model=` (NOT `model=None`). 2026-05-13 cost
    audit found the previous `model=None` fallback honored the user's
    `agent_model` setting — for GPT-5.5 users this turned a "Haiku
    call" into a ~$28/day leak on the platform OpenAI key. Pinning
    here makes the cost predictable regardless of who's using it.
  - logged as operation_type="system.extension.suggest" (exempt from
    user spend caps)
  - capped at ~200 output tokens
  - 8 second timeout — the sidepanel falls back to a generic greeting if
    we miss the SLA so a slow Haiku never blocks the UX
  - cached by (user_id, url, content_hash) for 15 min: re-firing
    `chrome.tabs.onActivated` for a tab the user already visited this
    quarter-hour returns the cached suggestion without a model call.
    Tab-switching is the dominant trigger, so cache hit rate is high.
  - per-user rate-limited (60 calls / 15 min) as a backstop in case a
    misbehaving client (or a bot pretending to be one) tries to burn
    through the budget anyway.

JSON schema we ask for:

  {
    "greeting": "I can see you're looking at <thing>. Want me to ...",
    "chips": [
      {"label": "Find better prices",  "prefill": "Find better prices for this product"},
      {"label": "Check reviews",       "prefill": "Summarize the reviews for this product"},
      {"label": "Compare alternatives","prefill": "Suggest cheaper alternatives to this"}
    ]
  }

A model that returns malformed JSON or no chips is logged and the caller
returns a sentinel "no suggestion" — the sidepanel renders the static
greeting "What do you want to do today?" in that case.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from app.services.internal_llm import call_system_llm
from app.services.page_context_render import render_page_context

logger = logging.getLogger(__name__)

# Pinned model for /suggest. Cheapest Anthropic model that gives
# acceptable JSON quality for this task. Hardcoded here rather than
# pulling from settings so it survives an env-var nudge.
_SUGGEST_MODEL = "claude-haiku-4-5-20251001"

# Trim readable_content to this many chars before rendering the context
# block. The renderer's own cap is 8000 (good for the chat path); for
# /suggest we don't need that much — 2500 chars is enough to identify
# what the page is about for a 2-chip suggestion.
_SUGGEST_READABLE_CAP = 2500

# In-memory cache: key = (user_id, content_hash) → (result, expires_at).
# Bounded to 2k entries with FIFO eviction so a runaway extension
# can't OOM the process. Hit rate is the main lever here — most cost
# savings come from avoiding the model call entirely.
_CACHE: Dict[Tuple[str, str], Tuple[Optional[Dict[str, Any]], float]] = {}
_CACHE_ORDER: Deque[Tuple[str, str]] = deque()
_CACHE_MAX = 2_000
_CACHE_TTL_S = 15 * 60  # 15 minutes

# Per-user sliding window rate limit: max N calls per WINDOW_S seconds.
# A user opening sidepanel on 60 different pages in 15 minutes is the
# upper end of reasonable; beyond that we return None (the sidepanel
# falls back to its static greeting, no model call).
_RATELIMIT_MAX = 60
_RATELIMIT_WINDOW_S = 15 * 60
_USER_HITS: Dict[str, Deque[float]] = {}


def _now() -> float:
    return time.monotonic()


def _content_hash(url: str, body: str) -> str:
    """Cache key — URL + first ~1000 chars of readable. Catches
    re-loads of the same page even when the URL has cache-busting
    query params, but treats a substantively different page (different
    article on the same template) as a cache miss."""
    raw = (url or "")[:512] + "\x1f" + (body or "")[:1000]
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()


def _cache_get(user_id: str, key_hash: str) -> Optional[Dict[str, Any]]:
    entry = _CACHE.get((user_id, key_hash))
    if not entry:
        return None
    val, exp = entry
    if exp < _now():
        # Stale — evict and miss.
        _CACHE.pop((user_id, key_hash), None)
        return None
    return val


def _cache_set(user_id: str, key_hash: str, val: Optional[Dict[str, Any]]) -> None:
    while len(_CACHE_ORDER) >= _CACHE_MAX:
        old = _CACHE_ORDER.popleft()
        _CACHE.pop(old, None)
    k = (user_id, key_hash)
    if k not in _CACHE:
        _CACHE_ORDER.append(k)
    _CACHE[k] = (val, _now() + _CACHE_TTL_S)


def _rate_limited(user_id: str) -> bool:
    """Drop the oldest hit if it's outside the window, then check the
    count. Returns True if the user has spent their quota for this
    window — caller should return None without invoking the model."""
    q = _USER_HITS.setdefault(user_id, deque())
    cutoff = _now() - _RATELIMIT_WINDOW_S
    while q and q[0] < cutoff:
        q.popleft()
    if len(q) >= _RATELIMIT_MAX:
        return True
    q.append(_now())
    return False

# Block "wow moment" for hosts that almost always mean the user is on
# private/sensitive territory and a chirpy bot popping up feels wrong.
_PRIVACY_BLOCK_FRAGMENTS = (
    "bankofamerica.com", "chase.com", "wellsfargo.com", "paypal.com",
    "stripe.com", "coinbase.com", "1password.com", "bitwarden.com",
    "lastpass.com", "/login", "/signin", "/sign-in", "/checkout",
    "/admin/", "/account/password", "/2fa",
)


def _is_blocked(url: Optional[str]) -> bool:
    if not url: return True
    u = url.lower()
    if u.startswith(("about:", "chrome://", "edge://", "view-source:", "file:")):
        return True
    return any(frag in u for frag in _PRIVACY_BLOCK_FRAGMENTS)


_SYSTEM = (
    "You are a concise assistant that surfaces 2–3 useful actions a user "
    "might want to take on the page they're currently viewing in their "
    "browser. Your output goes into the user's AI sidepanel as tappable "
    "chips, BEFORE they type anything.\n"
    "\n"
    "Rules:\n"
    " - Output JSON only: {greeting, chips:[{label, prefill}, ...]} — no prose around it.\n"
    " - `greeting`: ONE short sentence (max 110 chars) that names what the page is and what you can offer.\n"
    " - `chips`: 2–3 entries.\n"
    "     * `label`: 3–5 words, action-style ('Find better prices', not 'Prices').\n"
    "     * `prefill`: the EXACT user message we'll send when the user taps the chip — first person, complete sentence, ≤ 140 chars.\n"
    " - Do not invent specifics not present in the page (no fabricated prices/dates).\n"
    " - If the page looks generic (search results, homepage, blank tab), keep greeting and chips broad.\n"
    " - Never suggest actions that require credentials the user hasn't shared.\n"
    " - Never recommend purchasing, hitting 'submit', or any action with money/legal consequences.\n"
)


async def suggest_for_page(
    *,
    user_id: str,
    page_context: Dict[str, Any],
    timeout_s: float = 8.0,
) -> Optional[Dict[str, Any]]:
    """
    Return a dict {greeting, chips} or None if we should skip the
    proactive greeting (privacy block / model failure / timeout /
    cache miss for a privacy-restricted page).
    """
    url = page_context.get("url")
    if _is_blocked(url):
        logger.debug("[suggest] skip — privacy block (%s)", url)
        return None

    # Honor sidepanel privacy toggles the same way ws_chat does. Without
    # this, /suggest would see readable/selection/dom even when the user
    # disabled them in Options.
    _flags = page_context.get("_flags") or {}
    _filtered = dict(page_context)
    if _flags.get("hide_readable", False):
        _filtered["readable_content"] = ""
    if _flags.get("hide_selection", False):
        _filtered["selected_text"] = ""
    if _flags.get("hide_dom", False):
        _filtered["dom_summary"] = ""

    # Cost lever #1: trim readable_content BEFORE rendering so the
    # context block is small. The chat path's 8k cap was overkill for
    # a 2-chip greeting; 2.5k is plenty to identify the page.
    if _filtered.get("readable_content"):
        _filtered["readable_content"] = str(_filtered["readable_content"])[:_SUGGEST_READABLE_CAP]

    # Cost lever #2: cache by (user, content_hash). `chrome.tabs.onActivated`
    # fires on every tab switch — including switching back to a tab the
    # user already had a suggestion for. Cache hits avoid the model call
    # entirely. TTL 15 min so a page that changed substantially still
    # eventually re-fires.
    cache_key = _content_hash(
        url or "",
        (_filtered.get("readable_content") or "") + "|" + (_filtered.get("selected_text") or ""),
    )
    cached = _cache_get(user_id, cache_key)
    if cached is not None:
        logger.debug("[suggest] cache HIT user=%s url=%s", user_id, (url or "")[:80])
        return cached

    # Cost lever #3: rate limit before the LLM call. A misbehaving
    # client (or a bot) can't drive cost into the floor.
    if _rate_limited(user_id):
        logger.info("[suggest] rate-limited user=%s (>%d in %ds)", user_id, _RATELIMIT_MAX, _RATELIMIT_WINDOW_S)
        return None

    # Render the same hidden block we send into the chat — same view of
    # the page the agent would have.
    ctx_block = render_page_context(_filtered)
    if not ctx_block:
        _cache_set(user_id, cache_key, None)  # cache the "nothing to suggest"
        return None

    # Phase 6: fold in recent browsing memory so suggestions feel
    # continuous across days ("you looked at Sony headphones 3 days ago…").
    recent_block = ""
    try:
        from app.services.browsing_memory import recent_compact
        rc = await recent_compact(user_id, limit=8)
        if rc:
            recent_block = "\n\n[RECENT_BROWSING]\n" + rc + "\n[/RECENT_BROWSING]"
    except Exception:
        recent_block = ""

    user_msg = (
        ctx_block
        + recent_block
        + "\n\nReturn the JSON object as specified — nothing else."
    )

    try:
        raw = await asyncio.wait_for(
            call_system_llm(
                user_id=user_id,
                operation_type="system.extension.suggest",
                # Cost lever #4: PIN to Haiku. `model=None` honored the
                # user's chat model — for GPT-5.5 users this turned the
                # "Haiku call" into a $5/1M-input pipe. Hardcoded here
                # so cost is predictable regardless of who's using it.
                model=_SUGGEST_MODEL,
                max_tokens=200,
                system=_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                timeout=int(timeout_s),
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.info("[suggest] timed out for %s on %s", user_id, url)
        return None
    except Exception as exc:
        logger.warning("[suggest] LLM call failed: %s", exc)
        return None

    if not raw:
        _cache_set(user_id, cache_key, None)
        return None

    parsed = _safe_parse_json(raw)
    if not parsed:
        logger.info("[suggest] malformed JSON from model: %r", raw[:200])
        _cache_set(user_id, cache_key, None)
        return None

    out = _validate(parsed)
    if not out:
        logger.info("[suggest] validation failed for: %r", parsed)
        _cache_set(user_id, cache_key, None)
        return None

    _cache_set(user_id, cache_key, out)
    return out


# ─────────────────────────────────────────────────────────────────────
# Parsing / validation
# ─────────────────────────────────────────────────────────────────────
def _safe_parse_json(s: str) -> Optional[Any]:
    if not s: return None
    # Models sometimes emit ```json … ``` fences; strip them.
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Last-ditch: find the first {...} object.
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except json.JSONDecodeError: return None
        return None


def _validate(d: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(d, dict): return None
    greeting = d.get("greeting")
    chips    = d.get("chips")
    if not isinstance(greeting, str) or not greeting.strip(): return None
    if not isinstance(chips, list) or not chips: return None

    out_chips: List[Dict[str, str]] = []
    for c in chips:
        if not isinstance(c, dict): continue
        label   = (c.get("label")   or "").strip()
        prefill = (c.get("prefill") or "").strip()
        if not label or not prefill: continue
        out_chips.append({"label": label[:60], "prefill": prefill[:280]})
        if len(out_chips) >= 4: break
    if not out_chips: return None

    return {"greeting": greeting.strip()[:200], "chips": out_chips}
