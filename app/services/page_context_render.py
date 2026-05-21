"""
Render the Chrome sidepanel's `page_context` payload into a short hidden
preamble that the agent sees prepended to every user message.

The sidepanel injects this on every turn so the agent has automatic
awareness of the page the user is looking at — they never need to say
"look at this page", they just talk.

Shape of the incoming dict (best-effort; all fields optional):

    {
      "url":               "https://...",
      "title":             "...",
      "selected_text":     "...",            # may be empty
      "readable_content":  "...",            # up to ~8k chars, optional
      "dom_summary":       "...",            # very short structural summary
      "extension_version": "3.0.0",
      "captured_at":       1715520000
    }

Truncation policy (so we don't blow the context window):
  - title:            120 chars
  - url:              350 chars
  - selected_text:    1200 chars (the user explicitly highlighted this)
  - readable_content: 8000 chars
  - dom_summary:      400 chars

Returns "" if nothing meaningful is present — caller treats that as
"don't inject anything", so the agent stays clean for messages where
the user isn't on any page yet (e.g. talking to it from chrome://newtab).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# TKT-LAT-013: extension page-context bloat. The previous
# readable_content cap (8000 chars ≈ ~2000 tokens) prepended on every
# turn ate 6–8 % of the model context window for sidepanel users —
# whether they were asking about the page or not. The settings flag
# extension_page_context_compact (default ON) trims readable_content
# to a tighter cap so the agent still gets a useful page snapshot
# without dominating the prompt; users who want the legacy verbose
# version flip the flag off per tenant. The agent can still pull
# more of the page on demand via extension_read /
# browser_action(kind=dom_snapshot) — the system-prompt note at
# the bottom of this block already tells it how.
_LIMITS_COMPACT = {
    "title":            120,
    "url":              350,
    "selected_text":    1200,
    "readable_content": 2000,   # was 8000 — see TKT-LAT-013
    "dom_summary":      400,
}

_LIMITS_VERBOSE = {
    "title":            120,
    "url":              350,
    "selected_text":    1200,
    "readable_content": 8000,
    "dom_summary":      400,
}


def _active_limits() -> Dict[str, int]:
    """Pick truncation limits based on the per-tenant compaction flag."""
    try:
        from app.config import settings
        if getattr(settings, "extension_page_context_compact", True):
            return _LIMITS_COMPACT
    except Exception:
        # Settings not loaded yet (unit tests / agent boot edge): default
        # to the compact policy. Verbose mode is opt-in, not the default.
        pass
    return _LIMITS_COMPACT


# Back-compat alias — existing imports still work.
_LIMITS = _LIMITS_COMPACT


def _trunc(s: Any, n: int) -> str:
    s = ("" if s is None else str(s)).strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def render_page_context(payload: Optional[Dict[str, Any]]) -> str:
    if not payload or not isinstance(payload, dict):
        return ""

    limits = _active_limits()
    url   = _trunc(payload.get("url"),   limits["url"])
    title = _trunc(payload.get("title"), limits["title"])
    sel   = _trunc(payload.get("selected_text"),    limits["selected_text"])
    body  = _trunc(payload.get("readable_content"), limits["readable_content"])
    dom   = _trunc(payload.get("dom_summary"),      limits["dom_summary"])

    # If the user is on chrome://newtab or about:blank we don't bother.
    if not url or url.startswith(("about:", "chrome://", "edge://", "view-source:")):
        return ""

    lines = ["[PAGE_CONTEXT]"]
    if title: lines.append(f"title: {title}")
    if url:   lines.append(f"url:   {url}")
    if dom:   lines.append(f"dom:   {dom}")
    if sel:
        lines.append("selected_text: |")
        for ln in sel.splitlines():
            lines.append(f"  {ln}")
    if body:
        lines.append("readable_content: |")
        for ln in body.splitlines():
            lines.append(f"  {ln}")
    lines.append("[/PAGE_CONTEXT]")
    lines.append("")
    lines.append(
        "Note: the block above is hidden ambient context describing the page the user is "
        "currently viewing in their Chrome. They may or may not be asking about it — use "
        "your judgment. If you need more of the page than is included here, call "
        "extension_read or browser_action(kind=dom_snapshot) against the same URL. "
        "Do not echo this block back to the user."
    )
    return "\n".join(lines)
