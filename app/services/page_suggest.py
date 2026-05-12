"""
Page suggestions — the "wow moment" Haiku call.

When a user opens the Chrome sidepanel on any page, the extension hits
POST /api/extension/suggest with the current page_context. We ask Haiku
for 2–3 short contextual chips the user can tap to start a conversation
without typing anything.

The call is intentionally fast and cheap:
  - logged as operation_type="system.extension.suggest" (exempt from
    user spend caps)
  - capped at ~200 output tokens
  - 8 second timeout — the sidepanel falls back to a generic greeting if
    we miss the SLA so a slow Haiku never blocks the UX

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
import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.services.internal_llm import call_system_llm
from app.services.page_context_render import render_page_context

logger = logging.getLogger(__name__)

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
    proactive greeting (privacy block / model failure / timeout).
    """
    url = page_context.get("url")
    if _is_blocked(url):
        logger.debug("[suggest] skip — privacy block (%s)", url)
        return None

    # Render the same hidden block we send into the chat — same view of
    # the page the agent would have.
    ctx_block = render_page_context(page_context)
    if not ctx_block:
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
                model=None,            # honor user's default model resolver
                max_tokens=300,
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
        return None

    parsed = _safe_parse_json(raw)
    if not parsed:
        logger.info("[suggest] malformed JSON from model: %r", raw[:200])
        return None

    out = _validate(parsed)
    if not out:
        logger.info("[suggest] validation failed for: %r", parsed)
        return None
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
