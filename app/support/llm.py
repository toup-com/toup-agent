"""LLM helper for the support agent.

Wraps ``call_system_llm`` (system operation — exempt from user credit caps,
uses the platform default model when ``model`` is None — never pins Claude
while Anthropic is deactivated) and robustly extracts a JSON object from the
reply (tolerates ```json fences / surrounding prose).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from app.config import settings
from app.services.internal_llm import call_system_llm

logger = logging.getLogger("support.llm")

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _support_model() -> Optional[str]:
    """Model for support ops (config default gpt-4o-mini); None/"" ⇒
    platform default (settings.agent_model, gpt-5.5)."""
    return getattr(settings, "support_agent_model", None) or None


def _extract_json(text: str):
    if not text:
        return None
    m = _FENCE_RE.search(text)
    candidate = m.group(1) if m else None
    if candidate is None:
        # First {...} or [...] span in the text.
        start = min(
            (i for i in (text.find("{"), text.find("[")) if i != -1),
            default=-1,
        )
        if start == -1:
            return None
        candidate = text[start:]
    # Try progressively shorter suffixes ending at the last brace/bracket.
    for end_char in ("}", "]"):
        end = candidate.rfind(end_char)
        if end != -1:
            try:
                return json.loads(candidate[: end + 1])
            except Exception:
                continue
    try:
        return json.loads(candidate)
    except Exception:
        return None


async def ask_json(
    *,
    user_id: str,
    operation_type: str,
    system: str,
    prompt: str,
    max_tokens: int = 1500,
    timeout: int = 60,
) -> Optional[dict]:
    """Call the system LLM and parse a JSON object/array reply (or None)."""
    text = await call_system_llm(
        user_id=user_id,
        operation_type=operation_type,
        max_tokens=max_tokens,
        system=system + "\n\nReply with ONLY a single JSON object. No prose, no markdown fences.",
        messages=[{"role": "user", "content": prompt}],
        model=_support_model(),
        timeout=timeout,
    )
    if text is None:
        logger.warning("[support.llm] %s returned None (LLM unavailable)", operation_type)
        return None
    parsed = _extract_json(text)
    if parsed is None:
        logger.warning("[support.llm] %s: could not parse JSON from reply", operation_type)
    return parsed
