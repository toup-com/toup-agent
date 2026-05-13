"""
Vision-based action grounding.

When the agent wants to click/type/hover but no reliable selector is
available (canvas, PDF viewer, shadow DOM, obfuscated React class
names), the extension sends us a screenshot and a textual description
of the target. We ask a vision model where to click and return
{x, y, confidence, element_description}.

The model gets:
  - the page screenshot (already JPEG-encoded by the extension)
  - the viewport dimensions (we normalize to image pixels in case Chrome
    captured at a different DPR than the agent's mental model)
  - the textual target description from the agent

We deliberately keep the output schema tight so a half-confident model
doesn't ship low-quality coordinates that misclick. Confidence below
GROUNDING_MIN_CONFIDENCE results in a `grounding_failed` reply — the
agent then re-describes the target instead of acting on guesswork.

Cost containment: same `system.extension.suggest` exemption category so
this doesn't bill against the user's per-conversation budget. We cap
the model at 200 output tokens; the response is one JSON object.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from app.services.internal_llm import call_system_llm

logger = logging.getLogger(__name__)

GROUNDING_MIN_CONFIDENCE = 0.7
GROUNDING_TIMEOUT_S = 12.0

# The vision model needs an image content block. internal_llm.call_system_llm
# accepts the standard Anthropic messages shape, so we attach an image
# part using the `source: {type: "base64", ...}` convention.

_SYSTEM = (
    "You are a precise visual UI grounder. You will see a screenshot of a web "
    "page and a one-sentence description of an element the user wants to "
    "interact with. Return exactly one JSON object describing the most likely "
    "pixel coordinate of that element's clickable center, plus your "
    "confidence and a short description of what's at those coordinates.\n"
    "\n"
    "Output ONLY a JSON object. No prose, no markdown fences.\n"
    "Schema:\n"
    '  { "x": <int>, "y": <int>, "confidence": <0..1 float>, '
    '"element_description": "<≤80 chars>" }\n'
    "\n"
    "Rules:\n"
    " - x,y are in image pixel coordinates (top-left = 0,0).\n"
    " - Pick the geometric center of the target's visually clickable area.\n"
    " - If multiple candidates exist, pick the most prominently visible one.\n"
    " - If you cannot see a plausible target, return confidence ≤ 0.3 and a\n"
    "   description explaining what's missing.\n"
    " - Never return coordinates outside the image bounds (the request\n"
    "   includes width/height of the image — stay inside them)."
)


async def ground(
    *,
    user_id: str,
    screenshot_b64: str,
    image_width: int,
    image_height: int,
    target_description: str,
) -> Dict[str, Any]:
    """
    Returns a dict with one of two shapes:

      success:   {"ok": True,  "x": int, "y": int, "confidence": float,
                  "element_description": str}
      failure:   {"ok": False, "reason": "<short>", "confidence": float}
    """
    target = (target_description or "").strip()
    if not target:
        return {"ok": False, "reason": "empty target description", "confidence": 0.0}

    if not screenshot_b64 or not isinstance(screenshot_b64, str):
        return {"ok": False, "reason": "missing screenshot", "confidence": 0.0}
    if len(screenshot_b64) > 2_000_000:
        # Defensive cap — a JPEG@70 of a 1920x1080 viewport encodes well
        # under 1.5 MB base64. Anything larger is wasteful for grounding
        # and risks memory pressure under concurrent requests.
        return {"ok": False, "reason": "screenshot too large", "confidence": 0.0}

    user_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": screenshot_b64,
            },
        },
        {
            "type": "text",
            "text": (
                f"Image dimensions: width={image_width}px, height={image_height}px.\n"
                f"Target: {target}\n"
                "Return the JSON object as specified."
            ),
        },
    ]

    try:
        raw = await asyncio.wait_for(
            call_system_llm(
                user_id=user_id,
                operation_type="system.extension.ground",
                # Pinned to Haiku (vision-capable since Claude 3.5 Haiku).
                # The previous `model=None` honored the user's chat model
                # — for GPT-5.5/Opus tenants every click on a Notion/Gmail
                # surface billed at $5-15 per 1M input + a ~1k-token
                # screenshot per call. Haiku 4.5 vision is plenty for
                # "where is this UI element on this screenshot". Pinning
                # here so the agentic-browse loop has a predictable
                # per-step cost regardless of who's driving.
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                system=_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
                timeout=int(GROUNDING_TIMEOUT_S),
            ),
            timeout=GROUNDING_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.info("[vision-ground] timeout for %s", user_id)
        return {"ok": False, "reason": "timeout", "confidence": 0.0}
    except Exception as exc:
        logger.warning("[vision-ground] LLM call failed: %s", exc)
        return {"ok": False, "reason": f"llm_error: {exc}", "confidence": 0.0}

    if not raw:
        return {"ok": False, "reason": "empty response", "confidence": 0.0}

    parsed = _safe_parse_json(raw)
    if not isinstance(parsed, dict):
        return {"ok": False, "reason": "malformed_json", "confidence": 0.0}

    coords = _coerce_coords(parsed, image_width, image_height)
    if coords is None:
        return {"ok": False, "reason": "no_coords", "confidence": float(parsed.get("confidence") or 0.0)}

    x, y = coords
    confidence = float(parsed.get("confidence", 0.0))
    if confidence < GROUNDING_MIN_CONFIDENCE:
        return {
            "ok": False,
            "reason": "low_confidence",
            "confidence": confidence,
            "element_description": str(parsed.get("element_description") or "")[:200],
        }

    return {
        "ok": True,
        "x": x,
        "y": y,
        "confidence": confidence,
        "element_description": str(parsed.get("element_description") or "")[:200],
    }


def _safe_parse_json(s: str) -> Optional[Any]:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except json.JSONDecodeError: return None
    return None


def _coerce_coords(d: Dict[str, Any], w: int, h: int) -> Optional[Tuple[int, int]]:
    try:
        x = int(round(float(d["x"])))
        y = int(round(float(d["y"])))
    except (KeyError, TypeError, ValueError):
        return None
    # Clamp to the image bounds.
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    return (x, y)
