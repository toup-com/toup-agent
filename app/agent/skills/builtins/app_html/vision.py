"""Somebody looks at the app before the user does.

Round 20, item 3. Round 18 put a real browser in front of the publish gate,
which ended the class where a card said "100% · App built" over a page that
threw on line one. It cannot end the class next door: a page that throws
nothing, measures fine, and is unusable. White text on a white panel. A score
readout clipped by its own container. A modal that opens behind the board. An
empty grey box where a chart was meant to be. Every one of those passes
`node --check`, passes `pageerror`, passes the 44 px rule, and is the first
thing a person sees.

The only instrument that catches those is an eye, so the gate grows one: the
smoke run takes a PNG, and a vision model is asked the one question the other
passes cannot ask — *does this look broken?*

Three properties this has to have, in order of how badly it goes wrong
without them:

1.  **It must be honest about not having run.** No key, no vision model, a
    timeout: any of those returns ``ran=False``, and the step says it could
    not look. `Look.ran` exists so the caller can never read silence as
    approval. This is the round-18 lesson restated one layer up — a green
    check from machinery that never looked is worse than no check.
2.  **It must not be trigger-happy.** A reviewer that objects to the palette
    refuses every publish, and a gate that refuses everything gets turned
    off. The prompt asks for defects a person would call broken, each one
    named and located; taste is explicitly out of scope.

    Round 22 adds ONE judgement to that list and it is not taste: does the
    palette **contradict the app's stated purpose**. The distinction is the
    whole reason it is safe to ask. *Is this a nice orange* is taste and stays
    out of scope forever. *This is a children's counting game and it is
    black* is a mismatch anyone would name in one sentence without being
    asked, and it shipped seventeen times out of twenty-five because nothing
    in the pipeline was allowed to mention it. The bar is deliberately set at
    "gross and nameable", the reviewer is told to pass when unsure, and it is
    only ever asked when `purpose` is non-empty — with no stated purpose there
    is nothing to contradict, and a reviewer guessing the domain from a
    screenshot is exactly the trigger-happy failure above.
3.  **The user must never see it.** The findings are the model's repair list.
    They go into the tool result, which is model-facing, and the step's
    detail line says only how many there are.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Pinned, never `None`. `call_system_llm(model=None)` resolves to the
#: tenant's CHAT model — an Opus or a gpt-5.x — and this runs on every
#: publish with a 390×844 screenshot attached. gpt-4o-mini is vision-capable,
#: is what the rest of the background plumbing uses (`vision_grounding`,
#: `internal_llm`'s own Anthropic fallback), and is the right size for "look
#: at this and tell me what is broken".
REVIEW_MODEL = "gpt-4o-mini"

#: Wall clock for the whole look, including the model call. A publish must
#: not be held up longer than a person will wait, and a slow reviewer is a
#: reviewer that did not run rather than one that approved.
REVIEW_TIMEOUT_S = 25

#: Most problems to act on in one round. A screen can be wrong in many ways
#: at once; six is enough to fix a real mess and few enough that the model
#: gets a repair list rather than an essay.
MAX_PROBLEMS = 6

#: Screenshots are PNG at 390×844 — typically 40–120 KB, ~55–160 KB base64.
#: The cap is a guard against an app that paints a full-bleed photograph, not
#: a budget.
MAX_SCREENSHOT_BYTES = 4 * 1024 * 1024


_SYSTEM = (
    "You are reviewing a screenshot of a small self-contained web app, taken "
    "on a 390x844 phone screen. You are the last check before it is shown to "
    "the person who asked for it.\n"
    "\n"
    "Report ONLY defects a person would call broken on sight:\n"
    " - text that cannot be read: too small, or too close in colour to what "
    "is behind it\n"
    " - anything clipped, cut off at an edge, or overflowing its container\n"
    " - elements overlapping or sitting on top of each other\n"
    " - an empty box, blank panel or missing region where content clearly "
    "belongs\n"
    " - controls crushed together, overlapping, or pushed off screen\n"
    " - a layout that has collapsed: everything in one column at the top, "
    "content behind a bar, huge empty areas with content squeezed into a "
    "corner\n"
    " - placeholder text that was never replaced (lorem ipsum, TODO, "
    "'Your text here', 'undefined', 'NaN', '[object Object]')\n"
    " - a control cluster that has lost its symmetry: sibling controls at "
    "visibly different heights or off a shared baseline, a D-pad or keypad "
    "off-centre in its own area, one control of a group floating apart from "
    "the bar or cluster it belongs to (a lone sound button at a third "
    "height while the hint text and the D-pad sit elsewhere is the "
    "canonical case)\n"
    " - a stray mark that belongs to nothing: an orphan glyph or symbol "
    "sitting on the playfield or at a screen edge with no label and no "
    "function you can infer\n"
    " - instructions for hardware the device does not have: keyboard hints "
    "(WASD, arrow keys, spacebar) rendered as UI — this is a touch "
    "screen\n"
    "\n"
    "\n"
    "AND ONE JUDGEMENT ABOUT COLOUR, only when you have been told what the "
    "app is meant to be:\n"
    " - the palette CONTRADICTS the app's stated purpose — a children's game "
    "that opens sombre and near-black, a gym or sports tracker that looks "
    "funereal rather than energising, a sleep or wind-down app that is loud "
    "and bright, a cooking app that looks clinical. Report it as: what the "
    "app is for, what the palette says instead, and the direction to move "
    "in.\n"
    " - The bar is HIGH. Only report this when the mismatch is gross and you "
    "could name it in one sentence to the person who asked for the app. If "
    "you are weighing whether the hue is quite right, that is taste — say "
    "nothing. If you were not told what the app is meant to be, say nothing "
    "about colour at all.\n"
    "\n"
    "Do NOT report matters of taste. Whether the palette is attractive, the "
    "font choice, the "
    "amount of whitespace, whether you would have laid it out differently, "
    "whether it needs more features — none of those are defects. A plain app "
    "that is legible, correctly laid out, and not at odds with its purpose "
    "is a PASS.\n"
    "An app showing a start screen, a title and one button is a normal, "
    "correct state — not an empty screen.\n"
    "\n"
    "Answer with ONE JSON object and nothing else. No prose, no markdown "
    "fences.\n"
    '{"ok": true}   when nothing above is wrong\n'
    '{"ok": false, "problems": ["<what is wrong, WHERE on the screen, and '
    'what to change>", ...]}\n'
    "Each problem is one sentence naming the element and its location, so it "
    "can be found in the source. At most six."
)


@dataclass
class Look:
    """What came back from looking at the app — or why nothing did."""

    #: True only if a model actually saw the image and answered.
    ran: bool = False
    problems: List[str] = field(default_factory=list)
    #: Why it did not run. Empty when it did.
    reason: str = ""
    #: Set when the reviewer was also asked to confirm a change landed.
    change_confirmed: Optional[bool] = None

    @property
    def ok(self) -> bool:
        return self.ran and not self.problems

    def summary(self) -> str:
        """One line for the step's detail. Never claims a look that did not
        happen — the same rule `verify.Report.summary` follows."""
        if not self.ran:
            return "couldn't look at it here"
        if self.problems:
            n = len(self.problems)
            return f"found {n} thing{'' if n == 1 else 's'} to fix on screen"
        return "looks right"

    def as_error(self) -> str:
        return "\n".join(f"  - {p}" for p in self.problems)


def enabled() -> bool:
    """Off only if an operator turns it off, like the other two gates."""
    return (os.environ.get("TOUP_APP_VISUAL_REVIEW", "1") or "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def can_call_model() -> bool:
    """Is there any chance a model call succeeds from this process?

    Asked BEFORE every model call in this pipeline — the visual review, the
    icon, the backfill — because the alternative is a publish that pays a
    30-second timeout to find out the answer is no. On a container with no
    credential that is 30 seconds added to every single build, spent
    discovering the same fact each time.

    Three ways to reach a model, and this is deliberately the *cheap* check
    rather than a real one: bundle mode (the agent's normal path, authorised
    by ``TOUP_TOKEN``), or a key of either provider. A True here is "worth
    trying", never "will work" — the callers all degrade honestly on failure,
    which is what makes a cheap check safe.

    ``TOUP_APP_MODEL_CALLS=0`` turns the whole class off for an operator who
    wants a build pipeline that touches nothing external.
    """
    if (os.environ.get("TOUP_APP_MODEL_CALLS", "1") or "1").strip().lower() in (
        "0", "false", "no", "off",
    ):
        return False
    try:
        from app.config import settings
    except Exception:  # pragma: no cover
        return False
    if (getattr(settings, "llm_mode", None) == "bundle"
            and getattr(settings, "toup_token", None)):
        return True
    return bool(
        getattr(settings, "platform_openai_api_key", None)
        or getattr(settings, "openai_api_key", None)
        or getattr(settings, "platform_anthropic_api_key", None)
        or getattr(settings, "anthropic_api_key", None)
    )


def image_block(png: bytes, model: str) -> Dict[str, Any]:
    """One image content block, in the shape the RESOLVED PROVIDER accepts.

    This is not a detail. The two wire formats are not interchangeable —
    OpenAI's chat completions API rejects Anthropic's
    ``{"type": "image", "source": {...}}`` outright — and
    ``internal_llm.call_system_llm`` passes ``messages`` through verbatim to
    whichever provider the model id resolves to. Getting it wrong does not
    degrade the review; it turns every call into a 400 that the caller sees
    as a generic failure, i.e. a gate that is permanently and silently down.
    """
    b64 = base64.b64encode(png).decode("ascii")
    try:
        from app.services.model_resolver import is_openai_model
        openai = bool(is_openai_model(model))
    except Exception:  # pragma: no cover - resolver must not break the gate
        openai = model.lower().startswith(("gpt-", "o1", "o3", "o4"))
    if openai:
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": b64},
    }


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_verdict(raw: str) -> Optional[Dict[str, Any]]:
    """The reviewer's JSON, or None if it did not answer in JSON.

    Split out so the parsing rules can be tested without a model — the same
    reason `verify.counts_as_breakage` is its own function. A reviewer that
    returns prose is a reviewer that did not answer, and must degrade to
    ``ran=False`` rather than to an empty (i.e. approving) problem list.
    """
    text = (raw or "").strip()
    if not text:
        return None
    text = _FENCE_RE.sub("", text).strip()
    # A model that prefixes a sentence still usually emits one object; take
    # the outermost braces rather than failing the whole look over a preamble.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _problems_from(data: Dict[str, Any]) -> List[str]:
    if data.get("ok") is True:
        return []
    raw = data.get("problems")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        # `{"ok": false}` with no list is a reviewer that flagged something
        # and did not say what. That is not a pass, and it is not actionable
        # either — treat it as one unnamed problem so the model looks again
        # rather than publishing on it.
        return ["the screen has a visible problem the review could not name — "
                "open the app's own layout and check nothing is clipped, "
                "empty or unreadable"] if data.get("ok") is False else []
    out: List[str] = []
    for item in raw:
        line = " ".join(str(item).split())[:300]
        if line and line not in out:
            out.append(line)
        if len(out) >= MAX_PROBLEMS:
            break
    return out


async def review_screenshot(
    png: Optional[bytes],
    *,
    user_id: str,
    title: str,
    purpose: str = "",
    change: str = "",
) -> Look:
    """Look at the app and say what is wrong with it.

    ``purpose`` is one line from the app's own brief, so the reviewer knows
    what it is supposed to be — an empty board is right for a chess app and
    wrong for a dashboard. ``change`` is the edit that was just made, which
    turns the same call into "is this change on screen and does it look
    right", the second half of item 3.

    Never raises. Every failure path returns ``ran=False`` with a reason.
    """
    look = Look()
    if not enabled():
        look.reason = "visual review is switched off"
        return look
    if not can_call_model():
        look.reason = "no model is reachable from here"
        return look
    if not png:
        look.reason = "no screenshot was captured"
        return look
    if len(png) > MAX_SCREENSHOT_BYTES:
        look.reason = f"screenshot is {len(png)} bytes, too large to review"
        return look

    asks = [f"This is “{title or 'the app'}”."]
    if purpose:
        asks.append(f"It is meant to be: {purpose}")
    if change:
        asks.append(
            f"A change was just made: {change}. Say whether that change is "
            f"visible on this screen and looks right — if it is not there, "
            f"or looks wrong, that is a problem to report. Also add "
            f'"change_visible": true/false to your JSON.'
        )
    asks.append("Report only real defects, as specified. Answer with the JSON object.")

    content = [image_block(png, REVIEW_MODEL), {"type": "text", "text": "\n".join(asks)}]

    try:
        from app.services.internal_llm import call_system_llm
        raw = await asyncio.wait_for(
            call_system_llm(
                user_id=user_id or "",
                operation_type="system.app_html.visual_review",
                model=REVIEW_MODEL,
                max_tokens=500,
                system=_SYSTEM,
                messages=[{"role": "user", "content": content}],
                timeout=REVIEW_TIMEOUT_S,
            ),
            timeout=REVIEW_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        look.reason = "the review timed out"
        logger.warning("[app_html] visual review timed out for %r", title)
        return look
    except Exception as exc:  # noqa: BLE001 - a missing model must not fail a build
        look.reason = f"the review could not run ({type(exc).__name__})"
        logger.warning("[app_html] visual review could not run", exc_info=True)
        return look

    data = parse_verdict(raw or "")
    if data is None:
        # WARNING, not debug: a container whose reviewer never answers has
        # silently lost this gate, and the only way anyone finds out is if
        # the line is loud enough to notice.
        look.reason = "the review did not answer"
        logger.warning(
            "[app_html] visual review returned no usable verdict (%r)",
            (raw or "")[:200],
        )
        return look

    look.ran = True
    look.problems = _problems_from(data)
    if "change_visible" in data:
        look.change_confirmed = bool(data.get("change_visible"))
        if change and look.change_confirmed is False:
            line = (
                f"the change you just made (“{change}”) is not visible on the "
                f"screen. Check you edited the element the user is actually "
                f"looking at, and that nothing is hiding it."
            )
            if line not in look.problems:
                look.problems.insert(0, line)
    return look
