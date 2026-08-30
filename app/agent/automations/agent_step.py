"""The `agent` spec step — the run's own thinking, as a step (R38).

Until this round every v2 step was a connector call. The engine could
read and it could write, and the only judgement in a run happened
afterwards, in the narrator, over material it could no longer change.
An `agent` step is a step whose work is a model call: it takes what the
steps before it produced, answers one prompt about it, and hands the
answer back as a plain string that `executor_v2` binds to the step's
`output_var` — so a later write step's template, and the narration,
read it as `{{var.<name>}}` like anything else.

Determinism is the discipline, for the same reason the triage got it
this round (`28bede64`): the same inbox must produce the same answer,
because the user acts on it. So temperature 0.0, and a prompt that is
byte-stable over the same facts — `sort_keys=True`, bounded lists, no
clock and no ids that move between runs.

The seam is `complete(prompt) -> str`, injectable, so the protocol is
testable without a model. Same shape as `narrator.narrate_run`.

This module never decides what a failure MEANS. It raises; the step's
`on_error` (which defaults to `fail` for an agent step, because its
value is interpolated into later text and a missing one renders as an
empty string) is the engine's business.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .spec import render_value

logger = logging.getLogger(__name__)

#: Bounds on what one step may return and what it may be shown. All
#: three are part of the determinism contract, not just cost control: a
#: prompt that grows without limit is a prompt whose content depends on
#: how much the provider happened to return.
MAX_OUTPUT_CHARS = 4000
MAX_STEP_TEXT_CHARS = 4000
MAX_STEP_ITEMS = 25
_MAX_TOKENS = 2000


class AgentStepError(RuntimeError):
    """The step produced nothing usable. Never swallowed here."""


_RULES = """\
How to answer:
- Reply with the finished text and nothing else. No preamble, no "here \
is", no restating the ask. What you write is used verbatim by the rest \
of the run.
- Use ONLY the material below. Never invent a name, a number, a date, a \
link or a quote. If the material does not answer the ask, say exactly \
that in one sentence — a plausible answer built from nothing is the one \
outcome worse than no answer.
- You are thinking, not acting. Never claim anything was sent, posted, \
scheduled, changed or deleted.
- No emoji, no ISO timestamps, no raw tool identifiers."""


def _bounded_steps(ctx: dict) -> dict:
    """What the steps before this one produced, in a stable shape.

    Sorted by step id and bounded, because this is both serialized into
    the prompt AND what the ask is rendered against: two runs over the
    same data must build the same bytes, dict insertion order here is an
    accident of which step ran first, and a length that depends on how
    much a provider happened to return is not a stable prompt.

    `raw_fields` is dropped: it is the narrator's private material, and
    it says the same thing as `lines` in a shape nobody reads.
    """
    out: dict[str, dict] = {}
    for sid, res in sorted((ctx.get("steps") or {}).items()):
        if not isinstance(res, dict):
            continue
        entry: dict[str, Any] = {
            "ok": bool(res.get("ok", True)),
            "text": str(res.get("text") or "")[:MAX_STEP_TEXT_CHARS],
        }
        if isinstance(res.get("count"), int):
            entry["count"] = res["count"]
        lines = [str(line) for line in (res.get("lines") or [])]
        if lines:
            entry["lines"] = lines[:MAX_STEP_ITEMS]
        out[str(sid)] = entry
    return out


def build_prompt(automation_name: str, step, ctx: dict) -> str:
    """The whole prompt for one agent step. Pure and deterministic."""
    # The ask is rendered against the SAME bounded view that is printed
    # below it, so the model can never be shown two different versions
    # of one string — and the prompt's length is bounded by the spec
    # rather than by the size of an inbox.
    bounded = dict(ctx or {})
    bounded["steps"] = _bounded_steps(ctx)
    asked = str(
        render_value(getattr(step, "prompt", "") or "", bounded)
    ).strip()
    material = {
        "event": ctx.get("event") or {},
        "variables": {
            str(k): str(v)[:MAX_STEP_TEXT_CHARS]
            for k, v in (ctx.get("var") or {}).items()
        },
        "steps": bounded["steps"],
    }
    # R38: where the user pinned this automation to start, per account.
    # Included only when there ARE pins — an empty key in the material
    # is a fact the model has to read and cannot use, and this prompt's
    # whole discipline is that its bytes are bounded and stable.
    focus = {
        str(cid): str((entry or {}).get("labels") or "")
        for cid, entry in (ctx.get("focus") or {}).items()
        if isinstance(entry, dict) and entry.get("labels")
    }
    if focus:
        material["starts_at"] = focus
    return "\n\n".join([
        f'You are one step inside a run of the automation '
        f'"{automation_name}". The steps before you have already run; '
        f'everything they produced is below. Do the one thing you are '
        f'asked with it.',
        _RULES,
        "WHAT YOU WERE ASKED:\n" + asked,
        "WHAT THE RUN HAS SO FAR:\n" + json.dumps(
            material, ensure_ascii=False, indent=1, sort_keys=True,
            default=str,
        ),
    ])


async def run_agent_step(*, automation, step, ctx: dict, complete=None) -> str:
    """Run one agent step and return its answer.

    Raises `AgentStepError` when the model returns nothing — binding an
    empty string would put a hole in whatever interpolates it, and the
    step's `on_error` is the only thing entitled to decide that is
    acceptable.
    """
    if complete is None:
        complete = _default_complete
    prompt = build_prompt(
        getattr(automation, "name", "") or "this automation", step, ctx,
    )
    answer = str(await complete(prompt) or "").strip()
    if not answer:
        raise AgentStepError(
            f"agent step {getattr(step, 'id', '?')!r} produced nothing"
        )
    return answer[:MAX_OUTPUT_CHARS]


async def _default_complete(prompt: str) -> str:
    """The pinned-model call, at temperature 0.0.

    Same model resolution as the narrator — one background model for
    the whole automations engine, never `model=None` on a background
    path.
    """
    import os
    import re

    from app.config import settings
    from app.services.llm_service import get_llm_service

    model = getattr(settings, "automation_narrator_model", None) \
        or os.environ.get("AUTOMATION_NARRATOR_MODEL") \
        or getattr(settings, "memory_extraction_model", None)
    response = await get_llm_service().complete_with_json(
        messages=[{
            "role": "user",
            "content": prompt + (
                '\n\nReply ONLY as JSON: {"text": "<your answer>"}'
            ),
        }],
        model=model,
        temperature=0.0,
        max_tokens=_MAX_TOKENS,
    )
    raw = response.content if hasattr(response, "content") else response
    if isinstance(raw, str):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            raw = json.loads(stripped)
        except (ValueError, TypeError):
            # The model answered in prose instead of the envelope. That
            # prose IS the answer: the JSON wrapper is a parsing
            # convenience, and failing the step over a missing pair of
            # braces would throw away work that is right there. An
            # EMPTY answer still fails, in run_agent_step.
            return stripped
    if isinstance(raw, dict):
        return str(raw.get("text") or "")
    return str(raw or "")
