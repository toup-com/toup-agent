"""The Steps-sheet recompile (R30 §4.4, CONTRACTS §11) — C's seam.

The user edits the three human sentences on the Steps sheet;
`workflow.set_steps` persists the wording and calls this seam once,
debounced. The recompile is a MINIMAL EDIT of the existing spec: the
model rewrites the current spec v2 to realize the new steps, keeping
the trigger, variables and surviving grants untouched; deterministic
gates decide the rest — a new write without an approved grant returns
`needs_consent` (the consent flow runs first, §4.4), a plan the model
cannot compile degrades to the honest saved-wording sentence, and the
apply path is `service.update_automation`, so bindings recompile and an
armed automation drops to draft rather than silently firing an old
shape.

Return contract (§11): `{"recompiled": bool, "sentence": str,
"code"?: str, "extra"?: dict}` — `code` present means the caller raises
it as a 409.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SPEC_RULES = """\
Spec rules (v2): top-level {"version": 2, "name", "mode", "variables",
"trigger": {"sources": [...]}, "steps": [...]}. Keep the trigger, the
variables and the mode EXACTLY as they are unless a step sentence
plainly demands otherwise. Steps: up to 8 tool calls, reads first —
each read may "collect" items into {{steps.<id>.text}} /
{{steps.<id>.count}} — then at most 3 writes. A write step that already
exists KEEPS its "grant_id" verbatim. A NEW write step carries
"grant_id": null — NEVER invent one. String params may use
{{var.<name>}}, {{source.id}}, {{event.<field>}} and {{memory.<key>}}.
Emit the FULL updated spec as JSON, nothing else."""


def _pluralise(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _ungranted_write(spec: dict) -> Optional[dict]:
    """A write step the model marked as needing a grant it does not
    have — the prompt's contract: a read step carries no `grant_id` key
    at all; a write carries its surviving grant verbatim, or null when
    it is new."""
    candidates = [s for s in (spec.get("steps") or []) if isinstance(s, dict)]
    action = spec.get("action")
    if isinstance(action, dict):
        candidates.append(action)
    for step in candidates:
        if "grant_id" in step and not step.get("grant_id"):
            return step
    return None


async def recompile_steps(
    db, *, automation, user_id: str, steps: list[dict], complete=None,
) -> dict:
    """See module docstring. `complete` is the injectable LLM seam."""
    try:
        current = json.loads(automation.spec_json or "{}")
    except (ValueError, TypeError):
        current = {}
    if complete is None:
        complete = _default_complete

    human = [{"n": i + 1, "text": s.get("text"), "sub": s.get("sub")}
             for i, s in enumerate(steps)]
    prompt = (
        "You maintain one automation's plan. The user rewrote its "
        "steps in plain words; produce the updated spec.\n\n"
        + _SPEC_RULES
        + "\n\nCURRENT SPEC:\n" + json.dumps(current, ensure_ascii=False)
        + "\n\nTHE STEPS, AS THE USER WROTE THEM:\n"
        + json.dumps(human, ensure_ascii=False)
    )

    from app.agent.automations.service import update_automation
    from app.agent.automations.spec import SpecError

    last_errors: Any = None
    for attempt in (1, 2):
        try:
            payload = await complete(
                prompt if attempt == 1 else (
                    prompt + "\n\nYour previous spec was rejected — fix "
                    "every problem and emit the full spec again:\n"
                    + json.dumps(last_errors, ensure_ascii=False)[:2000]
                )
            )
        except Exception as e:  # noqa: BLE001 — degrade, never break the sheet
            logger.warning("[automations] recompile LLM failed: %s: %s",
                           type(e).__name__, str(e)[:200])
            break
        spec = payload if isinstance(payload, dict) else {}
        if "spec" in spec and isinstance(spec["spec"], dict):
            spec = spec["spec"]
        if not spec.get("steps") and not spec.get("action"):
            last_errors = ["the spec carries no steps"]
            continue

        ungranted = _ungranted_write(spec)
        if ungranted is not None:
            from app.services.automation_verbs import display_name

            connector = str(ungranted.get("connector_id") or "")
            name = display_name(connector) or "that account"
            return {
                "recompiled": False,
                "code": "needs_consent",
                "sentence": (
                    f"That plan needs your yes before {name} can make "
                    "changes — approve it and I will finish the change."
                ),
                "extra": {"account_id": connector},
            }
        try:
            await update_automation(
                db, automation_id=automation.id, user_id=user_id, spec=spec,
            )
            await db.refresh(automation)
            sentence = (
                f"Changed the plan — it now does what your "
                f"{_pluralise(len(steps), 'step')} say."
            )
            if automation.status == "draft":
                sentence += " It needs another look before it runs again."
            return {"recompiled": True, "sentence": sentence}
        except SpecError as e:
            last_errors = e.errors
            continue

    return {
        "recompiled": False,
        "sentence": (
            "Rewrote the steps — the wording is saved, but I could not "
            "recompile the plan to match. Tell me in the thread what "
            "should change and I will do it there."
        ),
    }


async def _default_complete(prompt: str) -> dict:
    from app.config import settings
    from app.services.llm_service import get_llm_service

    model = getattr(settings, "automation_narrator_model", None) \
        or getattr(settings, "memory_extraction_model", None)
    response = await get_llm_service().complete_with_json(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
    )
    raw = response.content if hasattr(response, "content") else response
    if isinstance(raw, str):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        return json.loads(raw)
    return raw if isinstance(raw, dict) else {}
