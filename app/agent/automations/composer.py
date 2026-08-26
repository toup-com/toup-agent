"""The workflow composer — a sentence becomes a change (R30 §4.4, §5.5).

"Tell the agent what to change…" is a real conversation with the same
agent: the sentence is classified into workflow changes with the fixed
change-kind vocabulary, the safe ones are applied (with a 10s undo),
anything that widens access comes back as `needs`, a rail is refused in
plain words, and an ambiguous sentence earns ONE question — never a
guess.

Split of authorship: this module owns classification and every sentence
the composer says; the `/workflow/ask` endpoint (A) applies the intents
it returns, writes the thread record (`user` turn → `EDITED` note →
`agent` confirmation), and repaints the canvas.

    classify_change(text, workflow, *, complete=None) ->
        {"applied": [Intent], "needs": [Need], "answer": str|None}

    Intent = {kind: rule|step|schedule|permission|account,
              sentence,               # the confirmation the sheet shows
              sheet,                  # which sheet opens
              ...kind-specific fields}
    Need   = {kind: consent|confirm, account_id, sentence}
    answer — the one agent line for a question, a rail refusal, or a
             sentence the agent cannot place (posted in the card).

`workflow` is the §4.4 GET shape (`fixtures/automations/workflow.json`
is the test fixture). The LLM extracts intents; POLICY — what may be
applied silently — is deterministic code here, so a model can never
widen access by classifying creatively.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

CHANGE_KINDS = ("rule", "step", "schedule", "permission", "account")

_SHEET_FOR = {"rule": "rules", "step": "agent", "schedule": "sched",
              "permission": "account", "account": "add"}


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def confirmation_sentence(intent: dict, workflow: dict) -> str:
    """The §5.5 templates — one sentence per applied change."""
    kind = intent.get("kind")
    if kind == "rule":
        return f"Added a rule — {_lower_first(str(intent.get('text') or '').rstrip('.'))}."
    if kind == "schedule":
        sentence = intent.get("sentence") or ""
        if not sentence:
            preset = next((p for p in (workflow.get("schedule") or {}).get("presets", [])
                           if p.get("id") == intent.get("preset_id")), None)
            sentence = (preset or {}).get("sentence") or "the new time"
        return f"Moved it to {_lower_first(sentence)}."
    if kind == "step":
        return f"Changed the step — {_lower_first(str(intent.get('text') or '').rstrip('.'))}."
    if kind == "permission":
        name = _account_name(workflow, intent.get("account_id"))
        phrase = _lower_first(str(intent.get("label") or "do that").rstrip("."))
        if intent.get("direction") == "grant":
            return f"{name} can now {phrase} — you approved it."
        return f"{name} can no longer {phrase}."
    if kind == "account":
        name = _account_name(workflow, intent.get("account_id"))
        return f"Took {name} out of this automation."
    return "Done."


def _account_name(workflow: dict, account_id: Optional[str]) -> str:
    for pool in (workflow.get("accounts") or [], workflow.get("available") or []):
        for acct in pool:
            if acct.get("account_id") == account_id:
                return acct.get("name") or str(account_id)
    return str(account_id or "the account")


def _find_permission(workflow: dict, account_id: str, needle: str) -> tuple[Optional[dict], Optional[str]]:
    """Locate a permission row by id or label across can/cant. Returns
    (row, side) with side in {"can", "cant"}."""
    needle_l = (needle or "").strip().lower()
    for acct in workflow.get("accounts") or []:
        if acct.get("account_id") != account_id:
            continue
        for side in ("can", "cant"):
            for row in acct.get(side) or []:
                if needle_l in (str(row.get("id", "")).lower(),
                                str(row.get("label", "")).lower()):
                    return row, side
    return None, None


# ---------------------------------------------------------------------------
# Policy — deterministic; the model proposes, this disposes
# ---------------------------------------------------------------------------

def apply_policy(intents: list[dict], workflow: dict) -> dict:
    """Sort extracted intents into applied / needs / answer per §5.5.
    Safe: rule add, schedule change, step wording, permission REVOKE,
    account REMOVE. Never silent: grants, account adds, rails."""
    applied: list[dict] = []
    needs: list[dict] = []
    answers: list[str] = []

    for intent in intents:
        kind = intent.get("kind")
        if kind not in CHANGE_KINDS:
            continue
        if kind == "rule":
            text = " ".join(str(intent.get("text") or "").split())
            if not text:
                continue
            # A rule is the user's own words — stored verbatim, exempt
            # from the copy guard like any quoted content.
            applied.append({"kind": "rule", "text": text,
                            "sheet": _SHEET_FOR["rule"]})
        elif kind == "schedule":
            applied.append({k: v for k, v in intent.items()
                            if k in ("kind", "preset_id", "sentence")}
                           | {"sheet": _SHEET_FOR["schedule"]})
        elif kind == "step":
            applied.append({k: v for k, v in intent.items()
                            if k in ("kind", "n", "text")}
                           | {"sheet": _SHEET_FOR["step"]})
        elif kind == "permission":
            account_id = intent.get("account_id")
            row, side = _find_permission(workflow, account_id,
                                          intent.get("permission") or "")
            if row is None:
                answers.append(
                    f"I could not find that permission on "
                    f"{_account_name(workflow, account_id)} — open the "
                    "account on the map and I will show you what it can do."
                )
                continue
            if intent.get("direction") == "grant":
                if row.get("kind") == "rail":
                    answers.append("It can never do this.")
                    continue
                needs.append({
                    "kind": "consent", "account_id": account_id,
                    "sentence": (
                        f"{_account_name(workflow, account_id)} would need "
                        f"your yes to {_lower_first(row.get('label', ''))} — "
                        "nothing changes until you approve."
                    ),
                    "permission_id": row.get("id"),
                })
            else:
                if side == "cant":
                    answers.append(
                        f"{_account_name(workflow, account_id)} already "
                        f"cannot {_lower_first(row.get('label', ''))}."
                    )
                    continue
                applied.append({
                    "kind": "permission", "direction": "revoke",
                    "account_id": account_id, "permission_id": row.get("id"),
                    "label": row.get("label"), "sheet": _SHEET_FOR["permission"],
                })
        elif kind == "account":
            account_id = intent.get("account_id")
            if intent.get("direction") == "add":
                needs.append({
                    "kind": "consent", "account_id": account_id,
                    "sentence": (
                        f"Adding {_account_name(workflow, account_id)} needs "
                        "your yes — it starts read-only once you connect it."
                    ),
                })
            else:
                applied.append({"kind": "account", "direction": "remove",
                                "account_id": account_id,
                                "sheet": _SHEET_FOR["account"]})

    for intent in applied:
        intent["sentence"] = confirmation_sentence(intent, workflow)
    return {"applied": applied, "needs": needs,
            "answer": " ".join(answers) if answers else None}


def _normalize_intent(intent: dict) -> dict:
    """Models like to nest an intent under its kind
    (`{"schedule": {...}}`); unfold that to the flat contract shape."""
    if "kind" in intent:
        return intent
    for kind in CHANGE_KINDS:
        if kind in intent:
            inner = intent[kind] if isinstance(intent[kind], dict) \
                else {"text": intent[kind]}
            return {"kind": kind, **inner}
    return intent


# ---------------------------------------------------------------------------
# Extraction — the LLM seam
# ---------------------------------------------------------------------------

_EXTRACTION_RULES = """\
You maintain one automation's settings. Classify the user's sentence
into zero or more changes. Kinds:
- rule: a standing line the agent must not cross ("skip anything from
  recruiters"). Carry the user's words as one clean sentence.
- schedule: a different time. Prefer a preset_id from the list; else
  give sentence ("Weekdays at 7:30").
- step: a change to what a numbered step does; give n and the new text.
- permission: one account may/may no longer do one thing; give
  account_id, permission (the label or id), direction grant|revoke.
- account: add or remove a whole account; give account_id, direction.
A sentence can mean several changes ("never post anywhere" = a rule AND
revoking the post permission). An imperative about what the automation
reads, skips or surfaces is ALWAYS at least a rule — "stop reading
#eng-general" → {"kind": "rule", "text": "Stop reading #eng-general."}
(plus a step change when a numbered step names that source). If the
sentence is genuinely ambiguous, emit no changes and ask ONE short
question instead. If it is not about this automation's settings at
all, emit no changes and answer in one short line that the thread is
the place for it. NEVER reply with empty intents AND no question AND
no answer — a silently dropped sentence is a defect.
Reply ONLY as JSON, every intent a flat object carrying its "kind":
{"intents": [
   {"kind": "rule", "text": "..."}
 | {"kind": "schedule", "preset_id": "..." , "sentence": "..."}
 | {"kind": "step", "n": 2, "text": "..."}
 | {"kind": "permission", "account_id": "...",
    "permission": "<the label or id>", "direction": "grant"|"revoke"}
 | {"kind": "account", "account_id": "...", "direction": "add"|"remove"}
 ], "question": null|"...", "answer": null|"..."}"""


def _extraction_prompt(text: str, workflow: dict) -> str:
    ctx = {
        "schedule": {k: (workflow.get("schedule") or {}).get(k)
                     for k in ("preset_id", "sentence")},
        "presets": [{"id": p.get("id"), "sentence": p.get("sentence")}
                    for p in (workflow.get("schedule") or {}).get("presets", [])],
        "steps": workflow.get("steps"),
        "rules": [r.get("text") for r in workflow.get("rules") or []],
        "accounts": [
            {"account_id": a.get("account_id"), "name": a.get("name"),
             "can": [r.get("label") for r in a.get("can") or []],
             "cant": [r.get("label") for r in a.get("cant") or []]}
            for a in workflow.get("accounts") or []
        ],
        "available": [
            {"account_id": a.get("account_id"), "name": a.get("name"),
             "state": a.get("state")}
            for a in workflow.get("available") or []
        ],
    }
    return (
        _EXTRACTION_RULES
        + "\n\nTHE AUTOMATION TODAY:\n"
        + json.dumps(ctx, ensure_ascii=False, indent=1)
        + f"\n\nTHE USER SAYS: {text[:800]}"
    )


async def classify_change(
    text: str, workflow: dict, *, complete=None
) -> dict:
    """The composer pass. Returns `{"applied", "needs", "answer"}`;
    `answer` set means post one agent line (a question, a refusal, or a
    redirect) and change nothing beyond `applied`."""
    text = (text or "").strip()
    if not text:
        return {"applied": [], "needs": [], "answer": None}
    if complete is None:
        complete = _default_complete
    try:
        payload = await complete(_extraction_prompt(text, workflow))
    except Exception as e:  # noqa: BLE001 — the composer never crashes the sheet
        logger.warning("[automations] composer extraction failed: %s: %s",
                       type(e).__name__, str(e)[:200])
        return {"applied": [], "needs": [], "answer": (
            "I could not work out what to change from that — "
            "tell me again in the thread and I will do it there."
        )}
    if isinstance(payload, str):
        # A double-encoded reply ("\"{…}\"") — one more decode attempt.
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            payload = {}
    intents = payload.get("intents") if isinstance(payload, dict) else None
    intents = [_normalize_intent(i) for i in intents if isinstance(i, dict)] \
        if isinstance(intents, list) else []
    result = apply_policy(intents, workflow)
    for key in ("question", "answer"):
        line = payload.get(key) if isinstance(payload, dict) else None
        if line and not result["applied"] and not result["needs"] \
                and not result["answer"]:
            result["answer"] = str(line)[:300]
            break
    # By construction the wire always carries applied, needs, or a
    # question — a model that returns nothing at all (F-1: ~50% on
    # "stop reading #eng-general" at one point) must never silently
    # drop the user's sentence.
    if not result["applied"] and not result["needs"] \
            and not result["answer"]:
        result["answer"] = (
            "I want to get this right — should that become a standing "
            "rule, a different time, a change to a step, or a change to "
            "what an account may do?"
        )
    return result


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
