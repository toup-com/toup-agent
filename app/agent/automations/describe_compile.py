"""Describe your own — one sentence becomes a draft automation (§4.6, §11).

`POST /automations/describe` hands the user's sentence here. The model
compiles it into a spec v2 draft against the user's real capability
registry (never a connector they lack); the draft is created in
`template_mode` (a write without a grant cannot ARM — nothing weakens),
stays `unarmed`, and its setup thread opens with the user's own
sentence, the compiled plan in plain words, and the §5.3 mode-aware
script. The day chat receives only the `automation_setup` notification
card (A's pipeline).

An uncompilable sentence raises `DescribeError` — the endpoint maps it
to a 422 with the honest fallback ("build it in chat"); a fabricated
spec is never an outcome.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

_SPEC_RULES = """\
Spec shape (v2): {"version": 2, "name": <short title case>, "mode":
"confirm"|"auto", "description": <one plain sentence of what it does>,
"variables": {...}, "trigger": {"sources": [{"id", "mode":
"poll"|"push"|"schedule", ...}]}, "steps": [...]}. A schedule source is
{"id": "when", "mode": "schedule", "schedule": {"cron_local": "..."}}
— compile the time the user STATED, in their words ("every Friday" →
"0 17 * * 5"-style cron_local). Poll sources need connector_id + event
+ poll_interval_s >= 300 + dedupe_key. Steps: up to 8, reads first
(each read may "collect" into {{steps.<id>.text}}/{{steps.<id>.count}}),
then at most 3 writes; a write step carries "grant_id": null (the user
approves it later — NEVER invent a grant id). Prefer reads-only or
drafts when the sentence does not clearly ask for a send-like action.
Only use connectors from the registry below. Reply ONLY as JSON:
{"spec": {...}, "domain": "work"|"university"|"personal"|null}"""


class DescribeError(Exception):
    def __init__(self, code: str, sentence: str):
        super().__init__(sentence)
        self.code = code
        self.sentence = sentence


async def compile_describe(db, *, user_id: str, text: str, complete=None) -> dict:
    """See module docstring. Returns `{"automation", "thread_id"}`."""
    from app.agent.automations import ledger as _ledger
    from app.agent.automations import registry as reg
    from app.agent.automations.service import (
        automation_payload, create_automation,
    )
    from app.agent.automations.spec import SpecError

    if complete is None:
        complete = _default_complete

    capability = await reg.fetch_registry(user_id)
    compact = _compact_registry(capability)
    prompt = (
        "Compile the user's sentence into one automation spec.\n\n"
        + _SPEC_RULES
        + "\n\nTHE USER'S CONNECTED CAPABILITY REGISTRY:\n"
        + json.dumps(compact, ensure_ascii=False)
        + f"\n\nTHE USER SAYS: {text[:1000]}"
    )

    # R38 — the build history. `trigger` covers the compile itself:
    # for a described automation the model's spec IS the decision about
    # when it runs, so that is honestly what those milliseconds bought.
    from app.agent.automations import build_ledger as _build
    rec = _build.BuildRecorder("described")

    automation = None
    last_errors = None
    with rec.phase("trigger"):
        for attempt in (1, 2):
            try:
                payload = await complete(
                    prompt if attempt == 1 else (
                        prompt + "\n\nYour previous spec was rejected — fix "
                        "every problem and emit the full reply again:\n"
                        + json.dumps(last_errors, ensure_ascii=False)[:2000]
                    )
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("[automations] describe LLM failed: %s: %s",
                               type(e).__name__, str(e)[:200])
                # R31-22. This used to read "Describe it to me in chat and
                # I will set it up there." — the sentence that sent a
                # founder's setup into the main chat, where the agent asked
                # its questions and no card was ever created. Setting one
                # up happens in its own thread; when nothing could be
                # created there is no thread to go to either, so this is an
                # error the sheet shows, not an invitation somewhere else.
                raise DescribeError(
                    "compiler_unavailable",
                    "I could not set that up just now. Try again in a moment.",
                )
            spec = (payload or {}).get("spec") if isinstance(payload, dict) else None
            if not isinstance(spec, dict):
                last_errors = ["reply carried no spec object"]
                continue
            try:
                automation, _ = await create_automation(
                    db, user_id=user_id, spec=spec,
                    domain=(payload.get("domain") if isinstance(payload, dict)
                            else None),
                    template_mode=True,
                )
                break
            except SpecError as e:
                last_errors = e.errors
                continue
    if automation is None:
        raise DescribeError(
            "cannot_compile",
            "I could not turn that into a plan. Try saying it another "
            "way — name what it should watch, and what it should do.",
        )

    # R38: the `agent` phase of the build history — opening the record
    # and stating the plan is exactly "wired the agent in".
    with rec.phase("agent"):
        thread = await _ledger.ensure_thread(
            db, user_id=user_id, automation_id=automation.id,
        )
        await _ledger.append_turn(
            db, user_id=user_id, thread=thread, run_id=None,
            kind="note", payload={"stamp": "added",
                                  "at": datetime.utcnow().isoformat() + "Z"},
        )
        await _ledger.append_turn(
            db, user_id=user_id, thread=thread, run_id=None,
            kind="user", payload={"text": text[:1000]},
        )
        plan_sentence = _plan_sentence(automation)
        await _ledger.append_turn(
            db, user_id=user_id, thread=thread, run_id=None,
            kind="agent", payload={"text": plan_sentence},
        )
    try:
        from app.agent.automations.setup_script import setup_turns
        from app.agent.automations.workflow import mode_of, schedule_block

        raw = json.loads(automation.spec_json or "{}")
        with rec.phase("output"):
            mode, label = mode_of(automation, raw)
            sched = schedule_block(automation, raw)
        # R31-22 / §5.3: the capability check said nothing here too.
        # R35: EVERY member, not `automation.connector_id` alone — a
        # described multi-account automation opened with one account's
        # check (or none), same defect as the template path.
        # R38: per-account verb truth — only the account whose step
        # writes wears the write-mode label (rec1 f007–f011).
        from app.agent.automations.setup_script import writer_connectors
        _writer_cids = writer_connectors(raw)
        account_scopes: list = []
        try:
            from app.agent.automations import permissions as _perms
            from app.agent.automations.setup_script import scope_lines_from
            from app.agent.automations.workflow import _member_connectors
            members = _member_connectors(raw) or (
                [automation.connector_id] if automation.connector_id else []
            )
            for cid in members:
                # R38: the account phase of the build history is this
                # exact resolution — the same call, measured once.
                with rec.phase(f"account:{cid}"):
                    resolved = await _perms.resolve(
                        db, automation=automation, account_id=cid,
                    )
                account_scopes.append({
                    "account_id": cid,
                    "writes": cid in _writer_cids,
                    "steps": scope_lines_from(resolved),
                })
        except Exception:  # noqa: BLE001 — a short list beats none,
            # and neither is worth losing the thread over.
            account_scopes = (
                [{"account_id": automation.connector_id,
                  "writes": automation.connector_id in _writer_cids,
                  "steps": []}]
                if automation.connector_id else []
            )
        for d in setup_turns(mode, label,
                             sched.get("sentence") or "when you arm it",
                             accounts=account_scopes):
            kind = d.get("kind")
            if kind in ("agent", "think"):
                await _ledger.append_turn(
                    db, user_id=user_id, thread=thread, run_id=None,
                    kind=kind, payload={"text": d.get("text") or ""},
                )
            elif kind == "tool":
                await _ledger.append_turn(
                    db, user_id=user_id, thread=thread, run_id=None,
                    kind="tool", payload={
                        "account_id": d.get("account_id")
                        or automation.connector_id or "",
                        "tool_kind": "read",
                        "action": d.get("action") or "Checked what I can do",
                        "detail": d.get("detail") or "",
                        "ok": True, "ms": 0,
                        "steps": d.get("steps") or [],
                        "items": [], "write_ids": [], "rest": "",
                    },
                )
    except Exception as e:  # noqa: BLE001 — the thread opens regardless
        logger.warning("[automations] describe setup script skipped: %s", e)

    await _build.record(db, automation=automation, recorder=rec)

    return {"automation": automation_payload(automation),
            "thread_id": thread.id,
            "build_history": _build.read(automation)}


def _plan_sentence(automation) -> str:
    try:
        from app.services.automation_verbs import rule_sentence

        raw = json.loads(automation.spec_json or "{}")
        sentence = rule_sentence(raw)
        if sentence:
            return (f"Here is the plan I built: {sentence} "
                    "Nothing runs until you say so.")
    except Exception:  # noqa: BLE001
        pass
    return ("Here is the plan I built from your sentence. "
            "Nothing runs until you say so.")


def _compact_registry(capability: dict) -> dict:
    """The registry, shrunk to what spec authoring needs."""
    out = {}
    for cid, entry in (capability or {}).items():
        if not isinstance(entry, dict):
            continue
        out[cid] = {
            "connected": bool((entry.get("connection") or {}).get("connected")
                              if isinstance(entry.get("connection"), dict)
                              else entry.get("connected")),
            "events": sorted((entry.get("events") or {}).keys())
            if isinstance(entry.get("events"), dict)
            else entry.get("events"),
            "writes": sorted((entry.get("writes") or {}).keys())
            if isinstance(entry.get("writes"), dict)
            else entry.get("writes"),
        }
    return out


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
