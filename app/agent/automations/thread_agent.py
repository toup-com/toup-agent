"""The thread's own agent turn — CONTRACTS-R31 §4.1 / §4.9.

Until R31 an automation's thread could not answer anything. `POST
/{id}/thread/messages` persisted the user's turn and stopped; its own
docstring said "the conversational reply rides the existing WS chat
(session-resolved to this automation)". That sentence is the whole of
F1 on the platform side, because the chat socket stamps every user
message with today's `day_chat_id` **before** it looks at `session_id`
— so a question asked in a thread was a day-chat row by construction,
and the reply the runner produced was an ordinary day-chat turn for the
same reason. The founder's 11:17 answer about "everything in all
channels" appeared in his main chat for exactly that reason, followed
by `Memory updated · 5 facts`.

So the thread answers for itself. Two shapes, decided by the question:

  1. **Answerable from what is already known** — the run ledger, the
     automation's own rules and accounts, its scoped memory plus global
     memory. One `agent` turn, streamed.
  2. **Needs new reading** — a `question` RUN in the same thread: one
     tool turn per automation ACCOUNT (never one the automation does
     not have — Teams cannot appear unless it is on the canvas, and it
     did), the E-1 lines with a fix button on every non-success one,
     then the answer.

What this module deliberately does NOT do is run the general chat
agent. A thread question is bounded: it may read the automation's own
accounts and nothing else, and it has no write tools at all. That
boundary is why "what is the latest in Gmail" asked inside the work
brief cannot quietly become an agent turn with fourteen connectors and
a memory writer attached.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Automation
from . import account_health, ledger

logger = logging.getLogger(__name__)

# How much of the thread the answer is allowed to see. The thread is the
# whole history of an automation and can be thousands of turns; the
# question is nearly always about the last run.
_CONTEXT_TURNS = 40
_ANSWER_MAX_TOKENS = 900

# Does this question need NEW reading, or can the ledger answer it?
#
# Deliberately conservative in one direction only: a false "needs
# reading" costs a few seconds and some connector calls, while a false
# "answerable" produces a confident answer from stale data — which is
# what "The latest Gmail is still mostly security/test mail" was, on a
# thread whose last read had failed.
_FRESH_RE = re.compile(
    r"\b(latest|now|today|new|unread|current|right now|just came|"
    r"check|re-?check|look again|any(thing)? new)\b",
    re.IGNORECASE,
)
_PAST_RE = re.compile(
    r"\b(why|what did|explain|last run|earlier|yesterday|how come|"
    r"did it|what happened|reason)\b",
    re.IGNORECASE,
)


def needs_fresh_read(text: str) -> bool:
    """§4.9(1) vs §4.9(2). A question about the PAST is answered from
    the ledger; a question about NOW needs a run."""
    t = str(text or "")
    if _PAST_RE.search(t):
        return False
    return bool(_FRESH_RE.search(t))


def _account_ids(automation: Automation) -> list[str]:
    """The automation's OWN accounts, from its spec.

    §4.9: "never one the automation does not have". On 26 August the
    thread's answer named Teams — with a real, correct reason about
    re-authentication — for an automation whose canvas has no Teams
    node. Every sentence in it was true and the paragraph should not
    have existed, and a user reading it has no way to tell those apart.
    """
    from .workflow import _member_connectors, _spec_raw
    try:
        return list(_member_connectors(_spec_raw(automation)))
    except Exception as e:  # noqa: BLE001
        logger.debug("[thread_agent] member read failed: %s", e)
        return []


async def _recent_turns(db: AsyncSession, thread_id: str) -> list[dict]:
    turns, _more = await ledger.list_turns(
        db, thread_id=thread_id, limit=_CONTEXT_TURNS,
    )
    return turns


def _grounding(automation: Automation, turns: list[dict],
               facts: list[dict]) -> str:
    """What the answer may use. Facts only — no instructions about
    voice; that is C's system prompt's job."""
    lines: list[str] = [
        f"Automation: {automation.name}",
        f"Accounts it can reach: "
        f"{', '.join(account_health.display_of(a) for a in _account_ids(automation)) or 'none'}",
    ]
    try:
        rules = json.loads(automation.rules_json or "[]")
        for r in rules:
            if isinstance(r, dict) and r.get("text"):
                lines.append(f"Rule the user set: {r['text']}")
    except (ValueError, TypeError):
        pass
    for f in facts[:20]:
        text = (f or {}).get("text")
        if text:
            lines.append(f"Known: {text}")
    lines.append("")
    lines.append("The thread so far, oldest first:")
    for t in turns[-_CONTEXT_TURNS:]:
        kind = t.get("kind")
        if kind in ("user", "agent", "think"):
            who = "The user" if kind == "user" else "You"
            lines.append(f"{who}: {t.get('text') or ''}")
        elif kind == "tool":
            ok = "" if t.get("ok", True) else " (this one failed)"
            lines.append(
                f"[read] {t.get('action') or ''} — "
                f"{t.get('detail') or ''}{ok}"
            )
        elif kind == "needs_you":
            lines.append(f"[needs the user] {t.get('sentence') or ''}")
        elif kind == "note":
            lines.append(f"[{t.get('stamp')}]")
    return "\n".join(lines)


async def _facts_for(db: AsyncSession, automation: Automation) -> list[dict]:
    """This automation's scoped facts plus the user's global ones."""
    try:
        from app.services import memory_v2_service as mem
        return await mem.recall(
            db, user_id=automation.user_id, scope=automation.id, limit=20,
        ) or []
    except Exception as e:  # noqa: BLE001 — memory is context, not a gate
        logger.debug("[thread_agent] recall skipped: %s", e)
        return []


async def _complete(prompt: str) -> str:
    """One plain-prose completion, on the pinned model.

    Never `model=None` on a background path (the repo's own rule — an
    unpinned background call silently downgrades).
    """
    import os
    from app.config import settings
    from app.services.llm_service import get_llm_service

    model = getattr(settings, "automation_narrator_model", None) \
        or os.environ.get("AUTOMATION_NARRATOR_MODEL") \
        or getattr(settings, "memory_extraction_model", None)
    response = await get_llm_service().complete(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.3,
        max_tokens=_ANSWER_MAX_TOKENS,
    )
    raw = response.content if hasattr(response, "content") else response
    return raw if isinstance(raw, str) else ""


_ANSWER_RULES = (
    "Answer in plain sentences. No markdown of any kind — no asterisks, "
    "no backticks, no bullet characters, no headings. Name accounts by "
    "their display name. Never name an account this automation does not "
    "have. If a read failed, say which account and why, using the "
    "sentence given to you word for word. Never invent a count."
)


async def answer_in_thread(
    db: AsyncSession, *, automation: Automation, thread, user_text: str,
    run_id: Optional[str] = None,
) -> Optional[dict]:
    """§4.9(1) — answer from the ledger and memory, streamed.

    Returns the persisted `agent` turn, or None if the model produced
    nothing. Every phase change goes out as `automation.activity` so the
    thread can show the agent-state ladder instead of three dots
    (R31-17), and the body streams through `automation.turn.delta`.
    """
    user_id = automation.user_id
    await ledger.emit_activity(
        user_id, automation_id=automation.id, thread_id=thread.id,
        run_id=run_id, phase="thinking",
    )
    turns = await _recent_turns(db, thread.id)
    facts = await _facts_for(db, automation)
    prompt = (
        f"{_ANSWER_RULES}\n\n"
        f"{_grounding(automation, turns, facts)}\n\n"
        f"The user just asked: {user_text}\n\n"
        f"Answer them."
    )
    await ledger.emit_activity(
        user_id, automation_id=automation.id, thread_id=thread.id,
        run_id=run_id, phase="writing",
    )
    t0 = time.monotonic()
    try:
        text = await _complete(prompt)
    except Exception as e:  # noqa: BLE001
        logger.warning("[thread_agent] answer failed: %s", e)
        text = ""
    if not (text or "").strip():
        # R31-17's silence is the one outcome worth refusing. A thread
        # that shows a live state and then nothing has told the user
        # their question was lost.
        text = (
            "I could not put an answer together just then. Ask me again "
            "and I will try once more."
        )
    turn = await ledger.append_turn(
        db, user_id=user_id, thread=thread, run_id=run_id,
        kind="agent", payload={"text": text[:4000]},
    )
    # The delta carries the whole body once rather than nothing at all:
    # `complete()` is not a streaming call, so there is one chunk. When
    # the streaming seam lands this becomes many, and B's renderer does
    # not change.
    await ledger.emit_turn_delta(
        user_id, automation_id=automation.id, thread_id=thread.id,
        turn_id=turn["id"], text=text[:4000],
    )
    await ledger.emit_activity(
        user_id, automation_id=automation.id, thread_id=thread.id,
        run_id=run_id, phase="done",
        detail=f"{int((time.monotonic() - t0) * 1000)}ms",
    )
    return turn


async def open_question_run(
    db: AsyncSession, *, automation: Automation, thread, user_text: str,
) -> Optional[str]:
    """§4.9(2) — a `question` run in the thread. Returns its run id.

    One tool turn per automation account, then the answer. A question
    run NEVER appears in `run_in_flight`, never notifies, and never
    posts a card to the main chat: it is the user asking, not the
    automation firing, and treating the two the same is how a thread
    question became a main-chat job card with a progress bar.
    """
    from app.agent.job_runner import JobRunner, TaskSpec
    from . import registry as reg, run_v3
    from app.services import automation_verbs as verbs

    user_id = automation.user_id
    accounts = _account_ids(automation)
    if not accounts:
        return None

    job = await JobRunner().create_job(
        db,
        TaskSpec(
            user_id=user_id,
            title=automation.name[:200],
            prompt="(automation question)",
            job_type="automation_run",
            source_kind="automation",
            source_id=automation.id,
        ),
    )
    await run_v3.open_run(
        db, automation=automation, job=job, kind="question",
        total_steps=len(accounts),
    )

    await ledger.append_turn(
        db, user_id=user_id, thread=thread, run_id=job.id, kind="agent",
        payload={"text": (
            f"Looking at your {len(accounts)} account"
            f"{'' if len(accounts) == 1 else 's'} now."
        )},
    )

    failed: list[dict] = []
    for n, account_id in enumerate(accounts, start=1):
        label = verbs.live_sentence(account_id, None)
        await ledger.emit_activity(
            user_id, automation_id=automation.id, thread_id=thread.id,
            run_id=job.id, phase="tool",
            tool={"account_id": account_id, "label": label},
        )
        t0 = time.monotonic()
        ok, count, reason, message = await _read_account(
            automation, account_id,
        )
        ms = int((time.monotonic() - t0) * 1000)
        if ok:
            act = verbs.turn_action(account_id, None, kind="read", ok=True,
                                    count=count)
            steps_lines: list[dict] = []
        else:
            code = account_health.classify(reason, message)
            act = verbs.failure_action(account_id, reason)
            steps_lines = []
            failed.append({"account_id": account_id, "reason_code": code})
        await ledger.append_turn(
            db, user_id=user_id, thread=thread, run_id=job.id, kind="tool",
            payload={
                "account_id": account_id, "tool_kind": "read",
                "action": act["action"], "detail": act["detail"],
                "ok": ok, "ms": ms, "steps": steps_lines,
                "items": [], "write_ids": [], "rest": "",
            },
        )
        await ledger.emit_progress(
            user_id, run_id=job.id, automation_id=automation.id,
            step=n, total=len(accounts), sentence=label,
            fraction=n / max(len(accounts), 1), status="running",
        )

    if failed:
        # The same cards a run writes, for the same reason: the user
        # asked a question and part of the answer is missing. E-1's
        # per-account lines render from these.
        from .executor_v2 import _append_needs_you_turns
        await _append_needs_you_turns(
            db, thread=thread, automation=automation, job_id=job.id,
            failed_sources=failed,
        )
        from .executor_v2 import merge_job_config
        await merge_job_config(
            db, job.id,
            accounts_failed=[f["account_id"] for f in failed],
            failed_sources=failed,
        )

    await answer_in_thread(
        db, automation=automation, thread=thread, user_text=user_text,
        run_id=job.id,
    )

    from .executor import _finalize_job
    await _finalize_job(
        db, job.id, status="completed",
        outcome="partial" if failed else "sent",
    )
    return job.id


async def _read_account(
    automation: Automation, account_id: str,
) -> tuple[bool, Optional[int], str, str]:
    """One read of one account, through the same dispatch a run uses.

    Returns `(ok, count, reason_token, provider_message)`. The tool is
    the connector's own declared read — a question run reads what the
    automation reads, so the answer and the run cannot disagree about
    what "the latest" means.
    """
    from . import registry as reg
    from .executor_v2 import _failure_reason
    try:
        tool = await _default_read_tool(automation.user_id, account_id)
        if not tool:
            return False, None, "unreachable", "no read tool declared"
        result = await reg.dispatch_via_platform(
            automation.user_id, connector_id=account_id, tool_name=tool,
            tool_input={}, automation_id=automation.id,
        )
        if result.get("kind") != "ok":
            return (False, None, str(result.get("kind") or ""),
                    str(result.get("message") or ""))
        try:
            content = json.loads(result.get("content") or "{}")
        except (ValueError, TypeError):
            content = {}
        count = None
        if isinstance(content, dict):
            for key in ("total", "count", "resultSizeEstimate"):
                if isinstance(content.get(key), int):
                    count = content[key]
                    break
            if count is None:
                for value in content.values():
                    if isinstance(value, list):
                        count = len(value)
                        break
        return True, count, "", ""
    except Exception as e:  # noqa: BLE001 — a read never raises out
        return False, None, _failure_reason(e), str(e)


async def _default_read_tool(user_id: str, account_id: str) -> Optional[str]:
    """The connector's declared read tool, from capability metadata."""
    from . import registry as reg
    try:
        registry = await reg.fetch_registry(user_id)
        entry = (registry or {}).get(account_id) or {}
        for ev in entry.get("events") or []:
            if ev.get("source_tool"):
                return str(ev["source_tool"])
    except Exception as e:  # noqa: BLE001
        logger.debug("[thread_agent] registry read failed: %s", e)
    return None
