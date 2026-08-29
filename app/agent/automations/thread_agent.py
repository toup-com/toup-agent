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

So the thread answers for itself — with the SAME agent loop the main
chat runs (round 33, item 8).

R31 split the answer in two: "answerable from the ledger" got a bare
`llm_service.complete()`, and "needs new reading" opened a `question`
run that resolved one read tool per account from the capability
registry's `source_tool`. Both halves failed the user for the same
reason — neither was the connector path the main chat uses. Gmail and
Slack declare no `source_tool` at all (Gmail's automation event is
push), so a thread question about Gmail could not read Gmail even when
the router fired; and the router was two regexes that "give me my last
five gmail" does not match, so it never fired. The founder asked for
his last five emails in the thread, was told "I could not read Gmail",
and got them in the main chat a minute later on the same account.

There is now ONE shape: the agent loop, over the same MCP connector
tools, reaching the same `connector_dispatcher.execute`. The bounds
this module always claimed are enforced by the channel rather than by
having no tools —
`prompt_profile.AUTOMATION_THREAD_DISABLED_TOOLS` withholds every
deferral tool, every memory writer and every routine/trigger mutator.
The connector surface it KEEPS, reads and writes alike: a thread turn
is attended, so a mutating call meets the same per-tool `elevation`
confirmation the main chat puts in front of it. Persistence
stays with the ledger, which is what keeps a thread question out of
the day chat (the R31 leak this module was written to close).
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


async def _legacy_question_run_ids(
    db: AsyncSession, turns: list[dict],
) -> set[str]:
    """Run ids among `turns` that were a THREAD QUESTION, not a run.

    R31's `open_question_run` — deleted above — answered "what is the
    latest" by minting a fake run and calling every account with an
    EMPTY tool input. For gmail and slack, whose manifests declare no
    `source_tool`, it never made a call at all: it wrote
    "Could not reach Gmail - I could not tell why" straight into the
    ledger from `_default_read_tool` returning None.

    Those turns are durable, and the grounding replays the newest
    `_CONTEXT_TURNS` of them verbatim — so without this, the thread
    keeps answering "the last run did not read any account data" with a
    failure that never happened, for as long as the rows stay in the
    window. Deleting user data to fix a wording is the wrong trade; the
    rule applied here is `open_question_run`'s OWN docstring, which said
    a question run "is the user asking, not the automation firing, and
    treating the two the same" is the defect. Nothing it wrote is a fact
    about the automation.

    Scoped to the ids actually present in the window: one indexed
    `IN (...)` on at most `_CONTEXT_TURNS` jobs, and no query at all for
    a thread that never used the old path.
    """
    ids = {str(t.get("run_id")) for t in turns if t.get("run_id")}
    if not ids:
        return set()
    try:
        from sqlalchemy import select
        from app.db.models import BuildJob
        rows = (await db.execute(
            select(BuildJob.id, BuildJob.config_json)
            .where(BuildJob.id.in_(ids))
        )).all()
    except Exception as e:  # noqa: BLE001 — grounding is context, not a gate
        logger.debug("[thread_agent] run-kind read failed: %s", e)
        return set()
    return {
        str(rid) for rid, cfg in rows
        if isinstance(cfg, dict) and cfg.get("run_kind") == "question"
    }


async def _recent_turns(db: AsyncSession, thread_id: str) -> list[dict]:
    turns, _more = await ledger.list_turns(
        db, thread_id=thread_id, limit=_CONTEXT_TURNS,
    )
    legacy = await _legacy_question_run_ids(db, turns)
    if not legacy:
        return turns
    # The fabricated reads and everything derived from them. The question
    # run's own `agent` turn ("Looking at your 4 accounts now.") is left
    # alone: the user saw it, and it claims nothing about an account.
    return [
        t for t in turns
        if not (t.get("kind") in ("tool", "needs_you")
                and str(t.get("run_id") or "") in legacy)
    ]


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
        # `recall` answers {"facts": [...], "episodes": [...]} — a DICT. This
        # returned it whole while the annotation said list[dict] and
        # `_grounding` sliced it, so every past-tense question died on
        # `KeyError: slice(None, 20, None)`. The annotation was the only thing
        # that ever described the intent.
        out = await mem.recall(
            db, user_id=automation.user_id, scope=automation.id, limit=20,
        ) or {}
        if isinstance(out, dict):
            return list(out.get("facts") or [])
        return list(out or [])
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
    """§4.9 — answer the user, in the thread, with real tools.

    ── Round 33, item 8 ────────────────────────────────────────────────
    This was one bare `llm_service.complete()` over a text grounding: no
    tools, no MCP client, no connector call. So "give me my last five
    gmail" asked inside an automation was answered "I could not read
    Gmail" — while the same sentence, on the same account, in the main
    chat a minute later, listed five emails. The thread was not using a
    DIFFERENT connector path; it had none.

    It now runs the SAME agent loop the main chat runs, in the same
    process, over the same MCP connector tools and the same
    `connector_dispatcher.execute`. What makes that safe is the channel:
    `automation_thread` (prompt_profile.AUTOMATION_THREAD_DISABLED_TOOLS)
    withholds every deferral tool, every memory writer and every
    routine/trigger mutator. The connector surface it keeps: a thread
    turn is attended, and a mutating call meets the same per-tool
    `elevation` confirmation the main chat puts in front of it. Persistence stays with the ledger
    (`save_user_message=False` / `save_assistant_message=False`): a
    thread turn saved as a day-chat row is the R31 leak, and
    `disable_post_processing=True` is what stops a thread's connector
    failures being curated into the user's memory (item 6).

    Every phase change goes out as `automation.activity` and the body
    streams through `automation.turn.delta`, so the thread shows the
    agent-state ladder and real text rather than an empty bubble.
    """
    user_id = automation.user_id
    await ledger.emit_activity(
        user_id, automation_id=automation.id, thread_id=thread.id,
        run_id=run_id, phase="thinking",
    )
    turns = await _recent_turns(db, thread.id)
    facts = await _facts_for(db, automation)
    grounding = _grounding(automation, turns, facts)
    t0 = time.monotonic()

    runner = _runner()
    if runner is None:
        # No in-process runner (a container that never wired one). The
        # toolless answer is strictly worse, but silence is worse still.
        logger.warning("[thread_agent] no agent runner — answering without tools")
        text = await _answer_without_tools(grounding, user_text)
        return await _persist_answer(
            db, automation=automation, thread=thread, run_id=run_id,
            text=text, t0=t0, streamed=False,
        )

    # The live surface, driven by the same tool events the main chat's
    # rail is driven by. `phase="writing"` is emitted on the FIRST
    # character, never before: the thread used to announce writing and
    # then sit on an empty bordered bubble for the whole model latency.
    wrote_any = {"v": False}

    async def _on_tool_start(tool_name: str) -> None:
        from app.services import automation_verbs as verbs
        account_id = _account_for_tool(automation, tool_name)
        label = (verbs.live_sentence(account_id, None) if account_id
                 else _tool_label(tool_name))
        try:
            await ledger.emit_activity(
                user_id, automation_id=automation.id, thread_id=thread.id,
                run_id=run_id, phase="tool",
                tool={"account_id": account_id or "", "label": label},
            )
        except Exception as e:  # noqa: BLE001 — a frame never fails a turn
            logger.debug("[thread_agent] tool frame failed: %s", e)

    async def _on_text_chunk(chunk: str) -> None:
        if not chunk:
            return
        if not wrote_any["v"]:
            wrote_any["v"] = True
            try:
                await ledger.emit_activity(
                    user_id, automation_id=automation.id,
                    thread_id=thread.id, run_id=run_id, phase="writing",
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("[thread_agent] writing frame failed: %s", e)

    try:
        response = await runner.run(
            user_message=(
                f"{_ANSWER_RULES}\n\n{grounding}\n\n"
                f"The user just asked, inside this automation's thread: "
                f"{user_text}"
            ),
            display_user_message=user_text,
            user_id=user_id,
            channel="automation_thread",
            save_user_message=False,
            save_assistant_message=False,
            disable_post_processing=True,
            on_tool_start=_on_tool_start,
            on_text_chunk=_on_text_chunk,
        )
        text = (getattr(response, "text", "") or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.warning("[thread_agent] answer failed: %s", e)
        text = ""

    return await _persist_answer(
        db, automation=automation, thread=thread, run_id=run_id,
        text=text, t0=t0, streamed=False,
    )


def _runner():
    """The in-process agent runner, or None on a container without one.

    The same object `/internal/agent-turn` uses — main.py wires it at
    boot through `set_api_v1_refs`. Read lazily: importing api_v1 at
    module scope would be a cycle through the router.
    """
    try:
        from app.api import api_v1
        return getattr(api_v1, "_agent_runner", None)
    except Exception as e:  # noqa: BLE001
        logger.debug("[thread_agent] runner lookup failed: %s", e)
        return None


def _account_for_tool(automation: Automation, tool_name: str) -> Optional[str]:
    """The automation account a `<connector>__<tool>` call belongs to."""
    name = str(tool_name or "")
    if "__" not in name:
        return None
    prefix = name.split("__", 1)[0]
    return prefix if prefix in set(_account_ids(automation)) else None


def _tool_label(tool_name: str) -> str:
    """A human phrase for a non-connector tool. Never the identifier —
    the same rule the app's activity registry states."""
    name = str(tool_name or "")
    if name.startswith("web_search"):
        return "Searching the web"
    if name.startswith("web_fetch"):
        return "Reading a page"
    if name.startswith("memory_"):
        return "Checking what it knows"
    return "Working on it"


async def _answer_without_tools(grounding: str, user_text: str) -> str:
    prompt = (
        f"{_ANSWER_RULES}\n\n{grounding}\n\n"
        f"The user just asked: {user_text}\n\nAnswer them."
    )
    try:
        return await _complete(prompt)
    except Exception as e:  # noqa: BLE001
        logger.warning("[thread_agent] fallback answer failed: %s", e)
        return ""


async def _persist_answer(
    db: AsyncSession, *, automation: Automation, thread, run_id, text: str,
    t0: float, streamed: bool,
) -> Optional[dict]:
    user_id = automation.user_id
    if not (text or "").strip():
        # R31-17's silence is the one outcome worth refusing. A thread
        # that shows a live state and then nothing has told the user
        # their question was lost.
        text = (
            "I could not put an answer together just then. Ask me again "
            "and I will try once more."
        )
    body = text[:4000]
    turn = await ledger.append_turn(
        db, user_id=user_id, thread=thread, run_id=run_id,
        kind="agent", payload={"text": body},
    )
    # The delta rides AFTER the persisted turn, naming the SAME id — the
    # client retires its live surface on `streamTurnId === turn.id`, and
    # a delta that arrived first (with no id yet on the client) left the
    # answer rendered twice for a beat.
    await ledger.emit_turn_delta(
        user_id, automation_id=automation.id, thread_id=thread.id,
        turn_id=turn["id"], text=body,
    )
    await ledger.emit_activity(
        user_id, automation_id=automation.id, thread_id=thread.id,
        run_id=run_id, phase="done",
        detail=f"{int((time.monotonic() - t0) * 1000)}ms",
    )
    return turn
