"""Run ledger v3 — typed turns, first-class threads, the v3 projection.

CONTRACTS-R30 §3/§4.2/§4.3/§4.10. The thread is where the agent works on
an automation with the user; the day chat receives only notification
cards (§4.10, D-05). A run stays a `build_jobs` row; this module owns
what the job row cannot say: the typed turns the canvas renders, their
validation (nothing raw escapes — the verb dictionary is the only source
of `action`/`detail` strings), the v3 status projection, the
completeness invariant, and the three live frames.

Hard rules enforced here, not in callers:
  - a turn kind outside AUTOMATION_TURN_KINDS never persists;
  - a tool turn whose `action` the dictionary does not serve never
    persists (dev: raise; prod: substituted with the safe generic and
    logged — the run keeps its record either way);
  - a result turn's tiers must be EXACTLY the fixed vocabulary (five
    for `brief`, three for `changes`), labels and tones verbatim;
  - job-sheet grouping is keyed by the first tool turn's server id —
    a render-index lookup is unrepresentable (D-04).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Automation, AutomationThread, AutomationTurn, AutomationWrite, BuildJob,
    AUTOMATION_TURN_KINDS, AUTOMATION_NOTE_STAMPS, AUTOMATION_RUN_KINDS,
    RESULT_VOCABULARIES,
)
from app.db.models.automation_ledger import AUTOMATION_FIXES
from app.services import automation_verbs as verbs

logger = logging.getLogger(__name__)

# Raw-identifier tells that must never appear in a served sentence.
_RAW_TOOL_RE = re.compile(r"\b\w+__\w+\b")
_ISO_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


class LedgerValidationError(ValueError):
    """A turn payload violated the v3 grammar."""


def _strict() -> bool:
    """Dev/test: raise on grammar violations. Prod: sanitize + log."""
    try:
        from app.config import settings
        return (getattr(settings, "environment", "") or "").lower() != "production"
    except Exception:  # noqa: BLE001
        return True


# ---------------------------------------------------------------- threads

async def ensure_thread(
    db: AsyncSession, *, user_id: str, automation_id: str,
) -> AutomationThread:
    """Get or create the automation's thread (one per automation)."""
    row = (
        await db.execute(
            select(AutomationThread).where(
                AutomationThread.automation_id == automation_id,
            )
        )
    ).scalar_one_or_none()
    if row is not None:
        return row
    row = AutomationThread(user_id=user_id, automation_id=automation_id)
    db.add(row)
    try:
        await db.flush()
    except IntegrityError:
        # Lost the race to a concurrent creator — read theirs.
        await db.rollback()
        row = (
            await db.execute(
                select(AutomationThread).where(
                    AutomationThread.automation_id == automation_id,
                )
            )
        ).scalar_one()
    return row


async def thread_for(
    db: AsyncSession, automation_id: str,
) -> Optional[AutomationThread]:
    return (
        await db.execute(
            select(AutomationThread).where(
                AutomationThread.automation_id == automation_id,
            )
        )
    ).scalar_one_or_none()


# ------------------------------------------------------------- validation

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise LedgerValidationError(msg)


def _clean_sentence(s: Any, field: str, *, allow_empty: bool = False) -> str:
    _require(isinstance(s, str), f"{field} must be a string")
    if not allow_empty:
        _require(bool(s.strip()), f"{field} must not be empty")
    _require(not _RAW_TOOL_RE.search(s), f"raw tool id in {field}: {s!r}")
    return s


# R31-24. `**paused**`, `- **Gmail:**` and backticks reached the founder's
# thread verbatim because nothing between the model and the bubble had an
# opinion about markdown: `copy_guard.scan("**bold**")` returns clean, and
# `plain_text.strip_markdown` was wired only to push transports.
#
# This STRIPS and never rejects, deliberately. A rejected agent turn is
# R31-17's silence — 40 seconds of "Looking at that now…" and then
# nothing — and the user would be paying for C's prompt drifting with a
# blank screen. The violation is logged instead, under one grep-able key,
# so C can find and fix the source.
#
# Only OUR prose goes through it. Item titles/subs and message texts are
# quoted vendor content: an email whose subject really is `**URGENT**`
# must keep its asterisks, or the thread misquotes the mail it read.
_MD_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{2,3})(?=\S)(.+?)(?<=\S)\1",
                             re.DOTALL)
_MD_CODE_RE = re.compile(r"`{1,3}([^`]+)`{1,3}", re.DOTALL)
_MD_BULLET_RE = re.compile(r"(?m)^[ \t]*(?:[-*+]|\d{1,2}[.)])[ \t]+")
_MD_HEADING_RE = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]+")


def strip_markdown_markers(s: str) -> tuple[str, bool]:
    """Return `(plain, found)`. Total — never raises, never drops text."""
    if not isinstance(s, str) or not s:
        return s or "", False
    out = _MD_CODE_RE.sub(r"\1", s)
    prev = None
    while prev != out:                      # nested **_bold italic_**
        prev = out
        out = _MD_EMPHASIS_RE.sub(r"\2", out)
    out = _MD_HEADING_RE.sub("", out)
    out = _MD_BULLET_RE.sub("", out)
    return out, out != s


def _plain(s: Any, field: str, *, allow_empty: bool = False) -> str:
    """`_clean_sentence` for agent-authored prose, markdown removed."""
    text = _clean_sentence(s, field, allow_empty=allow_empty)
    plain, found = strip_markdown_markers(text)
    if found:
        logger.warning("automation.copy.markdown field=%s", field)
    return plain


def mint_item_ids(items: list[dict]) -> list[dict]:
    """Assign server ids to items that lack them (idempotent)."""
    out = []
    for it in items or []:
        it = dict(it)
        it.setdefault("id", str(uuid.uuid4()))
        msgs = []
        for m in it.get("msgs") or []:
            msgs.append({
                "who": m.get("who", ""),
                "at": m.get("at", ""),
                "text": m.get("text", ""),
                "why": m.get("why", ""),
            })
        it["msgs"] = msgs
        out.append({
            "id": it["id"],
            "title": it.get("title", ""),
            "sub": it.get("sub", ""),
            "why": it.get("why", ""),
            "msgs": it["msgs"],
        })
    return out


def validate_turn_payload(kind: str, payload: dict) -> dict:
    """Validate + normalize one turn body against the v3 grammar.

    Returns the normalized payload. Raises LedgerValidationError on any
    violation — callers decide strict vs sanitize via `append_turn`.
    Item titles/subs and msg texts are QUOTED VENDOR CONTENT and are
    deliberately not screened beyond being strings (the copy guard's
    vendor-content exemption); `action`/`detail`/`why`/free text are
    ours and are screened.
    """
    _require(kind in AUTOMATION_TURN_KINDS, f"unknown turn kind {kind!r}")
    p = dict(payload or {})

    if kind == "note":
        _require(p.get("stamp") in AUTOMATION_NOTE_STAMPS,
                 f"unknown note stamp {p.get('stamp')!r}")
        return {
            "stamp": p["stamp"],
            "at": p.get("at"),
            "writes_count": int(p.get("writes_count") or 0),
        }

    if kind in ("agent", "think", "user"):
        # `user` keeps its own words verbatim — they are the person's,
        # not ours, and the copy contract only binds what we author.
        clean = _clean_sentence if kind == "user" else _plain
        out = {"text": clean(p.get("text"), f"{kind}.text")}
        if kind == "user" and p.get("client_msg_id"):
            # The replay key for `POST /thread/messages`. It rides the
            # turn because the turn is the only durable record of that
            # message — the alternative is a second table whose only job
            # is to remember what this row already knows.
            out["client_msg_id"] = str(p["client_msg_id"])[:64]
        return out

    if kind == "tool":
        action = _clean_sentence(p.get("action"), "tool.action")
        _require(verbs.is_served_action(action),
                 f"action not served by the dictionary: {action!r}")
        detail = p.get("detail") or ""
        _clean_sentence(detail, "tool.detail", allow_empty=True)
        _require(not _ISO_TS_RE.search(detail), "ISO timestamp in tool.detail")
        tool_kind = p.get("tool_kind")
        _require(tool_kind in ("read", "write"),
                 f"tool_kind must be read|write, got {tool_kind!r}")
        steps = []
        for s in p.get("steps") or []:
            steps.append({
                "text": _clean_sentence(s.get("text"), "tool.steps.text"),
                "ok": bool(s.get("ok")),
            })
        # R35: the executed tool calls behind the sentence — tool id,
        # main-chat label, ok, elapsed, one-line summary. Validated the
        # same way as everything here: bounded strings, typed fields,
        # nothing passed through unread.
        actions = []
        for a in p.get("actions") or []:
            tool = str(a.get("tool") or "")[:120]
            if not tool:
                continue
            actions.append({
                "tool": tool,
                "label": str(a.get("label") or "")[:120],
                "ok": bool(a.get("ok", True)),
                "ms": max(int(a.get("ms") or 0), 0),
                "summary": str(a.get("summary") or "")[:200],
            })
        # ── The per-account line and its button (round 33, item 4) ──────
        # The app's E-1 surface (`AccountLines`) opens with
        # `tools.filter((t) => !!t.line)` and then renders `t.tone`,
        # `t.line`, `t.fix` and `t.reason_code` — the four fields declared
        # on its `ToolTurn`. This whitelist dropped all four, so the
        # closer of the two fix affordances the round shipped could not
        # render on any device, ever. Validated like everything else here:
        # an unknown fix or tone is refused rather than passed through.
        out = {
            "account_id": str(p.get("account_id") or ""),
            "tool_kind": tool_kind,
            "action": action,
            "detail": detail,
            "ok": bool(p.get("ok", True)),
            "ms": int(p.get("ms") or 0),
            "steps": steps,
            "actions": actions,
            "items": mint_item_ids(p.get("items") or []),
            "write_ids": list(p.get("write_ids") or []),
            "rest": p.get("rest") or "",
        }
        line = p.get("line")
        if line:
            out["line"] = _clean_sentence(line, "tool.line")
            tone = p.get("tone")
            _require(tone in ("success", "warning", "danger", None),
                     f"unknown tool tone {tone!r}")
            if tone:
                out["tone"] = tone
            fix = p.get("fix")
            if fix:
                _require(fix in AUTOMATION_FIXES, f"unknown fix {fix!r}")
                out["fix"] = fix
            if p.get("reason_code"):
                out["reason_code"] = str(p["reason_code"])
        return out

    if kind == "result":
        vocab = p.get("vocabulary")
        _require(vocab in RESULT_VOCABULARIES,
                 f"unknown result vocabulary {vocab!r}")
        tiers = RESULT_VOCABULARIES.get(vocab or "")
        groups = p.get("groups") or []
        if tiers is None:
            # R36-7 `digest`: free-form groups. The serializer's job
            # here is shape, not vocabulary — sequential ranks, short
            # non-empty labels, tones the app can draw.
            from app.db.models.automation_ledger import RESULT_TONES
            _require(1 <= len(groups) <= 6,
                     f"digest needs 1-6 groups, got {len(groups)}")
            out_groups = []
            for i, g in enumerate(groups, start=1):
                _require(int(g.get("rank", i)) == i,
                         f"group {i} rank mismatch")
                label = str(g.get("label") or "").strip()
                _require(1 <= len(label) <= 48,
                         f"group {i} label must be 1-48 characters")
                tone = g.get("tone")
                _require(tone in RESULT_TONES,
                         f"group {i} tone {tone!r} outside {RESULT_TONES!r}")
                rows = []
                for r in g.get("rows") or []:
                    rows.append({
                        "text": _plain(r.get("text"), "result.row.text"),
                        "sub": strip_markdown_markers(r.get("sub") or "")[0],
                        "tag": r.get("tag") or "",
                        "item_refs": list(r.get("item_refs") or []),
                    })
                out_groups.append(
                    {"rank": i, "label": label, "tone": tone, "rows": rows}
                )
            return {
                "title": _plain(p.get("title"), "result.title"),
                "vocabulary": vocab,
                "groups": out_groups,
            }
        _require(len(groups) == len(tiers),
                 f"{vocab} needs exactly {len(tiers)} tiers, got {len(groups)}")
        out_groups = []
        for i, (g, (label, tone)) in enumerate(zip(groups, tiers), start=1):
            _require(int(g.get("rank", i)) == i, f"tier {i} rank mismatch")
            _require(g.get("label") == label,
                     f"tier {i} label must be {label!r}, got {g.get('label')!r}")
            _require(g.get("tone") == tone,
                     f"tier {i} tone must be {tone!r}, got {g.get('tone')!r}")
            rows = []
            for r in g.get("rows") or []:
                rows.append({
                    "text": _plain(r.get("text"), "result.row.text"),
                    "sub": strip_markdown_markers(r.get("sub") or "")[0],
                    "tag": r.get("tag") or "",
                    "item_refs": list(r.get("item_refs") or []),
                })
            out_groups.append(
                {"rank": i, "label": label, "tone": tone, "rows": rows}
            )
        return {
            "title": _plain(p.get("title"), "result.title"),
            "vocabulary": vocab,
            "groups": out_groups,
        }

    if kind == "draft":
        target = p.get("target") or {}
        return {
            "text": _plain(p.get("text"), "draft.text"),
            "target": {
                "account_id": str(target.get("account_id") or ""),
                "ref": target.get("ref"),
            },
            "sent_at": p.get("sent_at"),
        }

    if kind == "waiting":
        return {
            "pending_action_id": str(p.get("pending_action_id") or ""),
            "text": _plain(p.get("text"), "waiting.text"),
            "expires_at": p.get("expires_at"),
        }

    if kind == "memory":
        # The "Memory updated · N facts" chip. `sheet` is the deep link
        # the chip opens; the count is the only number it may show.
        count = int(p.get("count") or 0)
        _require(count > 0, "memory turn needs a positive count")
        return {
            "count": count,
            "sheet": str(p.get("sheet") or "memory"),
        }

    if kind == "needs_you":
        # §4.4. Every field is required because the whole point of this
        # turn is that "could not reach an account" never happens again:
        # a card without an account_id, a reason or a fix is the same
        # nameless failure in a new shape.
        fix = p.get("fix")
        _require(fix in AUTOMATION_FIXES, f"unknown fix {fix!r}")
        reason = str(p.get("reason_code") or "")
        _require(bool(reason), "needs_you turn needs a reason_code")
        account_id = str(p.get("account_id") or "")
        _require(bool(account_id), "needs_you turn needs an account_id")
        return {
            "account_id": account_id,
            "connector_id": str(p.get("connector_id") or account_id),
            "name": _clean_sentence(p.get("name"), "needs_you.name"),
            "reason_code": reason,
            "sentence": _plain(p.get("sentence"),
                               "needs_you.sentence"),
            "fix": fix,
            "fix_label": _clean_sentence(p.get("fix_label"),
                                         "needs_you.fix_label"),
            "approval_url": p.get("approval_url") or None,
        }

    raise LedgerValidationError(f"unhandled kind {kind!r}")  # pragma: no cover


def _sanitize_fallback(kind: str, payload: dict) -> dict:
    """Prod-lane fallback: keep the record, drop the offending strings."""
    if kind == "tool":
        p = dict(payload or {})
        cid = str(p.get("account_id") or "")
        generic = verbs.turn_action(cid, "", kind=p.get("tool_kind") or "read",
                                    ok=bool(p.get("ok", True)))
        p["action"], p["detail"] = generic["action"], generic["detail"]
        try:
            return validate_turn_payload(kind, p)
        except LedgerValidationError:
            pass
    # Last resort: an honest agent line, never a dropped turn.
    return {"text": "Something here could not be shown safely."}


# ------------------------------------------------------------ turn writes

def _serialize_row(row: AutomationTurn) -> dict:
    try:
        body = json.loads(row.payload_json)
    except (ValueError, TypeError):
        body = {}
    return {
        "id": row.id,
        "kind": row.kind,
        "run_id": row.run_id,
        "seq": row.seq,
        "at": row.created_at.isoformat() + "Z",
        **body,
    }


async def append_turn(
    db: AsyncSession,
    *,
    user_id: str,
    thread: AutomationThread,
    kind: str,
    payload: dict,
    run_id: Optional[str] = None,
    commit: bool = True,
    broadcast: bool = True,
) -> dict:
    """Validate, persist and (best-effort) broadcast one turn.

    Returns the serialized turn. Grammar violations raise in dev and
    sanitize in prod (`_strict()`), so a live run never loses its
    record to a bad string.
    """
    try:
        body = validate_turn_payload(kind, payload)
    except LedgerValidationError as e:
        if _strict():
            raise
        logger.error("[ledger] sanitized invalid %s turn: %s", kind, e)
        body = _sanitize_fallback(kind, payload)
        if "text" in body and kind not in ("agent", "tool"):
            kind = "agent"

    for attempt in (1, 2, 3):
        seq = (
            await db.execute(
                select(func.coalesce(func.max(AutomationTurn.seq), 0)).where(
                    AutomationTurn.thread_id == thread.id,
                )
            )
        ).scalar_one() + 1
        row = AutomationTurn(
            thread_id=thread.id, run_id=run_id, seq=seq, kind=kind,
            payload_json=json.dumps(body, default=str),
        )
        db.add(row)
        try:
            await db.flush()
            break
        except IntegrityError:
            await db.rollback()
            if attempt == 3:
                raise
    if commit:
        await db.commit()

    turn = _serialize_row(row)
    if broadcast:
        await _broadcast(user_id, {
            "type": "automation.turn",
            # R31 §4.1: EVERY automation frame carries automation_id.
            # This one did not, so a client showing one thread could not
            # attribute a frame without joining through thread_id it
            # might not hold yet — and the app-level bridge, which runs
            # with no thread open at all, had nothing to route on.
            "automation_id": thread.automation_id,
            "thread_id": thread.id,
            "run_id": run_id,
            "turn": turn,
        })
    return turn


async def replace_turn(
    db: AsyncSession, *, user_id: str, thread: AutomationThread,
    turn_id: str, kind: str, payload: dict, run_id: Optional[str] = None,
) -> Optional[dict]:
    """Rewrite one turn IN PLACE and re-broadcast it (§4.5).

    "An `automation.turn` whose `turn_id` already exists replaces that
    turn, and `GET /thread` returns the replaced version." This is what
    makes the per-source merge (§4.2a) visible without a reload: when a
    reconnected account's catch-up read lands, the run's RESULT turn is
    rewritten where it already sits, in the run it belongs to, instead
    of a second brief appearing under the first and leaving the user to
    work out which one is true.

    Keeps `seq`, so the turn does not jump position in the thread.
    Returns None if the turn is gone — a replacement for something that
    no longer exists is not an error, it is a race with a delete.
    """
    row = await db.get(AutomationTurn, turn_id)
    if row is None or row.thread_id != thread.id:
        return None
    try:
        body = validate_turn_payload(kind, payload)
    except LedgerValidationError as e:
        if _strict():
            raise
        logger.error("[ledger] sanitized invalid %s replacement: %s", kind, e)
        body = _sanitize_fallback(kind, payload)
    row.kind = kind
    row.payload_json = json.dumps(body, default=str)
    if run_id is not None:
        row.run_id = run_id
    await db.commit()
    turn = _serialize_row(row)
    await _broadcast(user_id, {
        "type": "automation.turn",
        "automation_id": thread.automation_id,
        "thread_id": thread.id,
        "run_id": row.run_id,
        "turn": turn,
    })
    return turn


async def _broadcast(user_id: str, frame: dict) -> None:
    """Best-effort live push; the persisted row is the durable record.
    NO `channel` key — the app's frame filter drops channeled frames."""
    try:
        from app.api.ws_chat import broadcast_to_user
        await broadcast_to_user(user_id, frame)
    except Exception as e:  # noqa: BLE001 — no live socket is normal
        logger.debug("[ledger] broadcast skipped: %s", e)


async def emit_progress(
    user_id: str, *, run_id: str, automation_id: str, step: int, total: int,
    sentence: str, fraction: float, status: str,
) -> None:
    await _broadcast(user_id, {
        "type": "automation.run.progress",
        "run_id": run_id, "automation_id": automation_id,
        "step": step, "total": total, "sentence": sentence,
        "fraction": round(float(fraction), 3), "status": status,
    })


async def emit_live(
    user_id: str, *, run_id: str, automation_id: str, text: str,
) -> None:
    await _broadcast(user_id, {
        "type": "automation.run.live",
        "run_id": run_id, "automation_id": automation_id, "text": text,
    })


ACTIVITY_PHASES = ("thinking", "tool", "writing", "done")


async def emit_activity(
    user_id: str, *, automation_id: str, thread_id: Optional[str],
    phase: str, run_id: Optional[str] = None,
    tool: Optional[dict] = None, detail: Optional[str] = None,
) -> None:
    """§4.5 — the thread's live state, one frame per phase change.

    R31-17. The thread showed `Looking at that now…` — three dots and a
    fixed string — for twenty to forty seconds while the main chat,
    two taps away, showed the agent-state orb walking its ladder with
    the tool glyph of whatever it was reading. The difference was not
    the component: it was that nothing on the wire told the thread what
    phase the turn was in. This is that information, and it is the same
    information the chat socket already gives ChatScreen.

    `tool.label` is the verb dictionary's PROGRESSIVE form ("reading
    your unread mail"), not the past-tense record form — the ladder is
    describing something still happening.
    """
    if phase not in ACTIVITY_PHASES:
        logger.debug("[ledger] unknown activity phase %r", phase)
        return
    frame = {
        "type": "automation.activity",
        "automation_id": automation_id,
        "thread_id": thread_id,
        "run_id": run_id,
        "phase": phase,
    }
    if tool:
        frame["tool"] = tool
    if detail:
        frame["detail"] = detail
    await _broadcast(user_id, frame)


async def emit_turn_delta(
    user_id: str, *, automation_id: str, thread_id: str, turn_id: str,
    text: str,
) -> None:
    """§4.5 — the appended CHUNK of a streaming agent turn.

    `text` is what to append, not the whole body: the thread's reply
    used to be accumulated in the client and painted only at `onDone`,
    so a forty-second answer arrived as forty seconds of nothing
    followed by a wall of text.
    """
    if not text:
        return
    await _broadcast(user_id, {
        "type": "automation.turn.delta",
        "automation_id": automation_id,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "text": text,
    })


async def emit_updated(
    db: AsyncSession, user_id: str, *, automation_id: str,
    workflow_rev: Optional[int] = None,
) -> None:
    """§4.6 — one automation's summary row changed.

    Emitted after EVERY workflow write, status write, delete and run
    transition. `summary` is the §4.1 per-automation object exactly as
    `GET /summary` serves it, so a client replaces one row instead of
    reloading a screen the user is looking at — which is what made
    every mutation cost a round trip on a tenant that can boot dark for
    a minute.

    Best-effort, and computed defensively: a summary that cannot be
    built must not take the write with it.
    """
    payload: Optional[dict] = None
    try:
        from .summary import summary_payload
        full = await summary_payload(db, user_id=user_id)
        for row in (full or {}).get("automations") or []:
            if row.get("id") == automation_id:
                payload = row
                break
    except Exception as e:  # noqa: BLE001 — see docstring
        logger.debug("[ledger] automation.updated summary skipped: %s", e)
    frame = {
        "type": "automation.updated",
        "automation_id": automation_id,
        "summary": payload,
    }
    if workflow_rev is not None:
        frame["workflow_rev"] = int(workflow_rev)
    await _broadcast(user_id, frame)


# -------------------------------------------------------------- reads

async def list_turns(
    db: AsyncSession, *, thread_id: str,
    before: Optional[str] = None, limit: int = 80,
) -> tuple[list[dict], bool]:
    """Oldest-first page ending just before `before` (a turn id)."""
    q = select(AutomationTurn).where(AutomationTurn.thread_id == thread_id)
    if before:
        anchor = (
            await db.execute(
                select(AutomationTurn.seq).where(
                    AutomationTurn.id == before,
                    AutomationTurn.thread_id == thread_id,
                )
            )
        ).scalar_one_or_none()
        if anchor is not None:
            q = q.where(AutomationTurn.seq < anchor)
    rows = list(
        (await db.execute(
            q.order_by(AutomationTurn.seq.desc()).limit(limit + 1)
        )).scalars()
    )
    has_more = len(rows) > limit
    rows = rows[:limit]
    rows.reverse()
    return [_serialize_row(r) for r in rows], has_more


async def run_turns(db: AsyncSession, *, run_id: str) -> list[dict]:
    rows = list(
        (await db.execute(
            select(AutomationTurn)
            .where(AutomationTurn.run_id == run_id)
            .order_by(AutomationTurn.seq.asc())
        )).scalars()
    )
    return [_serialize_row(r) for r in rows]


# --------------------------------------------------- the v3 projection

def run_kind_of(job: BuildJob) -> str:
    cfg = _cfg_of(job)
    kind = cfg.get("run_kind")
    if kind in AUTOMATION_RUN_KINDS:
        return kind
    if cfg.get("test_run"):
        return "run_now"
    return "scheduled"


def _cfg_of(job: BuildJob) -> dict:
    cfg = getattr(job, "config_json", None)
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, str) and cfg:
        try:
            return json.loads(cfg)
        except (ValueError, TypeError):
            return {}
    return {}


def run_v3_status(job: BuildJob) -> str:
    """Project (job.status, job.outcome) onto the §4.2 status enum."""
    status = (job.status or "").lower()
    outcome = (job.outcome or "").lower()
    if status in ("queued", "running"):
        return "running"
    if status in ("waiting_on_user", "paused"):
        return "waiting_on_user"
    if status == "completed":
        return "partial" if outcome == "partial" else "completed"
    if status == "cancelled":
        if outcome == "superseded":
            return "superseded"
        if outcome in ("stopped", "undone"):
            # `undone` = the user undid the staged write inside the 6s
            # window — the closest honest v3 reading is a user stop.
            return "stopped_by_user"
        return "skipped"
    return "failed"


def checkpoint_of(job: BuildJob) -> Optional[dict]:
    raw = getattr(job, "checkpoint_json", None)
    if not raw:
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (ValueError, TypeError):
        return None
    idx = data.get("step_index")
    return {"step_index": int(idx)} if idx is not None else None


async def run_v3_payload(
    db: AsyncSession, *, job: BuildJob, include_turns: bool = True,
) -> dict:
    cfg = _cfg_of(job)
    writes = list(
        (await db.execute(
            select(AutomationWrite)
            .where(AutomationWrite.run_id == job.id)
            .order_by(AutomationWrite.created_at.asc())
        )).scalars()
    )
    turns = await run_turns(db, run_id=job.id) if include_turns else []
    return {
        "id": job.id,
        "automation_id": job.source_id,
        "thread_id": cfg.get("thread_id"),
        "kind": run_kind_of(job),
        "started_at": job.created_at.isoformat() + "Z" if job.created_at else None,
        "finished_at": job.completed_at.isoformat() + "Z" if job.completed_at else None,
        "status": run_v3_status(job),
        "checkpoint": checkpoint_of(job),
        "accounts_touched": list(cfg.get("accounts_touched") or []),
        "accounts_failed": list(cfg.get("accounts_failed") or []),
        "writes": [
            {
                "id": w.id, "account_id": w.account_id, "what": w.what,
                "target": w.target, "audience": w.audience,
                "reversible": w.reversible, "undo_ref": w.undo_ref,
            }
            for w in writes
        ],
        "turns": turns,
    }


# ------------------------------------------- the completeness invariant

def _expected_vocabulary(automation, job) -> Optional[str]:
    """What this run's result turn should be speaking (R31-37).

    Derived from the automation's spec, which is the same source the
    narrator uses — so a mismatch means the two disagreed, not that
    this function has an opinion of its own. R36-7: a spec carrying a
    `narration.style` speaks that; and a write is a write by its TOOL
    (an ungranted template draft used to be invisible here).
    """
    try:
        from .narrator import vocabulary_for
        from app.services.automation_verbs import is_write_tool
        raw = json.loads(automation.spec_json or "{}")
        if raw.get("version") != 2:
            return None
        style = str(((raw.get("narration") or {}).get("style")) or "")
        if style in ("digest", "brief", "changes"):
            return style
        tools = [
            s.get("tool") for s in (raw.get("steps") or [])
            if isinstance(s, dict) and (s.get("grant_id")
                                        or is_write_tool(s.get("tool")))
        ]
        return vocabulary_for([t for t in tools if t])
    except Exception:  # noqa: BLE001
        return None


def _digest_title_of(automation) -> str:
    """The digest result's title — the spec's narration hint, then the
    automation's own name (R36-7)."""
    try:
        raw = json.loads(automation.spec_json or "{}")
        hint = str(((raw.get("narration") or {}).get("title")) or "").strip()
        if hint:
            return hint
    except Exception:  # noqa: BLE001
        pass
    return str(getattr(automation, "name", "") or "What this run found")


async def close_ledger(
    db: AsyncSession, *, user_id: str, job: BuildJob, automation: Automation,
) -> None:
    """Ledger-close verification + episode writes (§4.2/§4.5).

    Called after the run's terminal transition commits. Verifies the
    completeness invariant on completed/partial scheduled|run_now runs
    that read anything: every item id in the run's tool turns appears in
    exactly one result `item_refs`; misses are appended to the LAST tier
    as a count row and logged as `automation.ledger.unaccounted` (C's
    signal). Then stamps accounts_touched/failed and writes episodes.
    Best-effort by contract: a failure here never breaks the terminal.
    """
    v3 = run_v3_status(job)
    turns = await run_turns(db, run_id=job.id)
    tool_turns = [t for t in turns if t["kind"] == "tool"]

    touched: list[str] = []
    failed: list[str] = []
    read_ok: list[str] = []
    for t in tool_turns:
        acc = t.get("account_id") or ""
        if acc and acc not in touched:
            touched.append(acc)
        if acc and not t.get("ok", True) and acc not in failed:
            failed.append(acc)
        # R36-10: an account that READ successfully was read, whatever
        # else went wrong beside it. The home meta counted an account
        # with one ok read and one failed call as fully failed — "Ran
        # 18:54 · 0 of 1 accounts" about a run that had just listed
        # five threads.
        if (acc and t.get("ok", True) and t.get("tool_kind") == "read"
                and acc not in read_ok):
            read_ok.append(acc)

    result_rows = [t for t in turns if t["kind"] == "result"]
    invariant_applies = (
        v3 in ("completed", "partial")
        and run_kind_of(job) in ("scheduled", "run_now")
        and any(t["tool_kind"] == "read" and t.get("items") for t in tool_turns)
    )
    if invariant_applies and not result_rows:
        # Narration failed outright — the run still owes ONE result turn
        # (§4.2: every completed read-ful run emits exactly one). The
        # honest mechanical fallback: everything in the last tier as a
        # count row; C's rubric replaces it on the next healthy run.
        logger.warning(
            "automation.ledger.unaccounted run=%s automation=%s "
            "count=%d (no result turn — mechanical fallback)",
            job.id, automation.id,
            sum(len(t.get("items") or []) for t in tool_turns),
        )
        thread = await thread_for(db, automation.id)
        if thread is not None:
            all_ids = [it["id"] for t in tool_turns
                       for it in (t.get("items") or [])]
            n = len(all_ids)
            vocab = _expected_vocabulary(automation, job) or "brief"
            count_row = {
                "text": (f"{n} item(s) read — I could not rank them this "
                         "time").replace("(s)", "" if n == 1 else "s"),
                "sub": "Everything the run read is here, unranked.",
                "tag": str(n), "item_refs": all_ids,
            }
            if vocab == "digest":
                # R36-7: a digest automation's mechanical fallback keeps
                # its own title — never "Your morning, in order".
                title = _digest_title_of(automation)
                groups = [{"rank": 1, "label": "EVERYTHING IT READ",
                           "tone": "slate", "rows": [count_row]}]
            else:
                vocab = "brief"
                title = "Your morning, in order"
                tiers = RESULT_VOCABULARIES[vocab]
                groups = [
                    {"rank": i + 1, "label": label, "tone": tone, "rows": []}
                    for i, (label, tone) in enumerate(tiers)
                ]
                groups[-1]["rows"].append(count_row)
            appended = await append_turn(
                db, user_id=user_id, thread=thread, run_id=job.id,
                kind="result",
                payload={"title": title,
                         "vocabulary": vocab, "groups": groups},
            )
            result_rows = [appended]
            turns = await run_turns(db, run_id=job.id)
    if invariant_applies and result_rows:
        item_ids = {
            it["id"] for t in tool_turns for it in (t.get("items") or [])
        }
        referenced: set[str] = set()
        result = result_rows[-1]
        for g in result.get("groups") or []:
            for r in g.get("rows") or []:
                referenced.update(r.get("item_refs") or [])
        missing = sorted(item_ids - referenced)
        if missing:
            logger.warning(
                "automation.ledger.unaccounted run=%s automation=%s count=%d",
                job.id, automation.id, len(missing),
            )
            row_text = (
                f"{len(missing)} more item(s) the ranking missed"
                if len(missing) != 1 else "1 more item the ranking missed"
            )
            turn_row = await db.get(AutomationTurn, result["id"])
            if turn_row is not None:
                body = json.loads(turn_row.payload_json)
                body["groups"][-1]["rows"].append({
                    "text": row_text,
                    "sub": "Counted here so the account of the run stays complete.",
                    "tag": str(len(missing)),
                    "item_refs": missing,
                })
                turn_row.payload_json = json.dumps(body, default=str)

    # R31-37: the result turn's VOCABULARY must match what the run
    # actually did, and it is asserted here because here is where the
    # run's whole record exists at once.
    #
    # The founder's Morning work brief — a reads-only brief, posting one
    # line to Slack — rendered as `CHANGED YOUR WEEK · 1 item` /
    # `TOLD YOU ONLY · 0 items` / `LEFT ALONE ON PURPOSE · 2 items`,
    # and closed "That is everything I could change in this run." It
    # changed nothing. A brief that speaks the `changes` vocabulary
    # tells the user their week was altered by a run that only read.
    #
    # Logged, not corrected: the tiers differ in COUNT (five vs three)
    # as well as in wording, so a turn cannot be re-vocabularised
    # mechanically — its rows were ranked against labels that do not
    # exist in the other set. C's `vocabulary_for` is the fix at the
    # source; this is the tripwire that says when it drifted, on which
    # run, and in which direction.
    try:
        expected = _expected_vocabulary(automation, job)
        for t in [x for x in turns if x["kind"] == "result"]:
            if expected and t.get("vocabulary") != expected:
                logger.warning(
                    "automation.ledger.vocabulary run=%s automation=%s "
                    "served=%s expected=%s",
                    job.id, automation.id, t.get("vocabulary"), expected,
                )
    except Exception as e:  # noqa: BLE001 — a tripwire never fails a close
        logger.debug("[ledger] vocabulary check skipped: %s", e)

    # Stamp accounts onto the job config for cheap list reads.
    try:
        from app.agent.automations.executor_v2 import merge_job_config
        await merge_job_config(
            db, job.id,
            accounts_touched=touched, accounts_failed=failed,
            accounts_read_ok=read_ok,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("[ledger] accounts stamp skipped: %s", e)

    await _write_episodes(db, user_id=user_id, job=job, automation=automation,
                          turns=turns, v3_status=v3)
    await db.commit()


async def _write_episodes(
    db: AsyncSession, *, user_id: str, job: BuildJob, automation: Automation,
    turns: list[dict], v3_status: str,
) -> None:
    """Engine-written episodes (§4.5): one per run outcome, one per item
    that needed the user (tier 1–2 rows), one per write."""
    from app.db.models import MemoryEpisode
    cfg = _cfg_of(job)
    thread_id = cfg.get("thread_id")
    outcome_text = (
        automation.last_outcome_text
        or f"{automation.name} — {v3_status.replace('_', ' ')}"
    )
    db.add(MemoryEpisode(
        user_id=user_id, domain=automation.domain,
        automation_id=automation.id, run_id=job.id, thread_id=thread_id,
        text=outcome_text[:400], outcome=v3_status,
    ))
    for t in turns:
        if t["kind"] == "result":
            for g in (t.get("groups") or [])[:2]:
                for r in g.get("rows") or []:
                    db.add(MemoryEpisode(
                        user_id=user_id, domain=automation.domain,
                        automation_id=automation.id, run_id=job.id,
                        thread_id=thread_id, turn_id=t["id"],
                        item_ref=(r.get("item_refs") or [None])[0],
                        text=(r.get("text") or "")[:400],
                        outcome="needs_you",
                    ))
    writes = list(
        (await db.execute(
            select(AutomationWrite).where(AutomationWrite.run_id == job.id)
        )).scalars()
    )
    for w in writes:
        text = w.what if not w.target else f"{w.what} — {w.target}"
        db.add(MemoryEpisode(
            user_id=user_id, domain=automation.domain,
            automation_id=automation.id, run_id=job.id, thread_id=thread_id,
            text=text[:400], outcome="write",
        ))


# ------------------------------------------------------- legacy render

def legacy_turns(job: BuildJob, automation_name: str = "") -> list[dict]:
    """Render a pre-v3 run (steps_json only) as v3 turns, read-time.

    No items — the act page shows step lines only (§3.5 extension).
    Labels in steps_json were already minted through the dictionary
    (R29), so they are safe to serve as step-line text.
    """
    from app.agent.job_steps import parse_steps
    steps = parse_steps(getattr(job, "steps_json", None))
    v3 = run_v3_status(job)
    stamp = {"completed": "ran", "partial": "ran"}.get(v3, "tried")
    if v3 in ("running", "waiting_on_user"):
        stamp = "started"
    at = job.created_at.isoformat() + "Z" if job.created_at else None
    turns: list[dict] = [{
        "id": f"legacy:{job.id}:note", "kind": "note", "run_id": job.id,
        "stamp": stamp, "at": at, "writes_count": 0,
    }]
    by_brand: dict[str, list[dict]] = {}
    order: list[str] = []
    for s in steps:
        brand = s.get("brand") or ""
        if not brand:
            continue
        if brand not in by_brand:
            by_brand[brand] = []
            order.append(brand)
        ok = s.get("status") not in ("failed",)
        by_brand[brand].append({"text": s.get("label") or "", "ok": ok})
    for i, brand in enumerate(order):
        lines = by_brand[brand]
        ok = all(line["ok"] for line in lines)
        action = verbs.turn_action(
            brand, "", kind="read", ok=ok,
        )
        turns.append({
            "id": f"legacy:{job.id}:tool:{i}", "kind": "tool",
            "run_id": job.id, "account_id": brand, "tool_kind": "read",
            "action": action["action"], "detail": action["detail"],
            "ok": ok, "ms": 0, "steps": lines, "items": [],
            "write_ids": [], "rest": "",
        })
    return turns
