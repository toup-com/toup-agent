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
        return {"text": _clean_sentence(p.get("text"), f"{kind}.text")}

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
        return {
            "account_id": str(p.get("account_id") or ""),
            "tool_kind": tool_kind,
            "action": action,
            "detail": detail,
            "ok": bool(p.get("ok", True)),
            "ms": int(p.get("ms") or 0),
            "steps": steps,
            "items": mint_item_ids(p.get("items") or []),
            "write_ids": list(p.get("write_ids") or []),
            "rest": p.get("rest") or "",
        }

    if kind == "result":
        vocab = p.get("vocabulary")
        tiers = RESULT_VOCABULARIES.get(vocab or "")
        _require(tiers is not None,
                 f"unknown result vocabulary {vocab!r}")
        groups = p.get("groups") or []
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
                    "text": _clean_sentence(r.get("text"), "result.row.text"),
                    "sub": r.get("sub") or "",
                    "tag": r.get("tag") or "",
                    "item_refs": list(r.get("item_refs") or []),
                })
            out_groups.append(
                {"rank": i, "label": label, "tone": tone, "rows": rows}
            )
        return {
            "title": _clean_sentence(p.get("title"), "result.title"),
            "vocabulary": vocab,
            "groups": out_groups,
        }

    if kind == "draft":
        target = p.get("target") or {}
        return {
            "text": _clean_sentence(p.get("text"), "draft.text"),
            "target": {
                "account_id": str(target.get("account_id") or ""),
                "ref": target.get("ref"),
            },
            "sent_at": p.get("sent_at"),
        }

    if kind == "waiting":
        return {
            "pending_action_id": str(p.get("pending_action_id") or ""),
            "text": _clean_sentence(p.get("text"), "waiting.text"),
            "expires_at": p.get("expires_at"),
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
            "thread_id": thread.id,
            "run_id": run_id,
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
    for t in tool_turns:
        acc = t.get("account_id") or ""
        if acc and acc not in touched:
            touched.append(acc)
        if acc and not t.get("ok", True) and acc not in failed:
            failed.append(acc)

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
            vocab = "brief"
            tiers = RESULT_VOCABULARIES[vocab]
            n = len(all_ids)
            groups = [
                {"rank": i + 1, "label": label, "tone": tone, "rows": []}
                for i, (label, tone) in enumerate(tiers)
            ]
            groups[-1]["rows"].append({
                "text": (f"{n} item(s) read — I could not rank them this "
                         "time").replace("(s)", "" if n == 1 else "s"),
                "sub": "Everything the run read is here, unranked.",
                "tag": str(n), "item_refs": all_ids,
            })
            appended = await append_turn(
                db, user_id=user_id, thread=thread, run_id=job.id,
                kind="result",
                payload={"title": "Your morning, in order",
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

    # Stamp accounts onto the job config for cheap list reads.
    try:
        from app.agent.automations.executor_v2 import merge_job_config
        await merge_job_config(
            db, job.id,
            accounts_touched=touched, accounts_failed=failed,
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
