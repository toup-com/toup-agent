"""Engine-side memory — one namespaced state row per automation.

Round 28. The memory v3 product deliberately evicted scheduler-shaped
content from the user's memory files (`memory_v3_migration`'s
SCHEDULER_REF_KINDS: "a scheduler object is not a fact about a life"),
and every non-system MemoryFile enters the chat prompt via
`load_brain`. Automation state must live NEXT TO the memory system,
not inside the user's brain — so this module writes the retired-but-
intact row store directly (the `current_context.py` direct-write
precedent): ONE `Memory` row per automation, `ref_kind="automation"`,
`ref_id=<automation_id>`, `source_type="automation"`. The partial
UNIQUE index on (user_id, ref_kind, ref_id) makes the post-run write
an upsert; the row is invisible to `load_brain`, `search_files` and
the Memory UI, exactly like the document/media leg.

No curator, no LLM, no embedding — a deterministic engine write.

Contract (CONTRACTS-R28.md §6):
  - read at fire time  → `read_context` (exact indexed select),
    exposed to templates as {{memory.<key>}}
  - written after runs → `write_after_run` in its OWN session (a
    memory failure must never fail — or veto — a run)
  - deleted with the automation → `delete_for_automation`
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)

REF_KIND = "automation"

# Keys exposed to {{memory.<key>}}. last_counts is a compact JSON
# object (step id → count) rendered as a string.
_CONTEXT_KEYS = ("last_run_at", "last_outcome", "last_counts")


async def read_context(db, automation) -> dict:
    """The automation's memory namespace as a flat string dict for the
    template context. Missing namespace → {} (placeholders render as
    empty strings, never an error)."""
    try:
        from app.db.models.memory import Memory
        row = (await db.execute(
            select(Memory)
            .where(Memory.user_id == automation.user_id)
            .where(Memory.ref_kind == REF_KIND)
            .where(Memory.ref_id == automation.id)
            .where(Memory.is_deleted.is_(False))
        )).scalars().first()
        if row is None or not row.metadata_json:
            return {}
        meta = json.loads(row.metadata_json)
        if not isinstance(meta, dict):
            return {}
        out = {}
        for k in _CONTEXT_KEYS:
            v = meta.get(k)
            if v is None:
                continue
            out[k] = v if isinstance(v, str) else json.dumps(v, default=str)
        return out
    except Exception as e:  # noqa: BLE001 — memory is advisory, never fatal
        logger.warning("[automations] memory read failed for %s: %s",
                       automation.id, e)
        return {}


async def write_after_run(
    *,
    user_id: str,
    automation_id: str,
    automation_name: str,
    outcome: str,
    counts: Optional[dict] = None,
) -> None:
    """Upsert the state row AFTER the run's terminal commit, in this
    function's OWN session — a second-table write must never sit
    inside (and be able to veto) the run's transaction."""
    try:
        from app.db.database import async_session_maker
        from app.db.models.memory import Memory

        now_iso = datetime.utcnow().isoformat() + "Z"
        meta = {
            "last_run_at": now_iso,
            "last_outcome": outcome,
            "last_counts": counts or {},
        }
        content = (f"Automation {automation_name!r}: last run {outcome} "
                   f"at {now_iso}")[:500]

        async with async_session_maker() as db:
            row = (await db.execute(
                select(Memory)
                .where(Memory.user_id == user_id)
                .where(Memory.ref_kind == REF_KIND)
                .where(Memory.ref_id == automation_id)
                .where(Memory.is_deleted.is_(False))
            )).scalars().first()
            if row is not None:
                row.content = content
                row.metadata_json = json.dumps(meta, default=str)
                row.updated_at = datetime.utcnow()
                await db.commit()
                return
            row = Memory(
                user_id=user_id,
                brain_type="agent",
                content=content,
                category="automation",
                memory_type="engine_state",
                source_type="automation",
                ref_kind=REF_KIND,
                ref_id=automation_id,
                metadata_json=json.dumps(meta, default=str),
                importance=0.1,
            )
            db.add(row)
            try:
                await db.commit()
            except IntegrityError:
                # Concurrent first write won the unique index — update
                # theirs instead.
                await db.rollback()
                other = (await db.execute(
                    select(Memory)
                    .where(Memory.user_id == user_id)
                    .where(Memory.ref_kind == REF_KIND)
                    .where(Memory.ref_id == automation_id)
                    .where(Memory.is_deleted.is_(False))
                )).scalars().first()
                if other is not None:
                    other.content = content
                    other.metadata_json = json.dumps(meta, default=str)
                    await db.commit()
    except Exception as e:  # noqa: BLE001 — best-effort by contract
        logger.warning("[automations] memory write failed for %s: %s",
                       automation_id, e)


async def delete_for_automation(db, *, user_id: str, automation_id: str) -> None:
    """Soft-delete the namespace with the automation. Best-effort; the
    caller owns the session and the commit."""
    try:
        from app.db.models.memory import Memory
        rows = (await db.execute(
            select(Memory)
            .where(Memory.user_id == user_id)
            .where(Memory.ref_kind == REF_KIND)
            .where(Memory.ref_id == automation_id)
        )).scalars().all()
        for row in rows:
            row.is_deleted = True
            row.is_active = False
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] memory delete failed for %s: %s",
                       automation_id, e)
