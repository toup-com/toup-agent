"""
Browsing memory — per-user log of pages the agent visited and what it
found, persisted across sessions so the agent has "memory" of past
browsing when the user returns.

Schema is intentionally minimal — this is augmentary context, not the
canonical memory store. The full memory subsystem (with embeddings, FTS,
dedup, classification, etc.) lives in `memory.py`/`memory_service.py`
and is overkill for what we need here: a flat ring of recent browse
events keyed by user_id.

Created on first use via CREATE TABLE IF NOT EXISTS so no Alembic
migration is required — same pattern the codebase uses for other
opt-in tables (CLAUDE.md: "Base.metadata.create_all only creates NEW
tables; use ALTER … IF NOT EXISTS for column additions").
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from app.db.database import async_session_maker

logger = logging.getLogger(__name__)

_TABLE_READY = False

_DDL = """
CREATE TABLE IF NOT EXISTS extension_browsing_memory (
    id              BIGSERIAL PRIMARY KEY,
    user_id         VARCHAR(36) NOT NULL,
    occurred_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id      VARCHAR(36) NULL,
    task_id         VARCHAR(36) NULL,
    primary_url     TEXT NULL,
    primary_title   TEXT NULL,
    page_count      INTEGER NOT NULL DEFAULT 0,
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    agent_summary   TEXT NULL,
    visits_json     TEXT NULL
);
CREATE INDEX IF NOT EXISTS idx_extbrowse_user_ts
    ON extension_browsing_memory (user_id, occurred_at DESC);
"""

# SQLite fallback (some local dev DBs aren't Postgres).
_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS extension_browsing_memory (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    occurred_at     TEXT NOT NULL DEFAULT (datetime('now')),
    session_id      TEXT NULL,
    task_id         TEXT NULL,
    primary_url     TEXT NULL,
    primary_title   TEXT NULL,
    page_count      INTEGER NOT NULL DEFAULT 0,
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    agent_summary   TEXT NULL,
    visits_json     TEXT NULL
);
CREATE INDEX IF NOT EXISTS idx_extbrowse_user_ts
    ON extension_browsing_memory (user_id, occurred_at DESC);
"""


async def _ensure_table() -> None:
    global _TABLE_READY
    if _TABLE_READY:
        return
    try:
        async with async_session_maker() as db:
            url = str(db.bind.url) if hasattr(db, "bind") and db.bind else ""
            ddl = _DDL_SQLITE if "sqlite" in url else _DDL
            for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
                await db.execute(text(stmt))
            await db.commit()
        _TABLE_READY = True
    except Exception as exc:
        # Don't crash the request path if migration fails; just don't
        # remember this turn's browsing.
        logger.warning("[browsing_memory] table init failed: %s", exc)


async def record_task(
    *,
    user_id: str,
    session_id: Optional[str],
    task_id: Optional[str],
    primary_url: Optional[str],
    primary_title: Optional[str],
    page_count: int,
    duration_ms: int,
    agent_summary: Optional[str],
    visits: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Persist one completed browsing task. Returns True on success."""
    # Privacy opt-out: per-user flag stored on User.notification_preferences
    # JSONB (an existing column we can safely piggyback on without a
    # migration — same pattern as the Notifications card per CLAUDE.md).
    if await _user_opted_out(user_id):
        return False
    await _ensure_table()
    if not _TABLE_READY:
        return False
    try:
        async with async_session_maker() as db:
            await db.execute(
                text("""
                    INSERT INTO extension_browsing_memory
                        (user_id, session_id, task_id, primary_url, primary_title,
                         page_count, duration_ms, agent_summary, visits_json)
                    VALUES
                        (:user_id, :session_id, :task_id, :primary_url, :primary_title,
                         :page_count, :duration_ms, :agent_summary, :visits_json)
                """),
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "task_id": task_id,
                    "primary_url": (primary_url or "")[:2048] or None,
                    "primary_title": (primary_title or "")[:512] or None,
                    "page_count": int(page_count or 0),
                    "duration_ms": int(duration_ms or 0),
                    "agent_summary": (agent_summary or "")[:4000] or None,
                    "visits_json": json.dumps(visits[:30]) if visits else None,
                },
            )
            await db.commit()
        return True
    except Exception as exc:
        logger.warning("[browsing_memory] insert failed: %s", exc)
        return False


async def _user_opted_out(user_id: str) -> bool:
    """Check the per-user opt-out flag."""
    try:
        async with async_session_maker() as db:
            row = (await db.execute(
                text("SELECT notification_preferences FROM users WHERE id = :uid"),
                {"uid": user_id},
            )).first()
            if not row or not row[0]:
                return False
            prefs = row[0]
            if isinstance(prefs, str):
                prefs = json.loads(prefs or "{}")
            return bool(prefs.get("extension_memory_optout"))
    except Exception:
        return False


async def purge_user(user_id: str) -> int:
    """Delete every row for this user. Returns rows deleted."""
    await _ensure_table()
    if not _TABLE_READY:
        return 0
    try:
        async with async_session_maker() as db:
            r = await db.execute(
                text("DELETE FROM extension_browsing_memory WHERE user_id = :uid"),
                {"uid": user_id},
            )
            await db.commit()
            return int(r.rowcount or 0)
    except Exception as exc:
        logger.warning("[browsing_memory] purge failed: %s", exc)
        return 0


async def recent(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Return the user's most-recent browsing events, newest first."""
    if await _user_opted_out(user_id):
        return []
    await _ensure_table()
    if not _TABLE_READY:
        return []
    limit = max(1, min(limit, 200))
    try:
        async with async_session_maker() as db:
            rows = (await db.execute(
                text("""
                    SELECT id, occurred_at, primary_url, primary_title,
                           page_count, duration_ms, agent_summary
                      FROM extension_browsing_memory
                     WHERE user_id = :uid
                  ORDER BY occurred_at DESC
                     LIMIT :lim
                """),
                {"uid": user_id, "lim": limit},
            )).all()
    except Exception as exc:
        logger.warning("[browsing_memory] recent failed: %s", exc)
        return []
    out = []
    for r in rows:
        out.append({
            "id":             r[0],
            "ts":             _epoch_ms(r[1]),
            "url":            r[2],
            "title":          r[3],
            "page_count":     int(r[4] or 0),
            "duration_ms":    int(r[5] or 0),
            "summary":        r[6],
        })
    return out


async def recent_compact(user_id: str, limit: int = 8) -> str:
    """One-liner per entry — for embedding into the page-suggest prompt."""
    items = await recent(user_id, limit=limit)
    if not items:
        return ""
    lines = ["Recent browsing (newest first):"]
    for it in items:
        host = ""
        try:
            from urllib.parse import urlparse
            host = (urlparse(it["url"] or "").hostname or "").lstrip("www.")
        except Exception:
            pass
        title = (it["title"] or "").strip()
        summary = (it["summary"] or "").strip()
        bit = f"  - {host}"
        if title: bit += f" — {title[:80]}"
        if summary: bit += f" ({summary[:120]})"
        lines.append(bit)
    return "\n".join(lines)


def _epoch_ms(v: Any) -> int:
    """Turn a DB timestamp value into epoch-ms regardless of driver."""
    if v is None: return 0
    try:
        # Both datetime and ISO-string handled.
        if hasattr(v, "timestamp"):
            return int(v.timestamp() * 1000)
        import datetime as _dt
        return int(_dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return int(time.time() * 1000)


# Convenience used by extension_bridge: derive primary url/title from
# the assembled trace.
def derive_primary_from_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the most "important" URL from a task trace for the headline column."""
    steps = trace.get("steps") or []
    best_url, best_title, visits = None, None, []
    for s in steps:
        url   = s.get("url")
        title = s.get("title")
        if url:
            visits.append({
                "url":   url,
                "title": title,
                "kind":  s.get("kind") or s.get("phase"),
                "ts":    s.get("ts"),
            })
            if not best_url or s.get("phase") == "navigation":
                best_url   = url
                best_title = title or best_title
    return {
        "primary_url":   best_url,
        "primary_title": best_title,
        "page_count":    int(trace.get("page_count") or 0),
        "duration_ms":   int(trace.get("duration_ms") or 0),
        "visits":        visits,
    }
