"""Privacy-safe timing for tenant SQL calls and event-loop stall correlation.

SQLAlchemy's asyncpg JSONB decoder runs inside ``prepared_stmt.fetch``. A
large decode can therefore block the one Uvicorn event loop without leaving a
Python caller in the stall sampler's shallow stack. These engine hooks keep a
small, parameter-free identity for each in-flight statement and emit it when
the call takes at least two seconds. The loop sampler can photograph the same
identity while blocked. We never log SQL text, bound values, or result data.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import secrets
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import event

logger = logging.getLogger(__name__)

_WARN_S = 2.0
_ACTIVE_MAX_AGE_S = 120.0
_LOCK = threading.Lock()
_FINGERPRINT_KEY = secrets.token_bytes(32)


@dataclass(frozen=True)
class _Query:
    started: float
    fingerprint: str
    operation: str
    source: str


_active: dict[int, _Query] = {}


def _fingerprint(statement: str) -> str:
    # A keyed digest prevents a log reader from guessing a SQL literal (some
    # call sites use raw text rather than parameters) by offline dictionary.
    # It stays stable long enough to join SLOW_SQL to LOOP_STALL in one process.
    return hmac.new(
        _FINGERPRINT_KEY, statement.encode("utf-8", "replace"), hashlib.sha256,
    ).hexdigest()[:16]


def _source() -> str:
    """Return a source-code location, with no SQL text or request values."""
    def site(frame: Any) -> str | None:
        path = frame.f_code.co_filename.replace("\\", "/")
        if "/app/" in path and not path.endswith("/db/slow_query_probe.py"):
            module = path.rsplit("/app/", 1)[1].removesuffix(".py").replace("/", ".")
        elif "/tests/" in path:
            # The same hook is exercised from both /app/tests in CI and a
            # local checkout's backend/tests directory.
            module = "tests." + path.rsplit("/tests/", 1)[1].removesuffix(".py").replace("/", ".")
        else:
            return None
        return f"{module}:{frame.f_lineno}"

    fallback = "unknown"
    frame = sys._getframe(1)
    while frame is not None:
        location = site(frame)
        if location is not None:
            if not location.startswith("db."):
                return location
            if fallback == "unknown":
                fallback = location
        frame = frame.f_back
    # SQLAlchemy async calls run inside a greenlet. The greenlet's Python
    # stack can end at its runner; the owning Task's paused stack still names
    # the actual ``await db.execute(...)`` line.
    try:
        task = asyncio.current_task()
        if task is not None:
            for frame in reversed(task.get_stack(limit=24)):
                location = site(frame)
                if location is None:
                    continue
                if not location.startswith("db."):
                    return location
                if fallback == "unknown":
                    fallback = location
    except RuntimeError:
        pass
    return fallback


def _operation(statement: str) -> str:
    first = statement.lstrip().split(None, 1)
    if first and first[0].upper() in {"SELECT", "INSERT", "UPDATE", "DELETE", "WITH"}:
        return first[0].upper()
    return "OTHER"


def _before(_conn: Any, _cursor: Any, statement: str, _parameters: Any,
            context: Any, _executemany: bool) -> None:
    try:
        query = _Query(
            started=time.monotonic(),
            fingerprint=_fingerprint(statement),
            operation=_operation(statement),
            source=_source(),
        )
        with _LOCK:
            _active[id(context)] = query
    except Exception:
        # Observation must never alter a database operation.
        pass


def _finish(context: Any, *, failed: bool = False, warn_s: float = _WARN_S) -> None:
    try:
        with _LOCK:
            query = _active.pop(id(context), None)
        if query is None:
            return
        elapsed_ms = int((time.monotonic() - query.started) * 1000)
        if elapsed_ms >= warn_s * 1000:
            logger.warning(
                "[SLOW_SQL] ms=%d op=%s sql_hmac=%s source=%s failed=%d",
                elapsed_ms, query.operation, query.fingerprint,
                query.source, int(failed),
            )
    except Exception:
        pass


def active_snapshot(limit: int = 3) -> str:
    """Bounded, content-free in-flight SQL identities for ``[LOOP_STALL]``."""
    try:
        now = time.monotonic()
        with _LOCK:
            stale = [key for key, q in _active.items()
                     if now - q.started > _ACTIVE_MAX_AGE_S]
            for key in stale:
                _active.pop(key, None)
            queries = sorted(_active.values(), key=lambda q: q.started)[:max(0, limit)]
        return ",".join(
            f"{q.fingerprint}@{q.source}:{int((now - q.started) * 1000)}ms"
            for q in queries
        ) or "-"
    except Exception:
        return "-"


def install(engine: Any, *, warn_s: float | None = None) -> None:
    """Attach to one engine; safe across generic→tenant rebinds."""
    sync_engine = getattr(engine, "sync_engine", engine)
    if getattr(sync_engine, "_toup_slow_query_probe", False):
        return
    if warn_s is None:
        try:
            warn_s = max(0.0, float(os.getenv("TOUP_SLOW_SQL_WARN_S", _WARN_S)))
        except ValueError:
            warn_s = _WARN_S

    def after(conn: Any, cursor: Any, statement: str, parameters: Any,
              context: Any, executemany: bool) -> None:
        _finish(context, warn_s=warn_s)

    def on_error(exception_context: Any) -> None:
        context = getattr(exception_context, "execution_context", None)
        if context is not None:
            _finish(context, failed=True, warn_s=warn_s)

    event.listen(sync_engine, "before_cursor_execute", _before)
    event.listen(sync_engine, "after_cursor_execute", after)
    event.listen(sync_engine, "handle_error", on_error)
    sync_engine._toup_slow_query_probe = True
