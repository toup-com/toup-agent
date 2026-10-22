"""Per-call context cache for voice delegations (voice round 2026-10-06, D-T3).

A Live voice call hands the tenant one delegation per request — eight in the
2026-10-06 call — and every one of them rebuilt the same per-call context from
scratch in phase 1, measured from the tenant's own lines:

    load_agent_config     50–125 ms   (one SELECT, same row every time)
    load_day_context     337–2008 ms  (the whole day, re-read and re-rendered)
    build_system_prompt   49–835 ms   (`sys=b68cf42e` on ALL eight — identical)

This module remembers the three across the delegations of one call, and is
honest about when it may not:

* **Agent config** — a 60 s TTL, like ``supervisor_gate``'s overrides; a write
  to the row in this process drops it at once (below).
* **Day context** — reused only while ONE aggregate query over the day's rows
  (count, newest ``created_at``, newest ``occurred_at``, total content length,
  ``day_chats.summary_updated_at``) answers exactly what it answered when the
  context was built. That query runs on phase 1's own connection — no extra
  dial. Anything else (a row added, revised to a new length, re-stamped, a new
  summary) is a full reload; there is no incremental append, because the relay
  REVISES voice rows in place and ``messages`` has no ``updated_at``.
* **System prompt** — reused while the inputs it is built from are unchanged:
  the user, channel, client time zone, profile, the LOCAL DATE (the stable
  prefix is day-stable, not forever-stable) and a per-process MEMORY
  GENERATION that every write to a memory file, an identity (the soul) or the
  agent config bumps. The per-turn parts of the build — the user-brain block
  (its relevant files are chosen by THIS request), the requested-format
  section, the clock and the runtime envelope — are always recomputed, never
  reused (``AgentRunner._build_system_prompt``).

Every entry also expires ``IDLE_TTL_S`` after its last use, so a call that ends
releases its context, and nothing here outlives a process. Voice only; chat may
opt in later. ``settings.voice_call_context_cache`` is the kill switch.
"""
from __future__ import annotations

import copy
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Optional, Tuple

logger = logging.getLogger(__name__)

#: An entry not used for this long is dropped: a voice call's delegations are
#: seconds to a few minutes apart, a call that ended should not pin memory.
IDLE_TTL_S = 120.0
#: The agent config row's own bound (same as `supervisor_gate.OVERRIDES_TTL_S`).
AGENT_CONFIG_TTL_S = 60.0
#: Per-kind entry cap — one tenant process serves one user, a handful of
#: channels and time zones; this only stops a pathological growth.
MAX_ENTRIES = 32

_lock = threading.Lock()
_memory_generation = 0


def memory_generation() -> int:
    """The per-process generation of everything the system prompt reads that a
    user can change mid-call: memory files, identities (the soul), the agent
    config. Bumped by every ORM write to those tables in this process."""
    return _memory_generation


def bump_memory_generation(reason: str = "") -> None:
    global _memory_generation
    with _lock:
        _memory_generation += 1
    if reason:
        logger.debug("[CALLCTX] memory generation -> %d (%s)", _memory_generation, reason)


def enabled() -> bool:
    try:
        from app.config import settings
        return bool(getattr(settings, "voice_call_context_cache", True))
    except Exception:  # noqa: BLE001 — a missing setting is the default
        return True


@dataclass
class _Entry:
    value: Any
    stamp: Any
    created: float = field(default_factory=time.monotonic)
    used: float = field(default_factory=time.monotonic)


class _Store:
    def __init__(self) -> None:
        self._d: "OrderedDict[Hashable, _Entry]" = OrderedDict()

    def get(self, key: Hashable, *, max_age: Optional[float] = None) -> Optional[_Entry]:
        now = time.monotonic()
        with _lock:
            e = self._d.get(key)
            if e is None:
                return None
            if now - e.used > IDLE_TTL_S or (max_age is not None and now - e.created > max_age):
                self._d.pop(key, None)
                return None
            e.used = now
            self._d.move_to_end(key)
            return e

    def put(self, key: Hashable, value: Any, stamp: Any = None) -> None:
        with _lock:
            self._d[key] = _Entry(value=value, stamp=stamp)
            self._d.move_to_end(key)
            while len(self._d) > MAX_ENTRIES:
                self._d.popitem(last=False)

    def drop(self, key: Hashable) -> None:
        with _lock:
            self._d.pop(key, None)

    def clear(self) -> None:
        with _lock:
            self._d.clear()


_agent_configs = _Store()
_day_contexts = _Store()
_prompts = _Store()


def reset_for_test() -> None:
    global _memory_generation
    _agent_configs.clear()
    _day_contexts.clear()
    _prompts.clear()
    with _lock:
        _memory_generation = 0


# ── Agent config ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentConfigView:
    """The scalars phase 1 reads off the `AgentConfig` row — never the ORM
    object itself, whose session is closed by the time the next delegation
    arrives."""

    disabled_tools: Optional[str]
    preferred_provider: Optional[str]
    agent_name: Optional[str]
    onboarding_completed: Optional[bool]

    @classmethod
    def of(cls, row: Any) -> "AgentConfigView":
        return cls(
            disabled_tools=getattr(row, "disabled_tools", None),
            preferred_provider=getattr(row, "preferred_provider", None),
            agent_name=getattr(row, "agent_name", None),
            onboarding_completed=getattr(row, "onboarding_completed", None),
        )


def get_agent_config(
    user_id: str, *, max_age: float = AGENT_CONFIG_TTL_S,
) -> Tuple[bool, Optional[AgentConfigView]]:
    """``(hit, view)`` — a hit may carry ``None`` (the user has no row).

    ``max_age`` is the row's own bound; the speculative search's
    "is web_search enabled" check reads with the idle bound instead (voice
    round 2026-10-07, T1) — a ≤2 min stale answer to that one question is
    harmless, and an in-process write bumps the generation anyway."""
    e = _agent_configs.get((user_id, memory_generation()), max_age=max_age)
    if e is None:
        return False, None
    return True, e.value


def put_agent_config(user_id: str, row: Any) -> None:
    _agent_configs.put(
        (user_id, memory_generation()),
        AgentConfigView.of(row) if row is not None else None,
    )


# ── Day context ──────────────────────────────────────────────────────────

async def day_validator(db, day_chat_id: str) -> Optional[tuple]:
    """ONE aggregate query that changes whenever anything a day context is
    built from changes: the day's message count, newest created_at, newest
    occurred_at, total content length, and the day's summary stamp. None when
    it cannot be read (then the cache is simply not used)."""
    try:
        from sqlalchemy import func, select
        from app.db.models import DayChat, Message

        # A SAVEPOINT, like phase 1's own config read: a failure here must
        # roll back this read only, never the turn's session.
        async with db.begin_nested():
            summary_at = select(DayChat.summary_updated_at).where(
                DayChat.id == day_chat_id,
            ).scalar_subquery()
            row = (await db.execute(
                select(
                    func.count(Message.id),
                    func.max(Message.created_at),
                    func.max(Message.occurred_at),
                    func.coalesce(func.sum(func.length(Message.content)), 0),
                    summary_at,
                ).where(Message.day_chat_id == day_chat_id)
            )).one()
        return tuple(_plain(v) for v in row)
    except Exception as exc:  # noqa: BLE001 — a validator never costs a turn
        logger.debug("[CALLCTX] day validator unavailable: %s", type(exc).__name__)
        return None


def _plain(v: Any) -> Any:
    return v.isoformat() if hasattr(v, "isoformat") else v


def get_day_context(key: Hashable, validator: Optional[tuple]) -> Optional[dict]:
    """A COPY of the cached day context when ``validator`` matches, else None."""
    if validator is None:
        return None
    e = _day_contexts.get(key)
    if e is None:
        return None
    if e.stamp != validator:
        _day_contexts.drop(key)
        return None
    return copy.deepcopy(e.value)


def put_day_context(key: Hashable, validator: Optional[tuple], value: dict) -> None:
    if validator is None or not isinstance(value, dict):
        return
    _day_contexts.put(key, copy.deepcopy(value), stamp=validator)


# ── System prompt ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PromptEntry:
    prompt: str
    tz_name: str
    local_date: str
    envelope: Optional[dict]


def get_prompt(key: Hashable) -> Optional[PromptEntry]:
    e = _prompts.get(key)
    return e.value if e is not None else None


def put_prompt(key: Hashable, entry: PromptEntry) -> None:
    _prompts.put(key, entry)


def drop_prompt(key: Hashable) -> None:
    _prompts.drop(key)


# ── Invalidation: any write to what the prompt is built from ─────────────

#: Deliberately NOT `users`: a turn may touch that row (timezone capture), and
#: a generation that moves every turn is a cache that never hits. A rename
#: mid-call is bounded by IDLE_TTL_S instead.
_TRACKED_TABLES = frozenset({"memory_files", "identities", "agent_configs"})
_installed = False


def install_write_hooks() -> None:
    """Bump the memory generation on every ORM write to a tracked table in
    this process — flushed instances AND bulk ``update()``/``delete()``
    statements. Idempotent."""
    global _installed
    if _installed:
        return
    try:
        from sqlalchemy import event
        from sqlalchemy.orm import Session

        def _table_of(obj: Any) -> str:
            return str(getattr(getattr(obj, "__table__", None), "name", "") or "")

        @event.listens_for(Session, "after_flush")
        def _after_flush(session, _ctx):  # pragma: no cover — exercised via tests
            for obj in (*session.new, *session.dirty, *session.deleted):
                name = _table_of(obj)
                if name in _TRACKED_TABLES:
                    bump_memory_generation(f"flush:{name}")
                    return

        @event.listens_for(Session, "do_orm_execute")
        def _bulk(state):  # pragma: no cover — exercised via tests
            if not (state.is_update or state.is_delete or state.is_insert):
                return
            try:
                mapper = state.bind_mapper
                name = str(getattr(getattr(mapper, "local_table", None), "name", "") or "")
            except Exception:  # noqa: BLE001
                name = ""
            if not name:
                try:
                    name = str(getattr(state.statement.table, "name", "") or "")
                except Exception:  # noqa: BLE001
                    name = ""
            if name in _TRACKED_TABLES:
                bump_memory_generation(f"bulk:{name}")

        _installed = True
    except Exception:  # noqa: BLE001 — no hooks ⇒ the TTLs still bound staleness
        logger.warning("[CALLCTX] write hooks not installed", exc_info=True)


install_write_hooks()
