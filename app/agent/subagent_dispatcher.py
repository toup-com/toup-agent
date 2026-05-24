"""Sub-agent dispatcher — Phase 2 of the sub-agent spawning arc.

This module exposes the pre-spawn gating logic + the cancel-cascade
chokepoint. Nothing here is invoked yet — Phase 4 wires it into the
``spawn`` tool handler at ``app/agent/tool_executor.py``. Shipping
it standalone keeps the diff small and lets the gating logic be
unit-tested in isolation against an in-memory SQLite.

What lives here
---------------
- ``SpawnRejection`` — dataclass returned by ``pre_spawn_checks``
  when a spawn should be rejected. ``error_code`` is the closed
  enum the tool result surfaces to the LLM so it can react
  programmatically; ``message`` is the human-readable line.
- ``REJECTION_CODES`` — the closed enum of rejection reasons. Pinned
  by test so a typo or rename in one site lights up the other.
- ``derive_idempotency_key(parent_job_id, task)`` — content-hash key
  used by Phase 4 when calling ``JobRunner.create_job(...,
  idempotency_key=...)``. The existing composite UNIQUE on
  ``(source_id, idempotency_key)`` in build_jobs catches duplicate
  spawns automatically; we leverage that rather than building a
  separate cache.
- ``walk_parent_chain(db, job_id)`` — walks parent_job_id upward
  from a starting row, returning the chain including the start.
  Cycle-safe (an unexpected cycle bounded by chain length never
  loops infinitely).
- ``count_running_children(db, parent_job_id)`` — per-parent cap
  query.
- ``count_running_subagents_for_user(db, user_id)`` — per-user
  concurrent cap query.
- ``count_subagent_spawns_24h(db, user_id)`` — per-user 24h ceiling
  query.
- ``pre_spawn_checks(db, *, user_id, parent_job_id, task)`` — runs
  all gates in priority order: kill switch → depth → caps →
  idempotency-doesn't-collide. Returns ``None`` on green-light,
  otherwise a ``SpawnRejection``.
- ``cascade_cancel(db, job_id, *, reason)`` — walks descendants of
  ``job_id`` (rows with parent_job_id = job_id, transitively) and
  flips any that are ``queued`` or ``running`` to ``cancelled`` with
  ``error_message=reason``. Returns the list of newly-cancelled IDs.
  Cycle-bounded via a max-depth guard.
- ``transition_job_status(db, job_id, *, new_status, error_message,
  propagate_cancel)`` — the chokepoint for terminal transitions.
  Phase 4 routes sub-agent transitions through here so a parent
  failure / cancellation always reaches the child.

Threading + transaction model
-----------------------------
Every helper accepts a caller-owned ``AsyncSession`` and never
opens its own. Helpers that mutate (``cascade_cancel``,
``transition_job_status``) issue UPDATEs but DO NOT commit — the
caller commits in their own transaction scope. This matches the
``JobRunner.execute`` / ``_mark_failed`` pattern at
``app/agent/job_runner.py:264-280``.

What this module does NOT do
----------------------------
- Does not call ``JobRunner.create_job`` (that's Phase 4 — the spawn
  handler will assemble the ``TaskSpec`` and invoke create_job after
  ``pre_spawn_checks`` returns None).
- Does not acquire a ``LaneManager`` slot (Phase 4 wraps the actual
  ``agent_runner.run`` call in ``async with lane_manager.acquire(
  LaneType.SUBAGENT, …)``).
- Does not write Day-as-Chat messages (Phase 4's
  ``write_subagent_message``).
- Does not enforce the credit budget (Phase 3 wires the LLM-proxy
  hook).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Closed enum of rejection codes. Pinned by test — a typo in one site
# (tool handler vs. test) silently regresses if these strings drift.
# ──────────────────────────────────────────────────────────────────────


class REJECTION_CODES:
    DISABLED = "SUBAGENT_DISABLED"
    DEPTH_EXCEEDED = "SUBAGENT_DEPTH_EXCEEDED"
    PARENT_CAP = "SUBAGENT_PARENT_CAP"
    USER_CONCURRENT_CAP = "SUBAGENT_USER_CONCURRENT_CAP"
    USER_24H_CAP = "SUBAGENT_USER_24H_CAP"
    EMPTY_TASK = "SUBAGENT_EMPTY_TASK"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return (
            cls.DISABLED,
            cls.DEPTH_EXCEEDED,
            cls.PARENT_CAP,
            cls.USER_CONCURRENT_CAP,
            cls.USER_24H_CAP,
            cls.EMPTY_TASK,
        )


@dataclass(frozen=True)
class SpawnRejection:
    """Reason a pre_spawn_checks decided to reject. Phase 4's spawn
    handler turns this into the tool result string the LLM sees."""

    error_code: str
    message: str

    def to_dict(self) -> dict:
        return {"error": self.error_code, "message": self.message}


# ──────────────────────────────────────────────────────────────────────
# Sub-agent job type discriminator. Pinned here (not as a setting) so
# the value is the same across dispatcher / tests / migrations.
# ──────────────────────────────────────────────────────────────────────


SUBAGENT_JOB_TYPE = "subagent"


# Cycle guard on the parent-chain walk. Normal flow never produces a
# cycle (the FK is one-directional, populated only at insert time),
# but a malformed manual UPDATE could. Bound the walk so a bad row
# can't hang the dispatcher.
_MAX_CHAIN_DEPTH_GUARD = 64


# ──────────────────────────────────────────────────────────────────────
# Idempotency key
# ──────────────────────────────────────────────────────────────────────


def derive_idempotency_key(parent_job_id: Optional[str], task: str) -> str:
    """Content-hash key used with JobRunner.create_job's
    ``(source_id, idempotency_key)`` composite UNIQUE so a retried
    spawn tool call on the same parent + same task collapses to one
    row.

    16 hex chars from sha256 — collision-resistant enough for
    per-parent uniqueness, short enough to fit the VARCHAR(120)
    idempotency_key column with room to spare.
    """
    # Normalize: strip whitespace + lowercase so a stray space or
    # case change doesn't defeat the dedup.
    normalized = (task or "").strip().lower()
    h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    parent_tag = parent_job_id or "top"
    return f"sa:{parent_tag}:{h}"


# ──────────────────────────────────────────────────────────────────────
# Parent-chain walk (cycle-bounded)
# ──────────────────────────────────────────────────────────────────────


async def subagent_depth_of(db: AsyncSession, job_id: str) -> int:
    """Sub-agent nesting depth of a given job.

    Semantics:
      0 — "not a sub-agent." The user's main-turn job (an
          ``agent_task`` row, a ``chat_intent`` row, etc.) sits at
          depth 0 even though it can be the parent of a child. It is
          the root the depth counter starts above.
      1 — "first-level sub-agent." Spawned from a non-subagent
          parent (the user's main turn).
      2+ — "grandchild and beyond." Spawned from a depth-1+
           sub-agent.

    Walks ``parent_job_id`` while ancestors are themselves
    ``job_type='subagent'`` rows. Stops at the first non-subagent
    ancestor (the user's main turn) or at the chain end (an
    orphan / missing row).

    Bounded by ``_MAX_CHAIN_DEPTH_GUARD`` AND by a visited-set so
    an unexpected cycle doesn't spin.
    """
    from app.db.models import BuildJob

    depth = 0
    current: Optional[str] = job_id
    visited: set[str] = set()

    while current and depth < _MAX_CHAIN_DEPTH_GUARD:
        if current in visited:
            logger.warning(
                "[subagent_dispatcher] subagent_depth_of cycle at %s — "
                "short-circuiting",
                current,
            )
            break
        visited.add(current)

        row = (
            await db.execute(
                select(BuildJob.job_type, BuildJob.parent_job_id).where(
                    BuildJob.id == current
                )
            )
        ).first()
        if row is None:
            break

        job_type, parent_id = row
        if job_type != SUBAGENT_JOB_TYPE:
            # Hit the user's main turn (or any non-subagent ancestor).
            # The depth counter stops here — that row is the "root."
            break

        depth += 1
        current = parent_id

    return depth


async def walk_parent_chain(db: AsyncSession, job_id: str) -> List[str]:
    """Return the chain [job_id, parent, grandparent, ...] following
    ``parent_job_id``. Starting row's id is included as element 0.

    Returns an empty list if ``job_id`` doesn't exist (defensive —
    Phase 4's pre_spawn_checks may receive a parent_job_id from the
    tool executor's context that doesn't match a row).

    Cycle-safe: bounded by ``_MAX_CHAIN_DEPTH_GUARD`` AND by a
    visited-set so an unexpected loop short-circuits instead of
    spinning. The visited-set check is the primary safety; the depth
    cap is a belt-and-suspenders backstop.
    """
    from app.db.models import BuildJob

    chain: List[str] = []
    visited: set[str] = set()
    current: Optional[str] = job_id

    while current and len(chain) < _MAX_CHAIN_DEPTH_GUARD:
        if current in visited:
            logger.warning(
                "[subagent_dispatcher] parent_chain cycle detected at %s "
                "(visited=%s) — short-circuiting",
                current, sorted(visited),
            )
            break
        visited.add(current)
        chain.append(current)

        row = (
            await db.execute(
                select(BuildJob.parent_job_id).where(BuildJob.id == current)
            )
        ).scalar_one_or_none()
        # scalar_one_or_none returns None for both "no row" and
        # "row exists but parent_job_id IS NULL". The first iteration
        # is the only one where "no row" should affect the result —
        # but we already appended `current` to `chain`. If the very
        # first lookup returns None we trim to []:
        if len(chain) == 1:
            # Verify the row actually exists (separate query so we
            # don't have to over-fetch the row).
            exists = (
                await db.execute(
                    select(BuildJob.id).where(BuildJob.id == current)
                )
            ).scalar_one_or_none()
            if exists is None:
                return []
        current = row  # may be None → loop exits

    if len(chain) >= _MAX_CHAIN_DEPTH_GUARD:
        logger.warning(
            "[subagent_dispatcher] parent_chain hit guard %d at start=%s — "
            "truncating",
            _MAX_CHAIN_DEPTH_GUARD, job_id,
        )

    return chain


# ──────────────────────────────────────────────────────────────────────
# Cap-check queries
# ──────────────────────────────────────────────────────────────────────


async def count_running_children(
    db: AsyncSession, parent_job_id: str
) -> int:
    """Count rows where ``parent_job_id`` matches and status is
    currently active (``queued`` or ``running``). The composite
    index ``ix_build_jobs_parent_status`` (migration 056) covers
    this query."""
    from app.db.models import BuildJob

    return (
        await db.execute(
            select(func.count(BuildJob.id)).where(
                BuildJob.parent_job_id == parent_job_id,
                BuildJob.status.in_(("queued", "running")),
            )
        )
    ).scalar_one()


async def count_running_subagents_for_user(
    db: AsyncSession, user_id: str
) -> int:
    """Count active sub-agent rows for a user. ``job_type='subagent'``
    is the discriminator the Phase 4 dispatcher writes onto every
    sub-agent row it creates."""
    from app.db.models import BuildJob

    return (
        await db.execute(
            select(func.count(BuildJob.id)).where(
                BuildJob.user_id == user_id,
                BuildJob.job_type == SUBAGENT_JOB_TYPE,
                BuildJob.status.in_(("queued", "running")),
            )
        )
    ).scalar_one()


async def count_subagent_spawns_24h(
    db: AsyncSession, user_id: str
) -> int:
    """Rolling 24h count of sub-agent spawns (any terminal state
    included — the cap is on spawns, not on outcomes). Backstop
    against per-parent + concurrent caps that the user could chain
    around by spawning in series."""
    from app.db.models import BuildJob

    cutoff = datetime.utcnow() - timedelta(hours=24)
    return (
        await db.execute(
            select(func.count(BuildJob.id)).where(
                BuildJob.user_id == user_id,
                BuildJob.job_type == SUBAGENT_JOB_TYPE,
                BuildJob.created_at >= cutoff,
            )
        )
    ).scalar_one()


# ──────────────────────────────────────────────────────────────────────
# Pre-spawn gating
# ──────────────────────────────────────────────────────────────────────


async def _read_subagent_flag_for_user(user_id: str) -> bool:
    """Per-user spawn-enabled flag, fetched from the platform DB via
    HTTP. Mirrors ``tool_executor._read_subagent_flag_for_user`` (PR
    #100) so the dispatcher gate can honour the same per-user override
    as the outer ``_tool_spawn`` gate. Kept inline (not imported) to
    avoid a tool_executor → subagent_orchestrator → subagent_dispatcher
    → tool_executor import cycle. Fail-closed on any error — spawn
    stays gated rather than accidentally letting through under the
    global env-var-off default."""
    if not user_id:
        return False
    try:
        from app.config import settings as _settings
        import httpx
        platform_url = (getattr(_settings, "platform_api_url", "") or "").rstrip("/")
        agent_key = (getattr(_settings, "agent_api_key", "") or "").strip()
        if not platform_url or not agent_key:
            return False
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(
                f"{platform_url}/agent/runtime-flags",
                headers={"X-Agent-Key": agent_key, "X-Agent-User-Id": user_id},
            )
        if r.status_code != 200:
            return False
        return bool(r.json().get("subagent_spawning_enabled", False))
    except Exception:
        return False


async def pre_spawn_checks(
    db: AsyncSession,
    *,
    user_id: str,
    parent_job_id: Optional[str],
    task: str,
) -> Optional[SpawnRejection]:
    """Run the kill switch + depth + per-parent cap + per-user
    concurrent cap + 24h cap + non-empty-task check in priority
    order. Returns ``None`` if the spawn should proceed.

    Priority rationale:
      1. Kill switch first — if spawning is off, nothing else matters
         and the rejection saves the LLM a wasted tool call shape.
      2. Empty task next — cheap reject before any DB work.
      3. Depth — bounded walk, no DB writes; reject early if deep.
      4. Per-parent cap — single COUNT, indexed.
      5. Per-user concurrent — single COUNT, partially indexed.
      6. Per-user 24h — single COUNT, full scan in worst case but
         the user's 24h sub-agent volume is small (≤50 ceiling).

    All five DB queries are independent of each other and could
    theoretically run in parallel. We run them sequentially because
    most rejections happen at the cheaper earlier checks and the
    sequential cost is ~30ms on Postgres; gather() complexity isn't
    worth the saving.
    """
    from app.config import settings

    # 1. Kill switch — env-var first, per-user flag as fallback.
    # tool_executor._tool_spawn already did this two-step check before
    # calling into the orchestrator (PR #100), but the dispatcher used
    # to only check the env var — so a tenant whose per-user flag was
    # flipped via the admin endpoint would pass the outer gate, enter
    # Path A, then hit DISABLED here. Caught live 2026-05-24 for
    # mrvviinn@gmail.com after their per-user flag was set TRUE.
    # Mirror the outer fallback so both gates agree.
    if not settings.subagent_spawning_enabled:
        if not await _read_subagent_flag_for_user(user_id):
            return SpawnRejection(
                error_code=REJECTION_CODES.DISABLED,
                message=(
                    "Sub-agent spawning is disabled in this environment. "
                    "Operator must flip SUBAGENT_SPAWNING_ENABLED=true."
                ),
            )

    # 2. Empty task
    if not (task or "").strip():
        return SpawnRejection(
            error_code=REJECTION_CODES.EMPTY_TASK,
            message="Sub-agent task is empty — provide a concrete instruction.",
        )

    # 3. Depth — count consecutive ``job_type='subagent'`` ancestors
    # starting from the parent, then add 1 for the new child. The
    # user's main-turn job (any non-subagent ancestor) sits at depth
    # 0 and is the root the count starts above. So a child spawned
    # from a regular agent_task parent is at depth 1; a child
    # spawned from another sub-agent is at depth 2; etc.
    parent_subagent_depth = 0
    if parent_job_id:
        parent_subagent_depth = await subagent_depth_of(db, parent_job_id)
    child_depth = parent_subagent_depth + 1
    if child_depth > settings.subagent_max_depth:
        return SpawnRejection(
            error_code=REJECTION_CODES.DEPTH_EXCEEDED,
            message=(
                f"Sub-agent depth cap of {settings.subagent_max_depth} "
                f"reached (would land at depth {child_depth}); "
                "cannot spawn further."
            ),
        )

    # 4. Per-parent children cap
    if parent_job_id:
        running = await count_running_children(db, parent_job_id)
        if running >= settings.subagent_max_children_per_parent:
            return SpawnRejection(
                error_code=REJECTION_CODES.PARENT_CAP,
                message=(
                    f"Parent already has {running} active sub-agent(s); "
                    f"cap is {settings.subagent_max_children_per_parent}."
                ),
            )

    # 5. Per-user concurrent cap
    concurrent = await count_running_subagents_for_user(db, user_id)
    if concurrent >= settings.subagent_max_per_user_concurrent:
        return SpawnRejection(
            error_code=REJECTION_CODES.USER_CONCURRENT_CAP,
            message=(
                f"You already have {concurrent} active sub-agent(s); "
                f"cap is {settings.subagent_max_per_user_concurrent}."
            ),
        )

    # 6. Per-user 24h ceiling
    last24h = await count_subagent_spawns_24h(db, user_id)
    if last24h >= settings.subagent_max_per_user_24h:
        return SpawnRejection(
            error_code=REJECTION_CODES.USER_24H_CAP,
            message=(
                f"24h sub-agent spawn ceiling reached ({last24h}/"
                f"{settings.subagent_max_per_user_24h}); try again later."
            ),
        )

    return None


# ──────────────────────────────────────────────────────────────────────
# Cancel cascade — walks descendants and flips them
# ──────────────────────────────────────────────────────────────────────


# Statuses considered "active" (i.e. cancellable). A cascade that
# hits a terminal row leaves it alone — we don't overwrite an
# already-finalized outcome with a fresh "cancelled" stamp.
_ACTIVE_STATUSES = ("queued", "running")
_TERMINAL_STATUSES = ("completed", "failed", "cancelled", "timeout", "budget_exhausted")


async def cascade_cancel(
    db: AsyncSession,
    job_id: str,
    *,
    reason: str,
    _depth: int = 0,
) -> List[str]:
    """Walk descendants of ``job_id`` (rows with
    ``parent_job_id = job_id``, transitively) and flip any active
    rows to ``status='cancelled'`` with ``error_message=reason`` and
    ``completed_at=now()``.

    Returns the list of IDs that were actually flipped (so the
    caller can emit per-row WS events). Does NOT commit — caller's
    transaction scope owns the commit.

    Bounded by ``_MAX_CHAIN_DEPTH_GUARD`` to keep an unexpected
    cycle from looping. With ``subagent_max_depth=1`` in v1 there
    can be at most one level of children, but the helper is correct
    for arbitrary depth.
    """
    from app.db.models import BuildJob

    if _depth >= _MAX_CHAIN_DEPTH_GUARD:
        logger.warning(
            "[subagent_dispatcher] cascade_cancel hit guard depth %d at %s — "
            "stopping",
            _MAX_CHAIN_DEPTH_GUARD, job_id,
        )
        return []

    # Find all active children of this job.
    rows = (
        await db.execute(
            select(BuildJob.id).where(
                BuildJob.parent_job_id == job_id,
                BuildJob.status.in_(_ACTIVE_STATUSES),
            )
        )
    ).scalars().all()

    if not rows:
        return []

    now = datetime.utcnow()
    cancelled: List[str] = []
    for child_id in rows:
        # Flip the child. Use a targeted UPDATE so we don't race with
        # the child's own handler — if the handler finalizes the row
        # before this UPDATE lands, the WHERE clause's status filter
        # makes the UPDATE a no-op (existing terminal state preserved).
        result = await db.execute(
            update(BuildJob)
            .where(
                BuildJob.id == child_id,
                BuildJob.status.in_(_ACTIVE_STATUSES),
            )
            .values(
                status="cancelled",
                error_message=reason,
                completed_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        if (getattr(result, "rowcount", 0) or 0) > 0:
            cancelled.append(child_id)
            logger.info(
                "[subagent_dispatcher] cascade_cancel flipped child=%s parent=%s reason=%s",
                child_id, job_id, reason[:80] if reason else "",
            )

        # Recurse into the child's own children (grandchildren of
        # the original job_id).
        deeper = await cascade_cancel(
            db, child_id, reason=reason, _depth=_depth + 1,
        )
        cancelled.extend(deeper)

    return cancelled


# ──────────────────────────────────────────────────────────────────────
# Terminal-transition chokepoint
# ──────────────────────────────────────────────────────────────────────


_VALID_TERMINAL = {"completed", "failed", "cancelled", "timeout", "budget_exhausted"}


async def transition_job_status(
    db: AsyncSession,
    job_id: str,
    *,
    new_status: str,
    error_message: Optional[str] = None,
    propagate_cancel: bool = True,
) -> tuple[bool, List[str]]:
    """Terminal-transition chokepoint for sub-agent BuildJob rows.

    Returns ``(transitioned, cascaded_ids)``:
      - ``transitioned``: True if the UPDATE actually changed the
        row's status (False if the row was already in a terminal
        state — we preserve existing terminal stamps so a handler-
        owned ``error_message`` survives a later cascade attempt).
      - ``cascaded_ids``: list of descendant job_ids that were
        cancelled as a side-effect (empty unless
        ``propagate_cancel=True`` AND ``new_status`` is failure-shape).

    Does NOT commit — caller owns the transaction.

    Failure-shape statuses (failed / cancelled / timeout /
    budget_exhausted) propagate downward by default. Completed
    transitions do not propagate (a parent finishing successfully
    has no implication for the child, which runs on its own
    lifecycle).
    """
    from app.db.models import BuildJob

    if new_status not in _VALID_TERMINAL:
        raise ValueError(
            f"transition_job_status: new_status={new_status!r} is not a "
            f"valid terminal state ({sorted(_VALID_TERMINAL)})"
        )

    now = datetime.utcnow()
    # Only transition if the row is still active — terminal-state
    # preservation. The WHERE clause's status filter makes a race
    # with another writer a no-op rather than a clobber.
    result = await db.execute(
        update(BuildJob)
        .where(
            BuildJob.id == job_id,
            BuildJob.status.notin_(_TERMINAL_STATUSES),
        )
        .values(
            status=new_status,
            error_message=error_message,
            completed_at=now,
        )
        .execution_options(synchronize_session=False)
    )
    transitioned = (getattr(result, "rowcount", 0) or 0) > 0

    cascaded: List[str] = []
    if (
        transitioned
        and propagate_cancel
        and new_status in ("failed", "cancelled", "timeout", "budget_exhausted")
    ):
        cascade_reason = error_message or f"parent {new_status}"
        cascaded = await cascade_cancel(db, job_id, reason=cascade_reason)

    logger.info(
        "[subagent_dispatcher] transition_job_status job=%s new=%s "
        "transitioned=%s cascaded=%d",
        job_id, new_status, transitioned, len(cascaded),
    )
    return transitioned, cascaded
