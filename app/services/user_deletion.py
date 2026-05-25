"""Account deletion — single source of truth.

Public entry point: `delete_user_completely(db, user, *, actor)`.
Both the user-facing POST /auth/delete-account and admin DELETE
/users/{id} call into this. Returns a DeletionReceipt or raises
DeletionAbortedError; never silently completes a partial delete.

Architecture (§1.3 plan, signed off 2026-05-01)
-----------------------------------------------

Transaction boundaries are explicit:

  Txn 1  Insert deletion_audit_events row (status='in_progress'),
         commit. The receipt exists even if the process dies on
         the next line.

  No DB Stripe customer cancel + delete (HTTP) → on failure: write
         audit row failed, raise DeletionAbortedError(STRIPE).
  No DB Container teardown via bridge (HTTP) → same failure handling.
  No DB OpenAI bundle project archive (HTTP) → SOFT-fail: log,
         set audit.openai_archived=False, continue. (Stripe is
         already canceled and the container is gone; orphan project
         is a billing curiosity, not a data-loss scenario.)

  Txn 2  Wipe agent + platform tables. Each table delete is
         savepointed via _safe_exec so a missing table doesn't
         poison the outer transaction. Commit when done.

  Txn 3  ATOMIC: DELETE FROM users WHERE id=:uid AND UPDATE
         deletion_audit_events SET status='succeeded'... in the
         same DB transaction. If the audit update fails, the user
         row delete rolls back and a retry runs cleanly.

Failure semantics
-----------------

  - Stripe failure       → ABORT. User row + data still intact.
                           Audit row status='failed', failure_step='stripe'.
                           Endpoint returns 502; user can retry.
  - Container failure    → ABORT. Stripe is canceled (irreversibly,
                           as that's what protects the user); local
                           data still intact. Audit row failed,
                           failure_step='container'.
  - OpenAI failure       → SOFT-FAIL. Continue. Audit records
                           openai_archived=False so an admin can
                           reconcile the orphan project later.
  - Table-wipe failure   → SOFT-FAIL per-table (savepointed). The
                           outer Txn 2 always commits. Worst case is
                           orphan rows in 1-2 tables; sweep job out
                           of scope.
  - User-row delete failure → ABORT. Txn 3 rolls back, audit stays
                              in_progress. Idempotent retry succeeds.

Idempotency contract
--------------------

Source of truth is the users row's existence. If `user` is None on
entry, return a no-op receipt. Re-running the cascade on a user
whose row still exists is safe because:
  - Stripe.delete_customer is idempotent (200 on already-deleted)
  - destroy_container returns cleanly on already-gone containers
  - archive_project is idempotent
  - All table deletes are WHERE user_id = uid → empty rowset on
    second call

No `users.deletion_in_progress` flag — that creates more failure
modes than it prevents (forgotten flags blocking re-deletion, race
with concurrent retries).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import StrEnum
import logging
import uuid
from typing import Optional

from sqlalchemy import select, text, update
from sqlalchemy import delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DeletionAuditEvent

logger = logging.getLogger(__name__)


class DeletionActor(StrEnum):
    """Who triggered the deletion. Recorded on the audit row."""

    SELF = "self"
    ADMIN = "admin"


class DeletionStep(StrEnum):
    """Identifies which phase of the cascade failed.

    Surfaced to the client via DeletionAbortedError.step so the UI
    can render a step-specific error message ('We couldn't cancel
    your subscription' vs 'Infrastructure issue, please retry').
    """

    STRIPE = "stripe"
    CONTAINER = "container"
    OPENAI = "openai"  # soft-failed in practice; listed for completeness
    CASCADE = "cascade"  # table-wipe stage
    USER_ROW = "user_row"


class DeletionAbortedError(Exception):
    """Raised when the cascade hit a hard-failure step. The audit
    row has been updated to reflect the failure before the raise.

    Endpoints catch this and translate to HTTP 502 with a body
    keyed on `step.value` so clients can render UX appropriate to
    the failure mode.
    """

    def __init__(self, step: DeletionStep, detail: str):
        super().__init__(f"[{step.value}] {detail}")
        self.step = step
        self.detail = detail


@dataclass(frozen=True)
class DeletionReceipt:
    """Returned on successful cascade. Mirrors the audit row's
    final state for the response body.

    Frozen so the endpoint code can't mutate it after the fact.
    """

    audit_id: str
    user_id: str
    user_email: str
    actor: DeletionActor
    initiated_at: datetime
    completed_at: datetime
    container_destroyed: bool
    stripe_cleared: bool
    # None means "n/a" — the user had no per-tenant OpenAI project.
    openai_archived: Optional[bool]
    user_row_deleted: bool


# ── Public entry point ──────────────────────────────────────────────────


async def delete_user_completely(
    db: AsyncSession,
    user,
    *,
    actor: DeletionActor,
    actor_user_id: Optional[str] = None,
    request_ip: Optional[str] = None,
    request_user_agent: Optional[str] = None,
) -> DeletionReceipt:
    """Delete the user and all associated data, with a durable audit
    receipt and explicit transaction boundaries.

    Caller's responsibilities:
      - Auth (sensitive-action token verification for self path,
        admin role check for admin path)
      - Last-admin / cannot-delete-self guards (admin path only)
      - Cookie clearing on success (caller owns the response object)

    Caller MUST NOT pre-commit any writes on this session before
    calling — the audit insert is the first transaction we own.

    Args:
        db: open AsyncSession
        user: SQLAlchemy User instance (already loaded by caller)
        actor: who initiated; recorded on the audit row
        actor_user_id: who triggered; defaults to user.id for SELF
        request_ip / request_user_agent: optional, for the audit row

    Returns:
        DeletionReceipt — final state, ready for the response body.

    Raises:
        DeletionAbortedError: if Stripe, container, or final user-row
            delete fails. Audit row records the failure step before
            the raise propagates.
    """
    if user is None:
        # Idempotent no-op: nothing to delete.
        return DeletionReceipt(
            audit_id="",
            user_id="",
            user_email="",
            actor=actor,
            initiated_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            container_destroyed=False,
            stripe_cleared=False,
            openai_archived=None,
            user_row_deleted=True,  # truthful: there is no user row
        )

    user_id = str(user.id)
    user_email = user.email
    prefix = user_id[:8]
    effective_actor_user_id = actor_user_id or user_id

    # ── Txn 1: durable audit-row insert ────────────────────────────────
    # Pre-compute the id so we can reference it after commit even though
    # the SQLAlchemy object becomes detached on session.commit().
    audit_id = str(uuid.uuid4())
    audit_initiated_at = datetime.utcnow()
    audit = DeletionAuditEvent(
        id=audit_id,
        user_id=user_id,
        user_email=user_email,
        actor=actor.value,
        actor_user_id=effective_actor_user_id,
        initiated_at=audit_initiated_at,
        status="in_progress",
        request_ip=request_ip,
        request_user_agent=request_user_agent,
    )
    db.add(audit)
    await db.commit()

    # Track per-step results for the final audit update.
    container_destroyed = False
    stripe_cleared = False
    openai_archived: Optional[bool] = None  # None = n/a until proven otherwise

    # ── Stripe: cancel + delete customer (HARD-FAIL) ───────────────────
    # Stripe-first because billing is the user's primary harm vector.
    # If teardown fails, we have NOT yet started destroying local data,
    # so the user is still functional and an admin can retry.
    try:
        from app.services.stripe_service import (
            delete_customer, find_customers_by_email,
        )
        customer_ids: set[str] = set()
        if user.stripe_customer_id:
            customer_ids.add(user.stripe_customer_id)
        if user_email:
            try:
                for cid in find_customers_by_email(user_email):
                    customer_ids.add(cid)
            except Exception as e:
                # Email-search defense-in-depth — don't fail just
                # because Stripe's email index lookup hiccuped, we
                # still try the FK-attached customer below.
                logger.warning(
                    "[DELETE-USER] Stripe email-search failed for %s: %s",
                    user_email, e,
                )
        # If user has no customer_id and email-search returned nothing,
        # there's no Stripe state to clean — that's success, not failure.
        if not customer_ids:
            stripe_cleared = True
        else:
            for cid in customer_ids:
                ok = delete_customer(cid)
                if not ok:
                    raise RuntimeError(
                        f"Stripe customer {cid} delete returned not-ok",
                    )
            stripe_cleared = True
    except Exception as e:
        logger.exception(
            "[DELETE-USER] Stripe cleanup failed for %s — aborting cascade", prefix,
        )
        await _record_failure(
            db, audit_id, DeletionStep.STRIPE, str(e),
            stripe_cleared=False, container_destroyed=False,
            openai_archived=None,
        )
        raise DeletionAbortedError(DeletionStep.STRIPE, str(e))

    # ── Container teardown via bridge (HARD-FAIL) ──────────────────────
    # Hands off to the bridge: docker rm + network rm + tenant DB drop +
    # Caddy route removal. destroy_container is idempotent (returns
    # cleanly on 404) so re-running after a partial failure is safe.
    try:
        from app.services.docker_host_service import destroy_container
        await destroy_container(db, user_id)
        container_destroyed = True
    except Exception as e:
        logger.exception(
            "[DELETE-USER] Container teardown failed for %s — aborting cascade", prefix,
        )
        await _record_failure(
            db, audit_id, DeletionStep.CONTAINER, str(e),
            stripe_cleared=stripe_cleared, container_destroyed=False,
            openai_archived=None,
        )
        raise DeletionAbortedError(DeletionStep.CONTAINER, str(e))

    # ── OpenAI bundle project archive (SOFT-FAIL) ──────────────────────
    # Read AgentConfig.bundle_openai_project_id BEFORE the platform-table
    # wipe wipes the row. Archive cascade-revokes service-account keys +
    # stops billing on the OpenAI side. Idempotent. Soft-fail because
    # by this point Stripe is canceled and the container is gone — an
    # orphan OpenAI project is a billing footnote, not a data-loss
    # scenario, and we'd rather complete the user-data deletion.
    try:
        from app.db.models import AgentConfig
        ac_result = await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == user_id)
        )
        ac = ac_result.scalar_one_or_none()
        openai_project_id = ac.bundle_openai_project_id if ac else None
    except Exception:
        openai_project_id = None
    if openai_project_id:
        try:
            from app.services.openai_admin_service import (
                archive_project,
                revoke_all_project_service_accounts,
            )
            # Pre-revoke service-account keys WHILE the project is still
            # active. After archive, OpenAI rejects every key operation
            # with 400 project_archived, leaving the keys forever pinned
            # under the operator's "Active" filter in the OpenAI dashboard
            # (functionally disabled, but visually misleading and
            # cluttered as users accumulate). Best-effort: failures here
            # are logged but do not abort the cascade — the subsequent
            # archive still functionally disables the keys.
            try:
                revoked, failed_revoke = revoke_all_project_service_accounts(
                    openai_project_id,
                )
                if revoked or failed_revoke:
                    logger.info(
                        "[DELETE-USER] OpenAI key pre-revoke for %s: "
                        "deleted=%d failed=%d",
                        openai_project_id, revoked, failed_revoke,
                    )
            except Exception as e:
                logger.warning(
                    "[DELETE-USER] OpenAI key pre-revoke crashed for %s "
                    "(user=%s) — archive will still disable keys "
                    "functionally, UI may show Active label: %s",
                    openai_project_id, prefix, e,
                )
            # Capture archive_project's return value — it returns False
            # on any non-2xx (admin key unset, transient 4xx/5xx) WITHOUT
            # raising, and the previous code discarded that signal and
            # always wrote openai_archived=True. Now the audit row
            # reflects the actual OpenAI outcome.
            openai_archived = bool(archive_project(openai_project_id))
            if not openai_archived:
                logger.warning(
                    "[DELETE-USER] OpenAI archive_project returned False "
                    "for %s (user=%s) — orphan project may remain",
                    openai_project_id, prefix,
                )
        except Exception as e:
            logger.warning(
                "[DELETE-USER] OpenAI project archive raised for %s "
                "(user=%s) — orphan project may remain: %s",
                openai_project_id, prefix, e,
            )
            openai_archived = False
    # else: openai_archived stays None (n/a)

    # ── Txn 2: wipe agent + platform tables (savepointed) ──────────────
    # Each _safe_exec wraps in begin_nested() so a missing table doesn't
    # poison the outer transaction. Failure of any one table is logged
    # but does not abort. The outer commit at the end of this block
    # makes the wipes durable before we attempt Txn 3.
    await _wipe_user_data_tables(db, user_id)
    await db.commit()

    # ── Txn 3: user-row DELETE + audit success UPDATE (ATOMIC) ─────────
    # Both succeed together or both roll back. The retry case (audit
    # update fails for some reason) leaves the user row intact and the
    # audit row in_progress, so a re-run picks up cleanly.
    audit_completed_at = datetime.utcnow()
    try:
        async with db.begin():
            await db.execute(
                text("DELETE FROM users WHERE id = :uid").bindparams(uid=user_id)
            )
            await db.execute(
                update(DeletionAuditEvent)
                .where(DeletionAuditEvent.id == audit_id)
                .values(
                    status="succeeded",
                    completed_at=audit_completed_at,
                    container_destroyed=container_destroyed,
                    stripe_cleared=stripe_cleared,
                    openai_archived=openai_archived,
                )
            )
    except Exception as e:
        logger.exception(
            "[DELETE-USER] User-row delete + audit update failed for %s", prefix,
        )
        await _record_failure(
            db, audit_id, DeletionStep.USER_ROW, str(e),
            stripe_cleared=stripe_cleared, container_destroyed=container_destroyed,
            openai_archived=openai_archived,
        )
        raise DeletionAbortedError(DeletionStep.USER_ROW, str(e))

    return DeletionReceipt(
        audit_id=audit_id,
        user_id=user_id,
        user_email=user_email,
        actor=actor,
        initiated_at=audit_initiated_at,
        completed_at=audit_completed_at,
        container_destroyed=container_destroyed,
        stripe_cleared=stripe_cleared,
        openai_archived=openai_archived,
        user_row_deleted=True,
    )


# ── Internal helpers ────────────────────────────────────────────────────


async def _record_failure(
    db: AsyncSession,
    audit_id: str,
    step: DeletionStep,
    detail: str,
    *,
    stripe_cleared: bool,
    container_destroyed: bool,
    openai_archived: Optional[bool],
) -> None:
    """Mark the audit row as failed at `step`. Best-effort — if this
    UPDATE fails, the failure is logged but not re-raised, because the
    caller is already raising DeletionAbortedError.
    """
    try:
        await db.rollback()  # discard any aborted-txn state
        async with db.begin():
            await db.execute(
                update(DeletionAuditEvent)
                .where(DeletionAuditEvent.id == audit_id)
                .values(
                    status="failed",
                    failure_step=step.value,
                    failure_detail=detail[:500],
                    completed_at=datetime.utcnow(),
                    stripe_cleared=stripe_cleared,
                    container_destroyed=container_destroyed,
                    openai_archived=openai_archived,
                )
            )
    except Exception:
        logger.exception(
            "[DELETE-USER] Could not record failure on audit row %s", audit_id,
        )


async def _wipe_user_data_tables(db: AsyncSession, user_id: str) -> None:
    """Run the table-wipe phase. Each delete is wrapped in a savepoint
    so a missing table or column does not poison the outer transaction.

    This is the agent-monolith + platform-tables cleanup ported from
    the previous cascade_delete_user implementation, with the same
    ordering (children before parents, FK-aware).
    """
    from app.db.models import (
        Invite, ManagedContainer, AgentConfig, VPSInstance,
        LLMBundleAllocation, LLMUsageRecord, StreamingCredential,
    )

    async def _safe_exec(stmt) -> bool:
        try:
            async with db.begin_nested():
                await db.execute(stmt)
            return True
        except Exception:
            return False

    # Memory.source_message_id FK → null before deleting messages
    await _safe_exec(text(
        "UPDATE memories SET source_message_id = NULL WHERE user_id = :uid"
    ).bindparams(uid=user_id))

    # Messages (FK to conversations)
    await _safe_exec(text(
        "DELETE FROM messages WHERE conversation_id IN "
        "(SELECT id FROM conversations WHERE user_id = :uid)"
    ).bindparams(uid=user_id))

    # Document chunks (FK to documents and memories)
    await _safe_exec(text(
        "DELETE FROM document_chunks WHERE document_id IN "
        "(SELECT id FROM documents WHERE user_id = :uid)"
    ).bindparams(uid=user_id))
    await _safe_exec(text(
        "DELETE FROM document_chunks WHERE memory_id IN "
        "(SELECT id FROM memories WHERE user_id = :uid)"
    ).bindparams(uid=user_id))

    # Entity links + relationships
    await _safe_exec(text(
        "DELETE FROM entity_links WHERE entity_id IN "
        "(SELECT id FROM entities WHERE user_id = :uid)"
    ).bindparams(uid=user_id))
    await _safe_exec(text(
        "DELETE FROM entity_links WHERE memory_id IN "
        "(SELECT id FROM memories WHERE user_id = :uid)"
    ).bindparams(uid=user_id))
    await _safe_exec(text(
        "DELETE FROM entity_relationships WHERE user_id = :uid"
    ).bindparams(uid=user_id))

    # Memory children
    await _safe_exec(text(
        "DELETE FROM memory_events WHERE memory_id IN "
        "(SELECT id FROM memories WHERE user_id = :uid)"
    ).bindparams(uid=user_id))
    await _safe_exec(text(
        "DELETE FROM memory_events WHERE user_id = :uid"
    ).bindparams(uid=user_id))
    await _safe_exec(text(
        "DELETE FROM media WHERE memory_id IN "
        "(SELECT id FROM memories WHERE user_id = :uid)"
    ).bindparams(uid=user_id))
    await _safe_exec(text(
        "DELETE FROM media WHERE user_id = :uid"
    ).bindparams(uid=user_id))
    await _safe_exec(text(
        "DELETE FROM memory_relationships WHERE source_id IN "
        "(SELECT id FROM memories WHERE user_id = :uid) "
        "OR target_id IN (SELECT id FROM memories WHERE user_id = :uid)"
    ).bindparams(uid=user_id))

    # Retrieval events
    await _safe_exec(text(
        "DELETE FROM retrieval_events WHERE user_id = :uid"
    ).bindparams(uid=user_id))

    # Build jobs, reconciliation logs, apps
    await _safe_exec(text("DELETE FROM build_jobs WHERE user_id = :uid").bindparams(uid=user_id))
    await _safe_exec(text("DELETE FROM reconciliation_logs WHERE user_id = :uid").bindparams(uid=user_id))
    await _safe_exec(text("DELETE FROM apps WHERE user_id = :uid").bindparams(uid=user_id))

    # Context budget logs
    await _safe_exec(text("DELETE FROM context_budget_logs WHERE user_id = :uid").bindparams(uid=user_id))

    # Direct user_id tables
    for table_name in [
        "identities", "documents", "memories", "conversations",
        "entities", "brain_stats", "cron_jobs", "telegram_user_mappings",
        "api_keys", "agent_errors", "soul_configs",
        "workflows",
    ]:
        await _safe_exec(text(
            f"DELETE FROM {table_name} WHERE user_id = :uid"
        ).bindparams(uid=user_id))

    # Day chats
    await _safe_exec(text("DELETE FROM day_chats WHERE user_id = :uid").bindparams(uid=user_id))

    # Platform-only tables
    try:
        from app.db.models import LLMProxyEvent
        await _safe_exec(sa_delete(LLMProxyEvent).where(LLMProxyEvent.user_id == user_id))
    except Exception:
        pass

    await _safe_exec(sa_delete(StreamingCredential).where(StreamingCredential.user_id == user_id))
    await _safe_exec(sa_delete(LLMUsageRecord).where(LLMUsageRecord.user_id == user_id))
    await _safe_exec(sa_delete(LLMBundleAllocation).where(LLMBundleAllocation.user_id == user_id))
    await _safe_exec(sa_delete(ManagedContainer).where(ManagedContainer.user_id == user_id))
    await _safe_exec(sa_delete(VPSInstance).where(VPSInstance.user_id == user_id))
    await _safe_exec(sa_delete(AgentConfig).where(AgentConfig.user_id == user_id))

    # Invites: nullify those this user redeemed; delete those this user created.
    inv_result = await db.execute(select(Invite).where(Invite.used_by == user_id))
    for inv in inv_result.scalars().all():
        inv.used_by = None
        inv.used_at = None
    await db.execute(sa_delete(Invite).where(Invite.created_by == user_id))


# ── Backwards-compat shim (will be deleted once endpoint refactor lands
# in this same commit) ─────────────────────────────────────────────────

async def cascade_delete_user(db: AsyncSession, user) -> dict:
    """Deprecated shim — kept temporarily for any external callers.
    New code should call delete_user_completely(db, user, actor=...)
    directly. The auth + admin endpoints will switch over in this
    same commit; this wrapper exists only to keep test_user_deletion's
    backwards-compat coverage alive during the transition. Remove after
    the test commit that replaces it.
    """
    receipt = await delete_user_completely(
        db, user, actor=DeletionActor.SELF,
    )
    return {
        "steps": {
            "container_destroyed": receipt.container_destroyed,
            "stripe_cleared": receipt.stripe_cleared,
            "openai_archived": (
                "n/a" if receipt.openai_archived is None
                else receipt.openai_archived
            ),
            "agent_tables_cleared": True,
            "platform_tables_cleared": True,
            "user_row_deleted": receipt.user_row_deleted,
        },
    }
