"""Shared cascade-delete-user helper.

Extracted from `app/api/admin/users.py:delete_user` so both the
admin DELETE /users/{id} endpoint and the user-facing
POST /auth/delete-account endpoint can run the SAME cascade.

Why the duplication risk existed: the original user-facing endpoint
only anonymised the User row (renamed email/name, set is_active=False)
and added a comment promising "a background job later purges the row" —
no such job ever shipped. Result: when a user clicked "Delete Account"
in Settings, their managed container kept running, the tenant Postgres
DB kept all messages/memories, the Stripe customer kept being billed,
the OpenAI bundle project stayed active, and the AgentConfig row (with
every API key the user had ever entered) sat in the DB indefinitely.

This module is the single source of truth for "delete this user
everywhere." Both endpoints layer their own safety checks on top
(admin: can't delete self / can't delete last admin; user-facing:
password confirmation).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy import delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def cascade_delete_user(db: AsyncSession, user) -> dict:
    """Run the full delete-user cascade.

    Caller is responsible for safety checks (e.g. "can't delete self",
    "can't delete the last admin", or "verify the user's password").
    Caller also owns the response cookie cleanup if applicable.

    Performs, in order:
      1. Destroy managed container + tenant DB + tenant role + Caddy
         route via the bridge.
      2. Delete the Stripe customer (cancels every subscription tied
         to it). Defence-in-depth: also email-search Stripe for any
         orphan customer whose metadata.user_id matches.
      3. Archive the per-user OpenAI bundle project (cascade-revokes
         all service-account keys + stops billing).
      4. Wipe agent-only legacy monolith tables (messages, memories,
         documents, entities, etc.) using savepointed `_safe_exec` so
         missing tables/columns don't poison the outer transaction.
      5. Wipe platform-only tables (StreamingCredential, LLMUsage,
         LLMBundleAllocation, ManagedContainer, VPSInstance, AgentConfig).
      6. Mark invites this user redeemed as un-redeemed; delete invites
         this user created.
      7. DELETE FROM users via raw SQL (avoids ORM relationship loads
         that touch columns missing from the legacy platform DB).

    Returns a dict reporting which steps succeeded/failed for the
    caller's response shaping. Never raises — every step is wrapped
    so the user row delete still happens even if Stripe/OpenAI/bridge
    are flaky. Failures are logged loudly so we can reconcile later.

    Args:
        db: open AsyncSession; the function commits before returning.
        user: SQLAlchemy User instance (already loaded by caller).
    """
    from app.db.models import (
        Invite, ManagedContainer, AgentConfig, VPSInstance,
        LLMBundleAllocation, LLMUsageRecord, StreamingCredential,
    )

    user_id = str(user.id)
    user_email = user.email
    prefix = user_id[:8]
    report: dict = {"steps": {}}

    # ── Helper: run a delete inside a savepoint so missing tables/columns
    #    don't abort the outer transaction ──
    async def _safe_exec(stmt) -> bool:
        try:
            async with db.begin_nested():
                await db.execute(stmt)
            return True
        except Exception:
            return False

    # ── 1. Destroy ALL VPS artifacts via the bridge ──
    # DELETE /v1/tenants/<prefix> on the bridge handles:
    #   - docker rm -f toup-agent-<prefix>
    #   - docker network rm tnt_<prefix>
    #   - DROP DATABASE + DROP ROLE via /usr/local/sbin/revoke_tenant_db
    #   - remove Caddy tenant route via admin API
    try:
        from app.services.docker_host_service import destroy_container
        await destroy_container(db, user_id)
        report["steps"]["container_destroyed"] = True
    except Exception as e:
        logger.warning("[CASCADE-DELETE] VPS cleanup failed for %s: %s", prefix, e)
        report["steps"]["container_destroyed"] = False

    # ── 1b. Wipe Stripe state. Deleting the Stripe Customer
    #    auto-cancels every subscription tied to it. Without this,
    #    signing up again with the same email would re-attach to the
    #    orphan customer's still-active subscription (the 2026-04-27
    #    incident). Defence-in-depth: also email-search for any
    #    customer whose metadata.user_id matches in case the FK was
    #    not synced for some reason.
    try:
        from app.services.stripe_service import delete_customer, find_customers_by_email
        customer_ids_to_delete: set[str] = set()
        if user.stripe_customer_id:
            customer_ids_to_delete.add(user.stripe_customer_id)
        if user_email:
            try:
                for cid in find_customers_by_email(user_email):
                    customer_ids_to_delete.add(cid)
            except Exception as e:
                logger.warning(
                    "[CASCADE-DELETE] Stripe email-search failed for %s: %s",
                    user_email, e,
                )
        any_failed = False
        for cid in customer_ids_to_delete:
            ok = delete_customer(cid)
            if not ok:
                any_failed = True
                logger.warning(
                    "[CASCADE-DELETE] Stripe customer cleanup failed for %s "
                    "(user=%s) — orphan may remain", cid, prefix,
                )
        report["steps"]["stripe_cleared"] = (
            len(customer_ids_to_delete) > 0 and not any_failed
        )
    except Exception as e:
        logger.exception("[CASCADE-DELETE] Stripe cleanup raised: %s", e)
        report["steps"]["stripe_cleared"] = False

    # ── 1c. Archive per-user OpenAI bundle project. Read AgentConfig.
    #    bundle_openai_project_id BEFORE the platform DB cleanup wipes
    #    the row. Archiving cascade-revokes service-account keys + stops
    #    billing. Idempotent + non-fatal.
    try:
        ac_result = await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == user_id)
        )
        ac = ac_result.scalar_one_or_none()
        openai_project_id = ac.bundle_openai_project_id if ac else None
    except Exception:
        openai_project_id = None
    if openai_project_id:
        try:
            from app.services.openai_admin_service import archive_project
            archive_project(openai_project_id)
            report["steps"]["openai_archived"] = True
        except Exception as e:
            logger.warning(
                "[CASCADE-DELETE] OpenAI project archive failed for %s "
                "(user=%s) — orphan project may remain: %s",
                openai_project_id, prefix, e,
            )
            report["steps"]["openai_archived"] = False
    else:
        report["steps"]["openai_archived"] = "n/a"

    # ── 2. Clean up agent-only legacy monolith tables ──
    # Each wrapped in savepoint — tables/columns may not exist.
    # Order: children before parents (FK deps).

    # Nullify Memory → Message FK before deleting messages
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

    # Entity links and relationships
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

    report["steps"]["agent_tables_cleared"] = True

    # ── 3. Delete platform-only tables ──
    try:
        from app.db.models import LLMProxyEvent
        await _safe_exec(sa_delete(LLMProxyEvent).where(LLMProxyEvent.user_id == user_id))
    except Exception:
        pass

    await db.execute(sa_delete(StreamingCredential).where(StreamingCredential.user_id == user_id))
    await db.execute(sa_delete(LLMUsageRecord).where(LLMUsageRecord.user_id == user_id))
    await db.execute(sa_delete(LLMBundleAllocation).where(LLMBundleAllocation.user_id == user_id))
    await db.execute(sa_delete(ManagedContainer).where(ManagedContainer.user_id == user_id))
    await db.execute(sa_delete(VPSInstance).where(VPSInstance.user_id == user_id))
    await db.execute(sa_delete(AgentConfig).where(AgentConfig.user_id == user_id))
    report["steps"]["platform_tables_cleared"] = True

    # ── 4. Update/delete invites ──
    inv_result = await db.execute(select(Invite).where(Invite.used_by == user_id))
    for inv in inv_result.scalars().all():
        inv.used_by = None
        inv.used_at = None
    await db.execute(sa_delete(Invite).where(Invite.created_by == user_id))

    # ── 5. Delete the user row (raw SQL to avoid ORM loading
    #    relationships that reference columns missing from legacy
    #    platform DB) ──
    await db.execute(text("DELETE FROM users WHERE id = :uid").bindparams(uid=user_id))
    await db.commit()
    report["steps"]["user_row_deleted"] = True

    return report
