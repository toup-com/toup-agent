"""
T1b — ConnectorVault service.

Encrypted CRUD on `connector_identities`, with an audit-then-act,
fail-closed contract on every write. Mirrors the streaming-credential
audit pattern in `app/api/streaming.py:128-141`.

Public surface (all `async`):

  put(db, user_id, connector_id, *, access_token, ...)
      Create or replace the identity for (user, connector). Audit row
      is `event_type` (default `EVENT_CONNECTED`). Returns the
      ConnectorIdentity row.

  get(db, user_id, connector_id) -> Optional[DecryptedIdentity]
      Return the decrypted identity in a typed struct. No audit —
      vault reads inside the dispatcher are an implementation detail
      of the dispatcher's outer audit row (T1e). Returns None when
      either the row doesn't exist OR ciphertext has been zeroed
      (disconnected case).

  update(db, identity_id, *, access_token=None, ...)
      Refresh-time partial update — typically called by the dispatcher
      after a successful provider.refresh(). Audit row is `event_type`
      (default `EVENT_REFRESH_SUCCEEDED`).

  mark_reauth_required(db, identity_id, *, reason)
      Flip status to 'reauth_required'. Audit row is `event_type`
      (default `EVENT_REFRESH_FAILED`). After the commit, fires the
      best-effort agent state hook (R30 §4.7) so the tenant emits the
      `connector.state` frame — fire-and-forget, never blocking.

  notify_agent_connector_state(user_id, connector_id, *, ok, error=None)
      The R30 §4.7 platform→agent state hook body (own session,
      fully best-effort). Public so other status-flip writers
      (e.g. the health probe's provider_down) can reuse it.

  disconnect(db, user_id, connector_id, *, event_type=EVENT_DISCONNECTED)
      Atomically: status='revoked' AND access_token_enc=NULL AND
      refresh_token_enc=NULL, all in one transaction (per
      `01-architecture.md` §3.5 — no 24h grace, the token is dead at
      the provider so the ciphertext is liability with no recovery
      benefit). Row is preserved for audit history. Audit-then-act
      pattern still applies: the disconnected event is committed in a
      prior transaction.

  list_active(db, user_id) -> list[ConnectorIdentity]
      All ConnectorIdentity rows for the user where status='active'.
      No decryption (caller doesn't need plaintext for the system-
      prompt connected-services line or for cache-invalidation
      decisions). Returns SQLAlchemy rows.

Audit-then-act, fail-closed:
  Every write method runs as TWO transactions on the same session:
    1. INSERT ConnectorEvent row → COMMIT.  (durable point)
    2. Vault write → COMMIT.

  If step 1 raises, the vault write never happens. The caller sees
  the audit-failure exception. This is the same fail-closed contract
  streaming credentials use: if we cannot record what we are about
  to do, we do not do it.

What this module does NOT do:
  * No HTTP, no provider knowledge ("what is Gmail?").
  * No concurrent-refresh coordination — that lock lives in T1e
    (the dispatcher). The vault is safe under concurrent writes
    via the unique constraint + DB transaction semantics; it does
    not own coalescing.
  * No reads at the dispatcher's tool-call audit layer — those are
    audited by the dispatcher itself.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import and_, select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    ConnectorEvent,
    ConnectorIdentity,
    EVENT_CONNECTED,
    EVENT_DISCONNECTED,
    EVENT_REFRESH_FAILED,
    EVENT_REFRESH_SUCCEEDED,
)
from app.services.credential_crypto import decrypt_str, encrypt_str

logger = logging.getLogger(__name__)


# ─── Public types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class DecryptedIdentity:
    """Frozen plaintext result struct from `vault.get`.

    Lives only inside the dispatcher's call frame. Never serialised,
    never logged, never sent over the wire.
    """

    id: str
    user_id: str
    connector_id: str
    provider_account_id: Optional[str]
    scopes: list[str] = field(default_factory=list)
    # Plaintext access token. Caller must not log this.
    access_token: str = ""
    # Plaintext refresh token. None when the provider didn't return one.
    refresh_token: Optional[str] = None
    access_expires_at: Optional[datetime] = None
    status: str = "active"
    connected_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # True when the user has flipped this connection to read-only.
    # The MCP tool filter and the dispatcher both gate mutating tool
    # calls on this flag.
    read_only: bool = False


class VaultAuditError(RuntimeError):
    """Raised when the audit-then-act gate fails. Caller should treat
    as fail-closed: the vault write did NOT happen."""


# ─── Internal helpers ─────────────────────────────────────────────────


async def _audit_then_commit(
    db: AsyncSession,
    *,
    user_id: str,
    connector_id: str,
    event_type: str,
    metadata: Optional[dict] = None,
    channel: Optional[str] = None,
    tool_name: Optional[str] = None,
    agent_request_id: Optional[str] = None,
) -> str:
    """Insert a ConnectorEvent row and commit. Returns the event id.

    On failure: rollback, log, raise VaultAuditError. The caller's
    follow-up vault write is implicitly skipped because the exception
    propagates."""
    event = ConnectorEvent(
        user_id=user_id,
        connector_id=connector_id,
        event_type=event_type,
        channel=channel,
        tool_name=tool_name,
        agent_request_id=agent_request_id,
        metadata_json=json.dumps(metadata) if metadata else None,
    )
    db.add(event)
    try:
        await db.commit()
    except Exception as e:
        logger.exception(
            "[connector_vault] audit insert failed for user=%s connector=%s "
            "event=%s — fail-closed, vault write will NOT happen",
            user_id[:8], connector_id, event_type,
        )
        await db.rollback()
        raise VaultAuditError(
            f"audit insert failed: {type(e).__name__}: {e}"
        ) from e
    return event.id


def _to_struct(row: ConnectorIdentity) -> DecryptedIdentity:
    """Decrypt the row into the typed result struct. Raises if the
    row's ciphertext is missing — caller is expected to gate on that."""
    return DecryptedIdentity(
        id=row.id,
        user_id=row.user_id,
        connector_id=row.connector_id,
        provider_account_id=row.provider_account_id,
        scopes=(json.loads(row.scopes_json) if row.scopes_json else []),
        access_token=decrypt_str(row.access_token_enc),
        refresh_token=(
            decrypt_str(row.refresh_token_enc) if row.refresh_token_enc else None
        ),
        access_expires_at=row.access_expires_at,
        status=row.status,
        connected_at=row.connected_at,
        last_used_at=row.last_used_at,
        metadata=(json.loads(row.metadata_json) if row.metadata_json else {}),
        read_only=bool(getattr(row, "read_only", False)),
    )


# ─── Public API ───────────────────────────────────────────────────────


async def put(
    db: AsyncSession,
    user_id: str,
    connector_id: str,
    *,
    access_token: str,
    refresh_token: Optional[str] = None,
    access_expires_at: Optional[datetime] = None,
    scopes: Optional[list[str]] = None,
    provider_account_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    event_type: str = EVENT_CONNECTED,
) -> ConnectorIdentity:
    """Create or replace the (user, connector) identity. Always
    transitions status to 'active' (a fresh OAuth dance overwrites a
    previously-revoked identity cleanly).

    The default `event_type=EVENT_CONNECTED` matches the OAuth-callback
    use case. Re-connecting after a `reauth_required` flip should pass
    `event_type=EVENT_REAUTH_COMPLETED` so the audit trail tells the
    right story.
    """
    if not access_token:
        raise ValueError("vault.put: access_token must be non-empty")

    # 1. Audit FIRST. If this raises, the vault write never happens.
    await _audit_then_commit(
        db,
        user_id=user_id,
        connector_id=connector_id,
        event_type=event_type,
        metadata={
            "provider_account_id": provider_account_id,
            "scopes": scopes or [],
        },
    )

    # 2. Vault write. UPSERT — if a previous identity exists for this
    #    (user, connector), reuse it; otherwise INSERT new. The unique
    #    constraint guarantees there's at most one.
    existing = (
        await db.execute(
            select(ConnectorIdentity).where(
                and_(
                    ConnectorIdentity.user_id == user_id,
                    ConnectorIdentity.connector_id == connector_id,
                )
            )
        )
    ).scalar_one_or_none()

    access_enc = encrypt_str(access_token)
    refresh_enc = encrypt_str(refresh_token) if refresh_token else None
    scopes_blob = json.dumps(scopes) if scopes else None
    metadata_blob = json.dumps(metadata) if metadata else None

    if existing is not None:
        existing.access_token_enc = access_enc
        existing.refresh_token_enc = refresh_enc
        existing.access_expires_at = access_expires_at
        existing.scopes_json = scopes_blob
        existing.provider_account_id = provider_account_id
        existing.metadata_json = metadata_blob
        existing.status = "active"
        existing.connected_at = datetime.utcnow()
        identity = existing
    else:
        identity = ConnectorIdentity(
            user_id=user_id,
            connector_id=connector_id,
            access_token_enc=access_enc,
            refresh_token_enc=refresh_enc,
            access_expires_at=access_expires_at,
            scopes_json=scopes_blob,
            provider_account_id=provider_account_id,
            metadata_json=metadata_blob,
            status="active",
        )
        db.add(identity)

    await db.commit()
    return identity


async def get(
    db: AsyncSession,
    user_id: str,
    connector_id: str,
) -> Optional[DecryptedIdentity]:
    """Return the decrypted identity for (user, connector), or None.

    Returns None when:
      - The row doesn't exist (user never connected).
      - The row exists but `access_token_enc` is NULL (disconnected /
        revoked — ciphertext was zeroed; nothing usable to return).

    Does NOT filter by status — the dispatcher checks status itself
    and routes `reauth_required` to a re-auth prompt.
    """
    row = (
        await db.execute(
            select(ConnectorIdentity).where(
                and_(
                    ConnectorIdentity.user_id == user_id,
                    ConnectorIdentity.connector_id == connector_id,
                )
            )
        )
    ).scalar_one_or_none()
    if row is None or row.access_token_enc is None:
        return None
    return _to_struct(row)


async def update(
    db: AsyncSession,
    identity_id: str,
    *,
    access_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
    access_expires_at: Optional[datetime] = None,
    event_type: str = EVENT_REFRESH_SUCCEEDED,
) -> ConnectorIdentity:
    """Partial update — typically called from the dispatcher after a
    successful provider.refresh(). At least one of `access_token`,
    `refresh_token`, `access_expires_at` must be provided.

    The default `event_type` reflects the refresh-success path. Pass a
    different value if you're calling for another reason (rare).
    """
    row = (
        await db.execute(
            select(ConnectorIdentity).where(ConnectorIdentity.id == identity_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise ValueError(f"vault.update: identity {identity_id!r} not found")
    if access_token is None and refresh_token is None and access_expires_at is None:
        raise ValueError("vault.update: at least one field must be provided")

    # 1. Audit FIRST.
    await _audit_then_commit(
        db,
        user_id=row.user_id,
        connector_id=row.connector_id,
        event_type=event_type,
    )

    # 2. Vault write.
    if access_token is not None:
        row.access_token_enc = encrypt_str(access_token)
    if refresh_token is not None:
        row.refresh_token_enc = encrypt_str(refresh_token)
    if access_expires_at is not None:
        row.access_expires_at = access_expires_at
    row.last_used_at = datetime.utcnow()
    await db.commit()
    return row


async def mark_reauth_required(
    db: AsyncSession,
    identity_id: str,
    *,
    reason: str,
    event_type: str = EVENT_REFRESH_FAILED,
) -> None:
    """Flip the identity's status to 'reauth_required'.

    Ciphertext is NOT zeroed — the user may still recover by walking
    through the OAuth flow again, and we'd rather preserve the audit
    trail of which refresh failed and why. Only `disconnect` zeroes
    ciphertext.
    """
    row = (
        await db.execute(
            select(ConnectorIdentity).where(ConnectorIdentity.id == identity_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise ValueError(
            f"vault.mark_reauth_required: identity {identity_id!r} not found"
        )

    # 1. Audit FIRST. Reason goes into metadata so ops can grep it.
    await _audit_then_commit(
        db,
        user_id=row.user_id,
        connector_id=row.connector_id,
        event_type=event_type,
        metadata={"reason": reason[:500]},
    )

    # 2. Vault write — status flip only.
    row.status = "reauth_required"
    await db.commit()

    # 3. Round 30 §4.7: best-effort platform→agent state hook, AFTER
    #    the flip is durable. The agent answers by emitting the
    #    `connector.state` frame (state "expired") so the app's cards
    #    flip without a poll — before this round no connector state
    #    change was visible to clients at all. Fire-and-forget on its
    #    OWN session inside the spawned task: this session is the
    #    dispatcher's, and the hook must never delay or break the
    #    vault write it decorates.
    try:
        from app.services.background_tasks import spawn as _spawn_bg
        _spawn_bg(notify_agent_connector_state(
            row.user_id, row.connector_id, ok=False, error="reauth_required",
        ))
    except Exception as e:  # noqa: BLE001 — the vault write already landed
        logger.warning(
            "[connector_vault] reauth state hook spawn failed for "
            "user=%s connector=%s: %s",
            row.user_id[:8], row.connector_id, e,
        )


async def notify_agent_connector_state(
    user_id: str,
    connector_id: str,
    *,
    ok: bool,
    error: Optional[str] = None,
) -> None:
    """POST a connector state flip to the tenant agent's
    `/api/automations/_connector_connected` hook (CONTRACTS-R30 §4.7).

    Reuses the exact client path OAuth-callback step 6a uses
    (`automations_proxy._get_agent_target` + the shared
    `agent_http` client) rather than inventing a second HTTP lane.
    Opens its OWN session: it runs as a spawned background task, after
    the caller's session has moved on (the repo's own-session-per-
    fallback convention). Fully best-effort — every failure is a WARN,
    never an exception; a dark tenant answers 404 and that is normal.

    Public on purpose: `mark_reauth_required` is the only vault-side
    status-flip to a blocked state today, but the health probe's
    `provider_down` flip (connector_health_probe.py) should call this
    same helper when it gains the hook.
    """
    try:
        from app.api.automations_proxy import _get_agent_target
        from app.db.database import async_session_maker
        from app.services.agent_http import get_agent_http_client

        async with async_session_maker() as adb:
            target = await _get_agent_target(user_id, adb)
        if target is None:
            return
        agent_url, agent_key = target
        resp = await get_agent_http_client().post(
            f"{agent_url.rstrip('/')}/api/automations/_connector_connected",
            json={"connector_id": connector_id, "ok": ok, "error": error},
            headers={"X-Agent-Key": agent_key},
            timeout=5.0,
        )
        if resp.status_code not in (200, 404):
            logger.warning(
                "[connector_vault] agent state hook %s for user=%s connector=%s",
                resp.status_code, user_id[:8], connector_id,
            )
    except Exception as e:  # noqa: BLE001 — the flip must never depend on it
        logger.warning(
            "[connector_vault] agent state hook failed for user=%s "
            "connector=%s: %s",
            user_id[:8], connector_id, e,
        )


async def disconnect(
    db: AsyncSession,
    user_id: str,
    connector_id: str,
    *,
    event_type: str = EVENT_DISCONNECTED,
) -> None:
    """Atomically transition an identity to disconnected:
    `status='revoked'` AND `access_token_enc=NULL` AND
    `refresh_token_enc=NULL`, all in one DB transaction (per
    architecture §3.5).

    No-op if no identity exists for (user, connector). The audit row
    is still written (ops gets the signal that disconnect was
    requested even when the identity was already gone).
    """
    # 1. Audit FIRST.
    await _audit_then_commit(
        db,
        user_id=user_id,
        connector_id=connector_id,
        event_type=event_type,
    )

    # 2. Vault write. Use a single UPDATE statement so status flip and
    #    ciphertext zero are guaranteed atomic at the DB level —
    #    architecture §3.5 explicitly requires "in a single transaction".
    await db.execute(
        sa_update(ConnectorIdentity)
        .where(
            and_(
                ConnectorIdentity.user_id == user_id,
                ConnectorIdentity.connector_id == connector_id,
            )
        )
        .values(
            status="revoked",
            access_token_enc=None,
            refresh_token_enc=None,
            last_used_at=datetime.utcnow(),
        )
    )
    await db.commit()


async def list_active(
    db: AsyncSession,
    user_id: str,
) -> list[ConnectorIdentity]:
    """Return all active identities for the user (no decryption).

    Used by the agent's system-prompt "Connected: Gmail, Calendar"
    line (T1f) and by the cache-invalidation logic (T1h). Caller does
    not need plaintext, so we hand back SQLAlchemy rows holding only
    ciphertext + metadata.
    """
    rows = (
        await db.execute(
            select(ConnectorIdentity)
            .where(
                and_(
                    ConnectorIdentity.user_id == user_id,
                    ConnectorIdentity.status == "active",
                )
            )
            .order_by(ConnectorIdentity.connector_id)
        )
    ).scalars().all()
    return list(rows)


# Statuses the MCP filter treats as "visible to the agent" — the user
# has set up this connector at least once, so the agent should be aware
# of the tools EVEN IF the identity is currently in trouble (expired
# tokens, transient provider outage). Surfacing the tool lets the
# dispatcher return `ConnectorReauthRequired` / `ConnectorProviderDown`
# to the LLM, which then tells the user "Please reconnect your Gmail"
# instead of silently falling back to the browser tool (which fails
# because Chromium isn't installed in the runtime container).
#
# Excludes `revoked` because that one is terminal — the user
# explicitly disconnected. Agent shouldn't pretend the tool exists.
_VISIBLE_STATUSES = ("active", "reauth_required", "provider_down")


async def list_visible_to_agent(
    db: AsyncSession,
    user_id: str,
) -> list[ConnectorIdentity]:
    """Return all identities the agent should be able to *see* — the
    set used by the MCP filter when deciding which connector tools to
    expose in `tools/list`.

    Wider than `list_active`: also includes `reauth_required` and
    `provider_down`. The dispatcher gates *invocation* on
    `status == "active"`; this method gates *discoverability*. The two
    being different is intentional — a hidden tool means the agent
    chooses a worse fallback (browser → fails); a visible-but-failing
    tool means the agent surfaces the structured error to the user,
    which is the production-grade UX.
    """
    rows = (
        await db.execute(
            select(ConnectorIdentity)
            .where(
                and_(
                    ConnectorIdentity.user_id == user_id,
                    ConnectorIdentity.status.in_(_VISIBLE_STATUSES),
                )
            )
            .order_by(ConnectorIdentity.connector_id)
        )
    ).scalars().all()
    return list(rows)
