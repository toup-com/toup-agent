"""
T1b — ConnectorVault tests.

Smoke matrix from the T1b spec:

  1. Round-trip: put → get → tokens decrypt to original plaintext.
  2. Disconnect zeroes ciphertext in same txn: query the row directly,
     both `_enc` columns are NULL, status is `revoked`, row exists with
     audit history.
  3. Audit-then-act fail-closed: monkey-patch the audit insert to raise
     → vault write never executes, no `ConnectorIdentity` row created.
  4. Encryption is real: read raw `access_token_enc` from DB, confirm
     it's not the plaintext (Fernet output starts with `gAAAAA`).
  5. KEK rotation: encrypt with primary, set primary=B + previous=A,
     decrypt still works.
  6. `list_active` excludes `revoked` and `reauth_required`.

Plus a few invariants worth nailing down explicitly:
  - `update` audits + writes; no-arg call rejected.
  - `mark_reauth_required` keeps ciphertext intact (only `disconnect`
    zeroes).
  - `get` returns None when ciphertext is zeroed (treats disconnected
    as "no usable token", per the typed-struct contract).
  - Re-`put` on an existing (user, connector) updates in place (no
    UNIQUE violation).

Each test seeds at most ONE user (avoids the SQLite is_canary trap)
and uses the inline `_ensure_sqlite_fks_on()` pattern from T1a where
needed.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from sqlalchemy import select, text

from app.config import settings
from app.db.database import async_session_maker, engine
from app.db.models import (
    ConnectorEvent,
    ConnectorIdentity,
    EVENT_CONNECTED,
    EVENT_DISCONNECTED,
    EVENT_REFRESH_FAILED,
    EVENT_REFRESH_SUCCEEDED,
    User,
)
from app.services import connector_vault as vault
from app.services.credential_crypto import _multi_fernet, looks_encrypted


# ─── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _provision_encryption_key():
    """Set a fresh Fernet key on the settings object for the test
    duration. The crypto module's `_multi_fernet` is lru_cached on the
    settings values, so we clear the cache after each test to avoid
    leaking key state into the next."""
    prev_primary = settings.platform_encryption_key
    prev_previous = settings.platform_encryption_key_previous
    settings.platform_encryption_key = Fernet.generate_key().decode()
    settings.platform_encryption_key_previous = ""
    _multi_fernet.cache_clear()
    try:
        yield
    finally:
        settings.platform_encryption_key = prev_primary
        settings.platform_encryption_key_previous = prev_previous
        _multi_fernet.cache_clear()


@pytest_asyncio.fixture
async def alice_user_id() -> str:
    """One user. Single user per test → SQLite is_canary unique index
    is fine (the trap that bit T0c)."""
    async with async_session_maker() as db:
        uid = str(uuid.uuid4())
        db.add(User(
            id=uid,
            email=f"{uid[:8]}@example.com",
            hashed_password="x",
            name="Alice",
        ))
        await db.commit()
    return uid


# ─── 1. Round-trip ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_put_then_get_roundtrips_plaintext(alice_user_id):
    """put() encrypts, get() decrypts — plaintext survives intact."""
    expires = datetime.utcnow() + timedelta(hours=1)
    async with async_session_maker() as db:
        await vault.put(
            db,
            alice_user_id,
            "gmail",
            access_token="ya29.access_token_value_xxx",
            refresh_token="1//refresh_token_value_yyy",
            access_expires_at=expires,
            scopes=["https://www.googleapis.com/auth/gmail.modify"],
            provider_account_id="alice@gmail.com",
            metadata={"granted_email": "alice@gmail.com"},
        )

    async with async_session_maker() as db:
        ident = await vault.get(db, alice_user_id, "gmail")

    assert ident is not None
    assert ident.access_token == "ya29.access_token_value_xxx"
    assert ident.refresh_token == "1//refresh_token_value_yyy"
    assert ident.scopes == ["https://www.googleapis.com/auth/gmail.modify"]
    assert ident.provider_account_id == "alice@gmail.com"
    assert ident.metadata == {"granted_email": "alice@gmail.com"}
    assert ident.status == "active"
    # Microsecond round-trip can lose precision on SQLite — compare to seconds.
    assert int(ident.access_expires_at.timestamp()) == int(expires.timestamp())


# ─── 2. Disconnect zeroes ciphertext in same txn ──────────────────────


@pytest.mark.asyncio
async def test_disconnect_zeroes_ciphertext_and_revokes(alice_user_id):
    """status='revoked' AND both `_enc` columns NULL — single UPDATE."""
    async with async_session_maker() as db:
        await vault.put(
            db, alice_user_id, "gmail",
            access_token="aaa", refresh_token="bbb",
        )

    async with async_session_maker() as db:
        await vault.disconnect(db, alice_user_id, "gmail")

    # Query the raw row directly — must show ciphertext is gone.
    async with async_session_maker() as db:
        row = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == alice_user_id
            )
        )).scalar_one()
        assert row.status == "revoked", "status must be 'revoked'"
        assert row.access_token_enc is None, "access ciphertext must be NULL"
        assert row.refresh_token_enc is None, "refresh ciphertext must be NULL"
        # Row must STILL EXIST — disconnect preserves audit history.
        assert row.id is not None

        # Audit history is intact — connected event, then disconnected event.
        events = (await db.execute(
            select(ConnectorEvent)
            .where(ConnectorEvent.user_id == alice_user_id)
            .order_by(ConnectorEvent.occurred_at)
        )).scalars().all()
        event_types = [e.event_type for e in events]
        assert EVENT_CONNECTED in event_types
        assert EVENT_DISCONNECTED in event_types


@pytest.mark.asyncio
async def test_get_returns_none_after_disconnect(alice_user_id):
    """Per the typed-struct contract, get() returns None when
    ciphertext is gone — disconnected identity has no usable token."""
    async with async_session_maker() as db:
        await vault.put(
            db, alice_user_id, "gmail",
            access_token="t", refresh_token="r",
        )
        await vault.disconnect(db, alice_user_id, "gmail")

    async with async_session_maker() as db:
        ident = await vault.get(db, alice_user_id, "gmail")
    assert ident is None


# ─── 3. Audit-then-act fail-closed ────────────────────────────────────


@pytest.mark.asyncio
async def test_put_skips_vault_write_when_audit_fails(
    alice_user_id, monkeypatch,
):
    """Monkey-patch the audit-commit helper to raise. Vault write must
    not execute → no ConnectorIdentity row exists for this (user, connector)."""

    async def boom(db, **kwargs):
        raise vault.VaultAuditError("simulated audit table write failure")

    monkeypatch.setattr(
        "app.services.connector_vault._audit_then_commit", boom,
    )

    async with async_session_maker() as db:
        with pytest.raises(vault.VaultAuditError):
            await vault.put(
                db, alice_user_id, "gmail",
                access_token="should_not_persist",
            )

    async with async_session_maker() as db:
        row = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == alice_user_id
            )
        )).scalar_one_or_none()
        assert row is None, (
            "audit failed → vault write must have been skipped, "
            "but a ConnectorIdentity row exists"
        )


@pytest.mark.asyncio
async def test_disconnect_skips_vault_write_when_audit_fails(
    alice_user_id, monkeypatch,
):
    """Same fail-closed contract for disconnect: if audit fails, the
    identity is NOT zeroed/revoked. Operator sees the audit-failure
    log and can retry."""
    # Seed an active identity.
    async with async_session_maker() as db:
        await vault.put(
            db, alice_user_id, "gmail",
            access_token="t", refresh_token="r",
        )

    async def boom(db, **kwargs):
        raise vault.VaultAuditError("simulated")

    monkeypatch.setattr(
        "app.services.connector_vault._audit_then_commit", boom,
    )

    async with async_session_maker() as db:
        with pytest.raises(vault.VaultAuditError):
            await vault.disconnect(db, alice_user_id, "gmail")

    # Identity must still be active with intact ciphertext.
    async with async_session_maker() as db:
        row = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == alice_user_id
            )
        )).scalar_one()
        assert row.status == "active"
        assert row.access_token_enc is not None
        assert row.refresh_token_enc is not None


# ─── 4. Encryption is real ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stored_ciphertext_is_not_plaintext(alice_user_id):
    """Read the raw column from the DB; it must NOT contain the
    plaintext token, and must look like Fernet ciphertext."""
    secret = "ya29.SECRET_TOKEN_DO_NOT_LEAK_xxxxxx"
    async with async_session_maker() as db:
        await vault.put(
            db, alice_user_id, "gmail", access_token=secret,
        )

    async with async_session_maker() as db:
        row = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == alice_user_id
            )
        )).scalar_one()
        assert row.access_token_enc is not None
        # Plaintext must NOT appear anywhere in the stored blob.
        assert secret not in row.access_token_enc
        # Must look like Fernet ciphertext (gAAAAAB prefix per
        # credential_crypto._FERNET_PREFIX).
        assert looks_encrypted(row.access_token_enc), (
            f"expected Fernet ciphertext (gAAAAAB...), got: "
            f"{row.access_token_enc[:32]!r}"
        )


# ─── 5. KEK rotation: decrypt across primary swap ────────────────────


@pytest.mark.asyncio
async def test_decrypt_works_after_kek_rotation(alice_user_id):
    """Encrypt with key A. Set primary=B + previous=A. decrypt_str
    inside vault.get must still return the original plaintext."""
    secret = "rotation_proof_token"
    # Encrypt under the current key (key A from the autouse fixture).
    async with async_session_maker() as db:
        await vault.put(
            db, alice_user_id, "gmail", access_token=secret,
        )
    key_a = settings.platform_encryption_key

    # Rotate: new key B as primary, A as previous.
    key_b = Fernet.generate_key().decode()
    settings.platform_encryption_key = key_b
    settings.platform_encryption_key_previous = key_a
    _multi_fernet.cache_clear()

    # decrypt_str (inside vault.get) must walk the multi-fernet chain
    # and find key A in the previous list.
    async with async_session_maker() as db:
        ident = await vault.get(db, alice_user_id, "gmail")
    assert ident is not None
    assert ident.access_token == secret


# ─── 6. list_active filters by status ────────────────────────────────


@pytest.mark.asyncio
async def test_list_active_excludes_non_active_statuses(alice_user_id):
    """Three identities for one user: gmail (active), gcal (revoked),
    github (reauth_required). list_active returns only gmail."""
    async with async_session_maker() as db:
        await vault.put(db, alice_user_id, "gmail", access_token="g_a")
        await vault.put(db, alice_user_id, "gcal", access_token="g_c")
        await vault.put(db, alice_user_id, "github", access_token="g_h")

    # Disconnect gcal → revoked.
    async with async_session_maker() as db:
        await vault.disconnect(db, alice_user_id, "gcal")

    # Mark github as needing reauth.
    async with async_session_maker() as db:
        github_id = (await db.execute(
            select(ConnectorIdentity.id).where(
                ConnectorIdentity.connector_id == "github"
            )
        )).scalar_one()
        await vault.mark_reauth_required(
            db, github_id, reason="provider returned 401",
        )

    async with async_session_maker() as db:
        active = await vault.list_active(db, alice_user_id)
    assert {i.connector_id for i in active} == {"gmail"}


# ─── Bonus invariants ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_re_put_updates_in_place_no_unique_violation(alice_user_id):
    """Calling put() twice for the same (user, connector) does not
    duplicate-row — the existing row is updated in place. New plaintext
    survives the round-trip."""
    async with async_session_maker() as db:
        await vault.put(
            db, alice_user_id, "gmail",
            access_token="first", refresh_token="r1",
        )
        await vault.put(
            db, alice_user_id, "gmail",
            access_token="second", refresh_token="r2",
        )

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == alice_user_id
            )
        )).scalars().all()
        assert len(rows) == 1, (
            "re-put must UPDATE in place, not INSERT a duplicate"
        )

        ident = await vault.get(db, alice_user_id, "gmail")
        assert ident.access_token == "second"
        assert ident.refresh_token == "r2"


@pytest.mark.asyncio
async def test_update_partial_refresh_preserves_other_fields(alice_user_id):
    """update() with only `access_token` set leaves refresh_token
    untouched. Validates the typical refresh-flow call shape."""
    async with async_session_maker() as db:
        identity = await vault.put(
            db, alice_user_id, "gmail",
            access_token="orig_access",
            refresh_token="orig_refresh",
            access_expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        identity_id = identity.id

    new_expires = datetime.utcnow() + timedelta(hours=2)
    async with async_session_maker() as db:
        await vault.update(
            db, identity_id,
            access_token="rotated_access",
            access_expires_at=new_expires,
            event_type=EVENT_REFRESH_SUCCEEDED,
        )

    async with async_session_maker() as db:
        ident = await vault.get(db, alice_user_id, "gmail")
        assert ident.access_token == "rotated_access"
        assert ident.refresh_token == "orig_refresh", (
            "partial update must NOT clobber refresh_token"
        )


@pytest.mark.asyncio
async def test_update_with_no_fields_rejected(alice_user_id):
    """Defensive: update() with no field args is a programming error,
    must raise."""
    async with async_session_maker() as db:
        identity = await vault.put(
            db, alice_user_id, "gmail", access_token="x",
        )
        with pytest.raises(ValueError, match="at least one field"):
            await vault.update(db, identity.id)


@pytest.mark.asyncio
async def test_mark_reauth_required_keeps_ciphertext(alice_user_id):
    """Only `disconnect` zeroes ciphertext. `mark_reauth_required`
    only flips status, so the user can recover by walking through the
    OAuth flow again without losing audit info."""
    async with async_session_maker() as db:
        identity = await vault.put(
            db, alice_user_id, "gmail",
            access_token="t", refresh_token="r",
        )
        identity_id = identity.id

    async with async_session_maker() as db:
        await vault.mark_reauth_required(
            db, identity_id, reason="refresh returned 401",
        )

    async with async_session_maker() as db:
        row = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == alice_user_id
            )
        )).scalar_one()
        assert row.status == "reauth_required"
        assert row.access_token_enc is not None
        assert row.refresh_token_enc is not None

        # Audit row is the refresh-failed type, with reason in metadata.
        event = (await db.execute(
            select(ConnectorEvent)
            .where(ConnectorEvent.event_type == EVENT_REFRESH_FAILED)
        )).scalar_one()
        meta = json.loads(event.metadata_json)
        assert meta["reason"] == "refresh returned 401"


@pytest.mark.asyncio
async def test_disconnect_is_noop_when_no_identity(alice_user_id):
    """disconnect() on a (user, connector) pair that doesn't exist
    must NOT raise — but the audit row IS still written (operators
    care that the user pressed the button even when there was
    nothing to disconnect)."""
    async with async_session_maker() as db:
        await vault.disconnect(db, alice_user_id, "gmail")

    async with async_session_maker() as db:
        # No identity row — and that's fine.
        row = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == alice_user_id
            )
        )).scalar_one_or_none()
        assert row is None
        # But the audit row IS there.
        event = (await db.execute(
            select(ConnectorEvent).where(
                ConnectorEvent.event_type == EVENT_DISCONNECTED
            )
        )).scalar_one()
        assert event.user_id == alice_user_id
        assert event.connector_id == "gmail"


@pytest.mark.asyncio
async def test_get_returns_none_for_unknown_pair(alice_user_id):
    async with async_session_maker() as db:
        ident = await vault.get(db, alice_user_id, "never_connected")
    assert ident is None
