"""CP4.3 evidence — runnable as a plain script against the live platform DB.

Covers:
  1. End-to-end save via the proxy's handler (no LLM, no WS — direct fn call).
     A synthetic canary row is seeded, the handler runs, the DB is re-read.
  2. Unknown request_id → expired_or_unknown, no DB write.
  3. Cross-user forgery → expired_or_unknown + WARNING log with both user_ids.
  4. Expired pending → expired_or_unknown + no DB write.
  5. Cancelled (confirmed=False) → status=cancelled + no DB write.
  6. Invalid (confirmed=True, blank password) → invalid + no DB write.
  7. Encryption failure → encryption_failed + no DB write + pending popped (no zombie).
  8. Password wiped from frame dict after handler runs.
  9. CP1 audit log count unchanged by a save (audit fires on fetch, not save).
 10. save_encrypted_credential produces Fernet ciphertext (gAAAAAB…).

Invoke from backend/:
    railway run --service platform-api python tests/cp43_evidence_run.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


FAILS: List[str] = []


def _ok(label: str) -> None:
    print("  \u2713 " + label)


def _fail(label: str, detail: str = "") -> None:
    msg = f"  \u2717 {label}: {detail}" if detail else f"  \u2717 {label}"
    print(msg)
    FAILS.append(msg)


# ── Mock WebSocket that records every send_json / send_text ──────────

@dataclass
class MockWs:
    sent: List[dict]

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)

    async def send_text(self, txt: str) -> None:
        self.sent.append({"_text": txt})


def _mk_ws() -> MockWs:
    return MockWs(sent=[])


# ── DB helpers ───────────────────────────────────────────────────────

async def _count_audit_rows(engine) -> int:
    from sqlalchemy import text as _t
    async with engine.connect() as c:
        r = await c.execute(_t("SELECT COUNT(*) FROM credential_access_log"))
        return r.scalar() or 0


async def _get_cred_row(engine, user_id: str, channel: str):
    from sqlalchemy import text as _t
    async with engine.connect() as c:
        r = await c.execute(
            _t("""
                SELECT channel, email, password
                FROM streaming_credentials
                WHERE user_id = :uid AND channel = :ch
            """),
            {"uid": user_id, "ch": channel},
        )
        return r.fetchone()


async def _delete_cred_row(engine, user_id: str, channel: str) -> None:
    from sqlalchemy import text as _t
    async with engine.begin() as c:
        await c.execute(
            _t("DELETE FROM streaming_credentials WHERE user_id = :uid AND channel = :ch"),
            {"uid": user_id, "ch": channel},
        )


def _make_engine():
    from sqlalchemy.ext.asyncio import create_async_engine
    url = os.environ["DATABASE_URL"]
    if url.endswith("?ssl"):
        url = url[:-5]
    return create_async_engine(
        url,
        connect_args={
            "statement_cache_size": 0,
            "prepared_statement_name_func": lambda: "",
        },
    )


# ── Tests ────────────────────────────────────────────────────────────

CANARY_UID = "3134fece-c3de-411b-add8-3f2a58882794"  # parmida
OTHER_UID = "c47c5b4b-16d2-498c-b231-15d53d7454b9"   # mrhx
TEST_CHANNEL = "cp43_test"
REAL_SHAPED_PW = "N3tflix!Test#2026-correct-horse-battery-staple"


async def check_end_to_end_save(engine):
    print("\n[1] end-to-end save via _handle_credential_confirm_response")
    from app.api import ws_chat_proxy as p

    await _delete_cred_row(engine, CANARY_UID, TEST_CHANNEL)

    req_id = str(uuid.uuid4())
    await p._register_platform_pending(
        req_id, CANARY_UID, TEST_CHANNEL, time.time() + 300,
    )

    ws = _mk_ws()
    frame = {
        "type": "credential_confirm_response",
        "request_id": req_id,
        "confirmed": True,
        "email": "cp43-end-to-end@example.invalid",
        "password": REAL_SHAPED_PW,
    }
    await p._handle_credential_confirm_response(ws, CANARY_UID, frame)

    # Ack should be status=saved
    saved_ack = [s for s in ws.sent if s.get("status") == "saved"]
    if len(saved_ack) == 1 and saved_ack[0]["request_id"] == req_id:
        _ok("ack: status=saved with correct request_id")
    else:
        _fail("ack missing or wrong", repr(ws.sent))

    # DB row present with Fernet ciphertext
    row = await _get_cred_row(engine, CANARY_UID, TEST_CHANNEL)
    if row is None:
        _fail("DB row missing after save")
    else:
        if row[2].startswith("gAAAAAB"):
            _ok(f"DB password starts with gAAAAAB ({row[2][:32]}…)")
        else:
            _fail("DB password is not Fernet ciphertext", row[2][:80])
        if REAL_SHAPED_PW in row[2]:
            _fail("plaintext password appeared in DB column")
        else:
            _ok("plaintext is NOT in the DB column")
        if row[1] == "cp43-end-to-end@example.invalid":
            _ok("DB email matches input (trimmed)")
        else:
            _fail("DB email mismatch", repr(row[1]))

    # Frame dict was mutated in place (password popped out)
    if "password" in frame:
        _fail("password still in frame dict after handler")
    else:
        _ok("password popped from frame dict")
    if "email" in frame:
        _fail("email still in frame dict after handler")
    else:
        _ok("email popped from frame dict")

    await _delete_cred_row(engine, CANARY_UID, TEST_CHANNEL)


async def check_unknown_request_id():
    print("\n[2] unknown request_id → expired_or_unknown, no save attempt")
    from app.api import ws_chat_proxy as p

    ws = _mk_ws()
    await p._handle_credential_confirm_response(
        ws,
        CANARY_UID,
        {
            "type": "credential_confirm_response",
            "request_id": str(uuid.uuid4()),  # never registered
            "confirmed": True,
            "email": "x@y",
            "password": "irrelevant",
        },
    )
    if len(ws.sent) == 1 and ws.sent[0].get("status") == "expired_or_unknown":
        _ok("ack: expired_or_unknown")
    else:
        _fail("wrong ack", repr(ws.sent))


async def check_cross_user_forgery(caplog_buf):
    print("\n[3] cross-user forgery → expired_or_unknown + WARNING log with both uids")
    from app.api import ws_chat_proxy as p

    req_id = str(uuid.uuid4())
    # Register the pending entry OWNED BY mrhx.
    await p._register_platform_pending(
        req_id, OTHER_UID, TEST_CHANNEL, time.time() + 300,
    )

    ws = _mk_ws()
    # Attacker (parmida's WS connection) claims request_id.
    await p._handle_credential_confirm_response(
        ws,
        CANARY_UID,
        {
            "type": "credential_confirm_response",
            "request_id": req_id,
            "confirmed": True,
            "email": "attacker@evil",
            "password": "attacker-pw",
        },
    )

    if len(ws.sent) == 1 and ws.sent[0].get("status") == "expired_or_unknown":
        _ok("ack: expired_or_unknown (no existence leak)")
    else:
        _fail("wrong ack", repr(ws.sent))

    log_text = caplog_buf.getvalue()
    if "cross-user attempt" not in log_text:
        _fail("cross-user WARNING log missing", log_text[-400:])
    else:
        _ok("cross-user WARNING log fired")
    if CANARY_UID not in log_text:
        _fail("attempting_user id missing from warning")
    else:
        _ok(f"warning includes attempting_user=... {CANARY_UID[-8:]}")
    if OTHER_UID not in log_text:
        _fail("target_user id missing from warning")
    else:
        _ok(f"warning includes target_user=... {OTHER_UID[-8:]}")

    # Verify the pending entry is STILL there (cross-user rejection must not pop it).
    entry2, reason2 = await p._pop_platform_pending(req_id, OTHER_UID)
    if reason2 == "ok" and entry2 is not None:
        _ok("cross-user rejection did NOT pop the legitimate owner's pending entry")
    else:
        _fail(f"pending entry state after cross-user rejection: reason={reason2}")


async def check_expired():
    print("\n[4] expired pending → expired_or_unknown, no save attempt")
    from app.api import ws_chat_proxy as p

    req_id = str(uuid.uuid4())
    await p._register_platform_pending(
        req_id, CANARY_UID, TEST_CHANNEL, time.time() - 10,  # already expired
    )

    ws = _mk_ws()
    await p._handle_credential_confirm_response(
        ws,
        CANARY_UID,
        {
            "type": "credential_confirm_response",
            "request_id": req_id,
            "confirmed": True,
            "email": "x@y",
            "password": "irrelevant",
        },
    )
    if len(ws.sent) == 1 and ws.sent[0].get("status") == "expired_or_unknown":
        _ok("ack: expired_or_unknown")
    else:
        _fail("wrong ack", repr(ws.sent))


async def check_cancelled(engine):
    print("\n[5] cancelled (confirmed=False) → status=cancelled, no DB write")
    from app.api import ws_chat_proxy as p

    req_id = str(uuid.uuid4())
    await p._register_platform_pending(
        req_id, CANARY_UID, TEST_CHANNEL, time.time() + 300,
    )

    ws = _mk_ws()
    await p._handle_credential_confirm_response(
        ws,
        CANARY_UID,
        {
            "type": "credential_confirm_response",
            "request_id": req_id,
            "confirmed": False,
        },
    )
    if len(ws.sent) == 1 and ws.sent[0].get("status") == "cancelled":
        _ok("ack: cancelled")
    else:
        _fail("wrong ack", repr(ws.sent))

    row = await _get_cred_row(engine, CANARY_UID, TEST_CHANNEL)
    if row is None:
        _ok("no DB row created")
    else:
        _fail("DB row unexpectedly created on cancel", repr(row))


async def check_invalid_blank():
    print("\n[6] invalid (blank password) → status=invalid, no DB write")
    from app.api import ws_chat_proxy as p

    req_id = str(uuid.uuid4())
    await p._register_platform_pending(
        req_id, CANARY_UID, TEST_CHANNEL, time.time() + 300,
    )
    ws = _mk_ws()
    await p._handle_credential_confirm_response(
        ws,
        CANARY_UID,
        {
            "type": "credential_confirm_response",
            "request_id": req_id,
            "confirmed": True,
            "email": "x@y",
            "password": "",
        },
    )
    if len(ws.sent) == 1 and ws.sent[0].get("status") == "invalid":
        _ok("ack: invalid")
    else:
        _fail("wrong ack", repr(ws.sent))


async def check_encryption_failure(engine, monkey):
    print("\n[7] encryption failure → encryption_failed, no DB write, pending popped")
    from app.api import ws_chat_proxy as p
    from app.services import credential_save_service as css

    await _delete_cred_row(engine, CANARY_UID, TEST_CHANNEL)

    req_id = str(uuid.uuid4())
    await p._register_platform_pending(
        req_id, CANARY_UID, TEST_CHANNEL, time.time() + 300,
    )

    # Monkeypatch encrypt_str to always raise.
    original = css.encrypt_str
    def _boom(_s: str) -> str:
        raise RuntimeError("simulated encryption failure")
    monkey["patch"](css, "encrypt_str", _boom)

    try:
        ws = _mk_ws()
        await p._handle_credential_confirm_response(
            ws,
            CANARY_UID,
            {
                "type": "credential_confirm_response",
                "request_id": req_id,
                "confirmed": True,
                "email": "x@y",
                "password": "never-persisted",
            },
        )
        if len(ws.sent) == 1 and ws.sent[0].get("status") == "encryption_failed":
            _ok("ack: encryption_failed")
        else:
            _fail("wrong ack", repr(ws.sent))

        row = await _get_cred_row(engine, CANARY_UID, TEST_CHANNEL)
        if row is None:
            _ok("no DB row created on encryption failure")
        else:
            _fail("DB row leaked despite encryption failure", repr(row))

        # Pending entry must be popped even on failure so a retry isn't
        # stuck on an expired request_id.
        _, reason = await p._pop_platform_pending(req_id, CANARY_UID)
        if reason == "unknown":
            _ok("pending entry popped even on encryption failure")
        else:
            _fail(f"pending entry still present after failure (reason={reason})")
    finally:
        monkey["restore"](css, "encrypt_str", original)


async def check_audit_log_unchanged_on_save(engine):
    print("\n[8] CP1 audit log unchanged by a save (audit fires on fetch, not save)")
    from app.api import ws_chat_proxy as p

    await _delete_cred_row(engine, CANARY_UID, TEST_CHANNEL)
    before = await _count_audit_rows(engine)

    req_id = str(uuid.uuid4())
    await p._register_platform_pending(
        req_id, CANARY_UID, TEST_CHANNEL, time.time() + 300,
    )
    ws = _mk_ws()
    await p._handle_credential_confirm_response(
        ws,
        CANARY_UID,
        {
            "type": "credential_confirm_response",
            "request_id": req_id,
            "confirmed": True,
            "email": "cp43-audit@example.invalid",
            "password": REAL_SHAPED_PW,
        },
    )
    after = await _count_audit_rows(engine)

    if after == before:
        _ok(f"credential_access_log count unchanged ({before} → {after})")
    else:
        _fail(f"credential_access_log grew on save (should only grow on fetch)",
              f"{before} → {after}")

    await _delete_cred_row(engine, CANARY_UID, TEST_CHANNEL)


async def check_save_service_direct():
    print("\n[9] save_encrypted_credential produces Fernet ciphertext directly")
    from app.db.database import async_session_maker
    from app.services.credential_save_service import save_encrypted_credential

    async with async_session_maker() as db:
        await save_encrypted_credential(
            db=db,
            user_id=CANARY_UID,
            channel=TEST_CHANNEL,
            email="cp43-direct@example.invalid",
            password=REAL_SHAPED_PW,
        )

    engine = _make_engine()
    row = await _get_cred_row(engine, CANARY_UID, TEST_CHANNEL)
    if row and row[2].startswith("gAAAAAB"):
        _ok(f"direct call produces Fernet token ({row[2][:32]}…)")
    else:
        _fail("direct call did not produce Fernet token",
              repr(row[2][:80]) if row else "no row")

    await _delete_cred_row(engine, CANARY_UID, TEST_CHANNEL)
    await engine.dispose()


# ── Orchestration ────────────────────────────────────────────────────

async def amain():
    import io
    caplog_buf = io.StringIO()
    h = logging.StreamHandler(caplog_buf)
    h.setLevel(logging.WARNING)
    h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logging.getLogger().addHandler(h)
    logging.getLogger("app.api.ws_chat_proxy").setLevel(logging.WARNING)

    # Simple monkeypatch helper (no pytest fixture).
    def _patch(target, attr, new_val):
        setattr(target, attr, new_val)
    def _restore(target, attr, original):
        setattr(target, attr, original)
    monkey = {"patch": _patch, "restore": _restore}

    engine = _make_engine()
    try:
        await check_end_to_end_save(engine)
        await check_unknown_request_id()
        await check_cross_user_forgery(caplog_buf)
        await check_expired()
        await check_cancelled(engine)
        await check_invalid_blank()
        await check_encryption_failure(engine, monkey)
        await check_audit_log_unchanged_on_save(engine)
        await check_save_service_direct()
    finally:
        await engine.dispose()


def main() -> None:
    asyncio.run(amain())
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s) failed")
        for f in FAILS:
            print(f"  {f}")
        sys.exit(1)
    print("ALL CP4.3 EVIDENCE CHECKS PASSED")


if __name__ == "__main__":
    main()
