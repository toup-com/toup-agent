"""`/credits/agent-deduct` no longer exempts a self-reported `system.*`.

The endpoint is manual/BYOK mode's metering surface: the agent talked to
the provider on its own key and REPORTS what that cost. Until 2026-09-30 a
body whose `operation_type` started with "system." returned
`system_op_exempt` before the bundle/manual guard — a client-controlled
exemption: any X-Agent-Key holder could label a report "system.*" and
have it not charged. The platform's own system calls never reach this
endpoint (openai_agent_service skips the deduct report for a system
operation), so the label was only ever worn by reports nobody should
trust. Synthetic users; no real accounts.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

import pytest
from sqlalchemy import select

pytestmark = pytest.mark.asyncio


@pytest.fixture
def credit_flags(monkeypatch):
    import app.services.credit_service as CS

    def _set(**flags):
        for name, value in flags.items():
            monkeypatch.setattr(CS.settings, name, value, raising=False)
    return _set


async def _mk_user() -> str:
    from app.db import async_session_maker, User
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"s-{uuid.uuid4().hex[:10]}@example.com", hashed_password="x",
                    name="t", email_verified_at=datetime.utcnow()))
        await db.commit()
    return uid


async def _wallet(uid: str, plan: str = "100"):
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    async with async_session_maker() as db:
        b = await credit_service.get_or_create_balance(db, uid)
        b.message_credits_remaining = Decimal(plan)
        b.purchased_credits_remaining = Decimal("0")
        b.message_credits_used_today = Decimal("0")
        await db.commit()


async def _ledger(uid: str):
    from app.db import async_session_maker, CreditLedger
    async with async_session_maker() as db:
        return list((await db.execute(select(CreditLedger).where(CreditLedger.user_id == uid))).scalars().all())


async def _deduct(uid: str, llm_mode: str, key_suffix: str, **body_kw):
    """Drive the route the way the agent does: X-Agent-Key auth against the
    user's AgentConfig row."""
    from app.api.credits import AgentDeductRequest, agent_deduct
    from app.db import async_session_maker
    from app.db.models import AgentConfig

    key = f"agent-key-{uid[:8]}-{key_suffix}"
    async with async_session_maker() as db:
        db.add(AgentConfig(user_id=uid, llm_mode=llm_mode, agent_api_key=key))
        await db.commit()
    body = AgentDeductRequest(user_id=uid, model="gpt-5.5", provider="openai",
                              input_tokens=20_000, output_tokens=2_000, **body_kw)
    async with async_session_maker() as db:
        return await agent_deduct(body, x_agent_key=key, db=db)


@pytest.mark.parametrize("label", ["system.cache_warm", "system.anything", "SYSTEM.free", "system."])
async def test_a_manual_report_labelled_system_is_charged_like_any_report(credit_flags, label):
    credit_flags(credit_enforcement_enabled=True, credit_cap_admission_control=False)
    from app.services.credit_service import tokens_to_credits
    expected = tokens_to_credits("gpt-5.5", 20_000, 2_000)
    assert expected > 0

    uid = await _mk_user()
    await _wallet(uid, "100")
    resp = await _deduct(uid, "manual", "m", operation_type=label, idempotency_key=f"rep-{label}")

    assert resp.reason != "system_op_exempt"
    assert Decimal(str(resp.amount_charged)) == expected, "the label bought nothing"
    assert resp.success is True and resp.balance_after == float(Decimal("100") - expected)
    # Wallet initialization writes plan-grant rows. Inspect only this report's
    # debit, identified by the idempotency key supplied to agent_deduct.
    rows = [r for r in await _ledger(uid) if r.idempotency_key == f"rep-{label}"]
    assert len(rows) == 1 and Decimal(rows[0].amount) == -expected
    # The label is still RECORDED, so an audit can see who claimed it.
    assert (rows[0].metadata_json or {}).get("operation_type") == label


async def test_a_bundle_report_labelled_system_still_takes_the_bundle_path(credit_flags):
    """Bundle mode is metered at the proxy; this endpoint never charges it
    (idempotent_hit=True, real balance returned). A system label changes
    nothing about that either — no charge, and no `system_op_exempt`."""
    credit_flags(credit_enforcement_enabled=True, credit_cap_admission_control=False)
    uid = await _mk_user()
    await _wallet(uid, "100")
    resp = await _deduct(uid, "bundle", "b", operation_type="system.cache_warm", idempotency_key="rep-bundle")
    assert resp.reason != "system_op_exempt"
    assert resp.idempotent_hit is True and resp.amount_charged == 0.0 and resp.balance_after == 100.0
    assert not [r for r in await _ledger(uid) if r.idempotency_key == "rep-bundle"]


async def test_a_user_report_is_unchanged(credit_flags):
    credit_flags(credit_enforcement_enabled=True, credit_cap_admission_control=False)
    from app.services.credit_service import tokens_to_credits
    uid = await _mk_user()
    await _wallet(uid, "100")
    resp = await _deduct(uid, "manual", "u", operation_type="user.chat", idempotency_key="rep-user")
    assert Decimal(str(resp.amount_charged)) == tokens_to_credits("gpt-5.5", 20_000, 2_000)
