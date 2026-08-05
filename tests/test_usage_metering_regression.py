"""Regression pins for the 2026-08-03 usage-tracking incident.

Two independent defects made the credit meter read as frozen while real money
was being spent. Both were confirmed against the production database before
being fixed; the numbers quoted below are measured, not illustrative.

DEFECT 1 — the hourly re-grant loop (the meter that never moved).
``reconcile_deferred_grants`` runs hourly and selected every balance with no
``grant_eligibility`` row whose ``first_user_id`` is that user. But the
tombstone is keyed by CANONICAL EMAIL HASH: a user whose hash was already
claimed by an earlier account of the same person can never own one —
``_claim_grant_tombstone`` hits the IntegrityError branch and, with
``free_grant_dedupe_enabled`` False (the default), returns "granted" WITHOUT
writing anything. So such a user stayed a candidate forever, and
``_apply_initial_grant`` ASSIGNS ``message_credits_remaining =
plan.message_credits_monthly`` rather than adding — silently resetting the
wallet to full every hour while leaving ``message_credits_used_today``
climbing. Production: two accounts with 48 and 117 repeat grants,
``integration_credits_remaining`` pinned at exactly 500.00, and a balance row
reading remaining=98.90 beside used_today=11.80.

DEFECT 2 — the daily cap zeroed costs already paid for.
A single heavy gpt-5.5 call quotes 26-28 credits; the free daily cap is 15. So
``_split_message_charge`` could NEVER satisfy the cap for exactly the calls
that cost the most: it denied, ``try_charge`` wrote an ``amount=0`` ledger row
carrying ``{"denied": true, "reason": "daily_cap_exceeded"}``, and
``llm_proxy._log_event`` never branched on ``result.success`` — so the answer
streamed anyway. Production: 274 such calls, $17.17 of provider spend, 59% of
all real-user LLM cost, served free.

The fix splits one policy in two: the cap gates ADMISSION of new work
(``check_balance`` is now cap-aware) and never zeroes an incurred cost
(``already_incurred=True``). Both halves are behind
``credit_cap_admission_control``, default False.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select

pytestmark = pytest.mark.asyncio


@pytest.fixture
def credit_flags(monkeypatch):
    """Set a credit flag on the settings object the code under test ACTUALLY
    reads, not on whatever ``app.config.settings`` currently resolves to.

    ``test_subagent_settings.py`` calls ``importlib.reload(app.config)``, which
    rebinds ``app.config.settings`` to a NEW instance while
    ``credit_service``'s module-level ``settings`` still references the old
    one. Patching via a fresh ``from app.config import settings`` therefore
    has no effect on the charge path whenever that file is collected first —
    the tests pass alone and fail in a full run. The same hazard is documented
    at test_subagent_credit_budget.py:142. Patch the module's own reference and
    the order stops mattering.
    """
    import app.services.credit_service as CS

    def _set(**flags):
        for name, value in flags.items():
            monkeypatch.setattr(CS.settings, name, value, raising=False)
    return _set


# ── helpers ───────────────────────────────────────────────────────────

async def _mk_user(email: str) -> str:
    from app.db import async_session_maker, User
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=uid, email=email.strip().lower(), hashed_password="x", name="t",
            email_verified_at=datetime.utcnow(),
        ))
        await db.commit()
    return uid


async def _balance(uid: str):
    from app.db import async_session_maker, CreditBalance
    async with async_session_maker() as db:
        return await db.get(CreditBalance, uid)


async def _grant_via_balance(uid: str) -> None:
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    async with async_session_maker() as db:
        await credit_service.get_or_create_balance(db, uid)
        await db.commit()


async def _plan_grant_rows(uid: str) -> int:
    from app.db import async_session_maker
    from app.db.models import CreditLedger, LEDGER_PLAN_GRANT
    async with async_session_maker() as db:
        return int((await db.execute(
            select(func.count()).select_from(CreditLedger).where(
                CreditLedger.user_id == uid,
                CreditLedger.event_type == LEDGER_PLAN_GRANT,
            )
        )).scalar() or 0)


async def _steal_tombstone(uid: str) -> None:
    """Reproduce the production shape: the user's canonical hash exists but is
    owned by a DIFFERENT (earlier) account, so this user can never own one."""
    from app.db import async_session_maker, User
    from app.db.models import GrantEligibility
    from app.services.email_canonical import canonical_email_hash
    async with async_session_maker() as db:
        user = await db.get(User, uid)
        h = canonical_email_hash(user.email)
        row = await db.get(GrantEligibility, h)
        assert row is not None, "expected the normal grant path to write one"
        row.first_user_id = str(uuid.uuid4())   # an earlier, now-deleted account
        await db.commit()


# ══════════════════════════════════════════════════════════════════════
# DEFECT 1 — the hourly re-grant loop
# ══════════════════════════════════════════════════════════════════════


async def test_already_granted_reads_the_ledger_not_only_the_tombstone():
    """The ledger is the honest PER-USER record. The tombstone is keyed by
    canonical email hash and provably cannot answer this question."""
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    uid = await _mk_user("loop-a@example.com")
    await _grant_via_balance(uid)
    assert await _plan_grant_rows(uid) == 2      # message + integration

    await _steal_tombstone(uid)

    async with async_session_maker() as db:
        # Tombstone says "not yours"; the ledger says "already granted".
        assert await credit_service._already_granted(db, uid) is True


async def test_reconcile_never_regrants_a_user_who_already_has_a_plan_grant():
    """The exact production loop: no owned tombstone + hourly sweep."""
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    uid = await _mk_user("loop-b@example.com")
    await _grant_via_balance(uid)
    await _steal_tombstone(uid)

    async with async_session_maker() as db:
        granted = await credit_service.reconcile_deferred_grants(db)
    assert granted == 0, "a user who already received a plan grant must not get another"
    assert await _plan_grant_rows(uid) == 2, "no new plan_grant rows"


async def test_regrant_loop_no_longer_wipes_the_period_spend():
    """The user-visible symptom: spend down, let the hourly sweep run, and the
    wallet must NOT jump back to the full monthly allotment.

    Pre-fix this asserted 100 and used_today stayed put — which is precisely
    the 'remaining 98.90 beside used_today 11.80' contradiction on the phone.
    """
    from app.db import async_session_maker, CreditBalance
    from app.services.credit_service import credit_service

    uid = await _mk_user("loop-c@example.com")
    await _grant_via_balance(uid)
    await _steal_tombstone(uid)

    async with async_session_maker() as db:
        b = await db.get(CreditBalance, uid)
        b.message_credits_remaining = Decimal("90.2")
        b.integration_credits_remaining = Decimal("480")
        b.message_credits_used_today = Decimal("9.8")
        await db.commit()

    # Three hourly ticks.
    for _ in range(3):
        async with async_session_maker() as db:
            await credit_service.reconcile_deferred_grants(db)

    b = await _balance(uid)
    assert Decimal(b.message_credits_remaining) == Decimal("90.2"), "wallet was re-granted"
    assert Decimal(b.integration_credits_remaining) == Decimal("480")
    assert Decimal(b.message_credits_used_today) == Decimal("9.8")


async def test_a_genuinely_ungranted_user_is_still_granted():
    """The sweep's real job must survive the fix: a deferred-but-eligible user
    (balance row exists, no grant ever fired) still gets their credits."""
    from app.db import async_session_maker, CreditBalance
    from app.db.models import CreditLedger
    from app.services.credit_service import credit_service

    uid = await _mk_user("deferred@example.com")
    await _grant_via_balance(uid)

    # Simulate "signed up while the verify gate was ON": strip the grant
    # entirely — no ledger rows, no tombstone, zeroed wallet.
    from app.db.models import GrantEligibility
    from sqlalchemy import delete
    async with async_session_maker() as db:
        await db.execute(delete(CreditLedger).where(CreditLedger.user_id == uid))
        await db.execute(delete(GrantEligibility))
        b = await db.get(CreditBalance, uid)
        b.message_credits_remaining = Decimal("0")
        b.integration_credits_remaining = Decimal("0")
        await db.commit()

    async with async_session_maker() as db:
        granted = await credit_service.reconcile_deferred_grants(db)
    assert granted == 1
    b = await _balance(uid)
    assert Decimal(b.message_credits_remaining) == Decimal("100")


# ══════════════════════════════════════════════════════════════════════
# DEFECT 2 — the daily cap vs a cost already paid for
# ══════════════════════════════════════════════════════════════════════


def test_split_reproduces_the_unsatisfiable_cap():
    """A 28-credit call against a 15-credit cap with a full wallet: the cap can
    never be satisfied, so the charge is denied even though the user can afford
    it. This is the shape that made 274 production calls free."""
    from app.services.credit_service import _split_message_charge

    args = (Decimal("99.8"), Decimal("0"), Decimal("0.2"), Decimal("15"), Decimal("28"))
    from_plan, from_purchased, feasible, reason = _split_message_charge(*args)
    assert feasible is False
    assert reason == "daily_cap_exceeded"

    # Already incurred: the money is spent at the provider, so it must land.
    from_plan, from_purchased, feasible, reason = _split_message_charge(
        *args, ignore_daily_cap=True,
    )
    assert (feasible, reason) == (True, None)
    assert from_plan == Decimal("28")
    assert from_purchased == Decimal("0"), "must not be pushed onto the IAP wallet"


def test_incurred_charge_still_denied_when_the_wallet_is_genuinely_empty():
    """Ignoring the cap must not ignore the BALANCE — the plan wallet stays the
    hard ceiling, otherwise this turns into unbounded free credit."""
    from app.services.credit_service import _split_message_charge

    _fp, _fpur, feasible, reason = _split_message_charge(
        Decimal("2"), Decimal("0"), Decimal("50"), Decimal("15"), Decimal("28"),
        ignore_daily_cap=True,
    )
    assert feasible is False
    assert reason == "insufficient_message_credits"


async def test_incurred_charge_lands_instead_of_being_zeroed(credit_flags):
    """End to end through try_charge: the ledger row must carry the real amount
    and the balance must move."""
    from app.db import async_session_maker, CreditBalance
    from app.db.models import BUCKET_MESSAGE, LEDGER_CHAT_MESSAGE
    from app.services.credit_service import credit_service

    credit_flags(credit_enforcement_enabled=True, credit_cap_admission_control=True)

    uid = await _mk_user("cap-a@example.com")
    await _grant_via_balance(uid)

    async with async_session_maker() as db:
        b = await db.get(CreditBalance, uid)
        b.message_credits_daily_cap = Decimal("15")
        await db.commit()

    async with async_session_maker() as db:
        result = await credit_service.try_charge(
            db, uid, LEDGER_CHAT_MESSAGE, BUCKET_MESSAGE, Decimal("28"),
            already_incurred=True,
        )
        await db.commit()

    assert result.success is True, "an incurred cost must never be zeroed by the cap"
    b = await _balance(uid)
    assert Decimal(b.message_credits_remaining) == Decimal("72")
    # used_today deliberately exceeds the cap — that over-cap value is what
    # makes the NEXT pre-flight refuse.
    assert Decimal(b.message_credits_used_today) == Decimal("28")


async def test_over_cap_used_today_makes_the_next_preflight_refuse(credit_flags):
    """The self-limiting half: once the cap is blown, admission stops. Without
    this the fix would just bill more without ever stopping the loop."""
    from app.db import async_session_maker, CreditBalance
    from app.db.models import BUCKET_MESSAGE
    from app.services.credit_service import credit_service

    credit_flags(credit_enforcement_enabled=True, credit_cap_admission_control=True)

    uid = await _mk_user("cap-b@example.com")
    await _grant_via_balance(uid)
    async with async_session_maker() as db:
        b = await db.get(CreditBalance, uid)
        b.message_credits_daily_cap = Decimal("15")
        b.message_credits_used_today = Decimal("28")   # a previous incurred charge
        await db.commit()

    async with async_session_maker() as db:
        peek = await credit_service.check_balance(db, uid, BUCKET_MESSAGE, Decimal("0.1"))
    assert peek.success is False
    assert peek.reason == "daily_cap_exceeded"


async def test_preflight_is_cap_blind_when_the_flag_is_off(credit_flags):
    """Default OFF must be byte-for-byte the old behaviour, so the fix can land
    in prod without changing anything until it is deliberately switched on."""
    from app.db import async_session_maker, CreditBalance
    from app.db.models import BUCKET_MESSAGE
    from app.services.credit_service import credit_service

    credit_flags(credit_enforcement_enabled=True, credit_cap_admission_control=False)

    uid = await _mk_user("cap-c@example.com")
    await _grant_via_balance(uid)
    async with async_session_maker() as db:
        b = await db.get(CreditBalance, uid)
        b.message_credits_daily_cap = Decimal("15")
        b.message_credits_used_today = Decimal("28")
        await db.commit()

    async with async_session_maker() as db:
        peek = await credit_service.check_balance(db, uid, BUCKET_MESSAGE, Decimal("0.1"))
    assert peek.success is True, "flag off must preserve the legacy cap-blind peek"


async def test_flag_off_preserves_the_old_cap_denial(credit_flags):
    from app.db import async_session_maker, CreditBalance
    from app.db.models import BUCKET_MESSAGE, LEDGER_CHAT_MESSAGE
    from app.services.credit_service import credit_service

    credit_flags(credit_enforcement_enabled=True, credit_cap_admission_control=False)

    uid = await _mk_user("cap-d@example.com")
    await _grant_via_balance(uid)
    async with async_session_maker() as db:
        b = await db.get(CreditBalance, uid)
        b.message_credits_daily_cap = Decimal("15")
        await db.commit()

    async with async_session_maker() as db:
        result = await credit_service.try_charge(
            db, uid, LEDGER_CHAT_MESSAGE, BUCKET_MESSAGE, Decimal("28"),
            already_incurred=True,
        )
        await db.commit()
    assert result.success is False
    assert result.reason == "daily_cap_exceeded"


# ══════════════════════════════════════════════════════════════════════
# The /status read path
# ══════════════════════════════════════════════════════════════════════


async def test_status_rolls_a_stale_daily_counter_without_writing():
    """`_reset_daily_if_needed` is only reachable from writes and there is no
    daily cron, so after a quiet night /status served YESTERDAY's used_today —
    the "Today: 11 of 15" that could not move. The GET must roll it for the
    response and leave the row alone."""
    from app.db import async_session_maker, CreditBalance
    from app.services.credit_service import credit_service

    uid = await _mk_user("roll@example.com")
    await _grant_via_balance(uid)
    async with async_session_maker() as db:
        b = await db.get(CreditBalance, uid)
        b.message_credits_used_today = Decimal("11.8")
        b.day_anchor_local_date = (
            datetime.utcnow().date() - timedelta(days=1)
        ).isoformat()
        await db.commit()

    async with async_session_maker() as db:
        view = await credit_service.get_balance_view(db, uid)
    assert Decimal(view.message_credits_used_today) == Decimal("0")

    # The row itself is untouched — a GET must not write.
    b = await _balance(uid)
    assert Decimal(b.message_credits_used_today) == Decimal("11.8")


async def test_status_keeps_todays_counter():
    from app.db import async_session_maker, CreditBalance
    from app.services.credit_service import credit_service

    uid = await _mk_user("today@example.com")
    await _grant_via_balance(uid)
    async with async_session_maker() as db:
        b = await db.get(CreditBalance, uid)
        b.message_credits_used_today = Decimal("11.8")
        b.day_anchor_local_date = datetime.utcnow().date().isoformat()
        await db.commit()

    async with async_session_maker() as db:
        view = await credit_service.get_balance_view(db, uid)
    assert Decimal(view.message_credits_used_today) == Decimal("11.8")


# ══════════════════════════════════════════════════════════════════════
# Brave: the shed-reason vocabulary must have exactly one definition
# ══════════════════════════════════════════════════════════════════════


def test_never_reached_brave_has_one_definition():
    """A capped attempt consumed no Brave quota. If the monitor and the gateway
    disagreed, capped rows would count as usage and hold the user over their
    ceiling permanently."""
    from app.api.search_proxy import NEVER_REACHED_BRAVE, RSN_USER_DAILY_CAP
    from app.services.search_quota_monitor import _never_reached_brave

    assert RSN_USER_DAILY_CAP in NEVER_REACHED_BRAVE
    assert _never_reached_brave() == NEVER_REACHED_BRAVE


def test_search_daily_cap_defaults_to_off():
    """Enforcement is opt-in: merging this must not start denying searches."""
    from app.config import settings
    assert settings.search_daily_cap_enabled is False
    assert settings.credit_cap_admission_control is False
    assert settings.voice_metering_charge is False


def test_agent_charge_bundle_guard_exempts_tool_calls():
    """Flipping `web_tool_metering_charge` on used to make the agent send
    meter_only=False, re-engage the bundle guard, and DELETE the web-usage
    series for the whole fleet — the flag advertised as "start billing" would
    have blinded the system instead."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "app" / "api" / "credits.py").read_text()
    assert "_PROXY_NEVER_BILLS = {LEDGER_TOOL_CALL}" in src
    assert "_bundle_exempt = body.meter_only or (body.event_type in _PROXY_NEVER_BILLS)" in src


def test_every_priced_flat_fee_has_a_writer():
    """A price with no charge call site is invisible, and seven of eleven
    FLAT_FEES entries were exactly that for the whole life of the credits
    system: `_flat_fee_for_tool` is called ONLY by connector_dispatcher, which
    never sees browser/doc tool names. Pin the coverage so a new price cannot
    be added without something that writes it.

    `reminder_fire` is the one deliberate exclusion — a reminder fire already
    runs a metered AgentRunner turn, so charging the flat fee on top would bill
    one event twice. See docs/credits/coverage.md.
    """
    from app.agent.tool_executor import ToolExecutor
    from app.services.credit_service import FLAT_FEES

    written_by_connector_dispatcher = {
        "connector_call", "connector_bulk_call",
    }
    written_by_meter_web_tool = {"web_search", "web_fetch"}
    written_by_meter_flat_tool = set(ToolExecutor._FLAT_FEE_TOOLS.values())
    deliberately_unwired = {"reminder_fire"}

    covered = (
        written_by_connector_dispatcher
        | written_by_meter_web_tool
        | written_by_meter_flat_tool
        | deliberately_unwired
    )
    orphans = sorted(set(FLAT_FEES) - covered)
    assert not orphans, (
        f"priced but nothing writes them: {orphans}. Either wire a charge call "
        f"site or document the exclusion in docs/credits/coverage.md."
    )

    # And every mapped tool must resolve to a real fee.
    unknown = sorted(v for v in written_by_meter_flat_tool if v not in FLAT_FEES)
    assert not unknown, f"_FLAT_FEE_TOOLS points at missing FLAT_FEES keys: {unknown}"


def test_flat_tool_metering_is_measure_only_by_default():
    from app.config import settings
    assert settings.flat_tool_metering_enabled is True
    assert settings.flat_tool_metering_charge is False


async def test_credit_health_monitor_catches_both_incident_signatures():
    """The incident was queryable for months and nobody looked. This asserts
    the monitor actually fires on the two shapes that defined it: a denied
    charge that still cost us provider money, and a duplicate one-time grant.
    """
    from app.db import async_session_maker
    from app.db.models import (
        BUCKET_MESSAGE, CreditLedger, LEDGER_CHAT_MESSAGE, LEDGER_PLAN_GRANT,
    )
    from app.services import credit_health_monitor as CHM

    uid = await _mk_user("health@example.com")
    await _grant_via_balance(uid)   # writes the legitimate plan_grant pair

    async with async_session_maker() as db:
        # A charge denied by the cap, for work a provider already billed us for.
        db.add(CreditLedger(
            id=str(uuid.uuid4()), user_id=uid, event_type=LEDGER_CHAT_MESSAGE,
            bucket=BUCKET_MESSAGE, amount=Decimal("0"), balance_after=Decimal("100"),
            underlying_cost_cents=Decimal("28"),
            metadata_json={"denied": True, "reason": "daily_cap_exceeded"},
        ))
        # A SECOND one-time grant — the re-grant loop's fingerprint.
        db.add(CreditLedger(
            id=str(uuid.uuid4()), user_id=uid, event_type=LEDGER_PLAN_GRANT,
            bucket=BUCKET_MESSAGE, amount=Decimal("100"), balance_after=Decimal("100"),
            metadata_json={"reason": "initial_grant"},
        ))
        await db.commit()

    sent: list[tuple] = []

    async def _capture(category, level, message, **kw):
        sent.append((category, level))
        return True
    CHM.send_infra_alert = _capture

    result = await CHM.check_credit_health()

    assert "duplicate_grants" in result["alerts"]
    assert result["readings"]["users_with_duplicate_grants"] >= 1
    assert result["readings"]["served_unbilled_calls"] >= 1
    assert result["readings"]["served_unbilled_usd"] >= 0.28
    assert any(a.startswith("served_unbilled") for a in result["alerts"])
    assert {c for c, _ in sent} >= {"credit-served-unbilled", "credit-duplicate-grants"}


async def test_credit_health_monitor_is_quiet_on_a_healthy_system():
    """It must not cry wolf — a clean account trips nothing."""
    from app.services import credit_health_monitor as CHM

    uid = await _mk_user("healthy@example.com")
    await _grant_via_balance(uid)

    sent: list[tuple] = []

    async def _capture(category, level, message, **kw):
        sent.append((category, level)); return True
    CHM.send_infra_alert = _capture

    result = await CHM.check_credit_health()
    assert result["alerts"] == [], f"unexpected alarms: {result}"
    assert sent == []


async def test_duplicate_grant_alarm_is_windowed_not_all_time():
    """A repaired account keeps its historical duplicate rows forever. An
    all-time count would fire on every run from then on, and an alarm that
    never clears is one people learn to ignore. Reproduced live: after the
    repair script ran, the alarm still read 2 users on a system whose loop was
    already dead.
    """
    from app.db import async_session_maker
    from app.db.models import BUCKET_MESSAGE, CreditLedger, LEDGER_PLAN_GRANT
    from app.services import credit_health_monitor as CHM

    uid = await _mk_user("windowed@example.com")
    await _grant_via_balance(uid)

    # A duplicate grant from LAST WEEK — real history, already remediated.
    async with async_session_maker() as db:
        db.add(CreditLedger(
            id=str(uuid.uuid4()), user_id=uid, event_type=LEDGER_PLAN_GRANT,
            bucket=BUCKET_MESSAGE, amount=Decimal("100"),
            balance_after=Decimal("100"),
            created_at=datetime.utcnow() - timedelta(days=7),
            metadata_json={"reason": "initial_grant"},
        ))
        await db.commit()

    async def _noop(*a, **k):
        return True
    CHM.send_infra_alert = _noop

    result = await CHM.check_credit_health()
    assert result["readings"]["users_with_duplicate_grants"] == 0
    assert "duplicate_grants" not in result["alerts"]


async def test_charge_ratio_ignores_corrections_and_grants():
    """`plan_change` carries admin corrections and the duplicate-grant
    clawback. Counting any negative amount as revenue made a REFUND look like
    income — the live reading jumped 12.1 -> 73.7 credits the moment the repair
    script ran, which would have masked exactly the undercharging this alarm
    exists to catch."""
    from app.db import async_session_maker
    from app.db.models import (
        BUCKET_MESSAGE, CreditLedger, LEDGER_CHAT_MESSAGE, LEDGER_PLAN_CHANGE,
    )
    from app.services import credit_health_monitor as CHM

    uid = await _mk_user("ratio@example.com")
    await _grant_via_balance(uid)

    async with async_session_maker() as db:
        db.add(CreditLedger(
            id=str(uuid.uuid4()), user_id=uid, event_type=LEDGER_CHAT_MESSAGE,
            bucket=BUCKET_MESSAGE, amount=Decimal("-5"),
            balance_after=Decimal("95"), underlying_cost_cents=Decimal("5"),
        ))
        # A clawback. Negative, but NOT revenue.
        db.add(CreditLedger(
            id=str(uuid.uuid4()), user_id=uid, event_type=LEDGER_PLAN_CHANGE,
            bucket=BUCKET_MESSAGE, amount=Decimal("-40"),
            balance_after=Decimal("55"),
            metadata_json={"reason": "duplicate_grant_reconciliation"},
        ))
        await db.commit()

    async def _noop(*a, **k):
        return True
    CHM.send_infra_alert = _noop

    result = await CHM.check_credit_health()
    assert result["readings"]["credits_charged"] == 5.0, (
        "the 40-credit clawback must not be counted as revenue"
    )


def test_web_fetch_cache_probe_symbol_exists():
    """web_fetch called `reader.page_cache_get`, which did not exist — every
    page fetch raised AttributeError and returned an error string."""
    from app.agent.smart_fetch import reader
    assert hasattr(reader, "page_cache_get")
    assert hasattr(reader, "page_cache_key")
    # The probe must hash the url exactly the way toup_read_page stores it, or
    # it misses forever and every cached page bills as a fresh fetch.
    reader._PAGE_CACHE.set(reader.page_cache_key("https://a/b", 100), "X")
    assert reader.page_cache_get("https://a/b", 100) == "X"
