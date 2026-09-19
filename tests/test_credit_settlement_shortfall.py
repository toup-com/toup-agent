"""The balance dimension's stop: a refused settlement still moves money.

Measured incident (2026-09-15, user 364ad12a on toup-agent-pool-88): 21
served-and-unbilled turns over 6h07m, terminal balance 0.00. The mechanism was
a quote/settle asymmetry with no bottom. Admission quotes a flat nominal 0.1
credit; settlement demands the turn's realized cost (~4 credits). When
settlement was infeasible ``try_charge`` wrote an ``amount=0`` denied row and
returned WITHOUT calling ``_apply_delta``, so the residual was never debited —
the wallet froze at a positive value, and the next 0.1-credit pre-flight
admitted the next turn. Forever.

The cap dimension has had a documented stop since 2026-08-03 (an incurred
charge lands past the cap, and the over-cap ``used_today`` is what refuses the
next pre-flight). It has been INERT since 2026-08-29, when every
``message_credits_daily_cap`` went NULL, and there was no balance-dimension
counterpart at all.

So the tests come in pairs around one boundary: work ALREADY DONE settles to
the limit of the wallet and drives it to zero (so the next pre-flight is the
stop), while a pre-flight refusal for work NOT yet done still debits nothing.
A settlement that moves money without stopping the loop, or a pre-flight that
charges for work it refused, are each worse than the bug.

WHICH PATHS THIS REACHES — every MESSAGE-bucket ``try_charge`` in the repo,
not just chat. ``test_usage_metering_regression.py``'s AST guard
(``test_every_message_bucket_settlement_is_marked_already_incurred``) makes
``already_incurred=True`` a repo rule on that bucket, so the settlement fires
on:

* ``llm_proxy._log_event`` — the chat/agent turn (the measured incident).
* ``llm_proxy``'s four image-generation charges (``LEDGER_IMAGE_GEN``) — where
  a SINGLE event can zero a wallet, which is the shape most unlike chat.
* ``/credits/agent-deduct`` — manual (BYOK) mode's only metering surface.
* ``ws_realtime``'s voice charge, the moment ``voice_metering_charge`` is
  flipped (it passes ``meter_only`` today, so it is inert, not exempt).
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
    """Set a credit flag on the settings object the charge path ACTUALLY reads.

    Same hazard as test_usage_metering_regression.py's identical fixture:
    ``test_subagent_settings.py`` reloads ``app.config``, rebinding
    ``app.config.settings`` while ``credit_service``'s module-level reference
    still points at the old instance — so a fresh ``from app.config import
    settings`` patches nothing and the file passes alone but fails in a full
    run.
    """
    import app.services.credit_service as CS

    def _set(**flags):
        for name, value in flags.items():
            monkeypatch.setattr(CS.settings, name, value, raising=False)
    return _set


# ── helpers ───────────────────────────────────────────────────────────


async def _mk_user(email: str | None = None) -> str:
    from app.db import async_session_maker, User
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=uid, email=(email or f"s-{uuid.uuid4().hex[:10]}@example.com"),
            hashed_password="x", name="t",
            email_verified_at=datetime.utcnow(),
        ))
        await db.commit()
    return uid


async def _wallet(uid: str, *, plan="100", purchased="0", cap=None, used="0"):
    """Seed the two MESSAGE wallets directly — the plan catalogue's own numbers
    are not the subject here, the arithmetic around them is."""
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    async with async_session_maker() as db:
        b = await credit_service.get_or_create_balance(db, uid)
        b.message_credits_remaining = Decimal(plan)
        b.purchased_credits_remaining = Decimal(purchased)
        b.message_credits_used_today = Decimal(used)
        b.message_credits_daily_cap = (Decimal(cap) if cap is not None else None)
        await db.commit()


async def _balance(uid: str):
    from app.db import async_session_maker, CreditBalance
    async with async_session_maker() as db:
        return await db.get(CreditBalance, uid)


async def _charge(uid: str, amount: str, *, incurred=True, key=None,
                  event_type: str | None = None):
    from app.db import async_session_maker
    from app.db.models import LEDGER_CHAT_MESSAGE
    from app.services.credit_service import BUCKET_MESSAGE, credit_service
    async with async_session_maker() as db:
        r = await credit_service.try_charge(
            db, uid, event_type or LEDGER_CHAT_MESSAGE,
            BUCKET_MESSAGE, Decimal(amount),
            idempotency_key=key, underlying_cost_cents=Decimal("400"),
            model="gpt-5.5", provider="openai",
            metadata={"endpoint": "/v1/chat/completions",
                      "operation_type": "user"},
            already_incurred=incurred,
        )
        await db.commit()
    return r


async def _rows(uid: str):
    from app.db import async_session_maker, CreditLedger
    async with async_session_maker() as db:
        return list((await db.execute(
            select(CreditLedger).where(CreditLedger.user_id == uid)
            .order_by(CreditLedger.created_at)
        )).scalars().all())


# ── the fix ───────────────────────────────────────────────────────────


async def test_incurred_shortfall_drains_the_wallet_and_still_denies(credit_flags):
    """The core of it: 104 credits of work against a 100-credit wallet takes
    the 100 and records the 4 — and is still a DENIAL, because the question
    `success` answers is "may more work happen?"."""
    credit_flags(credit_enforcement_enabled=True,
                 credit_cap_admission_control=False)
    from app.services.credit_service import REASON_INSUFFICIENT_MESSAGE

    uid = await _mk_user()
    await _wallet(uid, plan="100")

    r = await _charge(uid, "104")

    # Money first, bookkeeping second: on the pre-change code the wallet is
    # still sitting at 100 here, which is the defect itself.
    b = await _balance(uid)
    assert Decimal(b.message_credits_remaining) == Decimal("0")
    assert Decimal(b.purchased_credits_remaining) == Decimal("0")

    row = [x for x in await _rows(uid) if x.metadata_json
           and x.metadata_json.get("denied")][-1]
    assert Decimal(row.amount) == Decimal("-100")

    assert r.success is False, "a settlement that cannot be covered is still a refusal"
    assert r.reason == REASON_INSUFFICIENT_MESSAGE
    assert r.charged == Decimal("100")
    assert r.shortfall == Decimal("4")
    assert row.metadata_json["settled_incurred"] is True
    assert row.metadata_json["credits_quoted"] == "104.0000"
    assert row.metadata_json["credits_shortfall"] == "4.0000"
    # The caller's own metadata is not clobbered by the settlement keys.
    assert row.metadata_json["operation_type"] == "user"


async def test_an_image_generation_shortfall_settles_the_same_way(credit_flags):
    """Not a chat-only change. `already_incurred=True` is a repo rule on the
    MESSAGE bucket (test_usage_metering_regression's AST guard), so the four
    image charges in llm_proxy settle through this same block — and an image
    is the shape most unlike chat: ONE event, priced per-image, can zero a
    wallet on its own rather than walking it down over a conversation. The
    picture exists; refusing to bill for it does not un-generate it."""
    credit_flags(credit_enforcement_enabled=True,
                 credit_cap_admission_control=False)
    from app.db.models import LEDGER_IMAGE_GEN
    from app.services.credit_service import REASON_INSUFFICIENT_MESSAGE

    uid = await _mk_user()
    await _wallet(uid, plan="1.5", purchased="0.5")

    r = await _charge(uid, "20", event_type=LEDGER_IMAGE_GEN)

    b = await _balance(uid)
    assert Decimal(b.message_credits_remaining) == Decimal("0")
    assert Decimal(b.purchased_credits_remaining) == Decimal("0")
    assert r.success is False
    assert r.reason == REASON_INSUFFICIENT_MESSAGE
    assert r.charged == Decimal("2")
    assert r.shortfall == Decimal("18")
    row = [x for x in await _rows(uid) if x.metadata_json
           and x.metadata_json.get("denied")][-1]
    assert row.event_type == LEDGER_IMAGE_GEN
    assert Decimal(row.amount) == Decimal("-2")
    assert row.metadata_json["settled_incurred"] is True


async def test_the_next_preflight_refuses_after_a_shortfall_settlement(credit_flags):
    """The whole point. Before this change the wallet stayed positive, so the
    flat 0.1-credit pre-flight admitted the next turn and the give-away had no
    bottom (21 turns in 6h07m). Draining to zero IS the stop."""
    credit_flags(credit_enforcement_enabled=True,
                 credit_cap_admission_control=False)
    from app.credit_shadow import PREFLIGHT_QUOTE_CREDITS
    from app.db import async_session_maker
    from app.services.credit_service import BUCKET_MESSAGE, credit_service

    # Pinned, not merely imported: since 2026-09-18 this constant IS the live
    # 402 gate on /v1/chat/completions, /v1/responses and /credits/agent-deduct,
    # so changing it moves production admission.
    assert PREFLIGHT_QUOTE_CREDITS == Decimal("0.1")

    uid = await _mk_user()
    await _wallet(uid, plan="3")

    async with async_session_maker() as db:
        before = await credit_service.check_balance(
            db, uid, BUCKET_MESSAGE, PREFLIGHT_QUOTE_CREDITS)
    assert before.success is True, "sanity: the wallet admits work before the turn"

    await _charge(uid, "4")

    async with async_session_maker() as db:
        after = await credit_service.check_balance(
            db, uid, BUCKET_MESSAGE, PREFLIGHT_QUOTE_CREDITS)
    assert after.success is False, "the loop must stop at the NEXT pre-flight"


async def test_the_purchased_wallet_is_drained_too(credit_flags):
    """Both MESSAGE wallets are spendable, so both are part of "what the wallet
    holds" — otherwise a user with IAP credits keeps a positive balance and
    the next pre-flight admits again."""
    credit_flags(credit_enforcement_enabled=True,
                 credit_cap_admission_control=False)

    uid = await _mk_user()
    await _wallet(uid, plan="2", purchased="5")

    r = await _charge(uid, "20")

    b = await _balance(uid)
    assert Decimal(b.message_credits_remaining) == Decimal("0")
    assert Decimal(b.purchased_credits_remaining) == Decimal("0")
    assert r.charged == Decimal("7")
    assert r.shortfall == Decimal("13")


async def test_a_shortfall_settlement_never_overdraws(credit_flags):
    """Never debit more than the wallet holds, and never leave a negative
    balance the rest of the system does not expect."""
    credit_flags(credit_enforcement_enabled=True,
                 credit_cap_admission_control=False)

    uid = await _mk_user()
    await _wallet(uid, plan="3.5")

    r = await _charge(uid, "500")

    b = await _balance(uid)
    assert Decimal(b.message_credits_remaining) == Decimal("0")
    assert Decimal(b.purchased_credits_remaining) == Decimal("0")
    assert Decimal(b.message_credits_remaining) >= 0
    assert Decimal(b.purchased_credits_remaining) >= 0
    assert r.charged == Decimal("3.5")


async def test_an_empty_wallet_settles_nothing_and_stays_a_bare_denial(credit_flags):
    """Second and later turns of the same exhausted account: there is nothing
    to take, so the row must not claim a partial settlement it did not make —
    invariant 6 ("a zero charge always says why") reads `denied`, and a
    `settled_incurred` marker on a zero row would be a lie about money."""
    credit_flags(credit_enforcement_enabled=True,
                 credit_cap_admission_control=False)

    uid = await _mk_user()
    await _wallet(uid, plan="0")

    r = await _charge(uid, "4")

    assert r.charged == Decimal("0")
    assert r.shortfall == Decimal("4")
    row = [x for x in await _rows(uid) if x.metadata_json
           and x.metadata_json.get("denied")][-1]
    assert Decimal(row.amount) == Decimal("0")
    assert "settled_incurred" not in row.metadata_json
    assert row.metadata_json["reason"] is not None


# ── the boundaries it must not cross ──────────────────────────────────


async def test_a_preflight_refusal_still_debits_nothing(credit_flags):
    """Work NOT yet done. `already_incurred=False` means nobody has been paid,
    so a refusal has to be free — charging for a turn we just declined to run
    is the opposite bug and a worse one."""
    credit_flags(credit_enforcement_enabled=True,
                 credit_cap_admission_control=False)

    uid = await _mk_user()
    await _wallet(uid, plan="100")

    r = await _charge(uid, "104", incurred=False)

    assert r.success is False
    assert r.charged == Decimal("0")
    b = await _balance(uid)
    assert Decimal(b.message_credits_remaining) == Decimal("100")
    row = [x for x in await _rows(uid) if x.metadata_json
           and x.metadata_json.get("denied")][-1]
    assert Decimal(row.amount) == Decimal("0")
    assert "settled_incurred" not in row.metadata_json


async def test_a_cap_refusal_is_left_to_its_own_flag(credit_flags):
    """`daily_cap_exceeded` is a rate limit on money that EXISTS. Whether an
    incurred cost may outrun the cap is `credit_cap_admission_control`'s
    decision, and settling past the cap here would move that policy behind a
    second switch. Byte-for-byte the pre-2026-09-18 behaviour."""
    credit_flags(credit_enforcement_enabled=True,
                 credit_cap_admission_control=False)
    from app.services.credit_service import REASON_DAILY_CAP_EXCEEDED

    uid = await _mk_user()
    await _wallet(uid, plan="100", cap="15")

    r = await _charge(uid, "28")

    assert r.reason == REASON_DAILY_CAP_EXCEEDED
    assert r.charged == Decimal("0")
    b = await _balance(uid)
    assert Decimal(b.message_credits_remaining) == Decimal("100")


async def test_a_shortfall_settlement_outruns_a_non_null_cap(credit_flags):
    """The DECISION at the cap/balance boundary, recorded because it looks like
    a leak of the flag above.

    With a cap set, `credit_cap_admission_control` OFF and an amount larger
    than BOTH wallets, the gate's precedence answers `insufficient_message`
    rather than `daily_cap_exceeded` — so the settlement runs with
    `ignore_daily_cap=True` and `used_today` ends up past the cap, which is the
    very policy that flag exists to own.

    Intended (2026-09-18): an incurred cost is owed whatever the cap says, and
    the emptied wallet is a harder ceiling than the cap ever was — there is
    nothing left for a rate limit to ration. Unreachable in production while
    every `message_credits_daily_cap` is NULL (2026-08-29); this test is here
    so whoever re-introduces caps meets the decision instead of discovering it.
    """
    credit_flags(credit_enforcement_enabled=True,
                 credit_cap_admission_control=False)
    from app.services.credit_service import REASON_INSUFFICIENT_MESSAGE

    uid = await _mk_user()
    await _wallet(uid, plan="100", cap="15", used="0")

    r = await _charge(uid, "200")

    assert r.reason == REASON_INSUFFICIENT_MESSAGE, (
        "amount > plan+purchased is a wallet shortfall, not a cap refusal — "
        "the cap arm only owns amounts the wallet could actually cover"
    )
    assert r.charged == Decimal("100")
    b = await _balance(uid)
    assert Decimal(b.message_credits_remaining) == Decimal("0")
    assert Decimal(b.message_credits_used_today) == Decimal("100")
    assert Decimal(b.message_credits_used_today) > Decimal(
        b.message_credits_daily_cap), "the settlement is cap-blind, by decision"


async def test_an_unverified_email_refusal_debits_nothing(credit_flags):
    """`email_not_verified` is a policy gate with no wallet dimension. There is
    no shortfall to settle, so nothing moves."""
    credit_flags(credit_enforcement_enabled=True,
                 credit_cap_admission_control=False,
                 require_email_verification_for_credits=True,
                 email_verification_required_after_iso="")
    from app.db import async_session_maker, User
    from app.services.credit_service import REASON_EMAIL_NOT_VERIFIED

    uid = await _mk_user()
    await _wallet(uid, plan="100")
    async with async_session_maker() as db:
        u = await db.get(User, uid)
        u.email_verified_at = None
        await db.commit()

    r = await _charge(uid, "4")

    assert r.reason == REASON_EMAIL_NOT_VERIFIED
    assert r.charged == Decimal("0")
    b = await _balance(uid)
    assert Decimal(b.message_credits_remaining) == Decimal("100")


async def test_the_switch_restores_the_old_zeroing(credit_flags):
    """One flag, revertible from Railway without a rebuild. OFF is exactly the
    behaviour that shipped the give-away — which is the point: the operator
    gets today's semantics back in a restart, not a deploy.

    Driven through `settings.credit_settle_incurred_shortfall`, which is the
    ONE resolution path — on Railway that field is populated from the
    `CREDIT_SETTLE_INCURRED_SHORTFALL` variable at boot.
    """
    credit_flags(credit_enforcement_enabled=True,
                 credit_cap_admission_control=False,
                 credit_settle_incurred_shortfall=False)

    uid = await _mk_user()
    await _wallet(uid, plan="100")

    r = await _charge(uid, "104")

    assert r.success is False
    assert r.charged == Decimal("0")
    b = await _balance(uid)
    assert Decimal(b.message_credits_remaining) == Decimal("100")


async def test_the_switch_defaults_on():
    """An active give-away is stopped by default. A switch that had to be
    turned on would leave production leaking until someone remembered.

    The pin is the DECLARED default on the Settings class — the one thing no
    ambient variable can move. The second assertion is the wiring, and it is
    deliberately live: in a process started with
    `CREDIT_SETTLE_INCURRED_SHORTFALL=false` it reads False, because that
    variable IS the documented revert route.
    """
    import app.services.credit_service as CS
    from app.config import Settings

    field = Settings.model_fields["credit_settle_incurred_shortfall"]
    assert field.default is True
    assert CS._settle_incurred_shortfall_enabled() is True


async def test_the_switch_has_exactly_one_resolution_path(credit_flags):
    """`settings` is the only reader. A module-local environment read would be
    a second path that can disagree with the value every other reader of
    `settings` sees — and on Railway the variable lands in the declared field
    anyway, so the fallback buys nothing.

    Asserted on the compiled names, not the source text: a text search would
    match this very docstring.
    """
    import app.services.credit_service as CS

    credit_flags(credit_settle_incurred_shortfall=False)
    assert CS._settle_incurred_shortfall_enabled() is False
    credit_flags(credit_settle_incurred_shortfall=True)
    assert CS._settle_incurred_shortfall_enabled() is True

    names = CS._settle_incurred_shortfall_enabled.__code__.co_names
    assert "getenv" not in names and "environ" not in names, \
        f"one lookup: the switch must resolve from settings alone, saw {names}"


async def test_a_replay_does_not_debit_twice(credit_flags):
    """The proxy event id doubles as the idempotency key, and an SDK retry
    replays it. The second call must report the replay, not drain a wallet
    that has already been drained."""
    credit_flags(credit_enforcement_enabled=True,
                 credit_cap_admission_control=False)

    uid = await _mk_user()
    await _wallet(uid, plan="10")

    first = await _charge(uid, "40", key="evt-1")
    second = await _charge(uid, "40", key="evt-1")

    assert first.charged == Decimal("10")
    assert second.idempotent_hit is True
    assert second.charged == Decimal("0")
    b = await _balance(uid)
    assert Decimal(b.message_credits_remaining) == Decimal("0")


# ── the receipt the agent is handed ───────────────────────────────────


async def _call_agent_deduct(uid: str, body):
    """Drive `/credits/agent-deduct` the way the agent does: X-Agent-Key auth
    against an AgentConfig row, llm_mode='manual' (bundle short-circuits before
    the charge, by design). Same helper shape as
    test_usage_metering_regression.py's."""
    from app.db import async_session_maker
    from app.db.models import AgentConfig
    from app.api.credits import agent_deduct

    key = f"agent-key-{uid[:8]}"
    async with async_session_maker() as db:
        db.add(AgentConfig(user_id=uid, llm_mode="manual", agent_api_key=key))
        await db.commit()
    async with async_session_maker() as db:
        return await agent_deduct(body, x_agent_key=key, db=db)


async def test_agent_deduct_reports_the_money_it_actually_took(credit_flags):
    """A response that says nothing was charged while the ledger says otherwise
    is a lie to the agent.

    `/credits/agent-deduct` passes `already_incurred=True` on the MESSAGE
    bucket, so a wallet shortfall here settles exactly like the proxy path:
    money moves and `success` is still False. `amount_charged` used to read
    `float(credits) if result.success else 0.0`, which reported 0.0 on the one
    case this round taught `try_charge` to debit.
    """
    credit_flags(credit_enforcement_enabled=True,
                 credit_cap_admission_control=False)
    from app.api.credits import AgentDeductRequest
    from app.services.credit_service import (
        REASON_INSUFFICIENT_MESSAGE, tokens_to_credits,
    )

    uid = await _mk_user()
    await _wallet(uid, plan="2")
    # Guard the fixture, not the pricing: if a repricing ever puts this turn
    # under 2 credits it stops being a shortfall and the test would silently
    # start asserting the ordinary success path.
    assert tokens_to_credits("gpt-5.5", 20_000, 2_000) > Decimal("2")

    resp = await _call_agent_deduct(uid, AgentDeductRequest(
        user_id=uid, model="gpt-5.5", provider="openai",
        input_tokens=20_000, output_tokens=2_000,
        idempotency_key="shortfall-receipt-1",
    ))

    assert resp.amount_charged == 2.0, "the receipt must be what the wallet paid"
    assert resp.success is False, "and the agent must still be told to stop"
    assert resp.reason == REASON_INSUFFICIENT_MESSAGE
    assert resp.balance_after == 0.0
    b = await _balance(uid)
    assert Decimal(b.message_credits_remaining) == Decimal("0")


# ── the alarm still sees it ───────────────────────────────────────────


async def test_invariant_1_still_sees_a_partially_settled_row(credit_flags):
    """Requirement of record: a shortfall settlement is still visible to
    credit_health invariant 1, and invariants 6 and 7 stay quiet.

    Invariant 1 asks "was work served that a denial refused to bill", and the
    answer is yes for the residual — so the row must keep `denied`. Invariant 6
    ("a zero charge always says why") excludes denied rows, and invariant 7
    only fires when the refused account held the unlimited plan.

    Renamed 2026-09-18 (was `…_still_pages_on_…`): ONE settled row is a
    CROSSING by design — the bounded cost of the turn that crosses zero — so
    the alarm records it and does not page. Visibility is therefore asserted
    through the readings, which is where it now lives; the loop arm has its
    own coverage in test_credit_health_monitor.py. Pinning a PAGE here would
    pin the noise this round removed.
    """
    credit_flags(credit_enforcement_enabled=True,
                 credit_cap_admission_control=False)

    import app.services.credit_health_monitor as CHM

    uid = await _mk_user()
    await _wallet(uid, plan="100")
    await _charge(uid, "104")

    sent: list[tuple] = []

    async def _fake(category, level, message, *, subject=None, min_interval_s=600):
        sent.append((category, level, message, subject))
        return True

    _orig = CHM.send_infra_alert
    CHM.send_infra_alert = _fake
    try:
        out = await CHM.check_credit_health()
    finally:
        CHM.send_infra_alert = _orig

    r = out["readings"]
    assert r["served_unbilled_calls"] >= 1
    # The debit the wallet DID cover, which is the whole point of the change
    # this file guards: a settled refusal is not a give-away, and the alarm
    # reads how much came back.
    assert r["served_unbilled_recovered_credits"] > 0
    assert r["served_unbilled_crossing_calls"] >= 1
    assert r["served_unbilled_crossings"] >= 1
    assert r["served_unbilled_loop_accounts"] == 0
    assert not any(a.startswith("served_unbilled") for a in out["alerts"]), \
        "one settled row is a crossing: seen, counted, not paged"
    assert "credit-served-unbilled" not in {c for (c, _l, _m, _s) in sent}
    assert not any(a.startswith("silent_zero") for a in out["alerts"]), \
        "invariant 6 must not claim a denied row has no reason"
    assert "denied_unlimited" not in out["alerts"]
