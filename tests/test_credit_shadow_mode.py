"""Shadow mode for the credit admission gate — it must measure and do nothing.

#471 fixed three ways the cap gate would have failed the instant it was
switched on. What it did not do is answer the question that decides whether to
switch it on: **what would enforcement actually deny?** The only figure on
record — "274 calls, $17.17, 59% of all real-user LLM cost served free with
reason=daily_cap_exceeded" (2026-08-03) — was measured against the BROKEN gate,
the one that zeroed costs already incurred, so it does not describe the fixed
one.

Shadow mode asks the real gate and throws the answer away. The safety property
is not "it usually doesn't deny" — it is that the verdict has no caller. These
tests pin that from both sides:

  * a turn the gate WOULD refuse still succeeds, and the money moves exactly as
    it does with the flag off (byte-compared against a twin user);
  * a turn the gate would ADMIT still logs — the anti-vacuity control, without
    which "denies nothing" would also pass if the code never ran at all.

Plus: the flag defaults off and is inert when off, no line can carry an email
or an API key, and the rollup script reads what the emitter writes.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from decimal import Decimal

import pytest
from sqlalchemy import select

# No module-level `pytest.mark.asyncio`: pytest.ini sets `asyncio_mode = auto`,
# so async tests are collected without it and marking the sync ones only
# produces a PytestWarning per test.

SHADOW_LOGGER = "app.credit_shadow"


# ── fixtures / helpers ────────────────────────────────────────────────

@pytest.fixture
def credit_flags(monkeypatch):
    """Patch the settings object `credit_service` ACTUALLY holds.

    `test_subagent_settings.py` reloads `app.config`, rebinding
    `app.config.settings` to a new instance while `credit_service`'s
    module-level `settings` still points at the old one. Patching a freshly
    imported `settings` therefore silently does nothing whenever that file is
    collected first — the tests pass alone and fail in a full run. Same
    reasoning as test_usage_metering_regression.py:48.
    """
    import app.services.credit_service as CS

    def _set(**flags):
        for name, value in flags.items():
            monkeypatch.setattr(CS.settings, name, value, raising=False)
    return _set


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


async def _seed_capped_user(email: str, *, used_today: Decimal) -> str:
    """A granted user sitting at `used_today` against a 15-credit daily cap.

    The parent `users` row is real: `credit_balances.user_id` is a genuine FK
    and Postgres enforces it even though sqlite does not.
    """
    from app.db import async_session_maker, CreditBalance
    from app.services.credit_service import credit_service

    uid = await _mk_user(email)
    async with async_session_maker() as db:
        await credit_service.get_or_create_balance(db, uid)
        await db.commit()
    async with async_session_maker() as db:
        balance = await db.get(CreditBalance, uid)
        balance.message_credits_daily_cap = Decimal("15")
        balance.message_credits_used_today = used_today
        await db.commit()
    return uid


async def _charge(uid: str, amount: str = "28"):
    from app.db import async_session_maker
    from app.db.models import BUCKET_MESSAGE, LEDGER_CHAT_MESSAGE
    from app.services.credit_service import credit_service
    async with async_session_maker() as db:
        result = await credit_service.try_charge(
            db, uid, LEDGER_CHAT_MESSAGE, BUCKET_MESSAGE, Decimal(amount),
            already_incurred=True,
        )
        await db.commit()
    return result


async def _money(uid: str) -> dict:
    """Everything about this user's money that a charge can move."""
    from app.db import async_session_maker, CreditBalance
    from app.db.models import CreditLedger
    async with async_session_maker() as db:
        balance = await db.get(CreditBalance, uid)
        rows = (await db.execute(
            select(CreditLedger)
            .where(CreditLedger.user_id == uid)
            .order_by(CreditLedger.created_at, CreditLedger.id)
        )).scalars().all()
        return {
            "message_remaining": Decimal(balance.message_credits_remaining),
            "purchased_remaining": Decimal(
                getattr(balance, "purchased_credits_remaining", 0) or 0
            ),
            "integration_remaining": Decimal(balance.integration_credits_remaining),
            "used_today": Decimal(balance.message_credits_used_today),
            "daily_cap": Decimal(balance.message_credits_daily_cap),
            "ledger": [
                (r.event_type, r.bucket, Decimal(r.amount),
                 Decimal(r.balance_after), r.metadata_json)
                for r in rows
            ],
        }


def _shadow_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.name == SHADOW_LOGGER and "[CREDIT-SHADOW]" in r.getMessage()]


# ══════════════════════════════════════════════════════════════════════
# THE SAFETY TEST — shadow mode denies nothing and moves no money
# ══════════════════════════════════════════════════════════════════════

async def test_shadow_mode_denies_nothing_and_leaves_the_money_identical(
    credit_flags, caplog,
):
    """The non-negotiable one.

    Both users are in the PRODUCTION shape — enforcement off, cap admission
    off — and both are already over their daily cap, so the fully-enabled gate
    would refuse their next turn. The only difference is the shadow flag.

    The charge must succeed for both, and every number a charge can touch must
    be equal: wallet, purchased wallet, integration wallet, used_today, cap,
    and the full ledger (event type, bucket, amount, balance_after, metadata).
    """
    credit_flags(
        credit_enforcement_enabled=False,
        credit_cap_admission_control=False,
        credit_shadow_admission_logging=False,
    )
    control = await _seed_capped_user(
        "shadow-control@example.com", used_today=Decimal("28"),
    )
    control_result = await _charge(control)
    control_money = await _money(control)

    credit_flags(credit_shadow_admission_logging=True)
    with caplog.at_level(logging.INFO, logger=SHADOW_LOGGER):
        observed = await _seed_capped_user(
            "shadow-observed@example.com", used_today=Decimal("28"),
        )
        observed_result = await _charge(observed)
    observed_money = await _money(observed)

    # 1. The turn the gate would have refused still succeeded.
    assert control_result.success is True
    assert observed_result.success is True, (
        "shadow mode must never turn a would-deny verdict into a refusal"
    )
    assert observed_result.reason == control_result.reason

    # 2. Not one number moved differently.
    assert observed_money == control_money, (
        "shadow mode changed the money; the verdict has a caller somewhere"
    )

    # 3. …and it really did decide "deny", so this is not vacuous.
    from app.credit_shadow import parse_line
    lines = _shadow_lines(caplog)
    assert len(lines) == 1, f"expected exactly one shadow line, got {lines}"
    record = parse_line(lines[0])
    assert record is not None
    assert record["decision"] == "deny"
    assert record["reason"] == "daily_cap_exceeded"


async def test_shadow_mode_does_not_change_the_preflight_answer(credit_flags):
    """The other place a verdict could leak into behaviour: `check_balance`.

    `_admission_verdict` was extracted out of it so shadow mode could ask the
    same question; the extraction must not have moved the answer.
    """
    from app.db import async_session_maker
    from app.db.models import BUCKET_MESSAGE
    from app.services.credit_service import credit_service

    uid = await _seed_capped_user(
        "shadow-preflight@example.com", used_today=Decimal("28"),
    )
    answers = {}
    for shadow in (False, True):
        for enforcement, cap_admission in (
            (False, False), (True, False), (True, True),
        ):
            credit_flags(
                credit_enforcement_enabled=enforcement,
                credit_cap_admission_control=cap_admission,
                credit_shadow_admission_logging=shadow,
            )
            async with async_session_maker() as db:
                peek = await credit_service.check_balance(
                    db, uid, BUCKET_MESSAGE, Decimal("0.1"),
                )
            answers[(shadow, enforcement, cap_admission)] = (peek.success, peek.reason)

    for enforcement, cap_admission in ((False, False), (True, False), (True, True)):
        assert (answers[(False, enforcement, cap_admission)]
                == answers[(True, enforcement, cap_admission)]), (
            f"shadow flag changed the pre-flight at "
            f"enforcement={enforcement} cap_admission={cap_admission}"
        )
    # And the answers are the ones #471 pinned, so the equality above is not
    # equality between two broken values.
    assert answers[(False, False, False)] == (True, None)
    assert answers[(False, True, False)] == (True, None)
    assert answers[(False, True, True)] == (False, "daily_cap_exceeded")


# ══════════════════════════════════════════════════════════════════════
# THE VERDICT IS LOGGED, AND THE ALLOW SIDE IS THE ANTI-VACUITY CONTROL
# ══════════════════════════════════════════════════════════════════════

async def test_a_would_deny_turn_logs_every_field_the_rollup_needs(
    credit_flags, caplog,
):
    from app.credit_shadow import SHADOW_PREFIX, parse_line

    credit_flags(
        credit_enforcement_enabled=False,
        credit_cap_admission_control=False,
        credit_shadow_admission_logging=True,
    )
    uid = await _seed_capped_user("shadow-fields@example.com",
                                  used_today=Decimal("28"))
    with caplog.at_level(logging.INFO, logger=SHADOW_LOGGER):
        await _charge(uid, "26.5")

    lines = _shadow_lines(caplog)
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith(SHADOW_PREFIX), "the grep prefix must lead the line"

    record = parse_line(line)
    assert record is not None, f"the emitter wrote a line its own parser rejects: {line}"
    assert record["v"] == "1"
    assert record["event"] == "admission"
    assert record["user"] == uid
    assert record["decision"] == "deny"
    assert record["reason"] == "daily_cap_exceeded"
    assert record["amount"] == Decimal("26.5000"), "the REAL cost, not the quote"
    assert record["quote"] == Decimal("0.1000"), "what the live pre-flight asks"
    assert record["used_today"] == Decimal("28.0000")
    assert record["cap"] == Decimal("15.0000")
    assert record["plan_remaining"] == Decimal("100.0000")
    assert record["purchased_remaining"] == Decimal("0.0000")
    assert record["day"] == datetime.utcnow().date().isoformat()
    assert record["type"] == "chat_message"
    assert record["unlimited"] is False
    # The live flags ride along: without them a reader cannot tell a
    # counterfactual from a description of live behaviour.
    assert record["enforcement"] is False
    assert record["cap_admission"] is False


async def test_an_admissible_turn_logs_an_allow(credit_flags, caplog):
    """ANTI-VACUITY CONTROL.

    "Shadow mode denies nothing" is trivially true of code that never runs. A
    user well under their cap must produce a line too, and it must say allow.
    This test stays GREEN when the shadow hook is reverted only if the hook is
    actually reachable — so it is the control that proves the safety test above
    is testing something.
    """
    from app.credit_shadow import parse_line

    credit_flags(
        credit_enforcement_enabled=False,
        credit_cap_admission_control=False,
        credit_shadow_admission_logging=True,
    )
    uid = await _seed_capped_user("shadow-allow@example.com",
                                  used_today=Decimal("0"))
    with caplog.at_level(logging.INFO, logger=SHADOW_LOGGER):
        result = await _charge(uid, "1.25")

    assert result.success is True
    lines = _shadow_lines(caplog)
    assert len(lines) == 1
    record = parse_line(lines[0])
    assert record is not None
    assert record["decision"] == "allow"
    assert record["reason"] is None
    assert record["amount"] == Decimal("1.2500")


async def test_the_flag_off_writes_nothing_at_all(credit_flags, caplog):
    """Inertness. Default off must be silent, not merely quiet."""
    credit_flags(
        credit_enforcement_enabled=False,
        credit_cap_admission_control=False,
        credit_shadow_admission_logging=False,
    )
    uid = await _seed_capped_user("shadow-off@example.com",
                                  used_today=Decimal("28"))
    with caplog.at_level(logging.DEBUG):
        await _charge(uid)
    assert _shadow_lines(caplog) == []
    assert not [r for r in caplog.records if "[CREDIT-SHADOW]" in r.getMessage()]


def test_the_flag_ships_off():
    """A measurement flag that shipped on would be an ops surprise, not a PR."""
    from app.config import Settings
    assert Settings.model_fields["credit_shadow_admission_logging"].default is False


def test_this_pr_does_not_touch_the_enforcement_defaults():
    """G2 boundary, pinned in a test rather than in a promise."""
    from app.config import Settings
    assert Settings.model_fields["credit_enforcement_enabled"].default is False
    assert Settings.model_fields["credit_cap_admission_control"].default is False


# ══════════════════════════════════════════════════════════════════════
# WHAT A LINE MAY CARRY
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("hostile", [
    "chat_message victim@example.com",
    "sk-FAKE-not-a-real-key-0000000000000000000000000000",
    "eyJhbGciOiJIUzI1NiJ9.FAKE.FAKE",
    "remind me to call the clinic about my test results",
    "Bearer FAKEFAKEFAKE",
])
def test_a_free_form_field_is_rejected_whole_never_shortened(hostile):
    """User ids only — and the guarantee is REJECTION, not scrubbing.

    Truncation is not redaction (#419): `content[:50]` passed three green runs
    while seventeen sites leaked memory content, because the secret was 49
    characters. So a value that does not match its allowed shape is replaced by
    the word "rejected" and no fragment of it survives.
    """
    from app.credit_shadow import REJECTED, ShadowAdmission, format_line, parse_line

    line = format_line(ShadowAdmission(
        user_id="8f2a1c44-0000-4000-8000-000000000001",
        decision="deny",
        reason="daily_cap_exceeded",
        amount=Decimal("26.5"), quote=Decimal("0.1"),
        used_today=Decimal("28"), cap=Decimal("15"),
        plan_remaining=Decimal("100"), purchased_remaining=Decimal("0"),
        day="2026-08-06",
        # Nothing in production puts these in `type`; the point is that the
        # format could not carry them even if something did. Obviously-fake
        # values throughout.
        event_type=hostile,
        unlimited=False, enforcement_enabled=False, cap_admission_control=False,
    ))
    assert f"type={REJECTED} " in line
    assert "@" not in line
    assert "\n" not in line and "\r" not in line
    # No fragment of the hostile value survives in any emitted VALUE. (Keys
    # are checked separately by test_the_wire_format_is_pinned; "me" is a
    # substring of "plan_remaining", which is a key, not data.)
    values = [pair.split("=", 1)[1] for pair in line.split()[1:]]
    for word in hostile.replace("@", " ").replace(".", " ").split():
        if len(word) < 3:
            continue
        assert not any(word in value for value in values), (
            f"{word!r} leaked into {values!r}"
        )
    # Rejecting a field must not break the schema for the fields that are fine.
    record = parse_line(line)
    assert record is not None
    assert record["user"] == "8f2a1c44-0000-4000-8000-000000000001"
    assert record["amount"] == Decimal("26.5000")


def test_the_wire_format_is_pinned():
    """The rollup greps for these exact key names; changing one silently is how
    a probe starts reading a field that does not exist."""
    from app.credit_shadow import ShadowAdmission, format_line

    line = format_line(ShadowAdmission(
        user_id="8f2a1c44-0000-4000-8000-000000000001",
        decision="deny", reason="daily_cap_exceeded",
        amount=Decimal("26.5"), quote=Decimal("0.1"),
        used_today=Decimal("28"), cap=Decimal("15"),
        plan_remaining=Decimal("100"), purchased_remaining=Decimal("0"),
        day="2026-08-06", event_type="chat_message",
        unlimited=False, enforcement_enabled=False, cap_admission_control=False,
    ))
    assert line == (
        "[CREDIT-SHADOW] v=1 event=admission "
        "user=8f2a1c44-0000-4000-8000-000000000001 decision=deny "
        "reason=daily_cap_exceeded amount=26.5000 quote=0.1000 "
        "used_today=28.0000 cap=15.0000 plan_remaining=100.0000 "
        "purchased_remaining=0.0000 day=2026-08-06 type=chat_message "
        "unlimited=0 enforcement=0 cap_admission=0"
    )


# ══════════════════════════════════════════════════════════════════════
# THE ROLLUP
# ══════════════════════════════════════════════════════════════════════

def _line(user, decision, amount, day, *, reason=None, cap="15.0000",
          used="28.0000", etype="chat_message", unlimited=False):
    from app.credit_shadow import ShadowAdmission, format_line
    return format_line(ShadowAdmission(
        user_id=user, decision=decision,
        reason=(reason if reason is not None
                else ("daily_cap_exceeded" if decision == "deny" else None)),
        amount=Decimal(amount), quote=Decimal("0.1"),
        used_today=Decimal(used),
        cap=None if cap is None else Decimal(cap),
        plan_remaining=Decimal("100"), purchased_remaining=Decimal("0"),
        day=day, event_type=etype, unlimited=unlimited,
        enforcement_enabled=False, cap_admission_control=False,
    ))


def test_rollup_computes_the_share_of_spend_and_the_users_per_day():
    """Hand-computable fixture, so the assertion is arithmetic and not a
    re-implementation of the code under test.

    day-1: u1 deny 30, u1 allow 10, u2 allow 20      → 1 capped user
    day-2: u1 deny 40, u3 deny 20, u2 allow 5        → 2 capped users
    Totals: spend 125, denied 90 → 72.0%. Users seen 3, denied 2.
    Per-day capped users [1, 2] → median 1.5, mean 1.5, max 2.
    """
    from scripts.credit_shadow_rollup import rollup

    lines = [
        _line("u1", "deny", "30", "2026-08-05"),
        _line("u1", "allow", "10", "2026-08-05", used="0"),
        _line("u2", "allow", "20", "2026-08-05", used="0"),
        _line("u1", "deny", "40", "2026-08-06"),
        _line("u3", "deny", "20", "2026-08-06"),
        _line("u2", "allow", "5", "2026-08-06", used="0"),
        "some other log line entirely",
        "[PERF] ws_proxy_agent_wait_ms=12",
    ]
    result = rollup(lines)

    assert result["lines_parsed"] == 6
    assert result["lines_skipped"] == 2
    assert result["charges"] == 6
    assert result["spend_credits"] == Decimal("125.0000")
    assert result["would_deny_spend_credits"] == Decimal("90.0000")
    assert result["would_deny_spend_pct"] == pytest.approx(72.0)
    assert result["distinct_users"] == 3
    assert result["distinct_users_would_deny"] == 2
    assert result["days_observed"] == 2
    cap = result["users_hitting_cap_per_day"]
    assert cap["series"] == {"2026-08-05": 1, "2026-08-06": 2}
    assert cap["median"] == 1.5
    assert cap["mean"] == 1.5
    assert cap["max"] == 2
    assert result["would_deny_reasons"] == {"daily_cap_exceeded": 3}
    assert result["flag_regimes"] == {"enforcement=0 cap_admission=0": 6}


def test_rollup_excludes_unlimited_accounts_unless_asked():
    """Admins are never charged, so a would-deny line for one describes a
    refusal that would never have billed anybody."""
    from scripts.credit_shadow_rollup import rollup

    lines = [
        _line("u1", "deny", "30", "2026-08-06"),
        _line("admin", "deny", "70", "2026-08-06", unlimited=True),
    ]
    default = rollup(lines)
    assert default["spend_credits"] == Decimal("30.0000")
    assert default["distinct_users"] == 1
    assert default["lines_filtered_out"] == 1

    everything = rollup(lines, include_unlimited=True)
    assert everything["spend_credits"] == Decimal("100.0000")
    assert everything["distinct_users"] == 2


def test_rollup_can_restrict_to_one_event_type():
    from scripts.credit_shadow_rollup import rollup

    lines = [
        _line("u1", "deny", "30", "2026-08-06"),
        _line("u1", "deny", "70", "2026-08-06", etype="image_generation"),
    ]
    chat_only = rollup(lines, event_types={"chat_message"})
    assert chat_only["spend_credits"] == Decimal("30.0000")
    assert set(rollup(lines)["by_event_type"]) == {"chat_message", "image_generation"}


def test_rollup_refuses_a_line_it_cannot_read_rather_than_guessing():
    """A skipped line makes the number smaller, which is visible. A field read
    as zero makes it wrong, which is not."""
    from scripts.credit_shadow_rollup import rollup

    good = _line("u1", "deny", "30", "2026-08-06")
    future = good.replace("v=1", "v=99")
    truncated = good.replace(" day=2026-08-06", "")
    garbled = good.replace("amount=30.0000", "amount=thirty")

    result = rollup([good, future, truncated, garbled])
    assert result["lines_parsed"] == 1
    assert result["lines_skipped"] == 3
    assert result["spend_credits"] == Decimal("30.0000")


def test_rollup_is_honest_about_an_empty_log():
    from scripts.credit_shadow_rollup import _render, rollup

    result = rollup(["nothing to see here"])
    assert result["lines_parsed"] == 0
    assert result["would_deny_spend_pct"] is None
    assert "No shadow lines found" in _render(result)


def test_rollup_labels_every_figure_measured_or_modelled():
    """A modelled figure reads exactly like a measured one unless it is
    labelled. The renderer must label both."""
    from scripts.credit_shadow_rollup import _render, rollup

    text = _render(rollup([_line("u1", "deny", "30", "2026-08-06")]))
    assert "MEASURED" in text
    assert "MODELLED" in text


def test_rollup_warns_when_the_lines_were_written_under_live_enforcement():
    """Mixed regimes cannot be summed: under enforcement ON a would-deny turn
    may really have been refused, so its spend was never served free."""
    from app.credit_shadow import ShadowAdmission, format_line
    from scripts.credit_shadow_rollup import _render, rollup

    live = format_line(ShadowAdmission(
        user_id="u1", decision="deny", reason="daily_cap_exceeded",
        amount=Decimal("30"), quote=Decimal("0.1"), used_today=Decimal("28"),
        cap=Decimal("15"), plan_remaining=Decimal("100"),
        purchased_remaining=Decimal("0"), day="2026-08-06",
        event_type="chat_message", unlimited=False,
        enforcement_enabled=True, cap_admission_control=True,
    ))
    text = _render(rollup([live]))
    assert "enforcement=1 cap_admission=1" in text
    assert "WARNING" in text


async def test_the_rollup_reads_what_the_service_actually_wrote(
    credit_flags, caplog,
):
    """End to end, and the reason the parser lives beside the emitter.

    Real charges through `try_charge` → real log records → the real rollup. If
    the emitter ever renames a field, this fails here instead of quietly
    printing a plausible, uniform, fictional zero in production.
    """
    from scripts.credit_shadow_rollup import rollup

    credit_flags(
        credit_enforcement_enabled=False,
        credit_cap_admission_control=False,
        credit_shadow_admission_logging=True,
    )
    over = await _seed_capped_user("shadow-e2e-over@example.com",
                                   used_today=Decimal("28"))
    under = await _seed_capped_user("shadow-e2e-under@example.com",
                                    used_today=Decimal("0"))
    with caplog.at_level(logging.INFO, logger=SHADOW_LOGGER):
        await _charge(over, "30")
        await _charge(under, "10")

    result = rollup(_shadow_lines(caplog))
    assert result["lines_parsed"] == 2, "the rollup could not read the emitter"
    assert result["spend_credits"] == Decimal("40.0000")
    assert result["would_deny_spend_credits"] == Decimal("30.0000")
    assert result["would_deny_spend_pct"] == pytest.approx(75.0)
    assert result["distinct_users"] == 2
    assert result["distinct_users_would_deny"] == 1
    assert result["users_hitting_cap_per_day"]["max"] == 1
