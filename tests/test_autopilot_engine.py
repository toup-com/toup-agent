"""Autopilot engine core (Autopilot PR6).

Contracts under test:
- the handler NEVER raises and always returns status='success' so the
  runner's watermark mechanism persists mission state (a raise would
  burn the runner's retry budget on a turn that already spent tokens);
- tier-1 gates cost zero LLM calls (fake runner not invoked);
- terminal transitions disable the routine row (stops job minting) and
  notify through the durable outbox; waiting states keep it enabled;
- the strike system — not the model's marker honesty — terminates
  drifting loops;
- the per-kind fire idempotency key unblocks multi-tick days.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.agent.routines.autopilot_handler import (
    AutopilotHandler,
    MISSION_BLOCKED, MISSION_COMPLETED, MISSION_FAILED,
    MISSION_WAITING_INPUT,
    parse_tick_markers,
)
from app.db.models import Routine


# ── Marker parsing ────────────────────────────────────────────────


def test_parse_tick_markers_happy_and_tolerant():
    text = (
        "I did things.\n"
        "AUTOPILOT_STATUS: working extra words\n"
        "AUTOPILOT_SUMMARY: drafted the email\n"
        "AUTOPILOT_NOTE: send it after approval\n"
    )
    m = parse_tick_markers(text)
    assert m == {
        "status": "working",
        "summary": "drafted the email",
        "note": "send it after approval",
    }

    assert parse_tick_markers("no markers here") == {}
    # Last occurrence wins.
    m = parse_tick_markers("AUTOPILOT_STATUS: working\nAUTOPILOT_STATUS: done")
    assert m["status"] == "done"


# ── Fire idempotency key ──────────────────────────────────────────


def test_fire_idempotency_key_per_kind():
    from app.agent.routines.runner import RoutineRunner

    d = date(2026, 7, 8)
    instant = datetime(2026, 7, 8, 14, 5, 30)
    # Daily kinds keep the one-per-local-day contract.
    assert RoutineRunner._fire_idempotency_key("email_briefing", d, instant) == "2026-07-08"
    # Autopilot gets a fresh key per fire instant → multi-tick days.
    k1 = RoutineRunner._fire_idempotency_key("autopilot", d, instant)
    k2 = RoutineRunner._fire_idempotency_key(
        "autopilot", d, instant + timedelta(seconds=300),
    )
    assert k1 == "2026-07-08T140530" and k1 != k2
    # Retry of the SAME fire shares the key.
    assert RoutineRunner._fire_idempotency_key("autopilot", d, instant) == k1
    # Force-run (no APScheduler instant) still yields a usable key.
    assert RoutineRunner._fire_idempotency_key("autopilot", d, None)


def test_kind_enabled_flag_default_off(monkeypatch):
    from app.agent.routines.runner import RoutineRunner
    from app.config import settings

    assert RoutineRunner._kind_enabled("autopilot", settings) is False
    monkeypatch.setattr(settings, "autopilot_enabled", True)
    assert RoutineRunner._kind_enabled("autopilot", settings) is True


# ── Handler harness ───────────────────────────────────────────────


class FakeRunner:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if reply == "SLOW":
            await asyncio.sleep(5)
        return SimpleNamespace(
            text=reply, tokens_input=1000, tokens_output=500,
            model="gpt-4o-mini", asst_message_id="am-1",
        )


@pytest.fixture
def notify_calls(monkeypatch):
    calls = []

    async def fake_notify(**kwargs):
        calls.append(kwargs)
        return "nid"

    import app.services.agent_notify_client as anc
    monkeypatch.setattr(anc, "notify", fake_notify)
    return calls


@pytest.fixture(autouse=True)
def _platform_reachable(monkeypatch):
    """Default: credits fine, platform reachable."""
    import app.services.credit_reporter as cr

    monkeypatch.setattr(cr, "raise_if_exhausted", lambda: None)

    async def ok_preflight(**kwargs):
        return SimpleNamespace(network_ok=True)

    monkeypatch.setattr(cr, "check_balance_remote", ok_preflight)


async def _mk_mission(state=None, cfg=None, **routine_overrides) -> Routine:
    from app.db import async_session_maker
    from app.db.models import User
    from app.services.auth_service import get_password_hash

    user_id = str(uuid.uuid4())
    fields = dict(
        id=str(uuid.uuid4()),
        user_id=user_id,
        kind="autopilot",
        enabled=True,
        name="Test mission",
        schedule_cron_local="*/5 * * * *",
        schedule_kind="every",
        schedule_interval_seconds=300,
        config_json={"goal": "book the dentist", "budget_credits": 50},
        last_state_json=state or {},
    )
    fields.update(routine_overrides)
    if cfg is not None:
        fields["config_json"] = cfg
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"ap-{uuid.uuid4().hex[:8]}@example.com",
            hashed_password=get_password_hash("x" * 12), name="AP",
        ))
        db.add(Routine(**fields))
        await db.commit()
        routine = await db.get(Routine, fields["id"])
        db.expunge(routine)
    return routine


async def _routine_row(routine_id: str) -> Routine:
    from app.db import async_session_maker

    async with async_session_maker() as db:
        r = await db.get(Routine, routine_id)
        db.expunge(r)
        return r


def _run_shim():
    return SimpleNamespace(id=str(uuid.uuid4()), job_id=str(uuid.uuid4()))


async def _exec(handler, routine):
    from app.db import async_session_maker

    async with async_session_maker() as db:
        return await handler.execute(routine, _run_shim(), db)


# ── Tier-1 gates ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_not_due_is_free(notify_calls):
    future = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
    routine = await _mk_mission(state={"status": "active", "next_due_at": future})
    fake = FakeRunner([])
    result = await _exec(AutopilotHandler(fake), routine)
    assert result.status == "success" and result.outcome == "success_empty"
    assert fake.calls == [] and notify_calls == []


@pytest.mark.asyncio
async def test_waiting_status_noops(notify_calls):
    routine = await _mk_mission(state={"status": MISSION_WAITING_INPUT})
    fake = FakeRunner([])
    result = await _exec(AutopilotHandler(fake), routine)
    assert result.outcome == "success_empty" and fake.calls == []


@pytest.mark.asyncio
async def test_budget_ceiling_blocks_and_notifies(notify_calls):
    routine = await _mk_mission(
        state={"status": "active", "spent_credits": 55.0},
    )
    fake = FakeRunner([])
    result = await _exec(AutopilotHandler(fake), routine)
    assert result.new_watermark["status"] == MISSION_BLOCKED
    assert result.new_watermark["status_reason"] == "budget_exhausted"
    assert fake.calls == []
    assert notify_calls[0]["event_kind"] == "mission_failed"
    assert notify_calls[0]["priority"] == "high"
    # Terminal/blocked → routine disabled so job minting stops.
    assert (await _routine_row(routine.id)).enabled is False


@pytest.mark.asyncio
async def test_max_ticks_cap(notify_calls):
    routine = await _mk_mission(
        state={"status": "active", "ticks_run": 100},
    )
    result = await _exec(AutopilotHandler(FakeRunner([])), routine)
    assert result.new_watermark["status"] == MISSION_FAILED
    assert result.new_watermark["status_reason"] == "max_ticks_exceeded"


# ── Tick flow ─────────────────────────────────────────────────────


_WORKING = (
    "progress...\nAUTOPILOT_STATUS: working\n"
    "AUTOPILOT_SUMMARY: found three dentists, comparing\n"
    "AUTOPILOT_NOTE: call list in workspace/dentists.md"
)
_DONE = (
    "all set\nAUTOPILOT_STATUS: done\n"
    "AUTOPILOT_SUMMARY: booked Tuesday 10:00 with Dr. Kim"
)
_BLOCKED = (
    "need input\nAUTOPILOT_STATUS: blocked\n"
    "AUTOPILOT_SUMMARY: two equally good slots — which do you prefer?"
)


@pytest.mark.asyncio
async def test_working_tick_advances_state(notify_calls):
    routine = await _mk_mission()
    fake = FakeRunner([_WORKING])
    result = await _exec(AutopilotHandler(fake), routine)

    wm = result.new_watermark
    assert wm["ticks_run"] == 1
    assert wm["no_progress_streak"] == 0
    assert wm["spent_credits"] > 0
    assert wm["last_summary"] == "found three dentists, comparing"
    assert wm["note"] == "call list in workspace/dentists.md"
    # next due ≈ base interval (no backoff on progress).
    delta = (datetime.fromisoformat(wm["next_due_at"]) - datetime.utcnow())
    assert timedelta(minutes=4) < delta < timedelta(minutes=6)
    # A working tick emits exactly one progress heartbeat (drives the
    # phone's Live Activity bar) — never an alert kind: the dispatcher
    # suppresses kind=progress from every OS push path.
    assert [c["event_kind"] for c in notify_calls] == ["progress"]
    assert notify_calls[0]["data"]["mission_id"] == routine.id
    assert notify_calls[0]["priority"] == "low"
    assert "progress" in notify_calls[0]["data"]
    # The synthetic-turn invariants. channel='autopilot' keys the
    # PR7 policy layers (connector default-deny, vault strip, MCP
    # pending_channel).
    call = fake.calls[0]
    assert call["save_user_message"] is False
    assert call["channel"] == "autopilot"
    assert call["current_job_id"]


@pytest.mark.asyncio
async def test_done_completes_disables_and_notifies(notify_calls):
    routine = await _mk_mission()
    result = await _exec(AutopilotHandler(FakeRunner([_DONE])), routine)
    assert result.new_watermark["status"] == MISSION_COMPLETED
    assert notify_calls[0]["event_kind"] == "mission_completed"
    assert "Dr. Kim" in notify_calls[0]["body"]
    assert notify_calls[0]["data"]["mission_id"] == routine.id
    assert (await _routine_row(routine.id)).enabled is False


@pytest.mark.asyncio
async def test_blocked_waits_for_input_and_stays_enabled(notify_calls):
    routine = await _mk_mission()
    result = await _exec(AutopilotHandler(FakeRunner([_BLOCKED])), routine)
    wm = result.new_watermark
    assert wm["status"] == MISSION_WAITING_INPUT
    assert wm["next_due_at"] is None  # ticks paused until the user answers
    assert notify_calls[0]["event_kind"] == "needs_input"
    assert notify_calls[0]["priority"] == "high"
    # Resume path (PR7/8) needs the routine still schedulable.
    assert (await _routine_row(routine.id)).enabled is True


@pytest.mark.asyncio
async def test_no_progress_strikes_escalate(notify_calls):
    """Identical/missing summaries strike; threshold → ask the user.
    The marker contract is a hint — the strike system is enforcement."""
    handler = AutopilotHandler(FakeRunner(["no markers at all"] * 3))
    state = {}
    for expected_streak in (1, 2):
        r = await _mk_mission(state={**state, "status": "active"})
        # carry state forward manually (each execute persists via runner
        # in prod; here we thread the watermark ourselves)
        result = await _exec(handler, r)
        state = result.new_watermark
        assert state["no_progress_streak"] == expected_streak
        assert state.get("status", "active") == "active"
        # Backoff doubles with the streak.
        delay = datetime.fromisoformat(state["next_due_at"]) - datetime.utcnow()
        assert delay > timedelta(minutes=4) * (2 ** expected_streak)
        # Thread the state to the next mission tick as if the backoff
        # window already elapsed.
        state["next_due_at"] = None

    r = await _mk_mission(state={**state, "status": "active"})
    result = await _exec(handler, r)
    assert result.new_watermark["status"] == MISSION_WAITING_INPUT
    assert result.new_watermark["status_reason"] == "no_progress"
    assert notify_calls[-1]["event_kind"] == "needs_input"


@pytest.mark.asyncio
async def test_turn_exception_is_state_not_raise(notify_calls):
    routine = await _mk_mission()
    result = await _exec(
        AutopilotHandler(FakeRunner([RuntimeError("provider 500")])), routine,
    )
    assert result.status == "success"  # handler contract: never 'failed'
    wm = result.new_watermark
    assert wm["ticks_run"] == 1
    assert "RuntimeError" in wm["last_error"]
    assert wm["no_progress_streak"] == 1
    assert notify_calls == []


@pytest.mark.asyncio
async def test_tick_timeout_backs_off(notify_calls):
    routine = await _mk_mission(cfg={
        "goal": "g", "budget_credits": 50, "tick_timeout_s": 1,
    })
    result = await _exec(AutopilotHandler(FakeRunner(["SLOW"])), routine)
    wm = result.new_watermark
    assert "tick_timeout" in wm["last_error"]
    assert wm["next_due_at"] is not None


# ── Credit / reachability gate ────────────────────────────────────


@pytest.mark.asyncio
async def test_out_of_credits_blocks(monkeypatch, notify_calls):
    import app.services.credit_reporter as cr
    from app.services.credit_exhausted import OutOfCreditsError

    def broke():
        err = OutOfCreditsError.__new__(OutOfCreditsError)
        raise err

    monkeypatch.setattr(cr, "raise_if_exhausted", broke)
    routine = await _mk_mission()
    fake = FakeRunner([])
    result = await _exec(AutopilotHandler(fake), routine)
    assert result.new_watermark["status_reason"] == "insufficient_credits"
    assert fake.calls == []


@pytest.mark.asyncio
async def test_platform_unreachable_fails_closed(monkeypatch, notify_calls):
    """3 unreachable ticks → mission pauses itself. The default credit
    machinery is fail-open; an autonomous loop must not be.

    check_balance_remote returns None for BOTH unconfigured and
    network-failed — the gate only fail-closes when the platform is
    CONFIGURED (endpoint+key present) and the call still yields None."""
    import app.services.credit_reporter as cr

    async def dead(**kwargs):
        return None  # fail-open contract: None = no signal

    monkeypatch.setattr(cr, "check_balance_remote", dead)
    monkeypatch.setattr(cr, "_platform_endpoint", lambda p: f"https://toup.ai/api{p}")
    monkeypatch.setattr(cr, "_agent_key", lambda: "k" * 20)
    handler = AutopilotHandler(FakeRunner([]))

    state = {"status": "active"}
    for expected in (1, 2):
        routine = await _mk_mission(state=state)
        result = await _exec(handler, routine)
        state = result.new_watermark
        assert state["platform_fail_streak"] == expected
        state["next_due_at"] = None  # make next tick due immediately

    routine = await _mk_mission(state=state)
    result = await _exec(handler, routine)
    assert result.new_watermark["status"] == MISSION_BLOCKED
    assert result.new_watermark["status_reason"] == "platform_unreachable"


def test_subagent_token_field_names_fixed():
    """Pin the AgentResponse field-name fix (input_tokens → tokens_input
    etc.) — the old getattrs silently produced None forever."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent
        / "app" / "agent" / "subagent_orchestrator.py"
    ).read_text()
    assert 'getattr(response, "input_tokens"' not in src
    assert 'getattr(response, "tokens_input"' in src
    assert 'getattr(response, "model_used"' not in src


def test_autopilot_gate_allowlist(monkeypatch):
    """Staged rollout: master flag OR user allowlist activates the kind
    (and the start_mission tool) — see app/agent/autopilot_gate.py."""
    from types import SimpleNamespace
    from app.agent.autopilot_gate import autopilot_enabled_for
    from app.agent.routines.runner import RoutineRunner

    off = SimpleNamespace(autopilot_enabled=False, autopilot_user_allowlist="", user_id="u1")
    assert autopilot_enabled_for(off) is False

    master = SimpleNamespace(autopilot_enabled=True, autopilot_user_allowlist="", user_id="")
    assert autopilot_enabled_for(master) is True

    listed = SimpleNamespace(
        autopilot_enabled=False,
        autopilot_user_allowlist="aaa, u1 ,bbb",
        user_id="u1",
    )
    assert autopilot_enabled_for(listed) is True
    assert RoutineRunner._kind_enabled("autopilot", listed) is True

    unlisted = SimpleNamespace(
        autopilot_enabled=False, autopilot_user_allowlist="aaa,bbb", user_id="u1",
    )
    assert autopilot_enabled_for(unlisted) is False

    prebind = SimpleNamespace(
        autopilot_enabled=False, autopilot_user_allowlist="u1", user_id="",
    )
    assert autopilot_enabled_for(prebind) is False


def test_platform_endpoint_normalizes_api_suffix(monkeypatch):
    """Reporter URLs must hit /api/* — a platform_api_url without the
    suffix lands on the SPA catch-all (HTML with HTTP 200) and every
    reporter call silently no-ops (canary live find, 2026-07-09)."""
    from app.config import settings
    from app.services.credit_reporter import _platform_endpoint

    monkeypatch.setattr(settings, "platform_api_url", "https://toup.ai")
    assert _platform_endpoint("/credits/preflight") == "https://toup.ai/api/credits/preflight"

    monkeypatch.setattr(settings, "platform_api_url", "https://toup.ai/api/")
    assert _platform_endpoint("/credits/preflight") == "https://toup.ai/api/credits/preflight"

    monkeypatch.setattr(settings, "platform_api_url", "")
    assert _platform_endpoint("/credits/preflight") is None


def test_every_prompt_profile_covered_by_all_profile_maps():
    """Regression for the live KeyError: PromptProfile.AUTOPILOT tick
    crash — adding a PromptProfile member MUST extend every
    profile-keyed map, or run() raises at prompt-build time."""
    from app.agent.prompt_profile import (
        PromptProfile, _POST_BUILDER_ALLOWED, _SECTION_LISTS,
        allows_post_builder_blocks, disabled_tools_for, sections_for,
    )

    for profile in PromptProfile:
        assert profile in _SECTION_LISTS, f"_SECTION_LISTS missing {profile}"
        assert profile in _POST_BUILDER_ALLOWED, f"_POST_BUILDER_ALLOWED missing {profile}"
        sections_for(profile)
        allows_post_builder_blocks(profile)
        disabled_tools_for(profile)


# ── Budget hard-stop (2026-07-16): mid-tick ceiling + post-tick reconcile ──


class BudgetFakeRunner:
    """FakeRunner variant whose replies are full response namespaces —
    lets tests drive credits_spent / stopped_reason, and optionally
    feed the on_usage sink before timing out."""

    def __init__(self, replies, usage_events=None, slow=False):
        self.replies = list(replies)
        self.usage_events = usage_events or []
        self.slow = slow
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        on_usage = kwargs.get("on_usage")
        if on_usage:
            for (model, t_in, t_out) in self.usage_events:
                await on_usage(model, t_in, t_out)
        if self.slow:
            await asyncio.sleep(5)
        return self.replies.pop(0)


def _resp(text, *, credits=0.0, stopped=""):
    return SimpleNamespace(
        text=text, tokens_input=1000, tokens_output=500,
        model="gpt-4o-mini", asst_message_id="am-1",
        credits_spent=credits, stopped_reason=stopped,
    )


@pytest.mark.asyncio
async def test_tick_passes_remaining_budget_to_runner(notify_calls):
    routine = await _mk_mission(
        state={"status": "active", "spent_credits": 30.0},
    )
    fake = BudgetFakeRunner([_resp(
        "AUTOPILOT_STATUS: working\nAUTOPILOT_SUMMARY: step 1 done",
        credits=2.0,
    )])
    await _exec(AutopilotHandler(fake), routine)
    assert fake.calls, "tick should have run"
    assert fake.calls[0]["credit_budget"] == pytest.approx(20.0)
    assert callable(fake.calls[0]["on_usage"])


@pytest.mark.asyncio
async def test_midtick_budget_stop_blocks_mission(notify_calls):
    """Runner tripped its in-run ceiling → mission blocks
    budget_exhausted immediately, spend persisted, honest push."""
    routine = await _mk_mission(
        state={"status": "active", "spent_credits": 10.0},
    )
    fake = BudgetFakeRunner([_resp(
        "AUTOPILOT_STATUS: working\nAUTOPILOT_SUMMARY: got halfway",
        credits=45.0, stopped="credit_budget",
    )])
    result = await _exec(AutopilotHandler(fake), routine)
    wm = result.new_watermark
    assert wm["status"] == "blocked"
    assert wm["status_reason"] == "budget_exhausted"
    assert wm["spent_credits"] == pytest.approx(55.0)
    assert any(c["event_kind"] == "mission_failed" for c in notify_calls)


@pytest.mark.asyncio
async def test_working_tick_landing_over_budget_blocks_now(notify_calls):
    """A working tick whose spend lands >= budget must block in the
    same tick — not lie 'active' for 5 minutes until the next fire's
    pre-tick gate."""
    routine = await _mk_mission(
        state={"status": "active", "spent_credits": 48.0},
    )
    fake = BudgetFakeRunner([_resp(
        "AUTOPILOT_STATUS: working\nAUTOPILOT_SUMMARY: more findings",
        credits=6.0,  # lands at 54 >= 50, no in-run stop
    )])
    result = await _exec(AutopilotHandler(fake), routine)
    wm = result.new_watermark
    assert wm["status"] == "blocked"
    assert wm["status_reason"] == "budget_exhausted"
    assert "next_due_at" not in wm or wm["next_due_at"] is None


@pytest.mark.asyncio
async def test_done_over_budget_completes_with_overshoot(notify_calls):
    """Finished work is never blocked retroactively — the overshoot is
    recorded instead (bounded to ~one LLM call by the in-run ceiling)."""
    routine = await _mk_mission(
        state={"status": "active", "spent_credits": 45.0},
    )
    fake = BudgetFakeRunner([_resp(
        "AUTOPILOT_STATUS: done\nAUTOPILOT_SUMMARY: report finished",
        credits=8.0,  # lands at 53 > 50
    )])
    result = await _exec(AutopilotHandler(fake), routine)
    wm = result.new_watermark
    assert wm["status"] == "completed"
    assert wm["progress"] == 100
    assert wm["budget_overshoot"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_timeout_tick_meters_sink_spend(notify_calls):
    """A timed-out tick's LLM calls were genuinely charged — the
    on_usage sink must advance spent_credits (before 2026-07-16 they
    were invisible to the mission ledger)."""
    routine = await _mk_mission(
        cfg={"goal": "g", "budget_credits": 50, "tick_timeout_s": 1},
        state={"status": "active", "spent_credits": 5.0},
    )
    fake = BudgetFakeRunner(
        [_resp("never returned")], slow=True,
        usage_events=[("gpt-4o-mini", 100_000, 20_000)],
    )
    result = await _exec(AutopilotHandler(fake), routine)
    wm = result.new_watermark
    assert wm["ticks_run"] == 1
    assert wm["spent_credits"] > 5.0


# ── runner-side pure-function pins ──


def test_credits_for_llm_call_positive_for_priced_model():
    from app.agent.agent_runner import _credits_for_llm_call
    from app.config import settings

    model = next(iter(settings.pricing_per_1k))
    assert _credits_for_llm_call(model, 100_000, 20_000) > 0


def test_tokens_to_credits_raw_has_no_floor():
    from decimal import Decimal
    from app.services.credit_service import tokens_to_credits, tokens_to_credits_raw

    model = "gpt-4o-mini"
    floored = tokens_to_credits(model, 10, 5)
    raw = tokens_to_credits_raw(model, 10, 5)
    assert floored >= Decimal("0.1")
    assert raw < floored


# ── Clean mission chat (2026-07-16): headless ticks, one terminal message ──


def test_strip_tick_markers_removes_engine_contract_lines():
    from app.agent.routines.autopilot_handler import strip_tick_markers

    raw = (
        "Here is what I found about design tools.\n"
        "AUTOPILOT_STATUS: done\n"
        "Some more detail.\n"
        "autopilot_summary: finished the report\n"
        "AUTOPILOT_PROGRESS: 100\n"
        "AUTOPILOT_NOTE: private scratchpad\n"
    )
    out = strip_tick_markers(raw)
    assert "AUTOPILOT_" not in out.upper()
    assert "Here is what I found" in out
    assert "Some more detail." in out


@pytest.fixture
def chat_writes(monkeypatch):
    """Capture the routine message-writer seam (the real writer needs
    the pgvector-backed messages table, absent under SQLite)."""
    import app.agent.routines.message_writer as mw

    writes: list[dict] = []
    broadcasts: list[dict] = []

    async def fake_write(db, **kwargs):
        writes.append(kwargs)
        return f"msg-{len(writes)}", "dc-1"

    async def fake_broadcast(user_id, **kwargs):
        broadcasts.append({"user_id": user_id, **kwargs})
        return {"ws_count": 1, "channel_results": {}}

    monkeypatch.setattr(mw, "write_routine_message", fake_write)
    monkeypatch.setattr(mw, "broadcast_routine_message", fake_broadcast)
    return {"writes": writes, "broadcasts": broadcasts}


@pytest.mark.asyncio
async def test_ticks_run_headless(notify_calls, chat_writes):
    """The tick's raw reply (with its mandated AUTOPILOT_* trailer)
    must never persist as a chat message — working ticks are silent."""
    routine = await _mk_mission(state={"status": "active"})
    fake = BudgetFakeRunner([_resp(
        "AUTOPILOT_STATUS: working\nAUTOPILOT_SUMMARY: step 1",
    )])
    result = await _exec(AutopilotHandler(fake), routine)
    assert fake.calls[0]["save_assistant_message"] is False
    assert fake.calls[0]["disable_post_processing"] is True
    assert result.summary_message_id is None
    assert chat_writes["writes"] == [], "working tick wrote a chat message"


@pytest.mark.asyncio
async def test_done_writes_one_clean_chat_message(notify_calls, chat_writes):
    """Mission completion posts exactly ONE user-facing message with
    the cleaned results — no AUTOPILOT_ markers — and stamps its id
    on the RoutineResult so 'Open in chat' works."""
    routine = await _mk_mission(state={"status": "active"})
    fake = BudgetFakeRunner([_resp(
        "The final report is ready: Figma leads for teams.\n"
        "AUTOPILOT_STATUS: done\n"
        "AUTOPILOT_SUMMARY: report finished\n"
        "AUTOPILOT_PROGRESS: 100",
        credits=2.0,
    )])
    result = await _exec(AutopilotHandler(fake), routine)
    assert len(chat_writes["writes"]) == 1
    w = chat_writes["writes"][0]
    assert w["source"] == "autopilot"
    assert "Mission complete" in w["content"]
    assert "Figma leads for teams" in w["content"]
    assert "AUTOPILOT_" not in w["content"]
    assert w["extra_metadata"]["mission_id"] == routine.id
    assert result.summary_message_id == "msg-1"
    assert len(chat_writes["broadcasts"]) == 1


def test_read_paths_hide_autopilot_channel():
    """Historical raw tick rows are hidden at the two user-visible
    read paths (no data migration needed).

    R31 replaced the literal `!= "autopilot"` with the shared
    HIDDEN_DAY_CHANNELS tuple, a strictly BROADER predicate (autopilot
    + automation), so the contract this pin protects still holds. The
    pin was written against the spelling, not the contract, so it went
    red on a correct change — re-anchored on the tuple AND on autopilot
    being in it, which is the thing that actually has to stay true.
    """
    import inspect
    from app.api import day_chats, messages_recover
    from app.db.models.conversation import HIDDEN_DAY_CHANNELS

    assert "autopilot" in HIDDEN_DAY_CHANNELS

    pred = "Conversation.channel.notin_(HIDDEN_DAY_CHANNELS)"
    assert inspect.getsource(day_chats).count(pred) == 2
    assert pred in inspect.getsource(messages_recover)
