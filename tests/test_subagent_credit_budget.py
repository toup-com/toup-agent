"""Sub-agent credit-budget enforcement — Phase 9.

Closes the Phase 3 acceptance criterion that was left as a stub
through Phases 3-8:

  "A run that exceeds its credit_budget terminates and reports
   budget_exhausted."

The LLM-proxy credit hook on feat/credit-billing-system is the
canonical per-call enforcement. Until that merges, this is the
soft enforcement: compute cost from tokens + model pricing
(reusing the existing MODEL_PRICING table from
app/agent/token_tracker.py) at the end of the run, write the
result to BuildJob.credit_spent, and flip outcome to
'budget_exhausted' if we overran. Pinned here so the spec's
acceptance criterion is met without depending on the unmerged
hook.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

import pytest
import pytest_asyncio


# ──────────────────────────────────────────────────────────────────────
# _compute_run_cost — pure-function pin
# ──────────────────────────────────────────────────────────────────────


def test_compute_run_cost_known_model():
    """MODEL_PRICING for claude-3-5-haiku is $0.80/1M input,
    $4.00/1M output. 10k in + 5k out → 0.008 + 0.020 = $0.028.
    The multiplier defaults to 1.0 so the unscaled total is the
    output."""
    from app.agent.subagent_orchestrator import _compute_run_cost

    cost = _compute_run_cost(
        model="claude-3-5-haiku-20241022",
        tokens_in=10_000,
        tokens_out=5_000,
    )
    assert cost == 0.028


def test_compute_run_cost_unknown_model_returns_zero():
    """An unrecognised model name MUST return 0 — otherwise it
    silently trips budget exhaustion on every run. Operators
    add new models to MODEL_PRICING when they ship them."""
    from app.agent.subagent_orchestrator import _compute_run_cost

    assert _compute_run_cost(
        model="some-future-model-not-in-table",
        tokens_in=1_000_000, tokens_out=1_000_000,
    ) == 0.0


def test_compute_run_cost_zero_tokens_returns_zero():
    from app.agent.subagent_orchestrator import _compute_run_cost

    assert _compute_run_cost(
        model="claude-3-5-haiku-20241022", tokens_in=0, tokens_out=0,
    ) == 0.0


def test_compute_run_cost_applies_credit_multiplier(monkeypatch):
    """The subagent_credit_multiplier setting scales the result.
    Operators dial sub-agent spend up/down without redeploying."""
    from app.agent.subagent_orchestrator import _compute_run_cost
    from app.config import settings

    monkeypatch.setattr(settings, "subagent_credit_multiplier", 2.0)
    cost_2x = _compute_run_cost(
        model="claude-3-5-haiku-20241022",
        tokens_in=10_000, tokens_out=5_000,
    )
    assert cost_2x == 0.056  # 0.028 * 2.0

    monkeypatch.setattr(settings, "subagent_credit_multiplier", 0.5)
    cost_half = _compute_run_cost(
        model="claude-3-5-haiku-20241022",
        tokens_in=10_000, tokens_out=5_000,
    )
    assert cost_half == 0.014


# ──────────────────────────────────────────────────────────────────────
# End-to-end: budget exhaustion via the orchestrator
# ──────────────────────────────────────────────────────────────────────


@dataclass
class FakeAgentResponse:
    """Field names MUST match the real AgentResponse — see the guard below.

    They did not. This mock declared `input_tokens` / `output_tokens` /
    `model_used`, while `AgentResponse` declares `tokens_input` /
    `tokens_output` / `model`. `_run_child` reads the real names via getattr
    with a None default, so every read returned None, every cost computed to
    0.0, and the budget could never be exceeded. The tests below passed by
    measuring nothing.
    """
    text: str
    session_id: str = "subagent:fake"
    tokens_input: int = 100_000  # default: expensive run
    tokens_output: int = 50_000
    model: str = "claude-3-5-haiku-20241022"


def test_fake_response_field_names_match_the_real_one():
    """A mock that names fields the reader does not read tests nothing.

    `_run_child` pulls token counts off the response with
    `getattr(response, "tokens_input", None)`. A mock with a different
    spelling returns None for all of them, the cost is 0.0, and every budget
    assertion below silently passes no matter what the code does. So pin the
    three names the orchestrator actually reads against the real dataclass.
    """
    import dataclasses

    from app.agent.agent_runner import AgentResponse

    real = {f.name for f in dataclasses.fields(AgentResponse)}
    fake = {f.name for f in dataclasses.fields(FakeAgentResponse)}
    read_by_orchestrator = {"tokens_input", "tokens_output", "model"}

    missing_from_real = read_by_orchestrator - real
    assert not missing_from_real, (
        "subagent_orchestrator._run_child reads these off AgentResponse and "
        f"the dataclass no longer declares them: {sorted(missing_from_real)}"
    )
    missing_from_fake = read_by_orchestrator - fake
    assert not missing_from_fake, (
        "FakeAgentResponse does not declare the fields the orchestrator reads, "
        "so every cost in this file computes to 0.0 and the budget assertions "
        f"measure nothing: {sorted(missing_from_fake)}"
    )


class FakeAgentRunner:
    def __init__(self, *, response: FakeAgentResponse = None):
        self.calls: list[dict[str, Any]] = []
        self.response = response or FakeAgentResponse(text="result")

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


@pytest_asyncio.fixture
async def db():
    from app.db.database import async_session_maker
    async with async_session_maker() as s:
        yield s


@pytest.fixture
def enable_spawning(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "subagent_spawning_enabled", True)
    monkeypatch.setattr(settings, "subagent_credit_multiplier", 1.0)
    return settings


@pytest.fixture(autouse=True)
def reset_lane_manager():
    import app.agent.lanes as lanes
    lanes._lane_manager = None
    yield
    lanes._lane_manager = None


@pytest.fixture(autouse=True)
def pin_credit_multiplier_to_unity(monkeypatch):
    """Pin ``settings.subagent_credit_multiplier`` to 1.0 for every
    test in this file. Defends against the test_subagent_settings.py
    leak: that file calls ``importlib.reload(app.config)`` with
    ``SUBAGENT_CREDIT_MULTIPLIER=1.5`` in the env to verify env
    override works; pydantic-settings burns the value into the
    reloaded module's ``settings`` instance, and monkeypatch's env
    teardown doesn't undo a module reload. If pytest collects that
    file before this one, every cost computation here would be
    multiplied by 1.5 and the assertions would all fail.
    Pinning explicitly per test makes this file order-independent."""
    from app.config import settings
    monkeypatch.setattr(settings, "subagent_credit_multiplier", 1.0)
    yield


@pytest.fixture
def patch_writer(monkeypatch):
    calls: dict[str, Any] = {"write": [], "broadcast": []}

    async def _fake_write(db, **kwargs):
        calls["write"].append(kwargs)
        return ("msg-" + str(uuid.uuid4())[:8], "dc-fake")

    async def _fake_broadcast(user_id, **kwargs):
        calls["broadcast"].append({"user_id": user_id, **kwargs})
        return {"ws_count": 1, "channel_results": {}}

    import app.agent.subagent_message_writer as mw
    monkeypatch.setattr(mw, "write_subagent_message", _fake_write)
    monkeypatch.setattr(mw, "broadcast_subagent_message", _fake_broadcast)
    return calls


async def _seed_user(db, user_id: str):
    from app.db.models import User
    from sqlalchemy import select
    existing = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if existing is None:
        db.add(User(
            id=user_id,
            email=f"{user_id[:8]}@test.local",
            hashed_password="x" * 60,
            name="Test",
        ))
        await db.commit()


# Wall-clock ceilings in this file are SCAFFOLDING, not assertions. Every
# test here asserts WHAT happens — outcome, credit_spent,
# credit_budget_allocated, the announce-back row — and none asserts HOW
# FAST. They were 5s (poll) and 10s (the job's own timeout), which is a
# measurement of the host: the self-hosted runners share the box with the
# 58-container agent fleet, and at load average 9-14 this file failed on
# THREE different tests across THREE unrelated PRs (#601, #604, #609),
# each time on a PR whose diff cannot reach it. Raising the ceilings
# removes the host dependency and keeps every behavioural assertion
# exactly as it was; a genuinely hung job still fails, just not a merely
# slow one.
_JOB_TIMEOUT_S = 120
_POLL_CEILING_S = 60.0


async def _wait_for_completion(job_id: str, *, timeout: float = _POLL_CEILING_S):
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
    from sqlalchemy import select

    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        async with async_session_maker() as db:
            row = (await db.execute(select(BuildJob).where(BuildJob.id == job_id))).scalar_one()
            if row.status in ("completed", "failed", "timeout", "cancelled", "budget_exhausted"):
                return row
        await asyncio.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not reach terminal status within {timeout}s")


@pytest.mark.asyncio
async def test_run_under_budget_completes_normally(enable_spawning, patch_writer):
    """A run that costs $0.028 with a $1.00 budget completes
    normally — outcome='success', credit_spent stamped, no
    budget_exhausted flag."""
    from app.agent.subagent_orchestrator import spawn_subagent
    from app.db.database import async_session_maker

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed_user(db, uid)

    # Run costs $0.028; budget is $1.00 — comfortably under.
    runner = FakeAgentRunner(response=FakeAgentResponse(
        text="ok", tokens_input=10_000, tokens_output=5_000,
        model="claude-3-5-haiku-20241022",
    ))
    result = await spawn_subagent(
        user_id=uid, task="cheap task", label="L", model=None,
        timeout_seconds=_JOB_TIMEOUT_S, parent_job_id=None,
        channel="web", telegram_chat_id=None,
        agent_runner=runner, credit_budget=1.00,
    )
    row = await _wait_for_completion(result["job_id"])
    assert row.status == "completed"
    assert row.outcome == "success"
    assert row.credit_spent == 0.028
    assert row.credit_budget_allocated == 1.00


@pytest.mark.asyncio
async def test_run_over_budget_terminates_with_budget_exhausted(
    enable_spawning, patch_writer,
):
    """The acceptance criterion. Run costs $0.028; budget is
    $0.01 — over → outcome='budget_exhausted', status carries
    through to BuildJob, partial announce row written."""
    from app.agent.subagent_orchestrator import spawn_subagent
    from app.db.database import async_session_maker

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed_user(db, uid)

    runner = FakeAgentRunner(response=FakeAgentResponse(
        text="partial result before budget hit",
        tokens_input=10_000, tokens_output=5_000,
        model="claude-3-5-haiku-20241022",
    ))
    result = await spawn_subagent(
        user_id=uid, task="costly task", label="L", model=None,
        timeout_seconds=_JOB_TIMEOUT_S, parent_job_id=None,
        channel="web", telegram_chat_id=None,
        agent_runner=runner, credit_budget=0.01,
    )
    row = await _wait_for_completion(result["job_id"])
    assert row.status == "budget_exhausted"
    assert row.outcome == "budget_exhausted"
    assert row.credit_spent == 0.028
    assert row.credit_budget_allocated == 0.01
    assert "credit budget" in (row.error_message or "").lower()

    # An announce row was still posted — user wants to see the
    # partial work, not silent loss.
    assert patch_writer["write"], "announce-back fires on budget_exhausted"
    w = patch_writer["write"][0]
    assert w["outcome"] == "budget_exhausted"


@pytest.mark.asyncio
async def test_no_budget_set_skips_enforcement(enable_spawning, patch_writer):
    """credit_budget=None means no cap. The run completes
    regardless of how expensive."""
    from app.agent.subagent_orchestrator import spawn_subagent
    from app.db.database import async_session_maker

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed_user(db, uid)

    runner = FakeAgentRunner(response=FakeAgentResponse(
        text="ok", tokens_input=1_000_000, tokens_output=500_000,
        model="claude-3-5-haiku-20241022",
    ))
    result = await spawn_subagent(
        user_id=uid, task="big task", label="L", model=None,
        timeout_seconds=_JOB_TIMEOUT_S, parent_job_id=None,
        channel="web", telegram_chat_id=None,
        agent_runner=runner,  # credit_budget defaults to None
    )
    row = await _wait_for_completion(result["job_id"])
    assert row.status == "completed"
    assert row.outcome == "success"
    # Spend is still computed and stamped (operator visibility).
    assert row.credit_spent == 2.8  # 1M*0.80 + 0.5M*4 = $2.80
    assert row.credit_budget_allocated is None


@pytest.mark.asyncio
async def test_unknown_model_does_not_trip_budget(enable_spawning, patch_writer):
    """An unrecognised model returns cost=0 from MODEL_PRICING.
    A budget run with an unknown model must complete normally,
    not falsely trigger budget_exhausted."""
    from app.agent.subagent_orchestrator import spawn_subagent
    from app.db.database import async_session_maker

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed_user(db, uid)

    runner = FakeAgentRunner(response=FakeAgentResponse(
        text="ok", tokens_input=1_000_000, tokens_output=1_000_000,
        model="some-future-claude-not-in-pricing-table",
    ))
    result = await spawn_subagent(
        user_id=uid, task="t", label="L", model=None,
        timeout_seconds=_JOB_TIMEOUT_S, parent_job_id=None,
        channel="web", telegram_chat_id=None,
        agent_runner=runner, credit_budget=0.01,
    )
    row = await _wait_for_completion(result["job_id"])
    assert row.status == "completed"  # no budget trip
    assert row.outcome == "success"
    assert row.credit_spent == 0.0


@pytest.mark.asyncio
async def test_credit_budget_propagates_to_config_json(enable_spawning, patch_writer):
    """The BuildJob row carries the budget in config_json so the
    orphan sweep / introspection tools can read it without
    needing a second column join. credit_budget_allocated is the
    column for queries; config_json is the structured copy."""
    from app.agent.subagent_orchestrator import spawn_subagent
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
    from sqlalchemy import select

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed_user(db, uid)

    runner = FakeAgentRunner(response=FakeAgentResponse(text="ok"))
    result = await spawn_subagent(
        user_id=uid, task="t", label="L", model=None,
        timeout_seconds=_JOB_TIMEOUT_S, parent_job_id=None,
        channel="web", telegram_chat_id=None,
        agent_runner=runner, credit_budget=0.50,
    )
    # Inspect immediately — credit_budget_allocated is set BEFORE
    # the child runs.
    async with async_session_maker() as db:
        job = (await db.execute(
            select(BuildJob).where(BuildJob.id == result["job_id"])
        )).scalar_one()
        assert job.credit_budget_allocated == 0.50
        assert job.config_json is not None
        assert job.config_json["credit_budget"] == 0.50

    # Let the child finish so the task doesn't leak.
    await _wait_for_completion(result["job_id"])
