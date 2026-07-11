"""Autopilot approvals + unsupervised-action policy (Autopilot PR7).

Contracts:
- the AUTOPILOT prompt profile is a strict subset of FULL sections and
  denies the outward-mutation tool set;
- 'autopilot' is a connector default-deny channel and vault-blocked;
- a blocked tick writes a durable approval row + pauses the mission;
- a decision resumes the mission (active, due now, answer injected
  one-shot into the next tick prompt) and is idempotent on replay.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.db.models import (
    AutopilotApproval, APPROVAL_PENDING, APPROVAL_ANSWERED, Routine,
)


# ── Policy pins ───────────────────────────────────────────────────


def test_autopilot_profile_sections_strict_subset_of_full():
    from app.agent.prompt_profile import (
        PromptProfile, _AUTOPILOT_SECTIONS, _FULL_SECTIONS, sections_for,
    )

    assert set(_AUTOPILOT_SECTIONS) < set(_FULL_SECTIONS)
    assert sections_for(PromptProfile.AUTOPILOT) == _AUTOPILOT_SECTIONS
    # Foreground-only surfaces stay out of autonomous turns.
    for absent in ("media", "onboarding", "vibecoding", "activation"):
        assert absent not in _AUTOPILOT_SECTIONS


def test_autopilot_disabled_tools_policy():
    from app.agent.prompt_profile import (
        AUTOPILOT_DISABLED_TOOLS, PromptProfile, disabled_tools_for,
    )

    assert disabled_tools_for(PromptProfile.AUTOPILOT) == AUTOPILOT_DISABLED_TOOLS
    # Outward/automation mutators denied…
    for denied in (
        "memory_delete", "routines__delete", "triggers__create",
        "save_streaming_credential", "create_job",
    ):
        assert denied in AUTOPILOT_DISABLED_TOOLS
    # …but the tools missions need to make progress are NOT denied.
    for allowed in ("exec", "web_search", "web_fetch", "memory_store", "spawn"):
        assert allowed not in AUTOPILOT_DISABLED_TOOLS


def test_autopilot_channel_policy_pins():
    from app.services.connector_dispatcher import _MUTATES_DEFAULT_DENY_CHANNELS
    from app.agent.agent_runner import VAULT_TOOL_CHANNEL_BLOCK

    assert "autopilot" in _MUTATES_DEFAULT_DENY_CHANNELS
    assert "autopilot" in VAULT_TOOL_CHANNEL_BLOCK


def test_approvals_table_is_agent_only():
    from app.db.models.base import AGENT_ONLY_TABLES, PLATFORM_ONLY_TABLES

    assert "autopilot_approvals" in AGENT_ONLY_TABLES
    assert "autopilot_approvals" not in PLATFORM_ONLY_TABLES


# ── Blocked tick → approval row ───────────────────────────────────


@pytest.fixture(autouse=True)
def _quiet_infra(monkeypatch):
    import app.services.agent_notify_client as anc
    import app.services.credit_reporter as cr

    notified = []

    async def fake_notify(**kwargs):
        notified.append(kwargs)
        return "nid"

    async def ok_preflight(**kwargs):
        return SimpleNamespace(network_ok=True)

    monkeypatch.setattr(anc, "notify", fake_notify)
    monkeypatch.setattr(cr, "raise_if_exhausted", lambda: None)
    monkeypatch.setattr(cr, "check_balance_remote", ok_preflight)
    yield notified


async def _mk_mission(state=None) -> Routine:
    from app.db import async_session_maker
    from app.db.models import User
    from app.services.auth_service import get_password_hash

    user_id = str(uuid.uuid4())
    routine_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"apr-{uuid.uuid4().hex[:8]}@example.com",
            hashed_password=get_password_hash("x" * 12), name="APR",
        ))
        db.add(Routine(
            id=routine_id, user_id=user_id, kind="autopilot", enabled=True,
            name="Approval mission", schedule_cron_local="*/5 * * * *",
            schedule_kind="every", schedule_interval_seconds=300,
            config_json={"goal": "research and draft", "budget_credits": 50},
            last_state_json=state or {},
        ))
        await db.commit()
        routine = await db.get(Routine, routine_id)
        db.expunge(routine)
    return routine


class _BlockedRunner:
    def __init__(self):
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text=(
                "AUTOPILOT_STATUS: blocked\n"
                "AUTOPILOT_SUMMARY: which of the two flights should I book?"
            ),
            tokens_input=10, tokens_output=5, model="gpt-4o-mini",
            asst_message_id="am-1",
        )


@pytest.mark.asyncio
async def test_blocked_tick_creates_approval_and_policy_kwargs(_quiet_infra):
    from app.agent.routines.autopilot_handler import AutopilotHandler
    from app.agent.prompt_profile import PromptProfile
    from app.db import async_session_maker
    from sqlalchemy import select

    routine = await _mk_mission()
    fake = _BlockedRunner()
    handler = AutopilotHandler(fake)
    async with async_session_maker() as db:
        result = await handler.execute(
            routine, SimpleNamespace(id="r1", job_id="j1"), db,
        )

    # The turn ran under the autopilot policy surface.
    call = fake.calls[0]
    assert call["channel"] == "autopilot"
    assert call["prompt_profile"] == PromptProfile.AUTOPILOT

    # Durable approval row exists and is linked in state + notification.
    async with async_session_maker() as db:
        approval = (await db.execute(
            select(AutopilotApproval).where(
                AutopilotApproval.mission_id == routine.id,
            )
        )).scalar_one()
    assert approval.status == APPROVAL_PENDING
    assert "flights" in approval.title
    assert result.new_watermark["pending_approval_id"] == approval.id
    notified = _quiet_infra
    assert notified[0]["event_kind"] == "needs_input"
    assert notified[0]["data"]["approval_id"] == approval.id


# ── Decision endpoint ─────────────────────────────────────────────


def _api():
    from fastapi import FastAPI
    from app.api.autopilot import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return app


@pytest.mark.asyncio
async def test_decision_resumes_mission_and_is_idempotent(monkeypatch, _quiet_infra):
    from app.config import settings
    from app.db import async_session_maker

    routine = await _mk_mission(state={
        "status": "waiting_input",
        "status_reason": "agent_blocked",
        "next_due_at": None,
        "no_progress_streak": 2,
    })
    monkeypatch.setattr(settings, "user_id", routine.user_id)

    approval_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(AutopilotApproval(
            id=approval_id, mission_id=routine.id, user_id=routine.user_id,
            kind="question", title="which flight?",
        ))
        await db.commit()

    transport = ASGITransport(app=_api())
    async with AsyncClient(transport=transport, base_url="http://agent") as ac:
        # Pending list shows it.
        res = await ac.get("/api/autopilot/approvals")
        assert res.status_code == 200 and res.json()[0]["id"] == approval_id

        res = await ac.post(
            f"/api/autopilot/approvals/{approval_id}/decision",
            json={"decision": "answer", "answer_text": "the 9am one",
                  "decided_via": "push_action"},
        )
        assert res.status_code == 200, res.text
        assert res.json()["status"] == APPROVAL_ANSWERED

        # Idempotent replay (push buttons can double-fire).
        res2 = await ac.post(
            f"/api/autopilot/approvals/{approval_id}/decision",
            json={"decision": "deny"},
        )
        assert res2.status_code == 200
        assert res2.json()["status"] == APPROVAL_ANSWERED  # first decision kept

    async with async_session_maker() as db:
        resumed = await db.get(Routine, routine.id)
        state = resumed.last_state_json
    assert state["status"] == "active"
    assert state["next_due_at"] is None
    assert state["no_progress_streak"] == 0
    assert state["user_answers"][0]["answer"] == "the 9am one"
    assert resumed.enabled is True


@pytest.mark.asyncio
async def test_answers_inject_into_next_tick_prompt_one_shot(_quiet_infra):
    from app.agent.routines.autopilot_handler import AutopilotHandler
    from app.db import async_session_maker

    routine = await _mk_mission(state={
        "status": "active",
        "user_answers": [{"question": "which flight?", "answer": "the 9am one"}],
    })

    class _Working:
        def __init__(self):
            self.prompts = []

        async def run(self, **kwargs):
            self.prompts.append(kwargs["user_message"])
            return SimpleNamespace(
                text="AUTOPILOT_STATUS: working\nAUTOPILOT_SUMMARY: booked 9am",
                tokens_input=10, tokens_output=5, model="gpt-4o-mini",
                asst_message_id="am-2",
            )

    fake = _Working()
    handler = AutopilotHandler(fake)
    async with async_session_maker() as db:
        result = await handler.execute(
            routine, SimpleNamespace(id="r2", job_id="j2"), db,
        )

    assert "the 9am one" in fake.prompts[0]
    # One-shot: consumed answers do not persist in the new watermark.
    assert "user_answers" not in result.new_watermark


@pytest.mark.asyncio
async def test_decision_validation(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "user_id", str(uuid.uuid4()))
    transport = ASGITransport(app=_api())
    async with AsyncClient(transport=transport, base_url="http://agent") as ac:
        res = await ac.post(
            f"/api/autopilot/approvals/{uuid.uuid4()}/decision",
            json={"decision": "answer"},  # missing answer_text
        )
        assert res.status_code == 422
        res = await ac.post(
            f"/api/autopilot/approvals/{uuid.uuid4()}/decision",
            json={"decision": "approve"},
        )
        assert res.status_code == 404
