"""2026-05-16 production fix — Gmail trigger never firing.

Symptom the user reported: the "Summarize every new Gmail" trigger
showed `last_status='active'` (green Active pill) and 5 fire events,
but every event was a synthetic `test:*` row and every event was
Failed. Zero real Gmail messages had ever reached the trigger.

This module pins the four fixes that make the trigger's state honest
again:

  1. Synthetic test fires bypass the LLM / MCP / broadcast chain.
     The Test button proves wiring works — it must not fail because
     OpenAI is rate-limited or the MCP client hasn't been wired yet.

  2. `if not kept` no longer returns batch `status="success"`. When
     every event was filter-dropped or fetch-failed, the handler
     reports `success_empty` so the runner does NOT promote
     `last_status` to "active".

  3. The runner promotes `last_status="active"` ONLY on real event
     delivery. Test fires update `provider_state_json.last_test_at`,
     empty batches update `last_empty_batch_at`, real fires update
     `last_real_fired_at` AND `last_status`.

  4. The derived `health` field on the trigger response is the
     single source of truth for the UI pill. It's computed from
     `(last_status, provider_state_json, watch_provisioned)` so a
     stale `last_status` can't outvote the live signal.

There is also a `POST /api/triggers/{id}/reverify` endpoint that
re-runs the platform watch RPC for the trigger and returns the
refreshed row — recovery without delete+recreate.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


os.environ.setdefault("AGENT_API_KEY", "test-key-trigger-health")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000000c0")
os.environ.setdefault("TRIGGERS_EMAIL_ENABLED", "true")

CONTAINER_USER_ID = "00000000-0000-0000-0000-0000000000c0"

BACKEND = Path(__file__).resolve().parent.parent


# ── Source-grep guards — bug-recurrence pins ──────────────────────────


_HANDLER_SRC = (BACKEND / "app/agent/triggers/email_received_handler.py").read_text()
_RUNNER_SRC = (BACKEND / "app/agent/triggers/runner.py").read_text()
_API_SRC = (BACKEND / "app/api/triggers.py").read_text()


def test_handler_short_circuits_synthetic_batches_before_mcp():
    """The `_do_test_fire` shortcut must run BEFORE the
    `if mcp is None` early-return. Otherwise a freshly-booted agent
    where the MCP client hasn't been late-bound yet would still
    return `failed` on every Test click — the symptom that reproduced
    the "5 Failed events" state."""
    src = _HANDLER_SRC
    short_circuit_idx = src.find("self._do_test_fire(trigger, events, db)")
    mcp_guard_idx = src.find("if mcp is None:")
    assert short_circuit_idx != -1, (
        "Synthetic batch short-circuit missing from execute(). Test "
        "fires would run through the full MCP/LLM/broadcast path."
    )
    assert short_circuit_idx < mcp_guard_idx, (
        "Test-fire short-circuit must precede the MCP guard. "
        "Otherwise a missing MCP client kills the Test button."
    )


def test_empty_batch_does_not_report_success():
    """`if not kept` previously returned `status="success"`, which the
    runner read as "trigger is working" and stamped last_status=active.
    The fix reports `success_empty` — same DB outcome (no message
    written) but the runner knows not to false-promote."""
    src = _HANDLER_SRC
    assert 'status="success_empty"' in src, (
        "The `if not kept` branch must return success_empty, not "
        "success. Returning success bumps last_status to 'active' "
        "even though zero events delivered — the 2026-05-16 bug."
    )


def test_runner_only_promotes_active_on_real_success():
    """The runner's last_status update must distinguish real success
    from test_success and success_empty. The five-line `next_last_status`
    state machine is the only place this decision lives."""
    src = _RUNNER_SRC
    assert 'promote_active = res_status == "success"' in src, (
        "Runner must guard last_status='active' behind a real success "
        "check. Without it, test fires and empty batches mislead the UI."
    )
    assert 'new_state_patch["last_real_fired_at"]' in src, (
        "Runner must record last_real_fired_at on real success — the "
        "frontend `health=delivering` state machine reads this field."
    )
    assert 'new_state_patch["last_test_at"]' in src, (
        "Test fires must stamp last_test_at so the UI can render "
        "`health=test_passed` instead of `awaiting_event`."
    )


def test_api_exposes_health_field():
    src = _API_SRC
    assert "_derive_health" in src, (
        "The derived `health` state machine is the UI's source of "
        "truth. If `_derive_health` is gone, the pill falls back to "
        "stale `last_status` and the 'Active + Failed' mirage returns."
    )
    assert "/{trigger_id}/reverify" in src, (
        "The reverify endpoint must remain mounted — without it the "
        "user has no in-product recovery for a stuck trigger."
    )


# ── Behavior tests: synthetic test fire bypasses LLM/MCP ──────────────


@pytest.mark.asyncio
async def test_synthetic_test_fire_succeeds_without_mcp():
    """Clicking Test on a freshly-booted agent (MCP client not yet
    late-bound) must succeed — that was the original bug."""
    from app.agent.triggers.email_received_handler import EmailReceivedHandler

    writer = AsyncMock(return_value=("msg-test-1", "day-2026-05-16"))
    broadcaster = AsyncMock(return_value=1)

    handler = EmailReceivedHandler(
        mcp_client=None,             # ← critical: MCP not wired
        writer=writer,
        broadcaster=broadcaster,
    )
    trigger = SimpleNamespace(
        id="trig-1",
        user_id=CONTAINER_USER_ID,
        kind="email_received",
        action="summarize_and_post",
        name="Summarize every new Gmail",
        filter_json=None,
        config_json={"delivery_channels": ["website", "telegram"]},
        provider_state_json={},
    )
    events = [
        SimpleNamespace(id="ev-1", event_dedupe_id=f"test:{uuid.uuid4().hex}"),
    ]

    result = await handler.execute(trigger, events, db=MagicMock())

    # The handler reports its own non-DB status that the runner uses
    # to decide whether to promote last_status to 'active'.
    assert result.status == "test_success"
    # The per-event status that lands in the DB is success — the event
    # row genuinely was processed end-to-end against synthetic data.
    assert result.per_event_status == {"ev-1": "success"}
    # And critically: a real message was written and broadcast.
    assert writer.called, "Test fire must write to Day-as-Chat"
    assert broadcaster.called, "Test fire must broadcast to ws + extras"


@pytest.mark.asyncio
async def test_synthetic_test_fire_survives_broadcast_failure():
    """If Telegram fan-out fails, the Test button must still pass —
    the wiring check's promise is "your trigger can deliver to
    Day-as-Chat," not "every fan-out channel is healthy too."."""
    from app.agent.triggers.email_received_handler import EmailReceivedHandler

    writer = AsyncMock(return_value=("msg-test-2", "day-x"))
    broadcaster = AsyncMock(side_effect=RuntimeError("telegram 503"))
    handler = EmailReceivedHandler(
        mcp_client=None, writer=writer, broadcaster=broadcaster,
    )
    trigger = SimpleNamespace(
        id="trig-x", user_id=CONTAINER_USER_ID, kind="email_received",
        action="summarize_and_post", name=None,
        filter_json=None,
        config_json={"delivery_channels": ["website", "telegram"]},
        provider_state_json={},
    )
    events = [SimpleNamespace(id="ev-x", event_dedupe_id="test:abcd")]

    result = await handler.execute(trigger, events, db=MagicMock())
    assert result.status == "test_success"


@pytest.mark.asyncio
async def test_synthetic_test_fire_fails_loud_if_writer_breaks():
    """A DB write failure during the synthetic path must surface as
    `failed`. Silent recovery would make the UI lie about success
    while no message landed in chat."""
    from app.agent.triggers.email_received_handler import EmailReceivedHandler

    writer = AsyncMock(side_effect=RuntimeError("postgres connection lost"))
    handler = EmailReceivedHandler(mcp_client=None, writer=writer)
    trigger = SimpleNamespace(
        id="t", user_id=CONTAINER_USER_ID, kind="email_received",
        action="summarize_and_post", name=None,
        filter_json=None, config_json={}, provider_state_json={},
    )
    events = [SimpleNamespace(id="ev-q", event_dedupe_id="test:q")]
    result = await handler.execute(trigger, events, db=MagicMock())
    assert result.status == "failed"
    assert result.per_event_status == {"ev-q": "failed"}
    assert "postgres" in (result.error_detail or "").lower()


# ── Behavior tests: _derive_health state machine ──────────────────────


def _make_trigger_row(**overrides):
    """Build a minimal Trigger-shaped object for _derive_health tests
    without spinning up the SQLAlchemy session."""
    defaults = dict(
        last_status="never_fired",
        last_error=None,
        provider_state_json={},
        last_fired_at=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_health_delivering_when_real_fire_present():
    from app.api.triggers import _derive_health

    trig = _make_trigger_row(
        last_status="active",
        provider_state_json={
            "gmail_history_id": "12345",
            "last_real_fired_at": "2026-05-16T10:00:00Z",
        },
        last_fired_at=datetime(2026, 5, 16, 10, 0, 0),
    )
    assert _derive_health(trig, watch_provisioned=True) == "delivering"


def test_health_test_passed_when_only_synthetic_fires():
    """The exact bug the user hit: test fires bumped last_status to
    'active' even though no real event ever delivered. health must
    surface the truth — `test_passed` not `delivering`."""
    from app.api.triggers import _derive_health

    trig = _make_trigger_row(
        last_status="never_fired",  # runner now keeps it untouched on test fires
        provider_state_json={
            "gmail_history_id": "12345",
            "last_test_at": "2026-05-16T14:46:00Z",
            # NB: no last_real_fired_at — that's the whole point
        },
    )
    assert _derive_health(trig, watch_provisioned=True) == "test_passed"


def test_health_awaiting_event_on_armed_but_silent_watch():
    from app.api.triggers import _derive_health

    trig = _make_trigger_row(
        last_status="never_fired",
        provider_state_json={"gmail_history_id": "12345"},
    )
    assert _derive_health(trig, watch_provisioned=True) == "awaiting_event"


def test_health_needs_reauth_when_connector_revoked():
    from app.api.triggers import _derive_health

    trig = _make_trigger_row(last_status="skipped_reauth")
    assert _derive_health(trig, watch_provisioned=True) == "needs_reauth"


def test_health_setup_error_when_provisioning_failed():
    from app.api.triggers import _derive_health

    trig = _make_trigger_row(last_status="provisioning_failed")
    assert _derive_health(trig, watch_provisioned=False) == "setup_error"


def test_health_watch_expired_when_expiration_is_past():
    """Gmail watches expire ~7 days after start_watch. If the daily
    refresh missed (or hasn't run yet on a freshly-deployed agent),
    surface this distinctly from setup_error so the user knows the
    Verify button will fix it."""
    from app.api.triggers import _derive_health

    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    trig = _make_trigger_row(
        last_status="active",
        provider_state_json={
            "gmail_history_id": "x",
            "watch_expires_at": past,
            # No last_real_fired_at: we treat watch_expired as more
            # actionable than "delivering" — the expiration check is
            # explicit recovery guidance for the user.
        },
    )
    assert _derive_health(trig, watch_provisioned=True) == "watch_expired"


def test_health_last_fire_failed_when_real_success_then_crash():
    """When a trigger HAS delivered real events but the most recent
    fire crashed, surface `last_fire_failed` so the user notices
    instead of seeing a permanently green 'Delivering' pill."""
    from app.api.triggers import _derive_health

    earlier = datetime(2026, 5, 14, 10, 0, 0, tzinfo=timezone.utc)
    later = datetime(2026, 5, 16, 10, 0, 0)
    trig = _make_trigger_row(
        last_status="failed",
        provider_state_json={
            "gmail_history_id": "x",
            "last_real_fired_at": earlier.isoformat().replace("+00:00", "Z"),
        },
        last_fired_at=later,
    )
    assert _derive_health(trig, watch_provisioned=True) == "last_fire_failed"


# ── /api/triggers/{id}/reverify endpoint ──────────────────────────────


def _build_app() -> FastAPI:
    from app.api.triggers import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return app


@pytest_asyncio.fixture(autouse=True)
def _mock_auto_arm(monkeypatch):
    """Replace the platform→watch RPC with an in-process stub. We can
    inject different stubs per test by monkeypatching this fixture's
    target."""
    async def _noop(_trigger_id: str) -> None:
        return None
    from app.api import triggers as _t
    monkeypatch.setattr(_t, "_provision_email_watch", _noop)


@pytest_asyncio.fixture
async def _seed_container_user():
    """The triggers table has an FK to users; the autouse `_reset_database`
    fixture drops+recreates schema but does not seed users. Insert the
    container's owner so trigger creation doesn't 23503 on the FK."""
    from app.db.database import async_session_maker
    from app.db.models import User

    async with async_session_maker() as db:
        existing = await db.get(User, CONTAINER_USER_ID)
        if existing is None:
            db.add(User(
                id=CONTAINER_USER_ID,
                email=f"trigger-health-{CONTAINER_USER_ID[:8]}@example.com",
                hashed_password="x",
            ))
            await db.commit()
    yield


@pytest_asyncio.fixture
async def client(_seed_container_user) -> AsyncClient:
    app = _build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_reverify_404_when_trigger_missing(client):
    r = await client.post(f"/api/triggers/{uuid.uuid4()}/reverify")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_reverify_returns_refreshed_row_with_health(client, monkeypatch):
    """Happy path: reverify calls the platform RPC, the stub stamps
    `gmail_history_id` into provider_state_json, and the response
    carries a `health` field derived from the fresh state."""
    # Create the trigger first.
    create_r = await client.post("/api/triggers", json={
        "kind": "email_received",
        "action": "summarize_and_post",
        "name": "Watch inbox",
    })
    assert create_r.status_code == 201, create_r.text
    tid = create_r.json()["id"]

    # Swap in a real provision stub that stamps state.
    async def _stamp(trigger_id: str) -> None:
        from app.api.triggers import _stamp_trigger_state
        await _stamp_trigger_state(
            trigger_id,
            last_status="never_fired",
            provider_state_patch={
                "gmail_history_id": "999",
                "watch_expires_at": (
                    datetime.now(timezone.utc) + timedelta(days=7)
                ).isoformat(),
            },
        )
    from app.api import triggers as _t
    monkeypatch.setattr(_t, "_provision_email_watch", _stamp)

    r = await client.post(f"/api/triggers/{tid}/reverify")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["watch_provisioned"] is True
    # No real or test fires yet → awaiting_event.
    assert body["health"] == "awaiting_event"
    assert body["last_real_fired_at"] is None


@pytest.mark.asyncio
async def test_reverify_rejects_non_email_kinds(client):
    """The endpoint dispatches by kind — until other kinds have their
    own provisioning RPC, fail-loud rather than silently no-op."""
    # Manually insert a row with a fictional kind (bypass validation).
    from app.db.database import async_session_maker
    from app.db.models import Trigger

    tid = str(uuid.uuid4())
    async with async_session_maker() as db:
        row = Trigger(
            id=tid,
            user_id=CONTAINER_USER_ID,
            kind="calendar_event_made_up",
            action="summarize_and_post",
            enabled=True,
            last_status="never_fired",
        )
        db.add(row)
        await db.commit()

    r = await client.post(f"/api/triggers/{tid}/reverify")
    assert r.status_code == 400
    assert "calendar_event_made_up" in r.text
