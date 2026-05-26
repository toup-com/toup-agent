"""2026-05-16 production-hardening pass — five gap fixes.

Pinned bugs:

  Gap 1: `run_gmail_watch_refresh` updates ConnectorIdentity.metadata_json
         but never propagates `gmail_history_id` / `watch_expires_at`
         to `Trigger.provider_state_json`. After 7d, `_derive_health`
         reports `watch_expired` on every healthy trigger.

  Gap 2: `_derive_health` collapses every `last_status="provisioning_
         failed"` to `setup_error`. The specific `provision_error` codes
         (`needs_reauth`, `ops_blocked`, `scope_error`, `transient`,
         `feature_disabled`) are thrown away.

  Gap 3: No observability on the delivery path. Nobody can tell whether
         a missed event was lost at Pub/Sub, at the platform webhook,
         at the agent dispatch, or inside the handler.

  Gap 4: `_arm_gmail_watch_post_connect` failures are logged-and-
         swallowed. Trigger rows stay in `never_fired` with no signal.

  Gap 5: `triggers_email_enabled=true` with empty `gcp_project` /
         `pubsub_topic` / `pubsub_push_audience` ships a silently
         broken pipeline.

This module is the source-of-truth for each fix and the regression
test that would catch it coming back.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


os.environ.setdefault("AGENT_API_KEY", "test-key-prod-hardening")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000000d0")
os.environ.setdefault("TRIGGERS_EMAIL_ENABLED", "true")

CONTAINER_USER_ID = "00000000-0000-0000-0000-0000000000d0"

BACKEND = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────
# Source-grep guards — bug-recurrence pins.
# ──────────────────────────────────────────────────────────────────────

_SCHED_SRC = (BACKEND / "app/scripts/scheduled_tasks.py").read_text()
_TRIGGERS_SRC = (BACKEND / "app/api/triggers.py").read_text()
_WEBHOOK_SRC = (BACKEND / "app/api/webhooks_gmail_pubsub.py").read_text()
_HANDLER_SRC = (BACKEND / "app/agent/triggers/email_received_handler.py").read_text()
_PROVISION_SRC = (BACKEND / "app/api/triggers_provision.py").read_text()
_OAUTH_SRC = (BACKEND / "app/api/oauth.py").read_text()
_PLATFORM_MAIN_SRC = (BACKEND / "platform_main.py").read_text()


def test_grep_gap1_refresh_fans_out_to_triggers():
    """The fix MUST update trigger rows inside the refresh loop. Search
    is keyed on the explicit invariant string we left in place — if
    someone refactors the loop without preserving the fan-out, the test
    catches it. The original bug: refresh wrote only to ConnectorIdentity."""
    assert "trigger_rows_synced" in _SCHED_SRC, (
        "run_gmail_watch_refresh must fan out the refreshed watch state "
        "to every Trigger.provider_state_json — otherwise _derive_health "
        "reports watch_expired on healthy triggers after 7 days."
    )
    assert "last_watch_refresh_at" in _SCHED_SRC, (
        "Refresh must stamp last_watch_refresh_at on the trigger row so "
        "the probe / UI can show when the row was last reconciled with "
        "the connector."
    )


def test_grep_gap2_health_reads_provision_error():
    assert "setup_blocked_ops" in _TRIGGERS_SRC
    assert "setup_scope_error" in _TRIGGERS_SRC
    assert "setup_transient" in _TRIGGERS_SRC
    assert "setup_feature_disabled" in _TRIGGERS_SRC
    assert 'provision_error' in _TRIGGERS_SRC


def test_grep_gap3_webhook_records_push_received():
    """Push timestamp MUST be written before JWT verify so a JWT-
    rejected push still leaves a footprint (the prerequisite for the
    `push_rejected_at_platform` diagnosis).

    We scope the search to the route handler body — the file's import
    block references `verify_pubsub_jwt` at the top, which would
    otherwise make any push-record call look "later" than it is."""
    handler_start = _WEBHOOK_SRC.find("async def gmail_pubsub_webhook")
    assert handler_start != -1, "Lost the gmail_pubsub_webhook route handler"
    body = _WEBHOOK_SRC[handler_start:]
    push_idx = body.find("await _record_push_received")
    jwt_idx = body.find("await verify_pubsub_jwt")
    assert push_idx != -1, "Webhook lost the _record_push_received call"
    assert jwt_idx != -1, "Webhook lost the verify_pubsub_jwt call"
    assert push_idx < jwt_idx, (
        "_record_push_received must run BEFORE verify_pubsub_jwt — "
        "otherwise JWT-rejected pushes leave no audit trail and "
        "/probe can't surface push_rejected_at_platform diagnosis."
    )


def test_grep_gap3_handler_stamps_last_completed():
    assert "last_handler_completed_at" in _HANDLER_SRC
    assert "last_handler_status" in _HANDLER_SRC


def test_grep_gap3_probe_endpoint_present():
    assert '/{trigger_id}/probe' in _TRIGGERS_SRC
    assert '_compute_diagnosis' in _TRIGGERS_SRC
    assert '_probe_watch' in _PROVISION_SRC


def test_grep_gap4_oauth_notifies_agent_on_failure():
    assert "_notify_agent_provision_failure" in _OAUTH_SRC, (
        "OAuth post-connect arm failure must notify the agent so the "
        "trigger row's provision_error reflects what happened. Silent "
        "failures were the 2026-05-16 regression Gap 4 closed."
    )
    # And the agent must accept the notification.
    assert "_record_provision_failure" in _TRIGGERS_SRC


def test_grep_gap5_boot_validation_present():
    assert "_validate_triggers_email_env" in _PLATFORM_MAIN_SRC
    assert "TriggersEmailMisconfigured" in _PLATFORM_MAIN_SRC


# ──────────────────────────────────────────────────────────────────────
# Gap 1 — daily refresh fans out to trigger rows
# ──────────────────────────────────────────────────────────────────────


def _trigger_row(**overrides):
    """Build a Trigger-shaped namespace for _derive_health unit tests."""
    defaults = dict(
        last_status="never_fired",
        last_error=None,
        provider_state_json={},
        last_fired_at=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_health_watch_expired_BEFORE_refresh_fanout_fix():
    """Pin: a 7-day-old trigger with stale `watch_expires_at` reports
    `watch_expired` until the refresh fans out. This is the symptom
    Gap 1 fixes."""
    from app.api.triggers import _derive_health

    stale_expires = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    trig = _trigger_row(
        last_status="active",
        provider_state_json={
            "gmail_history_id": "old",
            "watch_expires_at": stale_expires,
            "last_real_fired_at": "2026-05-01T00:00:00Z",
        },
    )
    assert _derive_health(trig, watch_provisioned=True) == "watch_expired"


def test_health_delivering_AFTER_refresh_fanout_writes_fresh_expiry():
    """After the daily refresh propagates a fresh `watch_expires_at`,
    health flips back to `delivering`. This pins the Gap 1 fix:
    refresh fan-out is what makes `_derive_health` honest for
    long-running triggers."""
    from app.api.triggers import _derive_health

    fresh_expires = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    trig = _trigger_row(
        last_status="active",
        provider_state_json={
            "gmail_history_id": "fresh",
            "watch_expires_at": fresh_expires,
            "last_watch_refresh_at": datetime.now(timezone.utc).isoformat() + "Z",
            "last_real_fired_at": "2026-05-16T00:00:00Z",
        },
    )
    assert _derive_health(trig, watch_provisioned=True) == "delivering"


@pytest_asyncio.fixture
def _gcp_settings_set(monkeypatch):
    """The refresh job early-returns when GCP env is unset (the
    operator-protection invariant). Tests of the fan-out behaviour
    need GCP set for the loop to actually execute."""
    from app.config import settings as _s
    monkeypatch.setattr(_s, "gcp_project", "test-project", raising=False)
    monkeypatch.setattr(_s, "pubsub_topic", "test-topic", raising=False)
    monkeypatch.setattr(_s, "pubsub_push_audience",
                        "https://toup.test/api/v1/webhooks/gmail", raising=False)
    monkeypatch.setattr(_s, "triggers_email_enabled", True, raising=False)
    return _s


@pytest.mark.asyncio
async def test_run_gmail_watch_refresh_updates_trigger_rows(_gcp_settings_set):
    """End-to-end: simulate the daily refresh job and assert every
    email_received trigger for the user gets fresh `gmail_history_id`,
    `watch_expires_at`, and `last_watch_refresh_at` written to its
    `provider_state_json`. One transaction per user — verified
    indirectly by the fact that all updates land before the next
    user's loop iteration."""
    from app.db.database import async_session_maker
    from app.db.models import ConnectorIdentity, Trigger, User
    from app.scripts.scheduled_tasks import run_gmail_watch_refresh
    from app.services.gmail_pubsub import WatchHandle
    import json

    user_id = str(uuid.uuid4())
    stale_expires = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

    # Seed the User row first + flush so the FK is satisfied before the
    # connector / trigger inserts land in the same session.
    async with async_session_maker() as db:
        db.add(User(id=user_id, email=f"refresh-test-{user_id[:8]}@example.com",
                    hashed_password="x"))
        await db.commit()

    async with async_session_maker() as db:
        db.add(ConnectorIdentity(
            id=str(uuid.uuid4()), user_id=user_id, connector_id="gmail",
            provider_account_id=f"refresh-test-{user_id[:8]}@example.com",
            status="active",
            metadata_json=json.dumps({"gmail_history_id": "old"}),
        ))
        for i in range(3):
            db.add(Trigger(
                id=str(uuid.uuid4()), user_id=user_id, kind="email_received",
                action="summarize_and_post", name=f"T{i}",
                enabled=True, last_status="never_fired",
                provider_state_json={
                    "gmail_history_id": "old",
                    "watch_expires_at": stale_expires,
                },
            ))
        await db.commit()

    fresh_handle = WatchHandle(
        history_id="99999",
        expiration_unix_ms=int((datetime.now(timezone.utc) + timedelta(days=7)).timestamp() * 1000),
    )

    # Patch the Google API call. Everything else (DB writes, fan-out) is
    # exercised exactly as in production.
    # `refresh_watch` is imported INSIDE run_gmail_watch_refresh (local
    # import to break a circular dep), so we patch at the source module.
    with patch("app.services.gmail_pubsub.refresh_watch",
               new=AsyncMock(return_value=fresh_handle)):
        await run_gmail_watch_refresh()

    async with async_session_maker() as db:
        from sqlalchemy import select
        triggers = (await db.execute(
            select(Trigger).where(Trigger.user_id == user_id)
        )).scalars().all()

    assert len(triggers) == 3
    for trig in triggers:
        ps = trig.provider_state_json or {}
        assert ps.get("gmail_history_id") == "99999", (
            f"Trigger {trig.id} did not get the refreshed history_id; "
            "fan-out is broken — _derive_health will report watch_expired."
        )
        assert ps.get("watch_expires_at"), "Refresh did not write watch_expires_at"
        assert ps.get("last_watch_refresh_at"), (
            "Refresh did not stamp last_watch_refresh_at — probe / UI "
            "won't know when the row was reconciled."
        )


@pytest.mark.asyncio
async def test_refresh_clears_stale_provision_error_on_success(_gcp_settings_set):
    """If a trigger row carries a stale `provision_error` from a prior
    failed auto-arm, a successful refresh demonstrates the watch is
    now alive — the error MUST be cleared so the UI stops showing
    'Setup error'."""
    from app.db.database import async_session_maker
    from app.db.models import ConnectorIdentity, Trigger, User
    from app.scripts.scheduled_tasks import run_gmail_watch_refresh
    from app.services.gmail_pubsub import WatchHandle
    import json

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=user_id, email=f"clear-{user_id[:8]}@example.com",
                    hashed_password="x"))
        await db.commit()

    async with async_session_maker() as db:
        db.add(ConnectorIdentity(
            id=str(uuid.uuid4()), user_id=user_id, connector_id="gmail",
            provider_account_id=f"clear-{user_id[:8]}@example.com",
            status="active",
            metadata_json=json.dumps({}),
        ))
        db.add(Trigger(
            id=str(uuid.uuid4()), user_id=user_id, kind="email_received",
            action="summarize_and_post", name="stuck",
            enabled=True, last_status="provisioning_failed",
            last_error="prior arm failed",
            provider_state_json={
                "provision_error": "transient",
                "provision_detail": "google 503 from yesterday",
            },
        ))
        await db.commit()

    handle = WatchHandle(
        history_id="abc", expiration_unix_ms=int(
            (datetime.now(timezone.utc) + timedelta(days=7)).timestamp() * 1000
        ),
    )

    with patch("app.services.gmail_pubsub.refresh_watch",
               new=AsyncMock(return_value=handle)):
        await run_gmail_watch_refresh()

    async with async_session_maker() as db:
        from sqlalchemy import select
        trig = (await db.execute(
            select(Trigger).where(Trigger.user_id == user_id)
        )).scalars().one()

    ps = trig.provider_state_json or {}
    assert "provision_error" not in ps, "Stale provision_error must be cleared"
    assert "provision_detail" not in ps
    assert trig.last_status == "never_fired", (
        "A row stuck in provisioning_failed must flip back to never_fired "
        "after a successful refresh — otherwise the UI keeps lying."
    )
    assert trig.last_error is None


# ──────────────────────────────────────────────────────────────────────
# Gap 2 — specific provisioning errors reach the UI
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "provision_error,expected_health",
    [
        ("needs_reauth", "needs_reauth"),
        ("ops_blocked", "setup_blocked_ops"),
        ("scope_error", "setup_scope_error"),
        ("transient", "setup_transient"),
        ("platform_unreachable", "setup_transient"),
        ("feature_disabled", "setup_feature_disabled"),
        ("http_503", "setup_transient"),
        ("http_403", "setup_scope_error"),
        ("http_999", "setup_error"),
        ("unsupported_connector", "setup_error"),
        ("malformed_response", "setup_error"),
        ("", "setup_error"),
    ],
)
def test_health_maps_each_provision_error_to_specific_state(
    provision_error, expected_health,
):
    """Each `provision_error` code written by `_provision_email_watch`
    must surface as a distinct health value so the UI can render the
    right CTA. Before Gap 2 fix, every code collapsed to `setup_error`."""
    from app.api.triggers import _derive_health

    trig = _trigger_row(
        last_status="provisioning_failed",
        provider_state_json={"provision_error": provision_error},
    )
    assert _derive_health(trig, watch_provisioned=False) == expected_health


# ──────────────────────────────────────────────────────────────────────
# Gap 3 — observability + probe diagnosis
# ──────────────────────────────────────────────────────────────────────


def test_compute_diagnosis_healthy():
    from app.api.triggers import _compute_diagnosis
    diag, _ = _compute_diagnosis(
        watch_alive=True, watch_alive_reason=None,
        push_age=60, dispatch_age=55, handler_age=50,
        handler_status="success",
    )
    assert diag == "healthy"


def test_compute_diagnosis_pubsub_silent():
    """No pushes in > 7 days = Pub/Sub isn't reaching us (subscription
    misconfig or topic IAM)."""
    from app.api.triggers import _compute_diagnosis
    diag, _ = _compute_diagnosis(
        watch_alive=True, watch_alive_reason=None,
        push_age=None,
        dispatch_age=None, handler_age=None, handler_status=None,
    )
    assert diag == "pubsub_not_pushing"


def test_compute_diagnosis_push_rejected_at_platform():
    """Fresh push exists but never reached dispatch — JWT audience
    mismatch or similar."""
    from app.api.triggers import _compute_diagnosis
    diag, _ = _compute_diagnosis(
        watch_alive=True, watch_alive_reason=None,
        push_age=10, dispatch_age=None, handler_age=None,
        handler_status=None,
    )
    assert diag == "push_rejected_at_platform"


def test_compute_diagnosis_dispatched_but_no_handler():
    """Recent push + recent dispatch + no handler completion = runner
    stuck."""
    from app.api.triggers import _compute_diagnosis
    diag, _ = _compute_diagnosis(
        watch_alive=True, watch_alive_reason=None,
        push_age=10, dispatch_age=8, handler_age=None,
        handler_status=None,
    )
    assert diag == "dispatched_but_no_handler_response"


def test_compute_diagnosis_handler_failing():
    from app.api.triggers import _compute_diagnosis
    diag, _ = _compute_diagnosis(
        watch_alive=True, watch_alive_reason=None,
        push_age=10, dispatch_age=8, handler_age=5,
        handler_status="failed",
    )
    assert diag == "handler_failing"


def test_compute_diagnosis_needs_reauth():
    from app.api.triggers import _compute_diagnosis
    diag, _ = _compute_diagnosis(
        watch_alive=False, watch_alive_reason="gmail_401",
        push_age=100, dispatch_age=None, handler_age=None,
        handler_status=None,
    )
    assert diag == "needs_reauth"


def test_compute_diagnosis_watch_expired():
    from app.api.triggers import _compute_diagnosis
    diag, _ = _compute_diagnosis(
        watch_alive=False, watch_alive_reason="watch_expired",
        push_age=None, dispatch_age=None, handler_age=None,
        handler_status=None,
    )
    assert diag == "watch_expired"


@pytest.mark.asyncio
async def test_handler_stamps_observability_timestamps_on_success():
    """The wrapper around `_execute_core` must attach
    `last_handler_completed_at` + `last_handler_status` to
    `new_provider_state` on every result, so the probe can read them."""
    from app.agent.triggers.email_received_handler import EmailReceivedHandler

    writer = AsyncMock(return_value=("msg-1", "day-1"))
    broadcaster = AsyncMock(return_value=1)
    handler = EmailReceivedHandler(mcp_client=None, writer=writer, broadcaster=broadcaster)
    trigger = SimpleNamespace(
        id="t", user_id=CONTAINER_USER_ID, kind="email_received",
        action="summarize_and_post", name="x",
        filter_json=None, config_json={}, provider_state_json={},
    )
    events = [SimpleNamespace(id="ev-1", idempotency_key="test:abc")]

    result = await handler.execute(trigger, events, db=MagicMock())
    ps = result.new_provider_state or {}
    assert "last_handler_completed_at" in ps
    assert ps.get("last_handler_status") == "test_success"


@pytest.mark.asyncio
async def test_handler_stamps_observability_timestamps_on_failure():
    """Same stamping must happen even when the handler fails — that's
    exactly what the probe's `handler_failing` diagnosis depends on."""
    from app.agent.triggers.email_received_handler import EmailReceivedHandler

    writer = AsyncMock(side_effect=RuntimeError("db down"))
    handler = EmailReceivedHandler(mcp_client=None, writer=writer)
    trigger = SimpleNamespace(
        id="t", user_id=CONTAINER_USER_ID, kind="email_received",
        action="summarize_and_post", name="x",
        filter_json=None, config_json={}, provider_state_json={},
    )
    events = [SimpleNamespace(id="ev-1", idempotency_key="test:q")]
    result = await handler.execute(trigger, events, db=MagicMock())
    ps = result.new_provider_state or {}
    assert "last_handler_completed_at" in ps
    assert ps.get("last_handler_status") == "failed"
    assert ps.get("last_handler_error_class") == "RuntimeError"


# ──────────────────────────────────────────────────────────────────────
# Gap 4 — silent OAuth post-connect failures surface to user
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_provision_failure_writes_to_all_email_triggers():
    """The agent endpoint MUST update every email_received trigger for
    the user — not just one. Without this, a user with two Gmail
    triggers gets a confusing partial-failure state."""
    from app.db.database import async_session_maker
    from app.db.models import Trigger, User
    from app.api.triggers import (
        record_provision_failure,
        RecordProvisionFailureReq,
    )
    from app.config import settings as _s

    # Other test files use os.environ.setdefault for AGENT_API_KEY +
    # USER_ID, so whichever imports first wins. Read live settings
    # rather than hard-coding so the test passes under any order.
    container_key = _s.agent_api_key
    user_id = _s.user_id
    async with async_session_maker() as db:
        existing = await db.get(User, user_id)
        if existing is None:
            db.add(User(id=user_id, email=f"oauth-fail-{user_id[:8]}@example.com",
                        hashed_password="x"))
            await db.commit()

    async with async_session_maker() as db:
        for i in range(3):
            db.add(Trigger(
                id=str(uuid.uuid4()), user_id=user_id, kind="email_received",
                action="summarize_and_post", name=f"T{i}",
                enabled=True, last_status="never_fired",
            ))
        await db.commit()

    req = RecordProvisionFailureReq(
        user_id=user_id, connector_id="gmail",
        error="needs_reauth",
        detail="Gmail refresh token is rejected — reconnect Gmail.",
    )
    response = await record_provision_failure(
        req, x_agent_key=container_key,
    )
    assert response.triggers_updated == 3

    async with async_session_maker() as db:
        from sqlalchemy import select
        triggers = (await db.execute(
            select(Trigger).where(Trigger.user_id == user_id)
        )).scalars().all()
    for trig in triggers:
        ps = trig.provider_state_json or {}
        assert ps.get("provision_error") == "needs_reauth"
        assert "Gmail refresh token" in (ps.get("provision_detail") or "")
        assert trig.last_status == "provisioning_failed"


@pytest.mark.asyncio
async def test_record_provision_failure_rejects_wrong_user():
    from app.api.triggers import (
        record_provision_failure,
        RecordProvisionFailureReq,
    )
    from app.config import settings as _s
    from fastapi import HTTPException

    req = RecordProvisionFailureReq(
        user_id=str(uuid.uuid4()),  # NOT the container owner
        connector_id="gmail", error="needs_reauth", detail="x",
    )
    with pytest.raises(HTTPException) as exc:
        await record_provision_failure(
            req, x_agent_key=_s.agent_api_key,
        )
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_record_provision_failure_rejects_bad_key():
    from app.api.triggers import (
        record_provision_failure,
        RecordProvisionFailureReq,
    )
    from app.config import settings as _s
    from fastapi import HTTPException

    req = RecordProvisionFailureReq(
        user_id=_s.user_id, connector_id="gmail",
        error="needs_reauth", detail="x",
    )
    with pytest.raises(HTTPException) as exc:
        await record_provision_failure(req, x_agent_key="DEFINITELY_NOT_THE_KEY_X")
    assert exc.value.status_code == 401


# ──────────────────────────────────────────────────────────────────────
# Gap 5 — boot-time GCP env validation
# ──────────────────────────────────────────────────────────────────────


def test_validate_triggers_env_passes_when_disabled():
    """Feature flag off → no validation, no error. Operators must be
    able to run the platform without GCP configured for non-trigger
    deployments."""
    from platform_main import _validate_triggers_email_env

    settings_off = SimpleNamespace(
        triggers_email_enabled=False,
        gcp_project="", pubsub_topic="", pubsub_push_audience="",
    )
    _validate_triggers_email_env(settings_off)


def test_validate_triggers_env_passes_when_all_set():
    from platform_main import _validate_triggers_email_env

    settings_ok = SimpleNamespace(
        triggers_email_enabled=True,
        gcp_project="my-project",
        pubsub_topic="gmail-pushes",
        pubsub_push_audience="https://toup.ai/api/v1/webhooks/gmail",
    )
    _validate_triggers_email_env(settings_ok)


def test_validate_triggers_env_raises_when_gcp_project_empty():
    from platform_main import _validate_triggers_email_env, TriggersEmailMisconfigured

    bad = SimpleNamespace(
        triggers_email_enabled=True,
        gcp_project="",
        pubsub_topic="x",
        pubsub_push_audience="https://x/y",
    )
    with pytest.raises(TriggersEmailMisconfigured) as exc:
        _validate_triggers_email_env(bad)
    assert "GCP_PROJECT" in str(exc.value)


def test_validate_triggers_env_raises_when_pubsub_topic_empty():
    from platform_main import _validate_triggers_email_env, TriggersEmailMisconfigured

    bad = SimpleNamespace(
        triggers_email_enabled=True,
        gcp_project="x",
        pubsub_topic="",
        pubsub_push_audience="https://x/y",
    )
    with pytest.raises(TriggersEmailMisconfigured) as exc:
        _validate_triggers_email_env(bad)
    assert "PUBSUB_TOPIC" in str(exc.value)


def test_validate_triggers_env_raises_when_audience_missing():
    from platform_main import _validate_triggers_email_env, TriggersEmailMisconfigured

    bad = SimpleNamespace(
        triggers_email_enabled=True,
        gcp_project="x",
        pubsub_topic="x",
        pubsub_push_audience="",
    )
    with pytest.raises(TriggersEmailMisconfigured) as exc:
        _validate_triggers_email_env(bad)
    assert "PUBSUB_PUSH_AUDIENCE" in str(exc.value)


def test_validate_triggers_env_raises_on_malformed_audience():
    """A typo like `toup.ai/api/v1/webhooks/gmail` (no scheme) means
    Pub/Sub signs the JWT against a different audience and 100% of
    pushes fail JWT verify silently."""
    from platform_main import _validate_triggers_email_env, TriggersEmailMisconfigured

    bad = SimpleNamespace(
        triggers_email_enabled=True,
        gcp_project="x",
        pubsub_topic="x",
        pubsub_push_audience="not-a-url",
    )
    with pytest.raises(TriggersEmailMisconfigured) as exc:
        _validate_triggers_email_env(bad)
    assert "malformed_url" in str(exc.value)


def test_validate_triggers_env_raises_with_whitespace_only_values():
    from platform_main import _validate_triggers_email_env, TriggersEmailMisconfigured

    bad = SimpleNamespace(
        triggers_email_enabled=True,
        gcp_project="   ",
        pubsub_topic="x",
        pubsub_push_audience="https://x/y",
    )
    with pytest.raises(TriggersEmailMisconfigured):
        _validate_triggers_email_env(bad)
