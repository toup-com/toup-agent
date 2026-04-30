"""
Phase 3 rollout service tests — pure-unit, no DB, no real bridge.

Covers:
  - Image-tag prefix validation at the service layer
  - RolloutInProgress raised when active rollout exists
  - Canary selection: finds user.is_canary=TRUE in candidate set
  - Canary selection: returns None when canary user isn't in set
  - Attempt status transitions: upgrade OK / failed / rolled_back /
    rollback_failed
  - AgentEnvContract serialization matches the Phase 2 .env format

End-to-end tests (bridge roundtrip, full rollout against a fake bridge)
belong in docs/new-vps/10-VERIFICATION.md §2-§3 — operator-driven, not
pytest.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.types import AgentEnvContract


# ─── AgentEnvContract ─────────────────────────────────────────────


def _valid_contract_kwargs(**overrides):
    base = dict(
        user_id="3134fece-c3de-411b-add8-3f2a58882794",
        agent_api_key="x" * 40,
        database_url="postgresql+asyncpg://toup_agent_3134fece:pw@host.docker.internal:6432/toup_agent_3134fece",
        platform_api_url="https://toup.ai/api",
    )
    base.update(overrides)
    return base


def test_agent_env_happy_path():
    c = AgentEnvContract(**_valid_contract_kwargs(
        openai_api_key="sk-abc", telegram_bot_token="123:abc",
    ))
    lines = c.to_env_lines()
    keys = {line.split("=", 1)[0] for line in lines}
    # Required keys present
    for k in ("RUN_MODE", "USER_ID", "AGENT_API_KEY", "DATABASE_URL",
              "AGENT_WORKSPACE_DIR", "PLATFORM_API_URL", "AGENT_PORT"):
        assert k in keys, f"{k} missing from env output"
    # Optional keys only included when set
    assert "OPENAI_API_KEY" in keys
    assert "ANTHROPIC_API_KEY" not in keys
    assert "TELEGRAM_BOT_TOKEN" in keys
    # Feature-flag booleans rendered lowercase
    assert "USE_DAY_CHAT_CONTEXT=true" in lines
    assert "ENABLE_DAY_RECALL=false" in lines


def test_agent_env_rejects_direct_pg_5432():
    with pytest.raises(ValueError, match="PgBouncer"):
        AgentEnvContract(**_valid_contract_kwargs(
            database_url="postgresql+asyncpg://u:p@host.docker.internal:5432/db",
        ))


def test_agent_env_rejects_http_platform_url():
    with pytest.raises(ValueError, match="https://"):
        AgentEnvContract(**_valid_contract_kwargs(
            platform_api_url="http://toup.ai/api",
        ))


def test_agent_env_rejects_missing_api_suffix():
    with pytest.raises(ValueError, match="/api"):
        AgentEnvContract(**_valid_contract_kwargs(
            platform_api_url="https://toup.ai",
        ))


def test_agent_env_rejects_non_asyncpg_driver():
    with pytest.raises(ValueError, match="asyncpg"):
        AgentEnvContract(**_valid_contract_kwargs(
            database_url="postgresql://u:p@host.docker.internal:6432/db",
        ))


def test_agent_env_short_user_id_rejected():
    with pytest.raises(ValueError, match="user_id"):
        AgentEnvContract(**_valid_contract_kwargs(user_id="short"))


def test_agent_env_to_env_string_is_newline_terminated():
    c = AgentEnvContract(**_valid_contract_kwargs())
    s = c.to_env_string()
    assert s.endswith("\n")
    # One key per line, no blank lines
    assert "" not in [line for line in s.split("\n") if line == ""][-1:] or s.endswith("\n")


# ─── rollout_service.start_rollout validation ────────────────────


@pytest.mark.asyncio
async def test_start_rollout_rejects_non_ghcr_tag():
    from app.services.rollout_service import start_rollout

    db = MagicMock()
    with pytest.raises(ValueError, match="ghcr.io/toup-com/toup-agent"):
        await start_rollout(db, image_tag="docker.io/evil/backdoor:latest", trigger="ci")


@pytest.mark.asyncio
async def test_start_rollout_raises_in_progress_when_active():
    from app.services.rollout_service import start_rollout, RolloutInProgress
    from app.db.models import Rollout

    mock_active = Rollout(
        id="active-id-123",
        image_tag="ghcr.io/toup-com/toup-agent:earlier1234",
        status="running",
        trigger="ci",
    )

    # Patch active_rollout to return a fake in-flight rollout. Also patch
    # _reconcile_once: the self-heal pass added to start_rollout (eb2ecc31487
    # incident fix) runs BEFORE this lock check, and we want the test to
    # exercise the lock-collision path, not the reconciliation path.
    with patch("app.services.rollout_service.active_rollout", AsyncMock(return_value=mock_active)), \
         patch("app.services.rollout_service._reconcile_once", AsyncMock(return_value=None)):
        db = MagicMock()
        with pytest.raises(RolloutInProgress) as exc_info:
            await start_rollout(db, image_tag="ghcr.io/toup-com/toup-agent:abcd1234", trigger="ci")
        assert exc_info.value.active_id == "active-id-123"
        assert exc_info.value.active_tag == "ghcr.io/toup-com/toup-agent:earlier1234"


@pytest.mark.asyncio
async def test_start_rollout_invokes_reconcile_before_lock_check():
    """start_rollout must self-heal — run reconcile BEFORE the lock check.

    Without this ordering, a single stuck rollout from a dead orchestrator
    blocks every subsequent CI push with HTTP 409 indefinitely. The
    reconciler loop catches it eventually, but the next CI push immediately
    after is racy. (eb2ecc314879 incident, 2026-04-29 — 5h stuck.)
    """
    from app.services.rollout_service import start_rollout

    call_order: list[str] = []

    async def _tracker_reconcile(db_arg=None):
        call_order.append("reconcile")

    async def _tracker_active(db_arg):
        call_order.append("active_rollout")
        return None

    with patch("app.services.rollout_service._reconcile_once", _tracker_reconcile), \
         patch("app.services.rollout_service.active_rollout", _tracker_active), \
         patch("app.scripts.scheduled_tasks.schedule_one_shot"):
        db = MagicMock()
        db.add = MagicMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        await start_rollout(db, image_tag="ghcr.io/toup-com/toup-agent:abcd1234", trigger="ci")

    assert call_order == ["reconcile", "active_rollout"], (
        f"expected reconcile before active_rollout, got: {call_order}"
    )


# ─── canary selection ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_canary_container_empty_candidates():
    from app.services.rollout_service import _canary_container

    db = MagicMock()
    result = await _canary_container(db, [])
    assert result is None


@pytest.mark.asyncio
async def test_canary_container_no_canary_user():
    from app.services.rollout_service import _canary_container

    # Candidates exist but no user flagged is_canary=True
    cand = [MagicMock(user_id="aaaaaaaa-...-...-..."), MagicMock(user_id="bbbbbbbb-...-...-...")]

    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)
    db.execute = AsyncMock(return_value=mock_result)

    result = await _canary_container(db, cand)
    assert result is None


@pytest.mark.asyncio
async def test_canary_container_canary_user_in_set():
    from app.services.rollout_service import _canary_container

    canary_uid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    cand = [
        MagicMock(user_id=canary_uid),
        MagicMock(user_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
    ]

    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=canary_uid)
    db.execute = AsyncMock(return_value=mock_result)

    result = await _canary_container(db, cand)
    assert result is cand[0]


@pytest.mark.asyncio
async def test_canary_container_canary_user_not_in_set():
    """User is_canary=TRUE exists but their container isn't in candidates —
    rollout MUST refuse to proceed."""
    from app.services.rollout_service import _canary_container

    cand = [
        MagicMock(user_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        MagicMock(user_id="cccccccc-cccc-cccc-cccc-cccccccccccc"),
    ]
    # canary user has id 'aaaa...' but isn't in the candidate list
    canary_uid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=canary_uid)
    db.execute = AsyncMock(return_value=mock_result)

    result = await _canary_container(db, cand)
    assert result is None, "canary absent from running set → rollout must refuse"


# ─── reconciler orphan paths ─────────────────────────────────────


@pytest.mark.asyncio
async def test_reconcile_orphans_pending_past_threshold():
    """A row in `pending` status past _STUCK_PENDING_THRESHOLD_MIN means
    APScheduler never fired. The reconciler must orphan it so subsequent
    CI pushes don't 409 forever."""
    from datetime import datetime, timedelta
    from app.services.rollout_service import _reconcile_once_in_session, _STUCK_PENDING_THRESHOLD_MIN
    from app.db.models import Rollout

    stuck = Rollout(
        id="pending-orphan-1",
        image_tag="ghcr.io/toup-com/toup-agent:abcd1234",
        status="pending",
        trigger="ci",
        started_at=datetime.utcnow() - timedelta(minutes=_STUCK_PENDING_THRESHOLD_MIN + 1),
    )

    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[stuck])))
    db.execute = AsyncMock(return_value=mock_result)
    db.commit = AsyncMock()

    with patch("app.services.rollout_service._send_telegram", AsyncMock()):
        await _reconcile_once_in_session(db)

    assert stuck.status == "aborted_orphan"
    assert stuck.completed_at is not None
    assert "pending" in (stuck.notes or "")
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_reconcile_leaves_fresh_pending_alone():
    """A `pending` row younger than the threshold is healthy in-flight work
    (APScheduler about to fire). Reconciler must NOT touch it."""
    from datetime import datetime, timedelta
    from app.services.rollout_service import _reconcile_once_in_session, _STUCK_PENDING_THRESHOLD_MIN
    from app.db.models import Rollout

    fresh = Rollout(
        id="pending-fresh-1",
        image_tag="ghcr.io/toup-com/toup-agent:abcd1234",
        status="pending",
        trigger="ci",
        started_at=datetime.utcnow() - timedelta(minutes=max(0, _STUCK_PENDING_THRESHOLD_MIN - 1)),
    )

    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[fresh])))
    db.execute = AsyncMock(return_value=mock_result)
    db.commit = AsyncMock()

    with patch("app.services.rollout_service._send_telegram", AsyncMock()):
        await _reconcile_once_in_session(db)

    assert fresh.status == "pending", "fresh pending row must not be orphaned"
    db.commit.assert_not_awaited()


# ─── force-orphan operator escape hatch ──────────────────────────


@pytest.mark.asyncio
async def test_force_orphan_active_returns_none_when_no_lock():
    """Idempotent no-op when no rollout is in flight."""
    from app.services.rollout_service import force_orphan_active

    with patch("app.services.rollout_service.active_rollout", AsyncMock(return_value=None)):
        db = MagicMock()
        result = await force_orphan_active(db, reason="t")

    assert result is None


@pytest.mark.asyncio
async def test_force_orphan_active_orphans_in_flight_lock():
    """Force-orphan flips the active rollout to aborted_orphan, freeing
    the lock immediately regardless of age. Used when operator knows the
    orchestrator is dead and the reconciler hasn't aged it out yet."""
    from datetime import datetime, timedelta
    from app.services.rollout_service import force_orphan_active
    from app.db.models import Rollout

    active = Rollout(
        id="lock-holder",
        image_tag="ghcr.io/toup-com/toup-agent:zzzzzzz",
        status="running",
        phase="canary_observing",
        trigger="ci",
        started_at=datetime.utcnow() - timedelta(minutes=2),  # under threshold
    )

    with patch("app.services.rollout_service.active_rollout", AsyncMock(return_value=active)), \
         patch("app.services.rollout_service._send_telegram", AsyncMock()):
        db = MagicMock()
        db.commit = AsyncMock()
        result = await force_orphan_active(db, reason="lock wedged")

    assert result is active
    assert active.status == "aborted_orphan"
    assert active.completed_at is not None
    assert "lock wedged" in (active.notes or "")


# ─── signal-based canary observation ────────────────────────────


@pytest.mark.asyncio
async def test_observe_canary_signal_passes_on_healthy():
    """Healthy canary: 3+ consecutive 200s, sustained healthy in stability
    hold → pass. Was: always-elapsed-time wait of canary_wait_minutes."""
    import httpx
    from app.services.rollout_service import _observe_canary_signal

    ok = httpx.Response(200, content=b'{"ok":true}')

    class _Client:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, url): return ok

    with patch("httpx.AsyncClient", _Client), \
         patch("asyncio.sleep", AsyncMock()):
        passed, reason = await _observe_canary_signal(
            "https://test.local",
            cap_seconds=2.0, boot_gate_s=1.0, boot_interval_s=0.0,
            required_ok=3, stability_hold_s=0.5, stability_interval_s=0.0,
        )

    assert passed, f"healthy canary must pass; got reason={reason}"
    assert "healthy" in reason


@pytest.mark.asyncio
async def test_observe_canary_signal_fails_when_boot_never_passes():
    """Canary that never returns 200 within the boot gate must fail — not
    silently succeed when the cap is wide."""
    import httpx
    from app.services.rollout_service import _observe_canary_signal

    bad = httpx.Response(503, content=b"unavailable")

    class _Client:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, url): return bad

    with patch("httpx.AsyncClient", _Client), \
         patch("asyncio.sleep", AsyncMock()):
        passed, reason = await _observe_canary_signal(
            "https://test.local",
            cap_seconds=2.0, boot_gate_s=1.0, boot_interval_s=0.05,
            required_ok=3, stability_hold_s=0.5, stability_interval_s=0.05,
        )

    assert not passed
    assert "boot gate failed" in reason


@pytest.mark.asyncio
async def test_observe_canary_signal_fails_on_post_boot_regression():
    """Canary boots fine (3 consecutive 200s) then degrades during
    stability hold. Must fail — without this, a canary that crashes
    after boot would be promoted before the stability hold catches it."""
    import httpx
    from app.services.rollout_service import _observe_canary_signal

    ok = httpx.Response(200, content=b'{"ok":true}')
    bad = httpx.Response(503, content=b"crashed after boot")

    # First 5 calls return 200 (boot gate passes after the 3rd), then
    # subsequent calls return 503 (stability hold sees a regression).
    call_count = {"n": 0}

    class _Client:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, url):
            call_count["n"] += 1
            return ok if call_count["n"] <= 5 else bad

    with patch("httpx.AsyncClient", _Client), \
         patch("asyncio.sleep", AsyncMock()):
        passed, reason = await _observe_canary_signal(
            "https://test.local",
            cap_seconds=5.0, boot_gate_s=1.0, boot_interval_s=0.0,
            required_ok=3, stability_hold_s=2.0, stability_interval_s=0.0,
        )

    assert not passed
    assert "stability hold failed" in reason


@pytest.mark.asyncio
async def test_observe_canary_signal_respects_cap_as_hard_deadline():
    """`cap_seconds` is a hard deadline. With cap=0 (degenerate operator
    setting), the function must exit immediately rather than burn the
    full default stability hold. This is what makes
    `canary_wait_minutes` an upper bound rather than a target."""
    import httpx
    from app.services.rollout_service import _observe_canary_signal

    ok = httpx.Response(200, content=b'{"ok":true}')

    class _Client:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, url): return ok

    # cap=0 means deadline already passed at function entry. Boot loop's
    # `while time.time() < boot_deadline` is false on first iteration, so
    # consecutive_ok stays 0 → returns boot-gate-failed. This proves the
    # cap is enforced as a hard deadline, not a "best effort" budget.
    with patch("httpx.AsyncClient", _Client), \
         patch("asyncio.sleep", AsyncMock()):
        passed, reason = await _observe_canary_signal("https://t", cap_seconds=0.0)

    assert not passed
    assert "boot gate failed" in reason


# ─── heartbeat-based orphan detection ────────────────────────────


@pytest.mark.asyncio
async def test_reconcile_orphans_running_with_stale_heartbeat():
    """The heartbeat fix: a `running` rollout whose `last_progress_at` is
    older than _STUCK_HEARTBEAT_MIN must be orphaned regardless of total
    age. Catches the rapid-redeploy case where each new platform-api
    boot kills the previous orchestrator before it can complete."""
    from datetime import datetime, timedelta
    from app.services.rollout_service import (
        _reconcile_once_in_session, _STUCK_HEARTBEAT_MIN,
    )
    from app.db.models import Rollout

    stuck = Rollout(
        id="hb-stale-1",
        image_tag="ghcr.io/toup-com/toup-agent:abcd1234",
        status="running",
        phase="canary_observing",
        trigger="ci",
        started_at=datetime.utcnow() - timedelta(minutes=10),  # under 30-min total
        last_progress_at=datetime.utcnow() - timedelta(minutes=_STUCK_HEARTBEAT_MIN + 1),
    )

    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[stuck])))
    db.execute = AsyncMock(return_value=mock_result)
    db.commit = AsyncMock()

    with patch("app.services.rollout_service._send_telegram", AsyncMock()):
        await _reconcile_once_in_session(db)

    assert stuck.status == "aborted_orphan", (
        "stale-heartbeat running rollout must be orphaned"
    )
    assert stuck.completed_at is not None
    assert "heartbeat stale" in (stuck.notes or "")


@pytest.mark.asyncio
async def test_reconcile_skips_running_with_fresh_heartbeat():
    """Fresh heartbeat means the orchestrator is making progress. The
    heartbeat path must NOT orphan it just because the rollout is
    several minutes old (e.g. genuine canary observation in progress)."""
    from datetime import datetime, timedelta
    from app.services.rollout_service import _reconcile_once_in_session
    from app.db.models import Rollout

    healthy = Rollout(
        id="hb-fresh-1",
        image_tag="ghcr.io/toup-com/toup-agent:abcd1234",
        status="running",
        phase="batching",
        trigger="ci",
        started_at=datetime.utcnow() - timedelta(minutes=2),
        last_progress_at=datetime.utcnow() - timedelta(seconds=10),
    )

    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[healthy])))
    db.execute = AsyncMock(return_value=mock_result)
    db.commit = AsyncMock()

    with patch("app.services.rollout_service._send_telegram", AsyncMock()):
        await _reconcile_once_in_session(db)

    # Heartbeat path runs FIRST in the reconciler. With fresh heartbeat
    # it must skip — and total-age (2 min) is also under the threshold,
    # so the rollout stays running. Phase != canary_observing means
    # the resume path also skips. Net: status unchanged.
    assert healthy.status == "running", "fresh-heartbeat row must not be orphaned"


@pytest.mark.asyncio
async def test_reconcile_falls_back_to_started_at_when_heartbeat_null():
    """Backward compat: rows created before `last_progress_at` column
    existed have NULL heartbeat. Reconciler must fall back to
    `started_at` so old stuck rows still get orphaned (NULL → infinite
    idle would be a footgun, but NULL → fresh would never orphan
    legacy state — neither is right; falling back to started_at is
    the only safe choice)."""
    from datetime import datetime, timedelta
    from app.services.rollout_service import (
        _reconcile_once_in_session, _STUCK_HEARTBEAT_MIN,
    )
    from app.db.models import Rollout

    legacy = Rollout(
        id="legacy-null-1",
        image_tag="ghcr.io/toup-com/toup-agent:legacy",
        status="running",
        phase="canary_observing",
        trigger="ci",
        started_at=datetime.utcnow() - timedelta(minutes=_STUCK_HEARTBEAT_MIN + 1),
        last_progress_at=None,
    )

    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[legacy])))
    db.execute = AsyncMock(return_value=mock_result)
    db.commit = AsyncMock()

    with patch("app.services.rollout_service._send_telegram", AsyncMock()):
        await _reconcile_once_in_session(db)

    assert legacy.status == "aborted_orphan"
    assert "heartbeat stale" in (legacy.notes or "")
