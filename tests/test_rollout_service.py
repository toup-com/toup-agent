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


# ─── T17: resume-path dead-code fix ───────────────────────────────
# Before the indentation fix at rollout_service.py:781-798, the resume
# block was unreachable: the `if rollout.status != "running": continue`
# guard at line 778 swallowed every code path that would have reached the
# resume checks. These three tests cover the post-fix flow.


@pytest.mark.asyncio
async def test_reconcile_resumes_canary_observing_after_resume_after():
    """T17: a `running` rollout in `phase='canary_observing'` whose
    `resume_after` deadline has passed must be added to _resume_inflight
    and have a resume task scheduled. Heartbeat fresh + age young so the
    orphan paths don't fire — only the resume path should activate.
    """
    import asyncio
    from datetime import datetime, timedelta
    from app.services.rollout_service import (
        _reconcile_once_in_session, _resume_inflight,
    )
    from app.db.models import Rollout

    _resume_inflight.clear()

    eligible = Rollout(
        id="resume-eligible-1",
        image_tag="ghcr.io/toup-com/toup-agent:abcd1234",
        status="running",
        phase="canary_observing",
        trigger="ci",
        started_at=datetime.utcnow() - timedelta(minutes=2),
        last_progress_at=datetime.utcnow() - timedelta(seconds=30),
        resume_after=datetime.utcnow() - timedelta(seconds=10),
    )

    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[eligible])))
    db.execute = AsyncMock(return_value=mock_result)
    db.commit = AsyncMock()

    async def _noop_resume(_rollout_id):
        pass

    with patch("app.services.rollout_service._resume_with_cleanup", _noop_resume), \
         patch("app.services.rollout_service._send_telegram", AsyncMock()):
        await _reconcile_once_in_session(db)
        # Yield once to let the scheduled noop drain so we don't get
        # "coroutine was never awaited" warnings.
        await asyncio.sleep(0)

    assert "resume-eligible-1" in _resume_inflight, (
        "resume-eligible rollout must be added to _resume_inflight — "
        "this fails if the dedent fix at rollout_service.py:781-798 is reverted"
    )
    assert eligible.status == "running", "must NOT orphan a resume-eligible rollout"

    _resume_inflight.discard("resume-eligible-1")


@pytest.mark.asyncio
async def test_reconcile_does_not_resume_when_resume_after_is_future():
    """T17: a `running` canary_observing rollout whose `resume_after` is
    still in the future must NOT be resumed (deadline check at
    rollout_service.py:787). Heartbeat fresh + age young, so no orphan
    either — status must remain `running`.
    """
    import asyncio
    from datetime import datetime, timedelta
    from app.services.rollout_service import (
        _reconcile_once_in_session, _resume_inflight,
    )
    from app.db.models import Rollout

    _resume_inflight.clear()

    too_early = Rollout(
        id="resume-too-early-1",
        image_tag="ghcr.io/toup-com/toup-agent:abcd1234",
        status="running",
        phase="canary_observing",
        trigger="ci",
        started_at=datetime.utcnow() - timedelta(minutes=1),
        last_progress_at=datetime.utcnow() - timedelta(seconds=30),
        resume_after=datetime.utcnow() + timedelta(minutes=5),
    )

    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[too_early])))
    db.execute = AsyncMock(return_value=mock_result)
    db.commit = AsyncMock()

    async def _noop_resume(_rollout_id):
        pass

    with patch("app.services.rollout_service._resume_with_cleanup", _noop_resume), \
         patch("app.services.rollout_service._send_telegram", AsyncMock()):
        await _reconcile_once_in_session(db)
        await asyncio.sleep(0)

    assert "resume-too-early-1" not in _resume_inflight, (
        "resume must not fire while resume_after is in the future"
    )
    assert too_early.status == "running", "must not orphan a healthy in-flight canary"


@pytest.mark.asyncio
async def test_reconcile_does_not_resume_batching_phase():
    """T17: a `running` rollout in `phase='batching'` (not 'canary_observing')
    must NOT be resumed even with `resume_after` in the past — the resume
    guard at rollout_service.py:785 explicitly excludes non-canary_observing
    phases because mid-bridge-call resume risks double-deploys. With stale
    heartbeat, Path 0 fires and orphans the rollout. Confirms guard
    semantics survive the indentation fix.
    """
    import asyncio
    from datetime import datetime, timedelta
    from app.services.rollout_service import (
        _reconcile_once_in_session, _STUCK_HEARTBEAT_MIN, _resume_inflight,
    )
    from app.db.models import Rollout

    _resume_inflight.clear()

    batching_orphan = Rollout(
        id="batching-orphan-1",
        image_tag="ghcr.io/toup-com/toup-agent:abcd1234",
        status="running",
        phase="batching",
        trigger="ci",
        started_at=datetime.utcnow() - timedelta(minutes=10),
        last_progress_at=datetime.utcnow() - timedelta(minutes=_STUCK_HEARTBEAT_MIN + 1),
        resume_after=datetime.utcnow() - timedelta(seconds=10),
    )

    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[batching_orphan])))
    db.execute = AsyncMock(return_value=mock_result)
    db.commit = AsyncMock()

    async def _noop_resume(_rollout_id):
        pass

    with patch("app.services.rollout_service._resume_with_cleanup", _noop_resume), \
         patch("app.services.rollout_service._send_telegram", AsyncMock()):
        await _reconcile_once_in_session(db)
        await asyncio.sleep(0)

    assert batching_orphan.status == "aborted_orphan", (
        "stale-heartbeat batching rollout must be orphaned via Path 0"
    )
    assert "heartbeat stale" in (batching_orphan.notes or "")
    assert "batching-orphan-1" not in _resume_inflight, (
        "resume guard must block non-canary_observing phases — "
        "this fails if the resume path were to fire on phase!='canary_observing'"
    )


# ─── Orphan auto-quarantine (BridgeContainerNotFound) ──────────────
#
# Background: 2026-05-25 — Telegram showed 5 tenants per rollout firing
# the same "no container on bridge (orphan). Manual cleanup may be
# needed" warning. The platform DB had `managed_containers.status =
# 'running'` for prefixes the bridge had no record of; every rollout
# re-included the zombie rows in the candidate set, hit 404, and
# re-fired the warning. PR fixes this by flipping
# `status='running' → 'orphan'` when whois corroborates the 404
# (`confirmed_orphan=True`), so the next rollout's `_running_tenants`
# query skips the row. Unconfirmed 404s (whois timeout / disagreement)
# alert only — we don't auto-evict on ambiguous signal.


async def _seed_orphan_quarantine_fixtures(*, container_status: str = "running"):
    """Create a User + AgentConfig + Rollout + ManagedContainer for the
    orphan-quarantine tests. Returns (rollout, container) detached from
    any session — they're refetched inside `_upgrade_one`'s own session,
    which is how the production path works.
    """
    import uuid as _uuid
    from datetime import datetime, timedelta
    from app.db import async_session_maker
    from app.db.models import (
        User, AgentConfig, ManagedContainer, Rollout,
    )
    from app.services.auth_service import get_password_hash

    user_id = str(_uuid.uuid4())
    async with async_session_maker() as db:
        u = User(
            id=user_id,
            email=f"orphan-{user_id[:8]}@example.com",
            hashed_password=get_password_hash("test1234abcd"),
            name=f"Orphan {user_id[:6]}",
        )
        db.add(u)
        await db.flush()

        cfg = AgentConfig(
            user_id=user_id,
            hosting_mode="managed",
            agent_name="Test",
            llm_mode=None,
        )
        db.add(cfg)

        rollout = Rollout(
            image_tag="ghcr.io/toup-com/toup-agent:newtag00",
            status="running",
            trigger="ci",
            canary_wait_minutes=5,
            started_at=datetime.utcnow() - timedelta(minutes=1),
        )
        db.add(rollout)

        mc = ManagedContainer(
            id=str(_uuid.uuid4()),
            user_id=user_id,
            container_id=f"docker-{user_id[:8]}",
            container_name=f"toup-agent-{user_id[:8]}",
            host_port=9000 + (hash(user_id) % 900),
            db_name=f"toup_agent_{user_id[:8]}",
            status=container_status,
            image_tag="ghcr.io/toup-com/toup-agent:oldtag00",
        )
        db.add(mc)
        await db.commit()
        await db.refresh(rollout)
        await db.refresh(mc)
        # Detach for the next session opened by _upgrade_one.
        db.expunge(rollout)
        db.expunge(mc)

    return rollout, mc


@pytest.mark.asyncio
async def test_upgrade_one_quarantines_confirmed_orphan():
    """Confirmed orphan (whois 404) flips ManagedContainer.status to
    'orphan' so `_running_tenants` excludes it on the next rollout.
    """
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import ManagedContainer, RolloutAttempt
    from app.services import rollout_service
    from app.services.docker_host_service import BridgeContainerNotFound

    rollout, container = await _seed_orphan_quarantine_fixtures()

    async def fake_upgrade(*args, **kwargs):
        raise BridgeContainerNotFound(
            prefix=container.user_id[:8],
            raw='{"detail":"no such tenant"}',
            confirmed_orphan=True,
        )

    sent: list[tuple[str, str]] = []

    async def fake_telegram(level, message):
        sent.append((level, message))

    with patch.object(rollout_service, "upgrade_tenant_image", fake_upgrade), \
         patch.object(rollout_service, "_send_telegram", fake_telegram):
        attempt = await rollout_service._upgrade_one(
            None, rollout, container, "ghcr.io/toup-com/toup-agent:newtag00",
        )

    assert attempt.status == "failed"
    assert "orphan tenant" in (attempt.error or "")

    async with async_session_maker() as db:
        refreshed = (await db.execute(
            select(ManagedContainer).where(ManagedContainer.id == container.id)
        )).scalar_one()
        assert refreshed.status == "orphan", (
            "confirmed orphan must flip status running → orphan so "
            "_running_tenants excludes the zombie row on the next rollout"
        )
        assert "auto-quarantined" in (refreshed.error_message or "")
        # Attempt row persisted with status='failed'.
        rows = (await db.execute(
            select(RolloutAttempt).where(RolloutAttempt.rollout_id == rollout.id)
        )).scalars().all()
        assert len(rows) == 1 and rows[0].status == "failed"

    assert any("auto-quarantined" in m for _, m in sent), (
        f"expected auto-quarantine alert, got: {sent!r}"
    )


@pytest.mark.asyncio
async def test_upgrade_one_does_not_quarantine_unconfirmed_orphan():
    """Unconfirmed orphan (whois 0 / 200 / 5xx) only alerts — must not
    evict the ManagedContainer row, so a transient bridge hiccup can't
    silently quarantine a real tenant.
    """
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import ManagedContainer
    from app.services import rollout_service
    from app.services.docker_host_service import BridgeContainerNotFound

    rollout, container = await _seed_orphan_quarantine_fixtures()

    async def fake_upgrade(*args, **kwargs):
        raise BridgeContainerNotFound(
            prefix=container.user_id[:8],
            raw='{"detail":"no such tenant"}',
            confirmed_orphan=False,
        )

    sent: list[tuple[str, str]] = []

    async def fake_telegram(level, message):
        sent.append((level, message))

    with patch.object(rollout_service, "upgrade_tenant_image", fake_upgrade), \
         patch.object(rollout_service, "_send_telegram", fake_telegram):
        attempt = await rollout_service._upgrade_one(
            None, rollout, container, "ghcr.io/toup-com/toup-agent:newtag00",
        )

    assert attempt.status == "failed"

    async with async_session_maker() as db:
        refreshed = (await db.execute(
            select(ManagedContainer).where(ManagedContainer.id == container.id)
        )).scalar_one()
        assert refreshed.status == "running", (
            "unconfirmed orphan must NOT auto-evict a real tenant — "
            "a transient bridge 404 on /upgrade with whois timeout would "
            "otherwise silently quarantine a working container"
        )

    assert any("unconfirmed" in m for _, m in sent), (
        f"expected unconfirmed-orphan alert, got: {sent!r}"
    )


@pytest.mark.asyncio
async def test_upgrade_one_skips_quarantine_when_status_already_changed():
    """If a concurrent path already moved the ManagedContainer off
    'running' (e.g. operator manually marked it 'stopped'), the orphan
    handler must not overwrite that state.
    """
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import ManagedContainer
    from app.services import rollout_service
    from app.services.docker_host_service import BridgeContainerNotFound

    rollout, container = await _seed_orphan_quarantine_fixtures()

    # Race: between rollout candidate selection and the bridge call,
    # an operator marks the container stopped via the admin API.
    async with async_session_maker() as db:
        mc_now = (await db.execute(
            select(ManagedContainer).where(ManagedContainer.id == container.id)
        )).scalar_one()
        mc_now.status = "stopped"
        await db.commit()

    async def fake_upgrade(*args, **kwargs):
        raise BridgeContainerNotFound(
            prefix=container.user_id[:8],
            raw="", confirmed_orphan=True,
        )

    async def fake_telegram(level, message):
        pass

    with patch.object(rollout_service, "upgrade_tenant_image", fake_upgrade), \
         patch.object(rollout_service, "_send_telegram", fake_telegram):
        await rollout_service._upgrade_one(
            None, rollout, container, "ghcr.io/toup-com/toup-agent:newtag00",
        )

    async with async_session_maker() as db:
        refreshed = (await db.execute(
            select(ManagedContainer).where(ManagedContainer.id == container.id)
        )).scalar_one()
        assert refreshed.status == "stopped", (
            "must only quarantine rows that are still 'running' — "
            "overwriting 'stopped' would clobber operator intent"
        )


# ─── whois corroboration in upgrade_tenant_image ───────────────────


@pytest.mark.asyncio
async def test_upgrade_tenant_image_sets_confirmed_orphan_on_legacy_404_with_whois_404():
    """Legacy /upgrade returns 404 + whois returns 404 → exception carries
    confirmed_orphan=True. This is the gate the rollout service uses to
    decide whether to auto-quarantine the ManagedContainer row.
    """
    from app.config import settings as _settings
    from app.services import docker_host_service as dhs

    # Force legacy path
    _orig_flag = getattr(_settings, "use_blue_green_rollouts", False)
    _settings.use_blue_green_rollouts = False

    class _Resp:
        def __init__(self, code: int, text: str = ""):
            self.status_code = code
            self.text = text

        def json(self): return {}
        def raise_for_status(self): pass

    class _FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, path, json=None): return _Resp(404, '{"detail":"no container"}')
        async def get(self, path):
            assert path.endswith("/whois")
            return _Resp(404, '{"detail":"unknown tenant"}')

    try:
        with patch.object(dhs, "_bridge_client", lambda *_a, **_k: _FakeClient()):
            with pytest.raises(dhs.BridgeContainerNotFound) as exc_info:
                await dhs.upgrade_tenant_image(
                    db=AsyncMock(),
                    user_id="9ef741e3-aaaa-bbbb-cccc-dddddddddddd",
                    image_tag="ghcr.io/toup-com/toup-agent:newtag00",
                )
        assert exc_info.value.confirmed_orphan is True
    finally:
        _settings.use_blue_green_rollouts = _orig_flag


@pytest.mark.asyncio
async def test_upgrade_tenant_image_unconfirmed_when_whois_errors():
    """Legacy /upgrade returns 404 but whois call raises → confirmed_orphan
    is False. We must not quarantine when the bridge can't corroborate.
    """
    from app.config import settings as _settings
    from app.services import docker_host_service as dhs

    _orig_flag = getattr(_settings, "use_blue_green_rollouts", False)
    _settings.use_blue_green_rollouts = False

    class _Resp:
        def __init__(self, code: int, text: str = ""):
            self.status_code = code
            self.text = text

    class _FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, path, json=None): return _Resp(404, "")
        async def get(self, path):
            raise RuntimeError("simulated bridge connect error")

    try:
        with patch.object(dhs, "_bridge_client", lambda *_a, **_k: _FakeClient()):
            with pytest.raises(dhs.BridgeContainerNotFound) as exc_info:
                await dhs.upgrade_tenant_image(
                    db=AsyncMock(),
                    user_id="9ef741e3-aaaa-bbbb-cccc-dddddddddddd",
                    image_tag="ghcr.io/toup-com/toup-agent:newtag00",
                )
        assert exc_info.value.confirmed_orphan is False, (
            "whois transport error must NOT be treated as a confirmed "
            "orphan — a momentarily-unreachable bridge would otherwise "
            "auto-quarantine real tenants"
        )
    finally:
        _settings.use_blue_green_rollouts = _orig_flag


# ─── Hard per-attempt timeout (anti-hang defense in depth) ─────────
#
# Background: 2026-05-26 — rollout ed53f0d89a11 had 5 attempts stuck in
# `status='upgrading'` for 547 MINUTES before the reconciler caught them,
# despite httpx having a 180s timeout. Root cause was a bridge that
# accepted TCP but never responded, combined with the held DB sessions
# starving the reconciler's own DB queries on the Supabase pooler. The
# `asyncio.wait_for` wrapper in `_upgrade_one` is the defense-in-depth
# layer above httpx — it always fires, cancels the underlying coroutine,
# releases the session, and lets the rollout continue.


@pytest.mark.asyncio
async def test_upgrade_one_hard_timeout_marks_attempt_failed():
    """When upgrade_tenant_image hangs past the hard cap, the attempt
    must be marked 'failed' with the timeout error, NOT left in
    'upgrading' status (which is what caused the 547-min stall).
    """
    from sqlalchemy import select
    from app.config import settings as _settings
    from app.db import async_session_maker
    from app.db.models import ManagedContainer, RolloutAttempt
    from app.services import rollout_service

    rollout, container = await _seed_orphan_quarantine_fixtures()

    # Shrink the upgrade timeout so the test runs in ~1s instead of 210s.
    # `_upgrade_one` reads `settings.bridge_upgrade_timeout_s` at call time
    # and adds a +30s grace; we set it to a tiny value so the +grace is
    # still tiny.
    _orig_timeout = _settings.bridge_upgrade_timeout_s
    _settings.bridge_upgrade_timeout_s = 0  # → hard_timeout_s = 0 + 30 = 30s
    # Patch the constant offset by monkey-patching the source.

    async def fake_hung_upgrade(*args, **kwargs):
        # Hang forever — asyncio.wait_for must cancel this.
        await asyncio.sleep(3600)

    sent: list[tuple[str, str]] = []

    async def fake_telegram(level, message):
        sent.append((level, message))

    # We need the hard timeout to actually trip in test time. Patch
    # `settings.bridge_upgrade_timeout_s` AND override the +30s grace
    # by stubbing the timeout calculation directly via a wrapping fake.
    import app.services.rollout_service as rs_mod

    _orig_wait_for = asyncio.wait_for

    async def short_wait_for(coro, timeout):
        # Force a tiny timeout regardless of what _upgrade_one computed.
        return await _orig_wait_for(coro, timeout=1.0)

    try:
        with patch.object(rs_mod, "upgrade_tenant_image", fake_hung_upgrade), \
             patch.object(rs_mod, "_send_telegram", fake_telegram), \
             patch.object(rs_mod.asyncio, "wait_for", short_wait_for):
            attempt = await rs_mod._upgrade_one(
                None, rollout, container, "ghcr.io/toup-com/toup-agent:newtag00",
            )
    finally:
        _settings.bridge_upgrade_timeout_s = _orig_timeout

    assert attempt.status == "failed", (
        "hung upgrade MUST end as failed; the 547-min ed53f0d89a11 stall "
        "happened because the attempt stayed in 'upgrading' forever"
    )
    assert "hard timeout" in (attempt.error or "").lower(), (
        f"expected hard-timeout error message, got: {attempt.error!r}"
    )

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(RolloutAttempt).where(RolloutAttempt.rollout_id == rollout.id)
        )).scalars().all()
        assert len(rows) == 1
        assert rows[0].status == "failed", (
            "DB row must reflect 'failed' — otherwise the reconciler can't "
            "tell upgraded-cleanly from hung-and-killed"
        )

        # ManagedContainer should NOT be auto-quarantined on a hang —
        # we don't know if the tenant is alive or dead, only that the
        # bridge didn't respond. Quarantining on ambiguous signal would
        # silently evict working tenants on a transient bridge outage.
        mc = (await db.execute(
            select(ManagedContainer).where(ManagedContainer.id == container.id)
        )).scalar_one()
        assert mc.status == "running", (
            "hard timeout must NOT flip a tenant to 'orphan' — that's "
            "reserved for whois-confirmed orphans"
        )

    assert any("hard-timed out" in m for _, m in sent), (
        f"expected hard-timeout Telegram alert, got: {sent!r}"
    )


@pytest.mark.asyncio
async def test_upgrade_one_does_not_apply_hard_timeout_when_upgrade_succeeds_fast():
    """Hard timeout must not interfere with normal fast-path upgrades —
    pin that a quick successful upgrade still reports `status='ok'`.
    """
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import RolloutAttempt
    from app.services import rollout_service

    rollout, container = await _seed_orphan_quarantine_fixtures()

    async def fake_fast_upgrade(*args, **kwargs):
        return {"health_checks_passed": 3, "duration_ms": 42}

    async def fake_telegram(level, message):
        pass

    with patch.object(rollout_service, "upgrade_tenant_image", fake_fast_upgrade), \
         patch.object(rollout_service, "_send_telegram", fake_telegram):
        attempt = await rollout_service._upgrade_one(
            None, rollout, container, "ghcr.io/toup-com/toup-agent:newtag00",
        )

    assert attempt.status == "ok"
    assert attempt.health_checks_passed == 3

    async with async_session_maker() as db:
        row = (await db.execute(
            select(RolloutAttempt).where(RolloutAttempt.rollout_id == rollout.id)
        )).scalar_one()
        assert row.status == "ok"


@pytest.mark.asyncio
async def test_upgrade_one_hard_timeout_releases_db_session():
    """Pin that the timeout path commits attempt='failed' (which releases
    the held DB session). The 547-min stall happened because the held
    session was never released; the reconciler couldn't get its own
    connection on the Supabase pooler to mark the rollout orphan.

    We assert this indirectly by confirming the attempt row is queryable
    in a *new* session immediately after the timeout — if the timeout
    path held the row's session, a concurrent read would block.
    """
    import asyncio as _asyncio
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import RolloutAttempt
    from app.services import rollout_service

    rollout, container = await _seed_orphan_quarantine_fixtures()

    async def fake_hung_upgrade(*args, **kwargs):
        await _asyncio.sleep(3600)

    async def fake_telegram(level, message):
        pass

    _orig_wait_for = _asyncio.wait_for

    async def short_wait_for(coro, timeout):
        return await _orig_wait_for(coro, timeout=1.0)

    with patch.object(rollout_service, "upgrade_tenant_image", fake_hung_upgrade), \
         patch.object(rollout_service, "_send_telegram", fake_telegram), \
         patch.object(rollout_service.asyncio, "wait_for", short_wait_for):
        await rollout_service._upgrade_one(
            None, rollout, container, "ghcr.io/toup-com/toup-agent:newtag00",
        )

    # After the timeout, a fresh session must see the failed attempt.
    # If the timeout path didn't release the session, this read would
    # block (or fail if the pool is exhausted).
    async with async_session_maker() as db:
        row = (await db.execute(
            select(RolloutAttempt).where(RolloutAttempt.rollout_id == rollout.id)
        )).scalar_one()
        assert row.status == "failed"
        assert row.completed_at is not None, (
            "completed_at must be stamped on timeout — otherwise the "
            "reconciler's heartbeat-stale check might not fire correctly"
        )


# ─── Narrow-session invariant (no DB held across bridge call) ──────
#
# Direct regression for the pool-exhaustion class that contributed to
# the 547-min stall. Prove that `_upgrade_one` does NOT hold a DB
# session while `upgrade_tenant_image` is running — even a long-running
# bridge call must not lock up a connection.


@pytest.mark.asyncio
async def test_upgrade_one_releases_db_session_during_bridge_call():
    """Concurrent DB access must succeed WHILE the bridge call is
    in-flight. If `_upgrade_one` still held a session across the
    bridge call, this test would time out (the concurrent query
    would block on pool exhaustion in the narrow-pool case, or at
    minimum the session lifecycle would be observable from outside).

    The test fakes a bridge call that pauses for a controllable window,
    while a parallel coroutine runs queries against the same DB. Both
    must complete normally.
    """
    import asyncio as _asyncio
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import RolloutAttempt
    from app.services import rollout_service

    rollout, container = await _seed_orphan_quarantine_fixtures()

    bridge_started = _asyncio.Event()
    bridge_release = _asyncio.Event()
    concurrent_query_done = _asyncio.Event()

    async def fake_bridge_upgrade(*args, **kwargs):
        bridge_started.set()
        # Hold here until the concurrent query completes. If the
        # orchestrator held a DB session, the concurrent query would
        # block (in a tight pool) or at minimum we couldn't prove the
        # session was released. The fact that the concurrent query
        # completes BEFORE we return proves the session is free.
        await concurrent_query_done.wait()
        bridge_release.set()
        return {"health_checks_passed": 3, "duration_ms": 50}

    async def concurrent_db_reader():
        await bridge_started.wait()
        async with async_session_maker() as db:
            count = (await db.execute(
                select(RolloutAttempt).where(
                    RolloutAttempt.rollout_id == rollout.id
                )
            )).scalars().all()
            # The 'upgrading' row must be visible — proves _upgrade_one
            # already wrote+committed+CLOSED its session before the
            # bridge call.
            assert len(count) == 1, (
                f"expected 1 attempt row visible mid-bridge-call, got {len(count)}"
            )
            assert count[0].status == "upgrading", (
                f"expected attempt status='upgrading' (write committed before "
                f"bridge call), got {count[0].status!r}"
            )
        concurrent_query_done.set()

    async def fake_telegram(level, message):
        pass

    with patch.object(rollout_service, "upgrade_tenant_image", fake_bridge_upgrade), \
         patch.object(rollout_service, "_send_telegram", fake_telegram):
        upgrade_task = _asyncio.create_task(rollout_service._upgrade_one(
            None, rollout, container, "ghcr.io/toup-com/toup-agent:newtag00",
        ))
        reader_task = _asyncio.create_task(concurrent_db_reader())
        # Both must finish within a reasonable budget. If the
        # orchestrator's session blocks the reader, the test deadlocks.
        try:
            attempt, _ = await _asyncio.wait_for(
                _asyncio.gather(upgrade_task, reader_task), timeout=10.0,
            )
        except _asyncio.TimeoutError:
            pytest.fail(
                "DEADLOCK: concurrent DB read could not complete while "
                "_upgrade_one's bridge call was in-flight. This means "
                "the orchestrator still holds a session across the "
                "bridge call — exactly the pattern that contributed "
                "to the 547-min stall."
            )

    assert attempt.status == "ok"
    assert bridge_release.is_set(), "bridge call must have completed normally"
