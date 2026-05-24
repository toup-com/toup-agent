"""Pins that ``upgrade_tenant_image`` does NOT call /restart.

PR #104 added a "defensive restart" right after the bridge upgrade
to guard against a bridge bug where the image swap doesn't restart
the uvicorn process. Two production rollouts on 2026-05-24
(74d636db, ece2c2e2) aborted at the canary boot gate because the
restart kicked the process mid-window:

  ``canary observation failed: boot gate failed:
    only 1/3 consecutive 200s within 30s``

The boot gate (rollout_service._observe_canary_signal) needs 3
consecutive /agent/health 200s in 30s. The defensive restart broke
that contract. This file used to pin the restart's presence; now
it pins its ABSENCE so the regression can't sneak back in.

The stale-process issue the defensive restart was trying to solve
is now addressed by:
  - PR #105: ``POST /api/admin/users/<user_id>/subagent-spawning``
    flips the flag AND restarts in one operation, called explicitly
    by operators when they need eager-apply.
  - Manual ``POST /v1/tenants/<prefix>/restart`` on the bridge for
    one-off operator action.
"""
from __future__ import annotations

import httpx
import pytest

from app.services import docker_host_service as dhs


class _FakeResp:
    def __init__(self, status_code: int = 200, json_body: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._json = json_body or {}
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=None, response=None)


class _RecordingClient:
    def __init__(self, post_responses):
        self.calls = []
        self._responses = post_responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def post(self, path, json=None, **_):
        self.calls.append((path, json))
        for suffix, resp in self._responses.items():
            if path.endswith(suffix):
                return resp
        return _FakeResp(404, text=f"no fake configured for {path}")

    async def get(self, path, **_):
        return _FakeResp(404)


@pytest.fixture(autouse=True)
def _disable_blue_green(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "use_blue_green_rollouts", False, raising=False)


@pytest.fixture
def fake_db():
    class _Res:
        def scalar_one_or_none(self):
            return None

    class _DB:
        async def execute(self, _stmt):
            return _Res()

        async def commit(self):
            pass

    return _DB()


@pytest.mark.asyncio
async def test_upgrade_does_not_call_restart(fake_db, monkeypatch):
    """Regression guard for the canary-boot-gate race. After PR #104
    was reverted (this commit), ``upgrade_tenant_image`` MUST NOT
    POST ``/v1/tenants/<prefix>/restart``. If a future contributor
    re-adds it, the canary boot gate will start failing again."""
    upgrade_resp = _FakeResp(200, {"status": "ok"})
    client = _RecordingClient({"/upgrade": upgrade_resp})

    def _stub(*a, **kw):
        return client

    monkeypatch.setattr(dhs, "_bridge_client", _stub)

    await dhs.upgrade_tenant_image(
        db=fake_db,
        user_id="aabbccdd-1111-2222-3333-444444444444",
        image_tag="ghcr.io/toup-com/toup-agent:test",
        rollout_id="r-1",
    )

    paths = [c[0] for c in client.calls]
    assert "/v1/tenants/aabbccdd/restart" not in paths, (
        "upgrade_tenant_image MUST NOT call /restart after upgrade — "
        "it races with rollout_service's canary boot gate (3 "
        "consecutive 200s in 30s). PR #104 introduced this and "
        "caused two production rollouts to abort 2026-05-24."
    )
    # The upgrade call itself is still expected.
    assert "/v1/tenants/aabbccdd/upgrade" in paths


@pytest.mark.asyncio
async def test_upgrade_returns_normally(fake_db, monkeypatch):
    """The function still has to return the bridge's payload after
    a successful upgrade — that's its actual job, separate from
    the restart question."""
    upgrade_resp = _FakeResp(200, {"status": "ok", "duration_ms": 42})
    client = _RecordingClient({"/upgrade": upgrade_resp})

    def _stub(*a, **kw):
        return client

    monkeypatch.setattr(dhs, "_bridge_client", _stub)

    result = await dhs.upgrade_tenant_image(
        db=fake_db,
        user_id="aabbccdd-1111-2222-3333-444444444444",
        image_tag="ghcr.io/toup-com/toup-agent:test",
    )
    assert result == {"status": "ok", "duration_ms": 42}
