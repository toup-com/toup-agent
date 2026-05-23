"""Defensive container restart after ``upgrade_tenant_image``.

Bridge bug observed 2026-05-23: ``/v1/tenants/<prefix>/upgrade`` pulls
the new image and updates bind-mounted code but does NOT restart the
uvicorn process. Audit across 4 production tenants showed processes
ranging from 27s to 1d11h old while their ``managed_containers
.image_tag`` column reported the new tag. The platform's defensive
fix: always POST ``/restart`` after a successful upgrade, idempotent
and quick.

These tests pin three behaviours:

  1. After a successful upgrade, ``/restart`` is called against the
     same tenant prefix.
  2. A 200 from ``/upgrade`` followed by a 5xx from ``/restart`` does
     NOT fail the upgrade — the image swap already happened; the
     restart is belt-and-suspenders.
  3. A network exception during ``/restart`` is swallowed and logged,
     never raised, for the same reason.

Drift in any of these surfaces a real production-impact regression.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.services import docker_host_service as dhs


# ─── Helpers ──────────────────────────────────────────────────────────


class _FakeResp:
    """Minimal httpx-Response stand-in. ``status_code`` is settable;
    everything else returns reasonable defaults."""

    def __init__(self, status_code: int = 200, json_body: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._json = json_body or {}
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=None,
            )


class _RecordingClient:
    """Captures every ``post()`` call onto ``calls`` so tests can
    assert the order of bridge endpoint hits."""

    def __init__(self, post_responses: dict[str, _FakeResp]):
        self.calls: list[tuple[str, dict | None]] = []
        self._responses = post_responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def post(self, path: str, json: dict | None = None, **_):
        self.calls.append((path, json))
        # Match by path suffix so tests don't have to know the full
        # /v1/tenants/<prefix>/ prefix construction.
        for suffix, resp in self._responses.items():
            if path.endswith(suffix):
                return resp
        return _FakeResp(404, text=f"no fake configured for {path}")

    async def get(self, path: str, **_):
        return _FakeResp(404, text="whois not configured")


@pytest.fixture(autouse=True)
def _disable_blue_green(monkeypatch):
    """Legacy /upgrade path keeps the test surface small. Blue-green
    drift is covered by separate tests."""
    from app.config import settings
    monkeypatch.setattr(settings, "use_blue_green_rollouts", False, raising=False)


@pytest.fixture
def fake_db():
    """Stand-in AsyncSession that returns no ManagedContainer row —
    upgrade_tenant_image only writes to it for record-keeping, and
    the unit test doesn't need that side-effect."""

    class _Res:
        def scalar_one_or_none(self):
            return None

    class _DB:
        async def execute(self, _stmt):
            return _Res()

        async def commit(self):
            pass

    return _DB()


# ─── Tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upgrade_then_defensive_restart_is_called(fake_db, monkeypatch):
    """Happy path: bridge /upgrade returns 200; defensive /restart
    fires against the same prefix."""
    upgrade_resp = _FakeResp(200, {"status": "ok"})
    restart_resp = _FakeResp(200, {"status": "restarting"})
    client = _RecordingClient({
        "/upgrade": upgrade_resp,
        "/restart": restart_resp,
    })

    def _stub(*a, **kw):
        return client
    monkeypatch.setattr(dhs, "_bridge_client", _stub)

    await dhs.upgrade_tenant_image(
        db=fake_db,
        user_id="aabbccdd-1111-2222-3333-444444444444",
        image_tag="ghcr.io/toup-com/toup-agent:test",
        rollout_id="r-1",
    )

    # Two bridge hits in order: upgrade then restart, both against
    # the 8-char prefix of the user_id.
    paths = [c[0] for c in client.calls]
    assert paths == [
        "/v1/tenants/aabbccdd/upgrade",
        "/v1/tenants/aabbccdd/restart",
    ], f"expected upgrade then restart; got {paths!r}"


@pytest.mark.asyncio
async def test_restart_failure_does_not_fail_upgrade(fake_db, monkeypatch):
    """Bridge /upgrade 200 + /restart 500: upgrade_tenant_image MUST
    return normally. The image swap already succeeded; failing the
    upgrade here would trigger a needless rollback for what is
    only a stale-process risk."""
    client = _RecordingClient({
        "/upgrade": _FakeResp(200, {"status": "ok"}),
        "/restart": _FakeResp(500, text="bridge restart broken"),
    })

    def _stub(*a, **kw):
        return client
    monkeypatch.setattr(dhs, "_bridge_client", _stub)

    # Should NOT raise. If this raises, the upgrade was marked failed
    # and the rollout would have rolled back unnecessarily.
    result = await dhs.upgrade_tenant_image(
        db=fake_db,
        user_id="aabbccdd-1111-2222-3333-444444444444",
        image_tag="ghcr.io/toup-com/toup-agent:test",
    )
    assert result == {"status": "ok"}
    # /restart WAS still called even though it failed — operator-
    # visible warning in logs is enough.
    assert any(p.endswith("/restart") for p, _ in client.calls)


@pytest.mark.asyncio
async def test_restart_network_exception_is_swallowed(fake_db, monkeypatch):
    """Bridge unreachable during /restart (timeout, ConnectError,
    etc): upgrade_tenant_image MUST return normally. Same reasoning
    as the 5xx case — the upgrade already succeeded."""

    class _ExplodingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def post(self, path, json=None, **_):
            if path.endswith("/restart"):
                raise httpx.ConnectError("bridge unreachable")
            return _FakeResp(200, {"status": "ok"})

        async def get(self, *a, **kw):
            return _FakeResp(404)

    def _stub(*a, **kw):
        return _ExplodingClient()
    monkeypatch.setattr(dhs, "_bridge_client", _stub)

    result = await dhs.upgrade_tenant_image(
        db=fake_db,
        user_id="aabbccdd-1111-2222-3333-444444444444",
        image_tag="ghcr.io/toup-com/toup-agent:test",
    )
    assert result == {"status": "ok"}
