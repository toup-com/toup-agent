import os, asyncio
os.environ.setdefault("ENVIRONMENT", "test")
import pytest
from unittest.mock import patch, AsyncMock


def test_healthy_pool_claim_does_not_reprovision():
    """If the claimed pool member is reachable, the guard returns fast and
    never calls provision_container (no wasteful recreate)."""
    from app.services import pool_service as ps
    with patch("app.services.prewarm_service._is_agent_actually_healthy",
               new=AsyncMock(return_value=True)) as health, \
         patch("app.services.docker_host_service.provision_container",
               new=AsyncMock()) as prov:
        asyncio.run(ps._verify_and_heal_pool_claim("u1234567", budget_s=5, interval_s=0.1))
    assert health.await_count >= 1
    assert prov.await_count == 0, "healthy claim must NOT be re-provisioned"


def test_unreachable_pool_claim_heals_via_recreate():
    """If the claimed pool member never becomes reachable within budget, the
    guard re-provisions via the reliable slow path (recreate=True)."""
    from app.services import pool_service as ps
    with patch("app.services.prewarm_service._is_agent_actually_healthy",
               new=AsyncMock(return_value=False)), \
         patch("app.services.docker_host_service.provision_container",
               new=AsyncMock()) as prov, \
         patch("app.db.database.async_session_maker") as smaker:
        smaker.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
        smaker.return_value.__aexit__ = AsyncMock(return_value=False)
        asyncio.run(ps._verify_and_heal_pool_claim("u1234567", budget_s=0.3, interval_s=0.1))
    assert prov.await_count == 1, "unreachable claim must heal via provision_container"
    # recreate=True is the proven heal
    assert prov.await_args.kwargs.get("recreate") is True


def test_guard_never_raises():
    """Fire-and-forget: any internal error is swallowed (the periodic
    reconciler is the durable backstop)."""
    from app.services import pool_service as ps
    with patch("app.services.prewarm_service._is_agent_actually_healthy",
               new=AsyncMock(side_effect=RuntimeError("boom"))):
        # must not raise
        asyncio.run(ps._verify_and_heal_pool_claim("u1234567", budget_s=0.3, interval_s=0.1))
