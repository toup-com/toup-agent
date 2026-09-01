import os, asyncio
os.environ.setdefault("ENVIRONMENT", "test")
import pytest
from unittest.mock import patch, AsyncMock
from datetime import datetime, timedelta, timezone


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


def _fake_session(container=None):
    """An async-context session whose `_load_container` lookup answers
    `container`. `pool_service._load_container` is patched by the callers
    below, so the session itself only has to be enterable."""
    smaker = patch("app.db.database.async_session_maker").start()
    smaker.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
    smaker.return_value.__aexit__ = AsyncMock(return_value=False)
    return smaker


def test_unreachable_pool_claim_is_restarted_IN_PLACE_first():
    """R40. This used to go straight to `provision_container(recreate=True)`.

    That call binds `toup_agent_<prefix>`, while a pool member's data lives in
    the slot's own `toup_agent_feedNNNN` — so the "reliable slow path" was also
    the path that silently emptied the account. A wedged member is usually just
    wedged; `/v1/pool/restart-member` fixes that and keeps the database.
    """
    from app.services import pool_service as ps

    class _MC:
        container_name = "toup-agent-pool-17"
        started_at = datetime.now(timezone.utc)

    with patch("app.services.prewarm_service._is_agent_actually_healthy",
               new=AsyncMock(return_value=False)), \
         patch("app.services.docker_host_service.provision_container",
               new=AsyncMock()) as prov, \
         patch("app.services.docker_host_service.restart_container",
               new=AsyncMock()) as restart, \
         patch("app.services.pool_service._load_container",
               new=AsyncMock(return_value=_MC())), \
         patch("app.db.database.async_session_maker") as smaker:
        smaker.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
        smaker.return_value.__aexit__ = AsyncMock(return_value=False)
        asyncio.run(ps._verify_and_heal_pool_claim("u1234567", budget_s=0.3, interval_s=0.1))

    assert restart.await_count == 1, "a wedged pool member must be restarted in place"
    # A FRESH claim may still fall through to the named recreate — at that
    # point the user has never sent anything, so there is nothing to strand.
    assert prov.await_count == 1
    assert prov.await_args.kwargs.get("recreate") is True
    assert prov.await_args.kwargs.get("allow_pool_swap") is True


def test_an_ESTABLISHED_pool_member_is_never_recreated():
    """The data-loss case. Same code path, one difference: the claim is old,
    so the slot database holds an account."""
    from app.services import pool_service as ps

    class _MC:
        container_name = "toup-agent-pool-17"
        started_at = datetime.now(timezone.utc) - timedelta(days=30)

    with patch("app.services.prewarm_service._is_agent_actually_healthy",
               new=AsyncMock(return_value=False)), \
         patch("app.services.docker_host_service.provision_container",
               new=AsyncMock()) as prov, \
         patch("app.services.docker_host_service.restart_container",
               new=AsyncMock()) as restart, \
         patch("app.services.pool_service._load_container",
               new=AsyncMock(return_value=_MC())), \
         patch("app.db.database.async_session_maker") as smaker:
        smaker.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
        smaker.return_value.__aexit__ = AsyncMock(return_value=False)
        asyncio.run(ps._verify_and_heal_pool_claim("u1234567", budget_s=0.3, interval_s=0.1))

    assert restart.await_count == 1, "it must still be restarted in place"
    assert prov.await_count == 0, (
        "an established pool member was re-provisioned through the named path — "
        "that binds an empty toup_agent_<prefix> and strands the slot's data"
    )


def test_a_NAMED_tenant_still_heals_the_old_way():
    """The guard must not have become pool-only: a named tenant that never
    came up is re-provisioned exactly as before."""
    from app.services import pool_service as ps

    class _MC:
        container_name = "toup-agent-u1234567"
        started_at = datetime.now(timezone.utc) - timedelta(days=30)

    with patch("app.services.prewarm_service._is_agent_actually_healthy",
               new=AsyncMock(return_value=False)), \
         patch("app.services.docker_host_service.provision_container",
               new=AsyncMock()) as prov, \
         patch("app.services.docker_host_service.restart_container",
               new=AsyncMock()) as restart, \
         patch("app.services.pool_service._load_container",
               new=AsyncMock(return_value=_MC())), \
         patch("app.db.database.async_session_maker") as smaker:
        smaker.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
        smaker.return_value.__aexit__ = AsyncMock(return_value=False)
        asyncio.run(ps._verify_and_heal_pool_claim("u1234567", budget_s=0.3, interval_s=0.1))

    assert restart.await_count == 0, "a named tenant needs no pool restart"
    assert prov.await_count == 1
    assert prov.await_args.kwargs.get("recreate") is True


def test_guard_never_raises():
    """Fire-and-forget: any internal error is swallowed (the periodic
    reconciler is the durable backstop)."""
    from app.services import pool_service as ps
    with patch("app.services.prewarm_service._is_agent_actually_healthy",
               new=AsyncMock(side_effect=RuntimeError("boom"))):
        # must not raise
        asyncio.run(ps._verify_and_heal_pool_claim("u1234567", budget_s=0.3, interval_s=0.1))
