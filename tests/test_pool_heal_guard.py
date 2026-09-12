"""Containment pins for the September 12 pool/named split incident."""

import asyncio
import os
from unittest.mock import AsyncMock, patch

os.environ.setdefault("ENVIRONMENT", "test")


def test_healthy_claim_is_observed_without_mutation():
    from app.services import pool_service as ps

    with patch(
        "app.services.prewarm_service._is_agent_actually_healthy",
        new=AsyncMock(return_value=True),
    ) as health, patch(
        "app.services.docker_host_service.restart_container", new=AsyncMock()
    ) as restart, patch(
        "app.services.docker_host_service.provision_container", new=AsyncMock()
    ) as provision:
        asyncio.run(
            ps._verify_and_heal_pool_claim(
                "u1234567", budget_s=0.1, interval_s=0.01
            )
        )

    assert health.await_count == 1
    restart.assert_not_awaited()
    provision.assert_not_awaited()


def test_readiness_timeout_never_restarts_or_cold_swaps():
    """Slow DNS/TLS/boot is uncertainty, even for a seconds-old signup."""
    from app.services import pool_service as ps

    with patch(
        "app.services.prewarm_service._is_agent_actually_healthy",
        new=AsyncMock(return_value=False),
    ), patch(
        "app.services.docker_host_service.restart_container", new=AsyncMock()
    ) as restart, patch(
        "app.services.docker_host_service.provision_container", new=AsyncMock()
    ) as provision:
        asyncio.run(
            ps._verify_and_heal_pool_claim(
                "u1234567", budget_s=0.01, interval_s=0.02
            )
        )

    restart.assert_not_awaited()
    provision.assert_not_awaited()


def test_late_health_within_observation_window_still_never_mutates():
    from app.services import pool_service as ps

    with patch(
        "app.services.prewarm_service._is_agent_actually_healthy",
        new=AsyncMock(side_effect=[False, True]),
    ) as health, patch(
        "app.services.docker_host_service.restart_container", new=AsyncMock()
    ) as restart, patch(
        "app.services.docker_host_service.provision_container", new=AsyncMock()
    ) as provision:
        asyncio.run(
            ps._verify_and_heal_pool_claim(
                "u1234567", budget_s=0.1, interval_s=0
            )
        )

    assert health.await_count == 2
    restart.assert_not_awaited()
    provision.assert_not_awaited()


def test_guard_never_raises():
    from app.services import pool_service as ps

    with patch(
        "app.services.prewarm_service._is_agent_actually_healthy",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        asyncio.run(
            ps._verify_and_heal_pool_claim(
                "u1234567", budget_s=0.1, interval_s=0
            )
        )
