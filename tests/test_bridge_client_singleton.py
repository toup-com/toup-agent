"""Regression tests for TKT-LAT-006 — bridge AsyncClient singleton.

Verifies that:
  * ``get_bridge_client()`` returns the same AsyncClient instance across
    calls (no new TLS+mTLS handshake per bridge request).
  * The ``_bridge_client()`` lease wrapper hands out the shared singleton
    when no timeout is supplied (and does NOT close it on context exit).
  * A custom ``timeout_s`` argument forces a fresh client that IS closed
    on context exit.
  * If the singleton is disposed (e.g. process-wide reset), a fresh one
    is built on the next call.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from app.services import docker_host_service as dhs


def _fake_build_client(timeout_s=None):
    """Stand-in for ``_build_bridge_client`` that needs no mTLS certs."""
    return httpx.AsyncClient(base_url="https://bridge.test")


@pytest.fixture(autouse=True)
async def _reset_singleton():
    """Drop the singleton + monkey-patch the builder to a cert-free stub."""
    await dhs._close_bridge_client_for_test()
    with patch.object(dhs, "_build_bridge_client", side_effect=_fake_build_client):
        yield
    await dhs._close_bridge_client_for_test()


async def test_get_bridge_client_returns_singleton():
    c1 = await dhs.get_bridge_client()
    c2 = await dhs.get_bridge_client()
    assert c1 is c2
    assert not c1.is_closed


async def test_lease_wrapper_default_uses_shared_singleton():
    c1 = await dhs.get_bridge_client()
    async with dhs._bridge_client() as leased:
        assert leased is c1
    # Lease exit must NOT close the shared client.
    assert not c1.is_closed


async def test_lease_wrapper_custom_timeout_builds_fresh_and_closes():
    shared = await dhs.get_bridge_client()
    async with dhs._bridge_client(timeout_s=5) as fresh:
        assert fresh is not shared
        assert not fresh.is_closed
    # Custom-timeout lease closes its private client on exit.
    assert fresh.is_closed
    # Shared singleton untouched.
    assert not shared.is_closed


async def test_singleton_rebuilds_after_close():
    c1 = await dhs.get_bridge_client()
    await c1.aclose()
    c2 = await dhs.get_bridge_client()
    assert c2 is not c1
    assert not c2.is_closed
