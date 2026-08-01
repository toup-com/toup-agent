"""The agent side of the search gateway: tier order, degrade, breaker.

WHY THIS EXISTS

app/api/search_proxy.py has been live and correct for a while, and
tests/test_search_gateway.py covers it thoroughly. It was also connected to
nothing: `/api/search/health` reported `tenants_seen: 0` because no agent ever
called it. The first attempt to switch the agent over (bd141177) was reverted
(a67e7407) after two canary rollouts failed to boot, and the agent kept using
the topology the gateway was built to replace — the shared Brave key as
BRAVE_API_KEY inside all 42 containers.

Measured before this change, on production:

  * 42/42 tenants hold connect_token AND llm_token_hash — the gateway
    credential is already universal, so wiring it needs no backfill.
  * 0 tenants hold their own brave_api_key, so nothing here is BYOK.
  * exactly 1 of 42 tenants had run a web_search in 30 days, all on
    tier=brave_api. That is NOT evidence the other 41 are healthy: a degraded
    search is never metered, so a keyless container leaves no row at all.
  * BRAVE_API_KEY is injected only at container-CREATE time
    (bridge/pool_addon.py), so a container older than the key never gets one
    until it is recreated.

This wiring is deliberately ADDITIVE — the gateway tier goes in FRONT of the
existing Brave tier and nothing was deleted. The reverted attempt removed the
whole in-container path in the same commit that added models, a migration and
config; this one adds a method and a tier, so the blast radius of a bad rollout
is one tool call falling through to the tier that already worked.

Run:
  pytest backend/tests/test_search_gateway_agent_tier.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.agent.tool_executor as TE  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Harness
# ──────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status=200, payload=None, text_body=None):
        self.status_code = status
        self._payload = payload
        self._text = text_body

    def json(self):
        if self._text is not None:
            raise ValueError("not json")
        return self._payload


class _Client:
    """Stand-in for httpx.AsyncClient used as an async context manager."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kw):
        self.calls.append((url, kw))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


@pytest.fixture(autouse=True)
def _reset_breaker():
    """Module-level breaker state leaks between tests otherwise."""
    TE._gw_fails = 0
    TE._gw_skip_until = 0.0
    TE._gateway_unconfigured_warned = False
    yield
    TE._gw_fails = 0
    TE._gw_skip_until = 0.0


def _executor():
    """A ToolExecutor without running __init__ — we only exercise one method."""
    ex = TE.ToolExecutor.__new__(TE.ToolExecutor)
    return ex


def _wire(monkeypatch, result, *, token="tok-abc", base="https://toup.ai/api"):
    client = _Client(result)
    monkeypatch.setattr(TE.settings, "toup_token", token, raising=False)
    monkeypatch.setattr(TE.settings, "platform_api_url", base, raising=False)
    monkeypatch.setattr(TE.httpx, "AsyncClient", lambda **kw: client)
    return client


_OK = {
    "served": True,
    "results": [
        {"title": "T1", "url": "https://a.example/1", "description": "d1",
         "extra_snippets": ["s1", "s2"]},
        {"title": "T2", "url": "https://b.example/2", "description": "d2"},
    ],
}


# ──────────────────────────────────────────────────────────────
# The happy path
# ──────────────────────────────────────────────────────────────

class TestServed:
    @pytest.mark.asyncio
    async def test_returns_formatted_results(self, monkeypatch):
        _wire(monkeypatch, _Resp(200, _OK))
        served, text, reason = await _executor()._gateway_search("q", 5)
        assert served is True and reason is None
        assert "1. T1" in text and "https://a.example/1" in text
        assert "s1" in text          # extra_snippets surfaced for snippet-first reasoning

    @pytest.mark.asyncio
    async def test_sends_bearer_token_not_a_provider_key(self, monkeypatch):
        c = _wire(monkeypatch, _Resp(200, _OK), token="tok-xyz")
        await _executor()._gateway_search("q", 5)
        hdrs = c.calls[0][1]["headers"]
        assert hdrs["Authorization"] == "Bearer tok-xyz"
        # The whole point: no provider credential ever leaves the platform.
        assert not any("subscription" in k.lower() for k in hdrs)

    @pytest.mark.asyncio
    async def test_posts_to_the_gateway_path(self, monkeypatch):
        c = _wire(monkeypatch, _Resp(200, _OK))
        await _executor()._gateway_search("q", 5)
        assert c.calls[0][0] == "https://toup.ai/api/search/web"

    @pytest.mark.asyncio
    async def test_missing_api_suffix_is_normalised(self, monkeypatch):
        """Tenants provisioned before the /api suffix was normalised would
        otherwise hit the SPA catch-all and get HTML with a 200."""
        c = _wire(monkeypatch, _Resp(200, _OK), base="https://toup.ai")
        await _executor()._gateway_search("q", 5)
        assert c.calls[0][0] == "https://toup.ai/api/search/web"


# ──────────────────────────────────────────────────────────────
# Degrade, never deny
# ──────────────────────────────────────────────────────────────

class TestDegradeNeverDeny:
    @pytest.mark.asyncio
    async def test_served_false_is_a_degrade_with_its_reason(self, monkeypatch):
        _wire(monkeypatch, _Resp(200, {"served": False, "degraded_reason": "fleet_headroom"}))
        served, text, reason = await _executor()._gateway_search("q", 5)
        assert served is False and text is None and reason == "fleet_headroom"

    @pytest.mark.asyncio
    async def test_empty_results_is_a_degrade(self, monkeypatch):
        _wire(monkeypatch, _Resp(200, {"served": True, "results": []}))
        served, _, reason = await _executor()._gateway_search("q", 5)
        assert served is False and reason == "empty_result"

    @pytest.mark.asyncio
    async def test_transport_error_never_raises(self, monkeypatch):
        _wire(monkeypatch, RuntimeError("connection reset"))
        served, _, reason = await _executor()._gateway_search("q", 5)
        assert served is False and reason.startswith("gateway_error:")

    @pytest.mark.asyncio
    async def test_html_body_is_caught(self, monkeypatch):
        _wire(monkeypatch, _Resp(200, None, text_body="<!doctype html>"))
        served, _, reason = await _executor()._gateway_search("q", 5)
        assert served is False and reason == "gateway_bad_body"

    @pytest.mark.asyncio
    async def test_no_token_returns_none_not_an_error(self, monkeypatch):
        """None means 'not configured here', which the caller treats the same
        way as a decline: fall to the next tier."""
        _wire(monkeypatch, _Resp(200, _OK), token="")
        assert await _executor()._gateway_search("q", 5) is None

    @pytest.mark.asyncio
    async def test_no_retry_on_a_decline(self, monkeypatch):
        """A client that retries into a throttle is what the gateway exists to
        prevent. The lower tiers are the retry."""
        c = _wire(monkeypatch, _Resp(200, {"served": False, "degraded_reason": "tenant_rate_limit"}))
        await _executor()._gateway_search("q", 5)
        assert len(c.calls) == 1


# ──────────────────────────────────────────────────────────────
# The breaker
# ──────────────────────────────────────────────────────────────

class TestBreaker:
    @pytest.mark.asyncio
    async def test_opens_after_consecutive_transport_failures(self, monkeypatch):
        _wire(monkeypatch, RuntimeError("boom"))
        assert TE._gateway_allowed() is True
        for _ in range(TE._GW_FAIL_MAX):
            await _executor()._gateway_search("q", 5)
        assert TE._gateway_allowed() is False

    @pytest.mark.asyncio
    async def test_a_decline_does_NOT_open_it(self, monkeypatch):
        """200/served=false is the gateway working as designed. If throttling
        opened the breaker, a busy minute would disable fast search for the
        next one — the opposite of what the tier is for."""
        _wire(monkeypatch, _Resp(200, {"served": False, "degraded_reason": "tenant_rate_limit"}))
        for _ in range(TE._GW_FAIL_MAX * 3):
            await _executor()._gateway_search("q", 5)
        assert TE._gateway_allowed() is True

    @pytest.mark.asyncio
    async def test_success_resets_the_counter(self, monkeypatch):
        _wire(monkeypatch, RuntimeError("boom"))
        for _ in range(TE._GW_FAIL_MAX - 1):
            await _executor()._gateway_search("q", 5)
        _wire(monkeypatch, _Resp(200, _OK))
        await _executor()._gateway_search("q", 5)
        assert TE._gw_fails == 0
        _wire(monkeypatch, RuntimeError("boom"))
        for _ in range(TE._GW_FAIL_MAX - 1):
            await _executor()._gateway_search("q", 5)
        assert TE._gateway_allowed() is True   # counter really did reset

    @pytest.mark.asyncio
    async def test_401_opens_it(self, monkeypatch):
        """A token that does not resolve will not start resolving by itself."""
        _wire(monkeypatch, _Resp(401, {}))
        for _ in range(TE._GW_FAIL_MAX):
            served, _, reason = await _executor()._gateway_search("q", 5)
            assert reason == "gateway_unauthorized"
        assert TE._gateway_allowed() is False

    @pytest.mark.asyncio
    async def test_4xx_that_is_our_bug_does_not_open_it(self, monkeypatch):
        """A 422 is a bad request from this side. Skipping the tier for a
        minute would hide the bug instead of surfacing it."""
        _wire(monkeypatch, _Resp(422, {}))
        for _ in range(TE._GW_FAIL_MAX * 2):
            await _executor()._gateway_search("q", 5)
        assert TE._gateway_allowed() is True

    @pytest.mark.asyncio
    async def test_5xx_opens_it(self, monkeypatch):
        _wire(monkeypatch, _Resp(503, {}))
        for _ in range(TE._GW_FAIL_MAX):
            await _executor()._gateway_search("q", 5)
        assert TE._gateway_allowed() is False


# ──────────────────────────────────────────────────────────────
# Tier order and metering — asserted against the real source
# ──────────────────────────────────────────────────────────────

class TestTierWiring:
    @pytest.fixture(scope="class")
    def src(self):
        return (Path(TE.__file__)).read_text()

    def test_gateway_tier_runs_before_the_container_key(self, src):
        gw = src.index("if _gateway_allowed():")
        brave = src.index("if settings.brave_api_key:")
        assert gw < brave, "the gateway must be tried before the in-container key"

    def test_the_container_key_path_still_exists(self, src):
        """Additive on purpose. Removing both the key and the gateway in one
        step would leave a tenant with no fast search if the gateway is down —
        and that is what the reverted attempt did."""
        assert "_brave_search_fallback" in src
        assert hasattr(TE.ToolExecutor, "_brave_search_fallback")

    def test_a_served_gateway_result_is_not_metered_again(self, src):
        """search_proxy writes its own row from its own observation. Metering
        here as well would double-count every search in the ledger."""
        i = src.index("if _gateway_allowed():")
        j = src.index("if settings.brave_api_key:")
        block = src[i:j]
        assert "_meter_web_tool" not in block

    def test_a_served_gateway_result_is_cached(self, src):
        i = src.index("if _gateway_allowed():")
        j = src.index("if settings.brave_api_key:")
        assert "cache_set" in src[i:j]

    def test_gateway_timeout_is_tighter_than_the_direct_client(self, src):
        """This tier runs first, so its timeout is latency every search pays
        before falling through."""
        assert "httpx.AsyncClient(timeout=8.0)" in src
        assert "httpx.AsyncClient(timeout=15)" in src   # the direct path, unchanged
