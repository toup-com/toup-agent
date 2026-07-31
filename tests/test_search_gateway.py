"""Tests for the platform search gateway (`app/api/search_proxy.py`).

The gateway is now the ONLY holder of the fleet's Brave key, so three of its
properties are load-bearing for every tenant at once and each has a section
below:

  * **It degrades, it does not deny.** A shed request must come back 200 with
    ``served=false`` + a ``degraded_reason``. The agent's ladder
    (`tool_executor._gateway_search`) reads exactly that pair to decide whether
    to fall to tiers 2/3. A 4xx/5xx from here turns a rate limit — a
    latency problem — into a user-visible failure.
  * **A bad token is 401, never 404.** A 404 would let a leaked-token probe
    enumerate which tokens exist.
  * **No part of the key reaches a caller.** It is one secret for 42 tenants
    now, so a leak is a fleet-wide rotation.

No network: the Brave upstream is faked at the httpx transport layer
(`httpx.AsyncHTTPTransport.handle_async_request`), which is the transport under
the gateway's own client but NOT under the ASGI test client — the same idiom
`test_analyze_image_proxy` uses.

Run: cd backend && python3 -m pytest tests/test_search_gateway.py -q
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Optional

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

import app.api.search_proxy as sp


# ── Fixtures ──────────────────────────────────────────────────────────


_TEST_KEY = "brv-SECRET-0123456789abcdef"


@pytest.fixture(autouse=True)
def _isolate_gateway_state(monkeypatch):
    """Give each test its own token cache, tenant buckets, fleet guard and key.

    The first three are module-level process state — a tripped breaker or a
    drained bucket would otherwise leak into every test that ran after it. The
    route reads each through the module global at call time, so rebinding the
    name is enough. `sp.settings` is its own Settings instance (config.py's
    get_settings() constructs a new one per call), so patching it here does not
    touch anything else's view of config.
    """
    monkeypatch.setattr(sp, "_TOKEN_CACHE", {})
    monkeypatch.setattr(sp, "_buckets", {})
    monkeypatch.setattr(sp, "_fleet", sp._FleetGuard())
    monkeypatch.setattr(sp.settings, "brave_api_key", _TEST_KEY)


class _FakeBrave:
    """Programmable stand-in for api.search.brave.com."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.status = 200
        self.payload: Optional[dict] = _brave_payload(3)
        self.text: Optional[str] = None
        self.headers: dict[str, str] = {"x-ratelimit-remaining": "49, 0"}
        self.raise_exc: Optional[Exception] = None

    async def serve(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        # A real call is never instantaneous and `latency_ms` is asserted below;
        # without this the measured latency floors to 0 on a fast machine.
        await asyncio.sleep(0.005)
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.text is not None:
            return httpx.Response(
                self.status, text=self.text, headers=self.headers, request=request
            )
        return httpx.Response(
            self.status, json=self.payload, headers=self.headers, request=request
        )


def _brave_payload(count: int = 3, *, snippets: Optional[list[str]] = None) -> dict:
    return {
        "web": {
            "results": [
                {
                    "title": f"Result {i}",
                    "url": f"https://example.com/{i}",
                    "description": f"Description {i}",
                    "extra_snippets": list(snippets) if snippets is not None else [],
                }
                for i in range(1, count + 1)
            ]
        }
    }


@pytest.fixture
def brave(monkeypatch) -> _FakeBrave:
    fake = _FakeBrave()

    async def _handle(transport_self, request):
        return await fake.serve(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _handle)
    return fake


@asynccontextmanager
async def _gateway_client():
    """The gateway router alone — conftest's shared test app doesn't mount it."""
    app = FastAPI()
    app.include_router(sp.router, prefix="/api")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def gw() -> AsyncClient:
    async with _gateway_client() as c:
        yield c


async def _make_tenant() -> tuple[str, str]:
    """A user + an AgentConfig whose llm_token_hash matches a fresh TOUP_TOKEN.

    Same credential the LLM proxy authenticates — there is no search-specific
    secret, which is the whole point of the topology.
    """
    from app.db import AgentConfig, User, async_session_maker
    from app.services.auth_service import get_password_hash

    user_id = str(uuid.uuid4())
    token = f"toup-tok-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"gw-{uuid.uuid4().hex[:12]}@example.com",
            hashed_password=get_password_hash("test-password-1234"),
            name="Gateway Test User",
        ))
        db.add(AgentConfig(
            user_id=user_id,
            llm_token_hash=hashlib.sha256(token.encode()).hexdigest(),
            bundle_status="active",
        ))
        await db.commit()
    return user_id, token


@pytest_asyncio.fixture
async def tenant() -> tuple[str, str]:
    return await _make_tenant()


async def _events(user_id: Optional[str] = None) -> list:
    from app.db import async_session_maker
    from app.db.models import SearchEvent

    stmt = select(SearchEvent)
    if user_id is not None:
        stmt = stmt.where(SearchEvent.user_id == user_id)
    async with async_session_maker() as db:
        return list((await db.execute(stmt)).scalars().all())


async def _search(gw: AsyncClient, token: str, query: str = "who is ada lovelace", **body):
    return await gw.post(
        "/api/search/web",
        json={"query": query, **body},
        headers={"Authorization": f"Bearer {token}"},
    )


# ── Auth ──────────────────────────────────────────────────────────────


async def test_missing_authorization_is_401(gw, brave):
    res = await gw.post("/api/search/web", json={"query": "hello"})
    assert res.status_code == 401, res.text
    assert not brave.requests, "an unauthenticated call must never reach Brave"


async def test_bad_token_is_401_never_404(gw, brave):
    """404 would confirm-or-deny a token to a probe; 401 tells it nothing."""
    res = await gw.post(
        "/api/search/web",
        json={"query": "hello"},
        headers={"Authorization": "Bearer toup-tok-not-a-real-token"},
    )
    assert res.status_code == 401, res.text
    assert res.status_code not in (404, 405)
    assert not brave.requests


async def test_non_bearer_scheme_is_401(gw):
    res = await gw.post(
        "/api/search/web",
        json={"query": "hello"},
        headers={"Authorization": "Basic toup-tok-whatever"},
    )
    assert res.status_code == 401, res.text


async def test_valid_token_attributes_the_event_to_its_own_tenant(gw, brave):
    a_user, a_token = await _make_tenant()
    b_user, b_token = await _make_tenant()

    assert (await _search(gw, a_token)).status_code == 200
    assert (await _search(gw, b_token)).status_code == 200

    assert [e.user_id for e in await _events(a_user)] == [a_user]
    assert [e.user_id for e in await _events(b_user)] == [b_user]


async def test_token_lookup_is_cached_within_ttl(gw, brave, tenant):
    """`llm_token_hash` carries no index and one turn fans several searches out
    in parallel, so the second lookup must not hit the DB.

    Proven by mutating the row out from under the cache: the second call still
    resolves, and only stops resolving once the cache is cleared.
    """
    user_id, token = tenant
    assert (await _search(gw, token)).json()["served"] is True

    from app.db import AgentConfig, async_session_maker
    async with async_session_maker() as db:
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == user_id)
        )).scalar_one()
        cfg.llm_token_hash = hashlib.sha256(b"rotated").hexdigest()
        await db.commit()

    cached = await _search(gw, token)
    assert cached.status_code == 200, cached.text
    assert cached.json()["served"] is True
    assert len(await _events(user_id)) == 2

    sp._TOKEN_CACHE.clear()
    assert (await _search(gw, token)).status_code == 401


def test_token_cache_cap_is_a_runaway_guard():
    """The cap clears wholesale rather than evicting — one entry per tenant
    means it should never be reached, so it must not silently start thrashing
    a real eviction policy nobody wrote."""
    for i in range(sp._TOKEN_CACHE_MAX):
        sp._cache_put_user(f"hash-{i}", f"user-{i}")
    assert len(sp._TOKEN_CACHE) == sp._TOKEN_CACHE_MAX

    sp._cache_put_user("hash-overflow", "user-overflow")
    assert len(sp._TOKEN_CACHE) == 1
    assert sp._cache_get_user("hash-overflow") == "user-overflow"


# ── Degrade, never deny ───────────────────────────────────────────────
#
# The single most important contract in this file. Every one of these is a
# condition that must cost the user latency (a lower tier answers) and never
# an error.


@pytest.mark.parametrize("setup,expected_reason,reaches_brave", [
    ("tenant_bucket_drained", "tenant_rate_limit", False),
    ("fleet_cooldown", "cooldown_after_429", False),
    ("brave_429", "http_429", True),
    ("brave_500", "upstream_error", True),
    ("brave_unreachable", "upstream_error", True),
    ("brave_empty", "empty_result", True),
    ("no_platform_key", "unconfigured", False),
])
async def test_every_failure_mode_is_200_with_a_reason(
    gw, brave, tenant, monkeypatch, setup, expected_reason, reaches_brave,
):
    _user_id, token = tenant
    priming_calls = 0

    if setup == "tenant_bucket_drained":
        monkeypatch.setattr(sp.settings, "brave_burst", 1)
        monkeypatch.setattr(sp.settings, "brave_rate_per_sec", 0.01)
        assert (await _search(gw, token)).json()["served"] is True
        priming_calls = 1
    elif setup == "fleet_cooldown":
        sp._fleet.trip(30.0)
    elif setup == "brave_429":
        brave.status = 429
    elif setup == "brave_500":
        brave.status = 503
    elif setup == "brave_unreachable":
        brave.raise_exc = httpx.ConnectError("no route to host")
    elif setup == "brave_empty":
        brave.payload = {"web": {"results": []}}
    elif setup == "no_platform_key":
        monkeypatch.setattr(sp.settings, "brave_api_key", "")

    res = await _search(gw, token)

    assert res.status_code == 200, f"{setup} must degrade, not fail: {res.text}"
    body = res.json()
    assert body["served"] is False
    assert body["degraded_reason"] == expected_reason
    assert body["results"] == []
    assert body["tier"] == sp.TIER_BRAVE
    # Non-vacuity: the shed cases must be shed BEFORE the outbound call, and
    # the upstream cases must actually have made one.
    assert len(brave.requests) == priming_calls + (1 if reaches_brave else 0)


async def test_malformed_upstream_json_degrades_rather_than_500s(gw, brave, tenant):
    """A 200 carrying HTML (a CDN interstitial) must not raise out of the route."""
    _user_id, token = tenant
    brave.text = "<html>we are having trouble</html>"

    res = await _search(gw, token)
    assert res.status_code == 200, res.text
    assert res.json()["degraded_reason"] == "empty_result"


async def test_a_dead_database_still_serves_the_search(gw, brave, tenant, monkeypatch):
    """Metering must never take a user's search away from them. Telemetry and
    the credit write are both fail-open — loudly, but open."""
    from sqlalchemy.ext.asyncio import AsyncSession

    _user_id, token = tenant

    async def _explode(self, *args, **kwargs):
        raise RuntimeError("database is gone")

    monkeypatch.setattr(AsyncSession, "flush", _explode)

    res = await _search(gw, token)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["served"] is True
    assert len(body["results"]) == 3


async def test_unconfigured_key_is_not_charged_to_a_tenant(gw, brave, tenant, monkeypatch):
    """A missing platform key is our misconfiguration, not the tenant's usage —
    recording it against them would poison the per-user series."""
    user_id, token = tenant
    monkeypatch.setattr(sp.settings, "brave_api_key", "")

    assert (await _search(gw, token)).json()["degraded_reason"] == "unconfigured"
    assert await _events(user_id) == []
    assert not brave.requests


# ── Rate limiting ─────────────────────────────────────────────────────


async def test_burst_is_allowed_then_the_next_call_is_shed(gw, brave, tenant, monkeypatch):
    _user_id, token = tenant
    monkeypatch.setattr(sp.settings, "brave_burst", 3)
    monkeypatch.setattr(sp.settings, "brave_rate_per_sec", 0.01)

    for i in range(3):
        assert (await _search(gw, token)).json()["served"] is True, f"call {i} shed early"

    shed = (await _search(gw, token)).json()
    assert shed["served"] is False
    assert shed["degraded_reason"] == "tenant_rate_limit"
    assert len(brave.requests) == 3, "a shed call must not spend fleet quota"


async def test_tenants_have_independent_buckets(gw, brave, monkeypatch):
    """A shared bucket would let one runaway loop throttle all 42 tenants."""
    monkeypatch.setattr(sp.settings, "brave_burst", 2)
    monkeypatch.setattr(sp.settings, "brave_rate_per_sec", 0.01)

    _a_user, a_token = await _make_tenant()
    _b_user, b_token = await _make_tenant()

    for _ in range(2):
        assert (await _search(gw, a_token)).json()["served"] is True
    assert (await _search(gw, a_token)).json()["degraded_reason"] == "tenant_rate_limit"

    assert (await _search(gw, b_token)).json()["served"] is True, (
        "tenant B was throttled by tenant A's traffic — the buckets are shared"
    )


async def test_token_bucket_refills_over_time():
    bucket = sp._TokenBucket(rate_per_sec=100.0, burst=1)
    assert bucket.take() is True
    assert bucket.take() is False

    await asyncio.sleep(0.05)
    assert bucket.take() is True, "tokens never refilled — the bucket is a one-shot fuse"


def test_token_bucket_never_exceeds_its_burst():
    bucket = sp._TokenBucket(rate_per_sec=1000.0, burst=2)
    bucket._last -= 10.0  # ten seconds of accrual at 1000/s
    assert bucket.take() and bucket.take()
    assert bucket.take() is False


# ── Fleet guard ───────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("49, 0", 49),        # Brave's real two-bucket shape: per-second, per-month
    ("12", 12),           # single bucket
    (" 7 , 0 ", 7),       # whitespace
    ("0, 0", 0),
])
def test_fleet_guard_parses_the_remaining_header(raw, expected):
    guard = sp._FleetGuard()
    assert guard.observe({"x-ratelimit-remaining": raw}) == expected
    assert guard.remaining == expected


@pytest.mark.parametrize("headers", [
    {},                                    # header absent
    None,                                  # no headers at all
    {"x-ratelimit-remaining": ""},         # present but empty
    {"x-ratelimit-remaining": "unknown"},  # not a number
    {"x-ratelimit-remaining": ", 0"},      # leading bucket missing
])
def test_fleet_guard_survives_a_missing_or_malformed_header(headers):
    """Brave changing this header must never take search down; an unreadable
    reading is simply no reading."""
    guard = sp._FleetGuard()
    assert guard.observe(headers) is None
    assert guard.remaining is None
    assert guard.allowed() == (True, None)


def test_fleet_guard_sheds_below_the_floor():
    guard = sp._FleetGuard()
    guard.observe({"x-ratelimit-remaining": str(sp.settings.brave_fleet_floor)})
    assert guard.allowed() == (False, "fleet_headroom")

    guard.observe({"x-ratelimit-remaining": str(sp.settings.brave_fleet_floor + 1)})
    assert guard.allowed() == (True, None)


def test_fleet_guard_ignores_a_stale_reading():
    """The window is one second, so a reading from six seconds ago says nothing
    about now. If it still counted, one quiet moment at the floor would block
    the fleet until traffic resumed — and traffic can't resume while blocked."""
    guard = sp._FleetGuard()
    guard.observe({"x-ratelimit-remaining": "0, 0"})
    assert guard.allowed()[0] is False

    guard._seen_at -= 6.0
    assert guard.allowed() == (True, None)


async def test_low_headroom_sheds_the_next_call(gw, brave, tenant):
    _user_id, token = tenant
    brave.headers["x-ratelimit-remaining"] = "1, 0"

    assert (await _search(gw, token)).json()["served"] is True

    shed = (await _search(gw, token)).json()
    assert shed["served"] is False
    assert shed["degraded_reason"] == "fleet_headroom"
    assert len(brave.requests) == 1


async def test_fleet_shed_does_not_burn_tenant_tokens(gw, brave, tenant, monkeypatch):
    """A fleet-wide condition must not also drain the tenant's own budget, or a
    busy fleet would leave every tenant throttled after it recovers."""
    _user_id, token = tenant
    monkeypatch.setattr(sp.settings, "brave_burst", 2)
    monkeypatch.setattr(sp.settings, "brave_rate_per_sec", 0.01)
    sp._fleet.trip(30.0)

    for _ in range(3):
        assert (await _search(gw, token)).json()["degraded_reason"] == "cooldown_after_429"

    monkeypatch.setattr(sp, "_fleet", sp._FleetGuard())
    for i in range(2):
        assert (await _search(gw, token)).json()["served"] is True, (
            f"call {i} was throttled — the fleet shed spent tenant tokens"
        )


# ── Breaker ───────────────────────────────────────────────────────────


async def test_429_trips_the_breaker_for_everyone(gw, brave):
    """Brave's ceiling is one account limit, so a 429 seen by one tenant is
    information about the whole fleet."""
    _a_user, a_token = await _make_tenant()
    _b_user, b_token = await _make_tenant()

    brave.status = 429
    assert (await _search(gw, a_token)).json()["degraded_reason"] == "http_429"

    brave.status = 200  # upstream is healthy again; the breaker is not
    shed = (await _search(gw, b_token)).json()
    assert shed["served"] is False
    assert shed["degraded_reason"] == "cooldown_after_429"
    assert len(brave.requests) == 1, "the breaker must stop the call, not just label it"


async def test_breaker_releases_after_the_cooldown(gw, brave, tenant, monkeypatch):
    _user_id, token = tenant
    # `trip()` floors at 1s, so this is the shortest real cooldown there is.
    monkeypatch.setattr(sp.settings, "brave_cooldown_s", 0.5)

    brave.status = 429
    assert (await _search(gw, token)).json()["degraded_reason"] == "http_429"
    brave.status = 200
    assert (await _search(gw, token)).json()["degraded_reason"] == "cooldown_after_429"

    await asyncio.sleep(1.05)
    assert (await _search(gw, token)).json()["served"] is True, (
        "the breaker never released — search is off for the fleet until restart"
    )


# ── Telemetry ─────────────────────────────────────────────────────────


async def test_served_search_writes_exactly_one_ok_row(gw, brave, tenant):
    user_id, token = tenant
    brave.payload = _brave_payload(4)

    res = await _search(gw, token, query="Ada Lovelace", channel="voice", count=4)
    assert res.json()["served"] is True

    rows = await _events(user_id)
    assert len(rows) == 1, f"expected one row, got {len(rows)}"
    row = rows[0]
    assert row.status == sp.ST_OK
    assert row.was_fallback is False
    assert row.degraded_reason is None
    assert row.tier == sp.TIER_BRAVE
    assert row.engine == "brave"
    assert row.channel == "voice"
    assert row.result_count == 4
    assert row.latency_ms >= 1, "latency floored to 0 — the timer is not measuring the call"
    assert row.brave_remaining == 49


async def test_shed_search_is_still_recorded(gw, brave, tenant, monkeypatch):
    """Usage that was shed has to stay visible, or the throttle rate — the one
    number that says whether the limits are set right — is unobservable."""
    user_id, token = tenant
    monkeypatch.setattr(sp.settings, "brave_burst", 1)
    monkeypatch.setattr(sp.settings, "brave_rate_per_sec", 0.01)

    await _search(gw, token)
    assert (await _search(gw, token)).json()["degraded_reason"] == "tenant_rate_limit"

    rows = await _events(user_id)
    assert len(rows) == 2
    shed = [r for r in rows if r.status == sp.ST_THROTTLED]
    assert len(shed) == 1
    assert shed[0].was_fallback is True
    assert shed[0].degraded_reason == "tenant_rate_limit"
    assert shed[0].result_count == 0
    assert shed[0].latency_ms == 0


async def test_upstream_error_is_recorded_as_error_not_throttled(gw, brave, tenant):
    user_id, token = tenant
    brave.status = 503

    await _search(gw, token)
    rows = await _events(user_id)
    assert len(rows) == 1
    assert rows[0].status == sp.ST_ERROR
    assert rows[0].was_fallback is True
    assert rows[0].degraded_reason == "upstream_error"


async def test_query_is_hashed_not_stored(gw, brave, tenant):
    """A search query is the most sensitive string a private agent handles. The
    hash exists only to spot a runaway loop repeating one query."""
    user_id, token = tenant
    query = "my landlord's home address"

    await _search(gw, token, query=query)
    row = (await _events(user_id))[0]

    assert len(row.query_sha256) == 16
    int(row.query_sha256, 16)  # hex, or this raises
    assert query not in row.query_sha256
    for word in query.split():
        assert word not in row.query_sha256


def test_query_hash_normalizes_but_still_separates():
    assert sp._query_hash("  Ada   LOVELACE ") == sp._query_hash("ada lovelace")
    assert sp._query_hash("ada lovelace") != sp._query_hash("ada lovelace jr")
    assert len(sp._query_hash("x")) == 16


async def test_served_search_meters_credits_in_dry_run(gw, brave, tenant):
    """`web_tool_metering_charge` defaults False, so the usage series exists
    before any money moves — the row must still carry the quoted credits."""
    from app.services.credit_service import FLAT_FEES

    user_id, token = tenant
    assert sp.settings.web_tool_metering_charge is False

    await _search(gw, token)
    row = (await _events(user_id))[0]
    assert row.credits == Decimal(FLAT_FEES["web_search"]["credits"])
    assert row.charged is False


async def test_metered_search_writes_one_ledger_row(gw, brave, tenant):
    from app.db import async_session_maker
    from app.db.models.credit import CreditLedger

    user_id, token = tenant
    await _search(gw, token)

    async with async_session_maker() as db:
        ledger = list((await db.execute(
            select(CreditLedger).where(CreditLedger.user_id == user_id)
        )).scalars().all())
    tool_rows = [r for r in ledger if r.event_type == "tool_call"]
    assert len(tool_rows) == 1, f"expected one tool_call ledger row, got {len(tool_rows)}"
    assert tool_rows[0].metadata_json.get("via") == "gateway"
    assert tool_rows[0].metadata_json.get("tool") == "web_search"


# ── Key containment ───────────────────────────────────────────────────


async def test_the_key_goes_upstream_and_nowhere_else(gw, brave, tenant):
    _user_id, token = tenant

    res = await _search(gw, token)
    assert res.status_code == 200

    sent = brave.requests[0]
    assert sent.headers["x-subscription-token"] == _TEST_KEY, (
        "the key never reached Brave — this test would pass vacuously"
    )
    _assert_no_key(res.text)


async def test_degraded_response_carries_no_key(gw, brave, tenant):
    _user_id, token = tenant
    brave.status = 503
    res = await _search(gw, token)
    _assert_no_key(res.text)


async def test_health_reports_posture_without_the_key(gw, brave):
    res = await gw.get("/api/search/health")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["configured"] is True
    assert body["fleet_allowed"] is True
    assert body["fleet_block_reason"] is None
    _assert_no_key(res.text)


async def test_health_reports_unconfigured_when_the_key_is_missing(gw, monkeypatch):
    monkeypatch.setattr(sp.settings, "brave_api_key", "   ")
    body = (await gw.get("/api/search/health")).json()
    assert body["configured"] is False


async def test_health_surfaces_the_breaker(gw):
    sp._fleet.trip(30.0)
    body = (await gw.get("/api/search/health")).json()
    assert body["fleet_allowed"] is False
    assert body["fleet_block_reason"] == "cooldown_after_429"


def _assert_no_key(text: str) -> None:
    assert _TEST_KEY not in text
    # A partial leak is still a rotation, so check a prefix too.
    assert _TEST_KEY[:10] not in text
    assert "subscription-token" not in text.lower()


# ── Result shaping ────────────────────────────────────────────────────


async def test_extra_snippets_survive_the_gateway(gw, brave, tenant):
    """They are why the agent can often answer WITHOUT a slow web_fetch —
    dropping them here would quietly raise the fetch count."""
    _user_id, token = tenant
    brave.payload = _brave_payload(1, snippets=["passage one", "passage two"])

    body = (await _search(gw, token)).json()
    assert body["results"][0]["extra_snippets"] == ["passage one", "passage two"]


async def test_extra_snippets_are_capped_at_four(gw, brave, tenant):
    """Four is what the consumer renders (tool_executor.py:1681); a fifth on the
    wire would be paid-for bytes nothing reads."""
    _user_id, token = tenant
    brave.payload = _brave_payload(1, snippets=[f"p{i}" for i in range(6)])

    body = (await _search(gw, token)).json()
    assert body["results"][0]["extra_snippets"] == ["p0", "p1", "p2", "p3"]


async def test_blank_snippets_are_dropped(gw, brave, tenant):
    _user_id, token = tenant
    brave.payload = _brave_payload(1, snippets=["", "real", ""])

    body = (await _search(gw, token)).json()
    assert body["results"][0]["extra_snippets"] == ["real"]


async def test_results_without_a_url_are_dropped(gw, brave, tenant):
    _user_id, token = tenant
    brave.payload = {"web": {"results": [
        {"title": "no url", "description": "d"},
        {"title": "good", "url": "https://example.com/1", "description": "d"},
    ]}}

    body = (await _search(gw, token)).json()
    assert [r["url"] for r in body["results"]] == ["https://example.com/1"]


async def test_request_shape_reaches_brave(gw, brave, tenant):
    _user_id, token = tenant
    await _search(gw, token, query="tide times", count=5, country="gb")

    params = brave.requests[0].url.params
    assert params["q"] == "tide times"
    assert params["count"] == "5"
    assert params["country"] == "gb"
    assert params["extra_snippets"] == "true", (
        "extra_snippets must be requested or the agent loses snippet-first answers"
    )


async def test_count_is_bounded_by_the_schema(gw, brave, tenant):
    _user_id, token = tenant
    res = await gw.post(
        "/api/search/web",
        json={"query": "x", "count": 999},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 422, res.text
    assert not brave.requests


async def test_empty_query_is_rejected(gw, brave, tenant):
    _user_id, token = tenant
    res = await gw.post(
        "/api/search/web",
        json={"query": ""},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 422, res.text
    assert not brave.requests
