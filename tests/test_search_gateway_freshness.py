"""Search gateway — freshness policy (incident 2026-08-18).

What the gateway now does per query class, tested at the wire:

  evergreen  → exactly ONE Brave call, exactly the pre-incident params
               ({q, count, extra_snippets}), page dates forwarded.
  recent     → web call WITH `freshness` (ladder head), plus the News index,
               plus a neutral discovery call for a site:-anchored query, all
               concurrent; ladder widens pm → py → none while the page is
               thin; results >18 months are dropped; merged round-robin.

Same harness as test_search_gateway.py (fake Brave at the httpx transport).

Run: cd backend && python3 -m pytest tests/test_search_gateway_freshness.py -q
"""
from __future__ import annotations

import asyncio
import hashlib
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import app.api.search_proxy as sp

_TEST_KEY = "brv-SECRET-0123456789abcdef"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(sp, "_TOKEN_CACHE", {})
    monkeypatch.setattr(sp, "_buckets", {})
    monkeypatch.setattr(sp, "_fleet", sp._FleetGuard())
    monkeypatch.setattr(sp.settings, "brave_api_key", _TEST_KEY)
    monkeypatch.setattr(sp.settings, "platform_replicas", 1)
    monkeypatch.setattr(sp.settings, "search_freshness_enabled", True)
    monkeypatch.setattr(sp.settings, "search_news_blend_enabled", True)
    monkeypatch.setattr(sp.settings, "search_site_discovery_enabled", True)
    monkeypatch.setattr(sp.settings, "search_stale_filter_enabled", True)
    monkeypatch.setattr(sp.settings, "search_stale_max_days", 548)
    monkeypatch.setattr(sp.settings, "search_freshness_min_results", 3)
    monkeypatch.setattr(sp.settings, "search_recency_append_year", False)
    for k in sp._COUNTERS:
        sp._COUNTERS[k] = 0


def _web(items):
    return {"web": {"results": items}}


def _r(i, url=None, *, page_age="2026-07-24T00:00:00", age="3 weeks ago", **kw):
    d = {"title": f"Result {i}", "url": url or f"https://example.com/{i}",
         "description": f"Description {i}", "extra_snippets": [], "page_age": page_age, "age": age}
    d.update(kw)
    return d


class _FakeBrave:
    """Programmable Brave: routes by endpoint + `freshness` param so a test
    can script the ladder (pm thin → py full) and the news/discovery legs."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.web_by_freshness: dict[Optional[str], dict] = {}
        self.news: dict = {"results": []}
        self.discovery: Optional[dict] = None       # served for a query WITHOUT site:
        self.default_web = _web([_r(1), _r(2), _r(3), _r(4)])
        self.status_for: dict[str, int] = {}          # endpoint-substring -> status

    async def serve(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        await asyncio.sleep(0.002)
        url = str(request.url)
        params = request.url.params
        for sub, st in self.status_for.items():
            if sub in url:
                return httpx.Response(st, json={}, request=request)
        hdrs = {"x-ratelimit-remaining": "49, 0"}
        if "/news/search" in url:
            return httpx.Response(200, json=self.news, headers=hdrs, request=request)
        q = params.get("q", "")
        if self.discovery is not None and "site:" not in q:
            return httpx.Response(200, json=self.discovery, headers=hdrs, request=request)
        fr = params.get("freshness")
        payload = self.web_by_freshness.get(fr, self.default_web)
        return httpx.Response(200, json=payload, headers=hdrs, request=request)

    # helpers
    def web_calls(self):
        return [r for r in self.requests if "/web/search" in str(r.url)]

    def news_calls(self):
        return [r for r in self.requests if "/news/search" in str(r.url)]


@pytest.fixture
def brave(monkeypatch) -> _FakeBrave:
    fake = _FakeBrave()

    async def _handle(transport_self, request):
        return await fake.serve(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _handle)
    return fake


@asynccontextmanager
async def _client():
    app = FastAPI()
    app.include_router(sp.router, prefix="/api")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def gw():
    async with _client() as c:
        yield c


async def _tenant() -> tuple[str, str]:
    from app.db import AgentConfig, User, async_session_maker
    from app.services.auth_service import get_password_hash
    user_id = str(uuid.uuid4())
    token = f"toup-tok-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(User(id=user_id, email=f"gwf-{uuid.uuid4().hex[:12]}@example.com",
                    hashed_password=get_password_hash("test-password-1234"), name="GW Fresh"))
        db.add(AgentConfig(user_id=user_id, llm_token_hash=hashlib.sha256(token.encode()).hexdigest(),
                           bundle_status="active"))
        await db.commit()
    return user_id, token


@pytest_asyncio.fixture
async def token():
    return (await _tenant())[1]


async def _search(gw, token, query, **body):
    res = await gw.post("/api/search/web", json={"query": query, **body},
                        headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200, res.text
    return res.json()


# ── Evergreen: byte-identical upstream call, dates now forwarded ────────

async def test_evergreen_is_one_call_with_the_pre_incident_params(gw, brave, token):
    body = await _search(gw, token, "who is ada lovelace", count=5)
    assert len(brave.requests) == 1
    p = brave.requests[0].url.params
    assert dict(p) == {"q": "who is ada lovelace", "count": "5", "extra_snippets": "true"}
    assert body["freshness_class"] == "evergreen"
    assert body["freshness_applied"] is None
    assert body["brave_calls"] == 1
    # dates that were always in Brave's payload now reach the agent
    assert body["results"][0]["page_age"] == "2026-07-24T00:00:00"
    assert body["results"][0]["age"] == "3 weeks ago"
    assert body["results"][0]["source"] == "web"


async def test_flag_off_restores_the_pre_incident_behaviour(gw, brave, token, monkeypatch):
    monkeypatch.setattr(sp.settings, "search_freshness_enabled", False)
    body = await _search(gw, token, "anthropic newest model")
    assert len(brave.requests) == 1
    assert "freshness" not in brave.requests[0].url.params
    assert body["freshness_class"] == "evergreen"


# ── Recent: freshness + news, concurrent ─────────────────────────────

async def test_recent_query_sends_freshness_and_blends_news(gw, brave, token):
    brave.news = {"results": [_r(90, "https://news.example/a", page_age="2026-08-14T00:00:00", age="4 days ago")]}
    body = await _search(gw, token, "anthropic newest model", count=8)
    web = brave.web_calls(); news = brave.news_calls()
    assert len(web) == 1 and len(news) == 1
    assert web[0].url.params["freshness"] == "pm", "event-shaped recency starts at a month"
    assert news[0].url.params["freshness"] == "pm"
    assert body["freshness_class"] == "recent"
    assert body["freshness_applied"] == "pm"
    assert body["brave_calls"] == 2
    assert body["news_count"] == 1
    urls = [r["url"] for r in body["results"]]
    assert "https://news.example/a" in urls
    assert [r["source"] for r in body["results"] if r["url"] == "https://news.example/a"] == ["news"]
    # round-robin: the news item lands right after the first web hit
    assert urls[1] == "https://news.example/a"


async def test_agent_supplied_class_is_honoured(gw, brave, token):
    # our patterns say evergreen; the agent says recent → recent (with the default ladder)
    body = await _search(gw, token, "python list comprehension", freshness_class="recent")
    assert body["freshness_class"] == "recent"
    assert brave.web_calls()[0].url.params["freshness"] == "pm"
    # and the other way round: agent says evergreen for a query we would call recent
    brave.requests.clear()
    body = await _search(gw, token, "anthropic newest model", freshness_class="evergreen")
    assert body["freshness_class"] == "evergreen"
    assert "freshness" not in brave.requests[0].url.params
    # garbage is ignored, not trusted
    brave.requests.clear()
    body = await _search(gw, token, "anthropic newest model", freshness_class="whatever")
    assert body["freshness_class"] == "recent"


async def test_state_shaped_query_starts_at_a_year(gw, brave, token):
    await _search(gw, token, "most capable AI model")
    assert brave.web_calls()[0].url.params["freshness"] == "py"


# ── Ladder widening ──────────────────────────────────────────────────

async def test_thin_page_widens_pm_to_py(gw, brave, token):
    brave.web_by_freshness = {
        "pm": _web([_r(1)]),                                     # thin
        "py": _web([_r(1), _r(2), _r(3), _r(4), _r(5)]),
    }
    body = await _search(gw, token, "anthropic newest model", count=8)
    fr = [r.url.params.get("freshness") for r in brave.web_calls()]
    assert fr == ["pm", "py"], fr
    assert body["freshness_applied"] == "py"
    assert len(body["results"]) == 5
    assert sp._COUNTERS["freshness_widened"] == 1


async def test_ladder_gives_up_the_filter_when_still_thin(gw, brave, token):
    brave.web_by_freshness = {"pm": _web([]), "py": _web([_r(1)]), None: _web([_r(1), _r(2), _r(3)])}
    body = await _search(gw, token, "anthropic newest model")
    fr = [r.url.params.get("freshness") for r in brave.web_calls()]
    assert fr == ["pm", "py", None]
    assert body["freshness_applied"] is None
    assert len(body["results"]) == 3


async def test_a_full_first_page_does_not_widen(gw, brave, token):
    await _search(gw, token, "anthropic newest model")
    assert [r.url.params.get("freshness") for r in brave.web_calls()] == ["pm"]


async def test_widening_failure_keeps_the_thin_page(gw, brave, token):
    """A widening attempt is auxiliary: if it errors we serve what we have."""
    brave.web_by_freshness = {"pm": _web([_r(1)])}
    orig = brave.serve

    async def serve(request):
        if request.url.params.get("freshness") == "py":
            return httpx.Response(500, json={}, request=request)
        return await orig(request)
    brave.serve = serve
    body = await _search(gw, token, "anthropic newest model")
    assert body["served"] is True and len(body["results"]) == 1


# ── Stale filter ─────────────────────────────────────────────────────

async def test_stale_results_are_dropped_for_recency_intent(gw, brave, token):
    """The incident's 'LMSYS Chatbot Arena leaderboard latest models' put a
    2023-05-25 blog post at #1. It cannot survive a recency query now."""
    brave.web_by_freshness = {None: _web([
        _r(1, "https://www.lmsys.org/blog/2023-05-25-leaderboard/", page_age="2023-05-25T00:00:00", age="May 25, 2023"),
        _r(2, "https://fresh.example/1"),
        _r(3, "https://undated.example/1", page_age=None, age=None),
    ])}
    # force the ladder to the no-filter rung so the stale page can appear at all
    brave.web_by_freshness["pm"] = _web([]); brave.web_by_freshness["py"] = _web([])
    body = await _search(gw, token, "LMSYS Chatbot Arena leaderboard latest models")
    urls = [r["url"] for r in body["results"]]
    assert "https://www.lmsys.org/blog/2023-05-25-leaderboard/" not in urls
    assert "https://undated.example/1" in urls, "undated pages are kept, not guessed stale"
    assert body["dropped_stale"] == 1
    assert sp._COUNTERS["stale_dropped"] == 1


async def test_stale_pages_survive_an_evergreen_query(gw, brave, token):
    brave.default_web = _web([_r(1, page_age="2023-05-25T00:00:00", age="May 25, 2023")])
    body = await _search(gw, token, "history of rome")
    assert len(body["results"]) == 1 and body["dropped_stale"] == 0


# ── site: discovery ──────────────────────────────────────────────────

async def test_site_anchored_recency_query_also_runs_neutral(gw, brave, token):
    """`site:anthropic.com/news newest Claude model` — the incident's turn-2
    query — also runs without the operator, and the two are merged."""
    brave.discovery = _web([_r(50, "https://www.cnbc.com/2026/06/09/anthropic-mythos-claude-fable-5.html",
                                page_age="2026-06-09T00:00:00", age="June 9, 2026")])
    body = await _search(gw, token, "site:anthropic.com/news newest Claude model August 2026", count=8)
    qs = [r.url.params["q"] for r in brave.web_calls()]
    assert "site:anthropic.com/news newest Claude model August 2026" in qs
    assert "newest Claude model August 2026" in qs
    assert all(r.url.params.get("freshness") == "pm" for r in brave.web_calls())
    urls = [r["url"] for r in body["results"]]
    assert "https://www.cnbc.com/2026/06/09/anthropic-mythos-claude-fable-5.html" in urls
    assert sp._COUNTERS["site_discovery"] == 1


async def test_site_discovery_is_not_run_for_evergreen_or_when_off(gw, brave, token, monkeypatch):
    await _search(gw, token, "site:docs.python.org list comprehension")
    assert len(brave.web_calls()) == 1
    brave.requests.clear()
    monkeypatch.setattr(sp.settings, "search_site_discovery_enabled", False)
    await _search(gw, token, "site:anthropic.com/news newest Claude model")
    assert len(brave.web_calls()) == 1


# ── Failure semantics unchanged ──────────────────────────────────────

async def test_primary_429_still_trips_the_breaker_even_with_auxiliaries(gw, brave, token):
    brave.status_for = {"/web/search": 429}
    body = await _search(gw, token, "anthropic newest model")
    assert body["served"] is False and body["degraded_reason"] == "http_429"
    assert not sp._fleet.allowed()[0]


async def test_news_failure_never_takes_the_search_down(gw, brave, token):
    brave.status_for = {"/news/search": 500}
    body = await _search(gw, token, "anthropic newest model")
    assert body["served"] is True and body["news_count"] == 0 and len(body["results"]) == 4


async def test_year_append_is_opt_in_and_measured_off_by_default(gw, brave, token, monkeypatch):
    from app.config import Settings
    assert Settings.model_fields["search_recency_append_year"].default is False
    monkeypatch.setattr(sp.settings, "search_recency_append_year", True)
    await _search(gw, token, "anthropic newest model")
    assert brave.web_calls()[0].url.params["q"] == "anthropic newest model 2026"


async def test_health_reports_freshness_posture_and_counters(gw, brave, token):
    await _search(gw, token, "anthropic newest model")
    await _search(gw, token, "who is ada lovelace")
    h = (await gw.get("/api/search/health")).json()
    assert h["freshness_enabled"] is True
    assert h["stale_filter_days"] == 548
    assert h["counters"]["recent_queries"] == 1
    assert h["counters"]["evergreen_queries"] == 1
    assert h["counters"]["brave_calls"] == 3      # 2 (web+news) + 1


async def test_served_row_and_charge_are_still_one_per_logical_search(gw, brave, token):
    """Two upstream calls (web + news) are ONE search to the user."""
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import SearchEvent
    body = await _search(gw, token, "anthropic newest model")
    assert body["brave_calls"] == 2
    async with async_session_maker() as db:
        rows = list((await db.execute(select(SearchEvent))).scalars().all())
    mine = [r for r in rows if r.query_sha256 == sp._query_hash("anthropic newest model")]
    assert len(mine) == 1 and mine[0].status == "ok"


# ── All stale: the answer, not a miss ────────────────────────────────

async def test_all_stale_page_is_reported_not_degraded_to_empty(gw, brave, token):
    """Every dated result older than the cutoff → `all_stale`, so the agent
    tells the user it could not verify instead of falling to undated tiers
    that would re-serve the same pages."""
    old = _web([_r(1, page_age="2023-05-25T00:00:00", age="May 25, 2023"),
                _r(2, page_age="2022-01-01T00:00:00", age="January 1, 2022")])
    brave.web_by_freshness = {"pm": old, "py": old, None: old}
    body = await _search(gw, token, "anthropic newest model")
    assert body["served"] is False
    assert body["degraded_reason"] == "all_stale"
    assert body["dropped_stale"] == 2 and body["results"] == []


async def test_truly_empty_page_is_still_empty_result(gw, brave, token):
    brave.web_by_freshness = {"pm": _web([]), "py": _web([]), None: _web([])}
    body = await _search(gw, token, "anthropic newest model")
    assert body["served"] is False and body["degraded_reason"] == "empty_result"
