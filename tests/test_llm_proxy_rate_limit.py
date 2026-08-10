"""G-20: the LLM proxy caps requests per tenant, not just cents per month.

Before this, the only bounds on the proxy path were the monthly cents caps
(readable through a 30s cache, so bursts overshoot) — and `_check_budget`
returns None for admin-role tenants, so a leaked admin token had NO ceiling
of any kind. The sliding-window limiter (`llm_proxy_rate_limit_per_min`,
default 90 vs a 30-day measured per-user peak of 33/min) closes both: the
101st runaway request 429s instead of billing.

The endpoint tests drive the real `proxy_chat` handler with the same
fake-auth pattern as test_proxy_cache_logging.py; on pre-limit code the
over-limit call succeeds instead of 429ing, so they fail there.
"""

from __future__ import annotations

import types

import pytest
from fastapi import HTTPException

from app.api import llm_proxy as lp
from app.config import settings


@pytest.fixture(autouse=True)
def _clean_windows():
    lp._rate_windows.clear()
    yield
    lp._rate_windows.clear()


# ── _check_rate_limit (pure) ─────────────────────────────────────────


def test_allows_up_to_limit_then_blocks(monkeypatch):
    monkeypatch.setattr(settings, "llm_proxy_rate_limit_per_min", 5, raising=False)
    for _ in range(5):
        assert lp._check_rate_limit("u1") is None
    retry = lp._check_rate_limit("u1")
    assert retry is not None and retry >= 1, "6th call in the window must block"


def test_windows_are_per_user(monkeypatch):
    monkeypatch.setattr(settings, "llm_proxy_rate_limit_per_min", 2, raising=False)
    assert lp._check_rate_limit("u1") is None
    assert lp._check_rate_limit("u1") is None
    assert lp._check_rate_limit("u1") is not None
    assert lp._check_rate_limit("u2") is None, "one user's burst must not block another"


def test_blocked_call_is_not_recorded(monkeypatch):
    """A 429'd request must not extend its own punishment: after a block,
    the window holds only the ALLOWED calls, so the client's Retry-After
    is honest rather than sliding away from them."""
    monkeypatch.setattr(settings, "llm_proxy_rate_limit_per_min", 2, raising=False)
    lp._check_rate_limit("u1")
    lp._check_rate_limit("u1")
    for _ in range(10):
        lp._check_rate_limit("u1")
    assert len(lp._rate_windows["u1"]) == 2


def test_zero_disables(monkeypatch):
    monkeypatch.setattr(settings, "llm_proxy_rate_limit_per_min", 0, raising=False)
    for _ in range(50):
        assert lp._check_rate_limit("u1") is None
    assert lp._rate_windows == {}, "disabled limiter must not accumulate state"


def test_window_expires(monkeypatch):
    monkeypatch.setattr(settings, "llm_proxy_rate_limit_per_min", 2, raising=False)
    t = {"now": 1000.0}
    monkeypatch.setattr(lp.time, "time", lambda: t["now"])
    assert lp._check_rate_limit("u1") is None
    assert lp._check_rate_limit("u1") is None
    assert lp._check_rate_limit("u1") is not None
    t["now"] += 61.0
    assert lp._check_rate_limit("u1") is None, "a fresh minute must admit again"


# ── proxy_chat end to end (fake auth/budget/backend) ─────────────────


class _FakeRequest:
    def __init__(self, body: dict, headers: dict | None = None):
        self._body = body
        self.headers = headers or {}

    async def json(self) -> dict:
        return self._body


@pytest.fixture
def proxied(monkeypatch):
    config = types.SimpleNamespace(user_id="deadbeef-0000-0000-0000-000000000000")

    async def fake_auth(request, db):
        return config

    async def fake_budget(cfg, provider, db):
        return None

    async def fake_log_event(*a, **kw):
        return None

    monkeypatch.setattr(lp, "_auth_agent", fake_auth)
    monkeypatch.setattr(lp, "_check_budget", fake_budget)
    monkeypatch.setattr(lp, "_log_event", fake_log_event)

    async def drive():
        resp = types.SimpleNamespace(
            status_code=200, headers={}, content=b"",
            json=lambda: {"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        )

        async def fake_chat(b, api_key):
            return resp

        backend = types.SimpleNamespace(name="openai", chat=fake_chat)
        monkeypatch.setattr(lp, "_route_chat", lambda model, cfg: (backend, "k"))
        return await lp.proxy_chat(
            _FakeRequest({"model": "gpt-5.5", "messages": []}), db=None
        )

    return drive


async def test_over_limit_chat_request_429s_with_retry_after(proxied, monkeypatch):
    monkeypatch.setattr(settings, "llm_proxy_rate_limit_per_min", 3, raising=False)
    for _ in range(3):
        await proxied()
    with pytest.raises(HTTPException) as exc:
        await proxied()
    assert exc.value.status_code == 429
    assert int(exc.value.headers["Retry-After"]) >= 1


async def test_under_limit_requests_flow_untouched(proxied, monkeypatch):
    monkeypatch.setattr(settings, "llm_proxy_rate_limit_per_min", 90, raising=False)
    for _ in range(5):
        await proxied()  # raises nothing → the limiter is invisible below the cap
