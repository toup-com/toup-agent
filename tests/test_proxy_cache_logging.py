"""W0.2b — proxy-side prompt-cache observability ([CACHE] log lines).

prompt_cache_retention="24h" is verified sent end-to-end, yet 6h-gap
requests measured 0/6 cache hits. These tests pin the read-only logging
that makes cache behavior auditable per call:

  (a) request-side line: user prefix + model + has_cache_key +
      8-char key hash + retention — NEVER the key itself, never content.
  (b) usage-side line: prompt_tokens + cached_tokens together so one
      grep yields the per-tenant hit-ratio series.
  (c) debug-level dump of upstream cache/routing headers (x-request-id)
      as evidence for the OpenAI escalation — auth headers never match.

No behavior change, no DB change — logging only.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import types

import pytest

from app.api import llm_proxy as lp

FAKE_KEY = "user-e5ec1759:2026-07-27:day-chat"
FAKE_RETENTION = "24h"


# ── _cache_log_fields (pure fn) ──────────────────────────────────────


def test_cache_fields_hashes_key_never_returns_raw():
    body = {"prompt_cache_key": FAKE_KEY, "prompt_cache_retention": FAKE_RETENTION}
    has_key, key_hash, retention = lp._cache_log_fields(body)
    assert has_key is True
    assert retention == "24h"
    assert key_hash == hashlib.sha256(FAKE_KEY.encode()).hexdigest()[:8]
    assert len(key_hash) == 8
    assert FAKE_KEY not in key_hash


def test_cache_fields_stable_across_calls():
    body = {"prompt_cache_key": FAKE_KEY}
    assert lp._cache_log_fields(body) == lp._cache_log_fields(dict(body))


def test_cache_fields_absent_key_and_retention():
    assert lp._cache_log_fields({}) == (False, "none", "none")


def test_cache_fields_non_string_key_treated_as_absent():
    """A malformed (non-str) key must not crash the request path."""
    assert lp._cache_log_fields({"prompt_cache_key": 123}) == (False, "none", "none")


# ── proxy_chat log lines (fake auth/budget/backend, non-stream) ──────


class _FakeRequest:
    """Stands in for a Starlette Request.

    `headers` is part of that shape — a real Request always has it, and the
    proxy reads the channel-attribution header from it (alembic 082). The stub
    omitted it, so it modelled a request that cannot exist; a test double that
    is missing an attribute production always has will fail for a reason that
    has nothing to do with what it is testing.
    """

    def __init__(self, body: dict, headers: dict | None = None):
        self._body = body
        self.headers = headers or {}

    async def json(self) -> dict:
        return self._body


def _fake_resp(resp_data: dict, headers: dict | None = None):
    return types.SimpleNamespace(
        status_code=200,
        headers=headers or {},
        content=b"",
        json=lambda: resp_data,
    )


@pytest.fixture
def proxied(monkeypatch):
    """Patch auth/budget/persistence out of proxy_chat; return a driver."""
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

    async def drive(body: dict, resp_data: dict, backend_name: str = "openai",
                    resp_headers: dict | None = None):
        resp = _fake_resp(resp_data, resp_headers)

        async def fake_chat(b, api_key):
            return resp

        backend = types.SimpleNamespace(name=backend_name, chat=fake_chat)
        monkeypatch.setattr(lp, "_route_chat", lambda model, cfg: (backend, "k"))
        return await lp.proxy_chat(_FakeRequest(body), db=None)

    return drive


_OPENAI_USAGE = {
    "usage": {
        "prompt_tokens": 11346,
        "completion_tokens": 40,
        "prompt_tokens_details": {"cached_tokens": 10240},
    },
}


async def test_openai_request_and_usage_cache_lines(proxied, caplog):
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    body = {
        "model": "gpt-5.5",
        "stream": False,
        "messages": [{"role": "user", "content": "TOP SECRET user content"}],
        "prompt_cache_key": FAKE_KEY,
        "prompt_cache_retention": FAKE_RETENTION,
    }
    await proxied(body, _OPENAI_USAGE)

    cache_lines = [r.getMessage() for r in caplog.records if "[CACHE]" in r.getMessage()]
    assert len(cache_lines) == 2, cache_lines

    req_line, usage_line = cache_lines
    expected_hash = hashlib.sha256(FAKE_KEY.encode()).hexdigest()[:8]
    assert "user=deadbeef" in req_line
    assert "model=gpt-5.5" in req_line
    assert "has_cache_key=True" in req_line
    assert f"cache_key_hash={expected_hash}" in req_line
    assert "retention=24h" in req_line

    assert "user=deadbeef" in usage_line
    assert "prompt_tokens=11346" in usage_line
    assert "cached_tokens=10240" in usage_line
    assert "retention=24h" in usage_line


async def test_no_key_material_or_content_in_any_log(proxied, caplog):
    """The raw cache key and message content must never appear in output."""
    caplog.set_level(logging.DEBUG, logger="app.api.llm_proxy")
    body = {
        "model": "gpt-5.5",
        "stream": False,
        "messages": [{"role": "user", "content": "TOP SECRET user content"}],
        "prompt_cache_key": FAKE_KEY,
        "prompt_cache_retention": FAKE_RETENTION,
    }
    await proxied(body, _OPENAI_USAGE, resp_headers={"x-request-id": "req_123"})
    everything = "\n".join(r.getMessage() for r in caplog.records)
    assert FAKE_KEY not in everything
    assert "TOP SECRET" not in everything


async def test_openai_missing_cache_fields_logged_as_none(proxied, caplog):
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    body = {"model": "gpt-5.5", "stream": False, "messages": []}
    await proxied(body, {"usage": {"prompt_tokens": 300, "completion_tokens": 7}})

    cache_lines = [r.getMessage() for r in caplog.records if "[CACHE]" in r.getMessage()]
    assert len(cache_lines) == 2
    assert "has_cache_key=False" in cache_lines[0]
    assert "cache_key_hash=none" in cache_lines[0]
    assert "retention=none" in cache_lines[0]
    assert "cached_tokens=0" in cache_lines[1]


async def test_anthropic_path_emits_no_cache_lines(proxied, caplog):
    """[CACHE] is the OpenAI escalation series — Anthropic stays quiet."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    body = {"model": "claude-sonnet-4-6", "stream": False, "messages": []}
    await proxied(
        body,
        {"usage": {"input_tokens": 5, "output_tokens": 5}},
        backend_name="anthropic",
    )
    assert not any("[CACHE]" in r.getMessage() for r in caplog.records)


# ── upstream header debug dump ───────────────────────────────────────


def test_upstream_headers_filters_to_cache_and_request_id(caplog):
    caplog.set_level(logging.DEBUG, logger="app.api.llm_proxy")
    lp._debug_log_upstream_cache_headers(
        {
            "x-request-id": "req_abc123",
            "X-Cache-Status": "MISS",
            "Authorization": "Bearer sk-proj-SECRET",
            "openai-organization": "org-x",
        },
        "gpt-5.5",
    )
    msgs = [r.getMessage() for r in caplog.records if "[CACHE] upstream" in r.getMessage()]
    assert len(msgs) == 1
    assert "req_abc123" in msgs[0]
    assert "X-Cache-Status" in msgs[0]
    assert "SECRET" not in msgs[0]
    assert "org-x" not in msgs[0]


def test_upstream_headers_silent_when_debug_disabled(caplog):
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    lp._debug_log_upstream_cache_headers({"x-request-id": "req_abc"}, "gpt-5.5")
    assert not any("[CACHE] upstream" in r.getMessage() for r in caplog.records)


# ── source pins for the SSE streaming variant ────────────────────────
# The stream path's httpx response lives inside OpenAIBackend.chat_stream
# and its usage log runs in a generator finally — pin both by source
# (same idiom as test_openai_cache_observability.py) rather than driving
# the full ASGI streaming machinery.


def test_stream_finally_emits_usage_cache_line():
    src = inspect.getsource(lp.proxy_chat)
    assert "[CACHE] user=%s model=%s has_cache_key=%s retention=%s" in src
    assert src.count("cached_tokens=%s") == 2  # SSE finally + non-stream twin


def test_openai_chat_stream_logs_upstream_headers():
    src = inspect.getsource(lp.OpenAIBackend.chat_stream)
    assert "_debug_log_upstream_cache_headers" in src
