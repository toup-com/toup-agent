"""Process-lifetime counters, and a reader that tolerates an image without them (D10).

The 2026-09-14 incident was silent on every dashboard: turns completed, rows
were written into a day chat nobody could reach, a media card was dropped on
every reload, and `/api/day-chats` 500ed 65 times over three hours. Nothing
counted any of it.

D10 adds `app/services/health_signals.py` — ints only, no ids and no text,
exposed under `/agent/health.health_signals`. Two properties decide whether it
is safe to ship:

  * it may NEVER make a tenant look unhealthy. The counters are diagnostics;
    a raise inside them must not change the health verdict, or a bug in the
    instrumentation restarts containers.
  * the READER must tolerate a body with no `health_signals` key. The fleet
    runs image 7edaed3ab644 and a rollout is gradual, so for the length of
    that rollout the monitor reads old bodies and new ones in the same tick.

Local run (from backend/):
    RUN_MODE=agent PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_agent_health_signals.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import pytest


def hs():
    return pytest.importorskip(
        "app.services.health_signals",
        reason="health_signals not landed yet — lane B3 (D10)",
    )


# D10's list, verbatim. Named rather than discovered: a counter that is
# incremented but never declared is one nobody will think to graph.
EXPECTED = [
    "future_dated_day_chats", "day_chats_5xx", "turns_completed",
    "assistant_rows_written", "media_expected", "media_persisted",
    "channel_events_claimed", "channel_events_deduped", "channel_events_persisted",
    "wa_sidecar_spawns", "channel_tag_prefixed_replies",
    "rebucket_runs", "rebucket_failures",
    # Review round 1: a NULL-day message is the incident's own shape, and the
    # gauge needed a monotonic partner so a heal cannot erase the evidence.
    "day_chat_resolve_failures", "future_dated_day_chats_seen",
    "wa_sidecar_adopted",
    # Round 44: the cached WhatsApp status was advanced by the reconciler
    # rather than by SSE — i.e. a `connection_open` reached nobody.
    "wa_status_reconciled",
]


@pytest.fixture(autouse=True)
def _reset():
    """Counters are process-lifetime by design, so they leak between tests.

    Deliberately NOT an importorskip: this fixture runs for every test in the
    file, including the reader cases that exercise container_monitor and can
    run today. Skipping here would hide them behind a module that has not
    landed.
    """
    try:
        from app.services import health_signals as m
    except Exception:
        yield
        return
    m.reset_for_tests()
    yield
    m.reset_for_tests()


def test_the_snapshot_declares_every_documented_counter():
    snap = hs().snapshot()
    missing = [k for k in EXPECTED if k not in snap]
    assert not missing, f"health_signals does not declare {missing}"


def test_every_value_is_an_int_and_nothing_carries_text():
    """The body is served publicly at /agent/health with no auth. A user id,
    a phone number or a message fragment in here is a data leak on a route
    that exists to be polled."""
    snap = hs().snapshot()
    for k, v in snap.items():
        assert isinstance(v, int) and not isinstance(v, bool), f"{k}={v!r} is not an int"


def test_incr_and_get_round_trip():
    m = hs()
    m.incr("turns_completed")
    m.incr("turns_completed", 4)
    assert m.get("turns_completed") == 5
    assert m.snapshot()["turns_completed"] == 5


def test_an_unknown_counter_never_raises():
    """Call sites are `health_signals.incr(...)` sprinkled through the hot
    path and imported lazily; a typo must cost a lost number, never a turn."""
    m = hs()
    m.incr("a_counter_nobody_declared")
    assert m.get("a_counter_that_does_not_exist") == 0


def test_reset_for_tests_clears_everything():
    m = hs()
    m.incr("media_expected", 3)
    m.reset_for_tests()
    assert m.get("media_expected") == 0


# ── the endpoint ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_health_exposes_the_signals():
    import agent_main

    hs()
    body = await agent_main.agent_health()
    if "health_signals" not in body:
        pytest.skip("/agent/health does not expose health_signals yet — lane B2 (D10)")
    assert all(isinstance(v, int) for v in body["health_signals"].values())


@pytest.mark.asyncio
async def test_agent_health_still_answers_when_the_counters_raise(monkeypatch):
    """Best-effort, in the strict sense: an exception inside the diagnostics
    must not make a healthy tenant report unhealthy — that would have the
    monitor restarting containers because a counter module broke."""
    import agent_main

    m = hs()

    def boom():
        raise RuntimeError("counters exploded")

    monkeypatch.setattr(m, "snapshot", boom)
    body = await agent_main.agent_health()
    assert body.get("status") in ("healthy", "ok"), (
        "a failing counter module changed the health verdict"
    )


# ── the reader ────────────────────────────────────────────────────────


def test_verdict_from_health_body_tolerates_an_old_image():
    """The fleet runs 7edaed3ab644 and a rollout is gradual: the monitor
    reads bodies with and without this key in the same tick."""
    from app.services.container_monitor import verdict_from_health_body

    assert verdict_from_health_body({"status": "healthy"}) == (True, None, None)


def test_the_signals_reader_answers_none_for_a_body_without_them():
    from app.services import container_monitor as cm

    reader = getattr(cm, "signals_from_health_body", None)
    if reader is None:
        pytest.skip("signals_from_health_body not landed yet — lane B3 (D10)")
    assert reader({"status": "healthy"}) is None


def test_the_signals_reader_parses_a_new_image_body():
    from app.services import container_monitor as cm

    reader = getattr(cm, "signals_from_health_body", None)
    if reader is None:
        pytest.skip("signals_from_health_body not landed yet — lane B3 (D10)")
    got = reader({"status": "healthy", "health_signals": {"turns_completed": 7}})
    assert got == {"turns_completed": 7}


@pytest.mark.parametrize("body", [
    {"status": "healthy", "health_signals": None},
    {"status": "healthy", "health_signals": "not-a-dict"},
    {"status": "healthy", "health_signals": []},
    {},
])
def test_the_signals_reader_never_raises_on_a_malformed_body(body):
    """A tenant answering nonsense is a tenant with a problem, not a reason
    for the MONITOR to stop checking everybody else in the loop."""
    from app.services import container_monitor as cm

    reader = getattr(cm, "signals_from_health_body", None)
    if reader is None:
        pytest.skip("signals_from_health_body not landed yet — lane B3 (D10)")
    assert reader(body) is None or isinstance(reader(body), dict)


# ══════════════════════════════════════════════════════════════════════
# The public surface: `/agent/health` is UNAUTHENTICATED
# ══════════════════════════════════════════════════════════════════════
#
# It is in `agent_main._PUBLIC_PATHS`, so anyone can GET
# `https://agent-<prefix>.agents.toup.ai/agent/health` — the repo's own
# runbook (bridge/DEPLOY-WAVE2.md) curls it with no credential. Before round
# 46 that payload carried the tenant's WhatsApp number in E.164 and their raw
# user uuid; round 46 was about to add `turn_ready_detail`, `deps.db`, `loop`
# and the full WhatsApp session state to the same anonymous body, which tells
# a stranger whether that person has linked WhatsApp, whether they have ever
# completed a turn, and whether their database is up.

E164_RE = __import__("re").compile(r"\+\d{8,15}")
UUID_RE = __import__("re").compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


class _Req:
    def __init__(self, key: str | None = None) -> None:
        self.headers = {"x-agent-key": key} if key else {}


def _blob(body) -> str:
    import json as _json

    return _json.dumps(body, default=str)


@pytest.mark.asyncio
async def test_the_public_health_body_names_nobody(monkeypatch):
    import agent_main

    monkeypatch.setattr(agent_main.settings, "agent_api_key", "the-tenant-key")
    body = await agent_main.agent_health(_Req())

    assert body.get("status") in ("healthy", "ok")
    assert "bound_user_id" not in body, "the raw tenant uuid is public"
    for k in ("turn_ready_detail", "serving", "deps", "loop", "channels_settled"):
        assert k not in body, f"{k} is public"
    wa = (body.get("channels") or {}).get("whatsapp")
    if isinstance(wa, dict):
        assert "self_e164" not in wa, "the linked phone number is public"
        assert not (set(wa) - agent_main._PUBLIC_WA_KEYS), (
            f"unvetted WhatsApp keys are public: {sorted(set(wa) - agent_main._PUBLIC_WA_KEYS)}"
        )
    assert not E164_RE.search(_blob(body)), _blob(body)
    # …and no raw uuid anywhere in the body, by shape: `bound_user_id` was the
    # known one, but the rule is "nothing that identifies a person", and a
    # tenant uuid reaches the platform DB, the bridge and every log line.
    assert not UUID_RE.search(_blob(body)), _blob(body)


@pytest.mark.asyncio
async def test_the_bridge_and_the_monitor_still_get_what_they_read(monkeypatch):
    """The whole point of the tiering is that nothing operational moves. The
    bridge reads `boot_progress.ready` and the status code; container_monitor
    reads `status`/`db_ok`/`health_signals`; the rollout reads `active_turns`;
    both readiness probes read `is_bound`, and a container can legitimately be
    probed before the platform has stored its key."""
    import agent_main

    monkeypatch.setattr(agent_main.settings, "agent_api_key", "the-tenant-key")
    body = await agent_main.agent_health(_Req())
    for k in ("status", "turn_ready", "active_turns", "db_ok", "boot_progress",
              "is_bound", "pool_generic", "version", "schema"):
        assert k in body, k


@pytest.mark.asyncio
async def test_the_key_bearing_platform_still_gets_the_detail(monkeypatch):
    """`agent_turn_readiness` reads `turn_ready_detail` and `bound_user_id`
    off THIS body, and all three platform probes send the key."""
    import agent_main

    monkeypatch.setattr(agent_main.settings, "agent_api_key", "the-tenant-key")
    body = await agent_main.agent_health(_Req("the-tenant-key"))
    assert "bound_user_id" in body
    # turn_ready_detail is only present when `_turn_readiness()` succeeded;
    # what must never happen is the reverse (present for an anonymous caller).
    assert ("turn_ready_detail" in body) or ("serving" not in body)


@pytest.mark.asyncio
async def test_a_wrong_key_is_not_a_key(monkeypatch):
    import agent_main

    monkeypatch.setattr(agent_main.settings, "agent_api_key", "the-tenant-key")
    body = await agent_main.agent_health(_Req("the-tenant-keY"))
    assert "bound_user_id" not in body


@pytest.mark.asyncio
async def test_an_unbound_container_with_no_key_is_never_trusted(monkeypatch):
    """A generic pool member has no `agent_api_key` yet. An empty expected key
    must not make every anonymous caller trusted — which is what a plain
    `provided == expected` would do."""
    import agent_main

    monkeypatch.setattr(agent_main.settings, "agent_api_key", "")
    assert agent_main._health_caller_is_trusted(_Req()) is False
    assert agent_main._health_caller_is_trusted(_Req("")) is False
    body = await agent_main.agent_health(_Req())
    assert "bound_user_id" not in body


@pytest.mark.asyncio
async def test_calling_it_with_no_request_at_all_is_untrusted():
    """Every existing in-process caller (and this file's older tests) invoke
    `agent_health()` with no argument. That must keep working, and must be the
    SAFE side of the fork."""
    import agent_main

    body = await agent_main.agent_health()
    assert body.get("status") in ("healthy", "ok")
    assert "bound_user_id" not in body


@pytest.mark.asyncio
async def test_the_header_actually_reaches_the_handler_over_http(monkeypatch):
    """The in-process tests above hand the handler a fake request. This one
    drives the real ASGI app, because the whole tiering hangs on FastAPI
    injecting `Request` into a parameter that also has a default — and a
    handler that never sees the header would serve the public tier to the
    platform, silently."""
    import json as _json

    import agent_main

    monkeypatch.setattr(agent_main.settings, "agent_api_key", "k1")

    async def call(headers: dict):
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "GET", "scheme": "http", "path": "/agent/health",
            "raw_path": b"/agent/health", "query_string": b"", "root_path": "",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 8001),
        }
        body = bytearray()
        status = {}

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(m):
            if m["type"] == "http.response.start":
                status["code"] = m["status"]
            elif m["type"] == "http.response.body":
                body.extend(m.get("body") or b"")

        await agent_main.app(scope, receive, send)
        return status.get("code"), _json.loads(bytes(body) or b"{}")

    code, anon = await call({})
    assert code == 200
    assert "bound_user_id" not in anon, anon

    code, keyed = await call({"X-Agent-Key": "k1"})
    assert code == 200
    assert "bound_user_id" in keyed, keyed


# ── fix lane A: the push-gauge surface (R46/L8) ───────────────────────
#
# `set_gauge` exists so a sampler that already owns a measurement can publish
# it without this module pulling on a health request. Its whole point is that
# 0 MEANS "the last probe failed" — a property with no assertion anywhere
# until now.


def test_a_pushed_gauge_is_never_pulled():
    m = hs()
    m.set_gauge("db_reachable", 0)
    # However stale it gets: nothing can pull it, so "due" has no meaning and
    # a due pushed gauge only invites a call to a `fn` it does not have.
    m._GAUGES["db_reachable"]["at"] = -1e9
    assert m.gauge_due("db_reachable") is False, (
        "a pushed gauge went due — refresh_gauges would try to call a `fn` it "
        "does not have"
    )


@pytest.mark.asyncio
async def test_a_forced_refresh_leaves_a_pushed_value_alone():
    m = hs()
    m.set_gauge("loop_lag_ms_max_30s", 412)
    await m.refresh_gauges(force=True)
    assert m.snapshot()["loop_lag_ms_max_30s"] == 412


def test_the_alarm_value_is_not_what_a_reset_restores():
    """`reset_for_tests` zeroes measurements. For `db_reachable`, 0 is the
    ALARM, so zeroing it would make every test process claim the DB is
    unreachable."""
    m = hs()
    m.set_gauge("db_reachable", 0)
    m.reset_for_tests()
    assert m.snapshot()["db_reachable"] == 1


@pytest.mark.asyncio
async def test_a_push_over_a_pull_gauge_keeps_the_refresher():
    """Two owners for one name is a bug either way, but the silent conversion
    (pull → push) is the one that stops a live measurement."""
    m = hs()

    async def _fn() -> int:
        return 7

    m.register_gauge("db_pool_checked_out", _fn, ttl_s=0.0)
    m.set_gauge("db_pool_checked_out", 3)
    assert m.snapshot()["db_pool_checked_out"] == 3
    await m.refresh_gauges(force=True)
    assert m.snapshot()["db_pool_checked_out"] == 7, (
        "the pull gauge lost its refresher to a push"
    )


def test_a_gauge_may_never_break_its_caller():
    hs().set_gauge("db_reachable", "not a number")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_turn_ready_requires_the_key(monkeypatch):
    """Its own docstring promised "behind X-Agent-Key" and the handler had no
    check of any kind. `AgentAPIKeyMiddleware` covers it only while
    `agent_api_key` is set — it waves everything through on a generic pool
    member and in the monolith mount — and this body names the bound user, the
    database's reachability and the loop's lag."""
    import agent_main

    monkeypatch.setattr(agent_main.settings, "agent_api_key", "the-tenant-key")

    refused = await agent_main.agent_turn_ready(_Req())
    assert getattr(refused, "status_code", None) == 401, refused
    assert getattr(refused, "status_code", None) == 401

    ok = await agent_main.agent_turn_ready(_Req("the-tenant-key"))
    assert isinstance(ok, dict)
    assert "turn_ready_detail" in ok and "serving" in ok

    # An empty configured key can never be "trusted": `hmac.compare_digest("",
    # "")` is True, and a lobby container has no key.
    monkeypatch.setattr(agent_main.settings, "agent_api_key", "")
    assert getattr(await agent_main.agent_turn_ready(_Req()), "status_code", None) == 401


@pytest.mark.asyncio
async def test_turn_ready_never_names_a_diagnostic_as_the_reason(monkeypatch):
    """D1: `llm_wire_warm` / `channels_settled` are reported, never blamed."""
    import agent_main

    monkeypatch.setattr(agent_main.settings, "agent_api_key", "the-tenant-key")
    out = await agent_main.agent_turn_ready(_Req("the-tenant-key"))
    assert out["not_ready_because"] not in ("llm_wire_warm", "channels_settled")
    detail = out["turn_ready_detail"]
    assert "llm_wire_warm" in detail and "channels_settled" in detail
