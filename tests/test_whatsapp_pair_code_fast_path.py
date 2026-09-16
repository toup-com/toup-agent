"""R46 F3 + F8 — the pairing-code request answers now, and a retry is a replay.

Incident 2026-09-15. Requesting a code took ~14 s of wall clock, of which the
platform's own contribution was an AWAITED bridge push: the route seeded the
user's number into the allowlist and then held the request inside
`_bridge_push_within_budget` (up to `agent_setup_sync_timeout_s` = 8 s) BEFORE
asking the agent for a code. The push existed only because the allowlist used
to restart the channel — which F1 removes — so the await buys nothing and
costs everything.

F8 is the other half: every retry of this endpoint reaches the sidecar, which
wipes the auth dir and opens a fresh socket. The two retries in the incident
were harmless only because both 503'd before reaching it. A retry that lands
one tick after a success burns the code the user is typing, and re-pairing
lands in WhatsApp's 30–60 minute rate limiter.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test \\
      pytest tests/test_whatsapp_pair_code_fast_path.py -q -p no:cacheprovider
"""
from __future__ import annotations

import asyncio
import pathlib
import time

import pytest

_HERE = pathlib.Path(__file__).resolve().parents[1]
_SIDECAR_SRC = (_HERE / "whatsapp_sidecar" / "sidecar.mjs").read_text()


class _Cfg:
    def __init__(self, allowlist=""):
        self.whatsapp_baileys_allowlist = allowlist
        self.whatsapp_mode = None
        self.whatsapp_session_status = None
        self.updated_at = None


class _Db:
    def __init__(self):
        self.commits = 0

    async def commit(self):
        self.commits += 1


class _User:
    id = "11111111-2222-3333-4444-555555555555"


_UID = _User.id


@pytest.fixture
def wired(monkeypatch):
    """The pair-code route with its agent hop, config and push mocked out."""
    from app.api import agent_setup as mod

    calls = {"proxy": [], "spawned": [], "awaited_push": 0}
    cfg = _Cfg()

    async def fake_get_or_create(user_id, db):
        return cfg

    async def fake_proxy(method, path, user_id, db, **kw):
        calls["proxy"].append((method, path, kw))
        return {"ok": True, "pairing_code": "ABCD1234", "phone": "+14155552671"}

    def fake_spawn(coro, name=None):
        calls["spawned"].append(name)
        coro.close()          # never actually run the push in a test
        return object()

    async def fake_budget(coro, *, user_id, what):
        calls["awaited_push"] += 1
        coro.close()
        return True

    def fake_push(user_id):
        async def _never():
            await asyncio.sleep(3600)
        return _never()

    monkeypatch.setattr(mod, "_get_or_create_config", fake_get_or_create)
    monkeypatch.setattr(mod, "_agent_qr_proxy", fake_proxy)
    monkeypatch.setattr(mod, "_spawn_bg", fake_spawn)
    monkeypatch.setattr(mod, "_bridge_push_within_budget", fake_budget)
    monkeypatch.setattr(mod, "_env_push_worker", fake_push)
    mod._pair_code_idem.clear()
    try:
        yield mod, calls, cfg
    finally:
        mod._pair_code_idem.clear()


async def _call(mod, *, key=None, phone="+14155552671"):
    body = mod._WhatsAppPairCodeBody(phone=phone)
    return await mod.whatsapp_pair_code(
        body, current_user=_User(), db=_Db(), idempotency_key=key,
    )


# ── F3: the push is not on the response path ──────────────────────────


@pytest.mark.asyncio
async def test_the_route_does_not_await_the_env_push(wired):
    mod, calls, _cfg = wired

    t0 = time.monotonic()
    out = await _call(mod)
    elapsed = time.monotonic() - t0

    assert out["pairing_code"] == "ABCD1234"
    assert calls["awaited_push"] == 0, (
        "the route awaited _bridge_push_within_budget — that is the 8 s the "
        "user spent watching 'Waking Aria and creating your code…'"
    )
    assert any("whatsapp-pair-code-allowlist" in (n or "") for n in calls["spawned"]), (
        "the push was dropped entirely; it must still run, just not on the "
        "response path"
    )
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_the_agent_is_still_asked_for_the_code(wired):
    mod, calls, _cfg = wired
    await _call(mod)
    assert [c[1] for c in calls["proxy"]] == ["/api/whatsapp/qr/pair-code"]


@pytest.mark.asyncio
async def test_the_allowlist_seed_is_idempotent_across_retries(wired):
    """A second tap must not push a second time — the number is already there."""
    mod, calls, cfg = wired
    await _call(mod)
    spawned_after_first = len(calls["spawned"])
    assert cfg.whatsapp_baileys_allowlist

    await _call(mod, key=None)
    assert len(calls["spawned"]) == spawned_after_first, (
        "a repeat pair-code pushed the env again for an unchanged allowlist"
    )


@pytest.mark.asyncio
async def test_the_sync_push_rollback_lever_still_awaits(wired, monkeypatch):
    """One release of rollback. If the flag stops working, the lever is a lie."""
    mod, calls, _cfg = wired
    from app.config import settings

    monkeypatch.setattr(settings, "whatsapp_pair_code_sync_push", True, raising=False)
    await _call(mod)
    assert calls["awaited_push"] == 1


# ── F8: idempotency ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_request_without_a_key_behaves_exactly_as_today(wired):
    """Every installed build (123–126) sends no header. Two keyless calls must
    both reach the agent, exactly as they do now — no cache, no replay, and no
    Idempotency-Key invented on their behalf."""
    mod, calls, _cfg = wired

    await _call(mod, key=None)
    await _call(mod, key=None)

    assert len(calls["proxy"]) == 2
    for _m, _p, kw in calls["proxy"]:
        assert not kw.get("extra_headers"), (
            "a keyless request grew an Idempotency-Key it never sent"
        )


@pytest.mark.asyncio
async def test_a_retry_with_the_same_key_replays_without_a_second_mint(wired):
    mod, calls, _cfg = wired

    first = await _call(mod, key="k-1")
    second = await _call(mod, key="k-1")

    assert first == second
    assert len(calls["proxy"]) == 1, (
        "the retry minted a second code — the first one, which the user may "
        "already be typing, is now dead"
    )


@pytest.mark.asyncio
async def test_a_different_key_mints_again(wired):
    """A genuinely new tap is a new pairing, not a replay of the old code."""
    mod, calls, _cfg = wired
    await _call(mod, key="k-1")
    await _call(mod, key="k-2")
    assert len(calls["proxy"]) == 2


@pytest.mark.asyncio
async def test_the_key_is_forwarded_to_the_agent(wired):
    mod, calls, _cfg = wired
    await _call(mod, key="k-1")
    _m, _p, kw = calls["proxy"][0]
    assert (kw.get("extra_headers") or {}).get("Idempotency-Key") == "k-1"


@pytest.mark.asyncio
async def test_an_expired_entry_is_not_replayed(wired, monkeypatch):
    mod, calls, _cfg = wired
    await _call(mod, key="k-1")
    # Age the entry past the TTL without sleeping.
    for k in list(mod._pair_code_idem):
        stamp, value = mod._pair_code_idem[k]
        mod._pair_code_idem[k] = (stamp - mod._PAIR_CODE_IDEM_TTL_S - 1, value)
    await _call(mod, key="k-1")
    assert len(calls["proxy"]) == 2


def test_the_platform_ttl_is_shorter_than_the_sidecar_pairing_window():
    """A replayed code that outlives its own pairing window is WORSE than the
    503 it replaces: the user types a dead code and re-pairs into WhatsApp's
    rate limiter."""
    from app.api import agent_setup as mod

    import re as _re

    # Tolerant: the RELATIONSHIP is the property, not the spelling. The old
    # literal probe ("pairingInProgress = false; }, 150000)") failed on a
    # reformat and passed on a retune, which is exactly backwards.
    m = _re.search(
        r"pairingInProgress\s*=\s*false;?\s*\}\s*,\s*(\d+)\)", _SIDECAR_SRC,
    )
    assert m, "the sidecar's pairing window timer is no longer recognisable"
    window_s = int(m.group(1)) / 1000.0
    m2 = _re.search(r"PAIR_CODE_IDEM_TTL_MS\s*=\s*(\d+)", _SIDECAR_SRC)
    assert m2, "the sidecar's replay TTL is no longer recognisable"
    replay_s = int(m2.group(1)) / 1000.0

    assert replay_s < window_s, "the sidecar would replay a dead code"
    assert mod._PAIR_CODE_IDEM_TTL_S < replay_s <= window_s


def test_the_sidecar_replays_only_inside_a_live_pairing_window():
    """The sidecar is where the real guarantee lives: the platform cache is
    per-process and there are two replicas."""
    assert "lastPairCodeKey === idemKey" in _SIDECAR_SRC
    assert "&& pairingInProgress" in _SIDECAR_SRC, (
        "the sidecar would replay a code after its pairing window closed"
    )
    import re as _re
    assert _re.search(r"PAIR_CODE_IDEM_TTL_MS\s*=\s*\d+", _SIDECAR_SRC)


def test_the_sidecar_does_not_start_a_second_mint_for_an_in_flight_key():
    """Two teardowns for one key is the socket race the whole fix is about."""
    assert "pairCodeInFlight.key === idemKey" in _SIDECAR_SRC
    assert "await pairCodeInFlight.promise" in _SIDECAR_SRC


def test_the_sidecar_mint_still_wipes_and_rebuilds_for_a_genuine_request():
    """The idempotency short-circuit must not have removed the teardown a REAL
    pairing needs."""
    assert "async function mintPairCode(digits)" in _SIDECAR_SRC
    mint = _SIDECAR_SRC.split("async function mintPairCode(digits)", 1)[1]
    head = mint[:3000]
    assert "wipeAuthDir();" in head
    assert "requestPairingCodeWhenReady(s, digits)" in head


# ── fix lane A: the platform HOP itself, and the body it is keyed on ──
#
# `_agent_qr_proxy` is monkeypatched out in every case above, so the one place
# that can silently drop both halves of the F2/F8 contract was never executed.


class _ProxyRow:
    agent_url = "https://agent-abc.agents.toup.ai"
    agent_api_key = "agent-key"


class _ProxyDb:
    async def execute(self, _stmt):
        class _R:
            def first(_s):
                return _ProxyRow()
        return _R()


class _ProxyResp:
    def __init__(self, code, payload=None, text=""):
        self.status_code = code
        self._payload = payload
        self.text = text
        self.content = b"{}"

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _patch_proxy_httpx(monkeypatch, resp, sent):
    import httpx

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, method, url, headers=None, json=None):
            sent.append({"method": method, "url": url,
                         "headers": dict(headers or {}), "json": json})
            return resp

    monkeypatch.setattr(httpx, "AsyncClient", _Client)


@pytest.mark.asyncio
async def test_the_proxy_forwards_a_typed_agent_503_verbatim(monkeypatch):
    """The agent answers `whatsapp_adapter_starting` with a `retry_after_s`
    the client honours instead of guessing a sleep. Rewriting it into the
    generic copy leaves every installed build on its 1800 ms fallback."""
    from fastapi import HTTPException
    from app.api import agent_setup as mod

    sent: list = []
    _patch_proxy_httpx(monkeypatch, _ProxyResp(503, {
        "detail": {"error": "whatsapp_adapter_starting",
                   "message": "still starting", "retry_after_s": 3},
    }), sent)

    with pytest.raises(HTTPException) as exc:
        await mod._agent_qr_proxy(
            "POST", "/api/whatsapp/qr/pair-code", _UID, _ProxyDb(),
        )
    assert exc.value.status_code == 503
    assert isinstance(exc.value.detail, dict)
    assert exc.value.detail.get("retry_after_s") == 3
    assert (exc.value.headers or {}).get("Retry-After") == "3"


@pytest.mark.asyncio
async def test_the_proxy_puts_the_idempotency_key_on_the_wire(monkeypatch):
    """`extra_headers` reaching the FAKE proxy proves nothing about whether
    the real one puts it on the request beside X-Agent-Key."""
    from app.api import agent_setup as mod

    sent: list = []
    _patch_proxy_httpx(monkeypatch, _ProxyResp(200, {"ok": True}), sent)

    out = await mod._agent_qr_proxy(
        "POST", "/api/whatsapp/qr/pair-code", _UID, _ProxyDb(),
        json_body={"phone": "+14155552671"},
        extra_headers={"Idempotency-Key": "k-1"},
    )
    assert out == {"ok": True}
    assert len(sent) == 1
    hdrs = sent[0]["headers"]
    assert hdrs.get("X-Agent-Key") == "agent-key"
    assert hdrs.get("Idempotency-Key") == "k-1", (
        "the key never left the platform; the sidecar can never replay"
    )
    assert sent[0]["json"] == {"phone": "+14155552671"}


@pytest.mark.asyncio
async def test_an_agent_503_without_a_typed_body_keeps_the_old_copy(monkeypatch):
    """Mixed fleet: image 1cd801aacb11 has no typed body."""
    from fastapi import HTTPException
    from app.api import agent_setup as mod

    sent: list = []
    _patch_proxy_httpx(monkeypatch, _ProxyResp(503, {"detail": "not active"}), sent)
    with pytest.raises(HTTPException) as exc:
        await mod._agent_qr_proxy("POST", "/api/whatsapp/qr/start", _UID, _ProxyDb())
    assert exc.value.status_code == 503
    assert isinstance(exc.value.detail, str)


# ── the idempotency key names a REQUEST, not a caller ─────────────────


@pytest.mark.asyncio
async def test_the_same_key_with_a_different_phone_is_not_a_replay(wired):
    """The response echoes `phone`. A body-blind cache hands back a code
    minted for the number the user just corrected, with that number on it."""
    mod, calls, _cfg = wired
    await _call(mod, key="k-1", phone="+14155552671")
    await _call(mod, key="k-1", phone="+14155559999")
    assert len(calls["proxy"]) == 2, (
        "the cache replayed a code minted for a different number"
    )


@pytest.mark.asyncio
async def test_the_same_key_and_the_same_phone_still_replays(wired):
    mod, calls, _cfg = wired
    a = await _call(mod, key="k-1", phone="+14155552671")
    b = await _call(mod, key="k-1", phone="+14155552671")
    assert a == b
    assert len(calls["proxy"]) == 1


def test_the_cache_key_does_not_store_the_number_in_the_clear():
    from app.api import agent_setup as mod

    slot = mod._pair_code_idem_slot("u-1", "k-1", "+14155552671")
    assert "4155552671" not in "".join(slot)


@pytest.mark.asyncio
async def test_an_overflow_evicts_the_oldest_not_everyone(wired):
    """`clear()` let one tenant's key burst flush every other tenant's live
    entry on this replica, turning their next retry into a second mint."""
    mod, calls, _cfg = wired
    await _call(mod, key="k-keep", phone="+14155552671")
    assert len(mod._pair_code_idem) == 1

    now = asyncio.get_event_loop().time()
    for i in range(mod._PAIR_CODE_IDEM_MAX):
        mod._pair_code_idem[("other", f"k{i}")] = (now, {"ok": True})
    mod._pair_code_idem_put("burst", "k-new", {"ok": True}, "+14155550000")

    survivors = len(mod._pair_code_idem)
    assert survivors > mod._PAIR_CODE_IDEM_MAX // 2, (
        f"only {survivors} entries survived an overflow — this is a clear()"
    )


def test_the_sidecar_binds_its_replay_to_the_number_too():
    """The durable half. The platform cache is per-process and there are two
    replicas; the sidecar is the only place that can refuse a second mint."""
    assert "lastPairCodeDigits === digits" in _SIDECAR_SRC, (
        "the sidecar replays a cached code for any number"
    )
    assert "pairCodeInFlight.digits === digits" in _SIDECAR_SRC, (
        "a request for a DIFFERENT number rides the in-flight mint"
    )
    assert "lastPairCodeDigits = digits" in _SIDECAR_SRC
