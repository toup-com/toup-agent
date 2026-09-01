"""R40 — a tenant that did not answer is a 503, and a pool member is never
re-provisioned through the named path.

THE INCIDENT (2026-08-31, agent-871bac24, measured in the platform trail):

    15:47:23  [pool_service] Claimed toup-agent-pool-17 for user 51d4ed2f
    15:48:18  [update_container_env] pool refresh failed for 51d4ed2f: .
              Falling through to recreate.                          (×2)
    15:48:19  free_tier_activation: container env push failed
              err=RuntimeError: bridge create_tenant failed: 500
    15:48:27  provisioned_container user=51d4ed2f prefix=51d4ed2f
    …a neighbouring tenant's container recreate saturates the shared host…
    15:51:06  Day-chats proxy https://agent-871bac24…/api/day-chats failed:
    15:51:06  [day_chats] local SELECT skipped (table absent in this DB)
    15:51:06  GET /api/day-chats -> 200 10099ms          ← 200. EMPTY. A LIE.
    15:51:07  GET /api/sessions?limit=120 -> 422         ← the fallback, dead
    15:51:07  GET /api/day-chats/2026-08-31/messages -> 200 10125ms
    15:51:18  GET /api/automations/summary -> 200 22677ms ← correct, too late

The user's phone showed "Beginning of your history" over an untouched
account and he reported his messages deleted. Nothing was deleted; the API
answered 200 with an empty body because the proxy helper flattened a
ten-second timeout into `None`, and `None` meant "nothing to say".

Three invariants are pinned here:

  1. A tenant read that did not succeed is 503 — never 200, never 404.
  2. `provision_container(recreate=True)` REFUSES a pool member. The named
     path binds `toup_agent_<user_prefix>`; a pool slot's data lives in
     `toup_agent_feedNNNN` (bridge/pool_addon.py::_pool_db_name). Swapping
     one for the other is silent, permanent data loss.
  3. The timeout hierarchy points the right way: the platform's upstream
     budget for a READ is below the mobile client's 15 s request budget, so
     the platform is always the side that answers first.
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import httpx
import pytest

os.environ.setdefault("ENVIRONMENT", "test")

BACKEND_DIR = Path(__file__).resolve().parent.parent


def code_only(src: str) -> str:
    """`src` with comments and DOCSTRINGS removed; ordinary literals kept.

    A source-level guard must read CODE. Three of the assertions below first
    fired on the prose explaining why the thing they forbid is forbidden — and
    one of them SURVIVED the mutation that deleted the call it was pinning,
    because the docstring above that call named the same endpoint. A guard
    that greps its own rationale passes and fails for the wrong reasons.

    Only triple-quoted strings are dropped: the URLs and log markers these
    tests pin are ordinary one-line literals, so stripping every STRING token
    would delete the evidence along with the prose.
    """
    import io as _io
    import tokenize as _tok

    triples = ('"' * 3, "'" * 3)
    kept = []
    try:
        for tok in _tok.generate_tokens(_io.StringIO(src).readline):
            if tok.type == _tok.COMMENT:
                continue
            if tok.type == _tok.STRING:
                body = tok.string.lstrip("rbfuRBFU")
                if body[:3] in triples:
                    continue
            kept.append((tok.start[0], tok.string))
    except Exception:
        return src
    lines = {}
    for ln, text in kept:
        lines.setdefault(ln, []).append(text)
    return "\n".join(" ".join(v) for _k, v in sorted(lines.items()))


def code_index(block: str, snippet: str) -> int:
    """Position of `snippet` in `block`'s CODE, ignoring whitespace.

    The plain `str.index` ORDER assertions kept firing on the docstrings that
    NAME the call they forbid — `_verify_and_heal_pool_claim`'s own docstring
    mentions `provision_container(recreate=True)` 3800 characters before the
    call itself.
    """
    return re.sub(r"\s+", "", code_only(block)).index(re.sub(r"\s+", "", snippet))


def has_code(block: str, snippet: str) -> bool:
    """Is `snippet` present in `block` as CODE, ignoring whitespace?

    `code_only` re-joins tokens with single spaces, so `if not is_pool:`
    comes back as `if not is_pool :`. Comparing on the whitespace-free forms
    keeps these assertions about structure rather than about formatting —
    a reflow must not fail a guard, and must not silently pass one either.
    """
    return re.sub(r"\s+", "", snippet) in re.sub(r"\s+", "", code_only(block))


_DAY_CHATS_SRC = (BACKEND_DIR / "app/api/day_chats.py").read_text(encoding="utf-8")
_SESSIONS_SRC = (BACKEND_DIR / "app/api/sessions.py").read_text(encoding="utf-8")
_DHS_SRC = (BACKEND_DIR / "app/services/docker_host_service.py").read_text(encoding="utf-8")
_AP_SRC = (BACKEND_DIR / "app/api/automations_proxy.py").read_text(encoding="utf-8")
_KR_SRC = (BACKEND_DIR / "app/services/agent_key_rotation.py").read_text(encoding="utf-8")
_RECOVER_SRC = (BACKEND_DIR / "app/api/messages_recover.py").read_text(encoding="utf-8")
_POOL_SRC = (BACKEND_DIR / "app/services/pool_service.py").read_text(encoding="utf-8")
_TP_SRC = (BACKEND_DIR / "app/api/tenant_proxy.py").read_text(encoding="utf-8")


# ── A fake agent HTTP client ─────────────────────────────────────────────

class _Resp:
    def __init__(self, status: int, payload=None, text: str = ""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class _FakeClient:
    """Replays a scripted list of outcomes, one per request, and counts them."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def get(self, url, **kw):
        self.calls += 1
        out = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(out, Exception):
            raise out
        return out


def _patch_client(monkeypatch, client):
    import app.services.agent_http as agent_http
    monkeypatch.setattr(agent_http, "get_agent_http_client", lambda: client)


class AsyncNoop:
    """An async callable that always returns `value`, whatever it is given."""

    def __init__(self, value):
        self.value = value

    async def __call__(self, *a, **k):
        return self.value


# ── 1. The proxy helper tells the truth ──────────────────────────────────

def test_day_chats_proxy_raises_on_timeout_and_retries_once(monkeypatch):
    """A ReadTimeout is not an empty history. It is also worth one retry —
    the failure this guards against is a transient host stall."""
    from app.api import day_chats

    client = _FakeClient([httpx.ReadTimeout("")])
    _patch_client(monkeypatch, client)

    with pytest.raises(day_chats.AgentUnreachable):
        asyncio.run(day_chats._proxy_day_chats("https://agent-x", "k"))
    assert client.calls == day_chats._PROXY_ATTEMPTS == 2, (
        "a transient tenant stall must be retried before it is believed"
    )


def test_day_chats_proxy_raises_on_5xx(monkeypatch):
    """A 500 from the tenant used to return None SILENTLY — the helper only
    logged inside `except`, so this failure mode had no trail at all."""
    from app.api import day_chats

    client = _FakeClient([_Resp(500, text="boom")])
    _patch_client(monkeypatch, client)

    with pytest.raises(day_chats.AgentUnreachable):
        asyncio.run(day_chats._proxy_day_chats("https://agent-x", "k"))
    assert client.calls == 2


def test_day_chats_proxy_forwards_a_tenant_4xx_without_retrying(monkeypatch):
    """`app-conversation/{id}` uses 404 to mean "no conversation yet". That is
    an ANSWER: forward it, do not retry it, and do not call it unreachable —
    a client that reads 503 as that 404 starts a second thread beside the one
    the user is already in."""
    from app.api import day_chats

    client = _FakeClient([_Resp(404, text="not found")])
    _patch_client(monkeypatch, client)

    with pytest.raises(day_chats.AgentSaidNo) as ei:
        asyncio.run(day_chats._proxy_day_chats("https://agent-x", "k", "app-conversation/a"))
    assert ei.value.status == 404
    assert client.calls == 1, "a considered 4xx must not be retried"


def test_day_chats_proxy_returns_the_body_on_200(monkeypatch):
    from app.api import day_chats

    client = _FakeClient([_Resp(200, payload=[{"local_date": "2026-08-31"}])])
    _patch_client(monkeypatch, client)
    got = asyncio.run(day_chats._proxy_day_chats("https://agent-x", "k"))
    assert got == [{"local_date": "2026-08-31"}]
    assert client.calls == 1


def test_a_late_success_is_still_a_success(monkeypatch):
    """First attempt stalls, second answers. The user must never see the
    stall — that is the whole point of the retry."""
    from app.api import day_chats

    client = _FakeClient([httpx.ConnectTimeout(""), _Resp(200, payload=[])])
    _patch_client(monkeypatch, client)
    assert asyncio.run(day_chats._proxy_day_chats("https://agent-x", "k")) == []
    assert client.calls == 2


def test_sessions_proxy_has_the_same_contract(monkeypatch):
    """`/api/sessions` is the FALLBACK the mobile client reaches for when
    day-chats fails. It degraded identically, so both doors answered
    "you have no history" in the same second."""
    from app.api import sessions

    client = _FakeClient([httpx.ReadTimeout("")])
    _patch_client(monkeypatch, client)
    with pytest.raises(sessions.SessionsAgentUnreachable):
        asyncio.run(sessions._proxy_sessions("https://agent-x", "k", ""))
    assert client.calls == sessions._SESSIONS_PROXY_ATTEMPTS == 2


# ── 2. The routes never fall through to the platform DB ──────────────────

class _User:
    id = "871bac24-c366-42b5-b224-8802c73aef3a"
    timezone = "America/Toronto"


class _ExplodingDB:
    """A platform DB that answers every SELECT the way the real one does for an
    AGENT_ONLY table: `UndefinedTableError`.

    If a route reaches this object after a failed tenant hop, the old code
    caught the error and returned `JSONResponse([])` — HTTP 200, empty. So a
    test that reaches it and still gets a 503 is proving the fall-through is
    gone, and one that gets a 200 is reproducing the incident.
    """

    def __init__(self):
        self.selects = 0

    async def execute(self, *a, **k):
        self.selects += 1
        from sqlalchemy.exc import ProgrammingError
        raise ProgrammingError("SELECT …", {}, Exception("UndefinedTableError"))

    async def rollback(self):
        pass

    def begin_nested(self):
        class _N:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *a):
                return False
        return _N()


def _run_route(coro):
    from fastapi import HTTPException
    try:
        return ("ok", asyncio.run(coro))
    except HTTPException as e:
        return ("http", e)


def test_list_day_chats_answers_503_and_never_touches_the_platform_db(monkeypatch):
    """THE INCIDENT, reproduced at the route. Tenant times out → the route used
    to fall through to a SELECT that raises UndefinedTable, catch it, and answer
    `200 []`. Measured in production as `GET /api/day-chats -> 200 10099ms`."""
    from app.api import day_chats

    monkeypatch.setattr(day_chats, "_get_agent_proxy_info",
                        AsyncNoop(("https://agent-871bac24", "k")))
    _patch_client(monkeypatch, _FakeClient([httpx.ReadTimeout("")]))
    db = _ExplodingDB()

    kind, res = _run_route(day_chats.list_day_chats(
        before=None, limit=30, current_user=_User(), db=db,
    ))
    assert kind == "http", f"the route returned {res!r} instead of raising — an empty 200 is the defect"
    assert res.status_code == 503
    assert res.headers.get("X-Toup-Reason") == "agent_unreachable"
    assert db.selects == 0, (
        "the route still reached the platform DB after a failed tenant hop — "
        "that SELECT is what became `200 []`"
    )


def test_day_messages_answers_503_and_never_touches_the_platform_db(monkeypatch):
    from app.api import day_chats

    monkeypatch.setattr(day_chats, "_get_agent_proxy_info",
                        AsyncNoop(("https://agent-871bac24", "k")))
    _patch_client(monkeypatch, _FakeClient([_Resp(502, text="bad gateway")]))
    db = _ExplodingDB()

    kind, res = _run_route(day_chats.get_day_chat_messages(
        date_str="2026-08-31", limit=500, current_user=_User(), db=db,
    ))
    assert kind == "http"
    assert res.status_code == 503
    assert db.selects == 0


def test_a_user_with_NO_agent_still_takes_the_local_path(monkeypatch):
    """The local path is not deleted — it is what a user without a tenant uses,
    and for them an empty answer is the truth. Only a user WITH an agent whose
    agent went silent gets the 503."""
    from app.api import day_chats

    monkeypatch.setattr(day_chats, "_get_agent_proxy_info", AsyncNoop(None))
    db = _ExplodingDB()
    kind, res = _run_route(day_chats.list_day_chats(
        before=None, limit=30, current_user=_User(), db=db,
    ))
    assert kind == "ok", "a user with no agent must not be told their agent is unreachable"
    assert db.selects >= 1, "the local path was skipped for a user who has only that"


def test_app_conversation_forwards_the_tenants_own_404(monkeypatch):
    """404 there means "no conversation for this app yet" and the client uses it
    to start one. Turning it into 503 leaves the user with no thread; turning
    503 into it starts a SECOND thread beside the one they were in."""
    from app.api import day_chats

    monkeypatch.setattr(day_chats, "_get_agent_proxy_info",
                        AsyncNoop(("https://agent-871bac24", "k")))
    _patch_client(monkeypatch, _FakeClient([_Resp(404, text="no conversation")]))
    kind, res = _run_route(day_chats.resolve_app_conversation(
        app_id="snake", current_user=_User(), db=_ExplodingDB(),
    ))
    assert kind == "http"
    assert res.status_code == 404, f"the tenant's 404 became {res.status_code}"




def test_no_day_chats_route_can_reach_a_local_select_after_a_failed_hop():
    """The structural half of invariant 1.

    Every `_proxy_day_chats` call must be wrapped in a try that converts
    `AgentUnreachable` into `_unreachable(...)`, and must RETURN on the
    proxy's answer rather than testing `if data is not None` — the latter is
    exactly the fall-through that produced the 200 [].
    """
    assert "if data is not None" not in _DAY_CHATS_SRC, (
        "the `if data is not None:` fall-through is back — a failed tenant "
        "hop reaches the platform-DB SELECT again, and that answers 200 []"
    )
    calls = _DAY_CHATS_SRC.count("await _proxy_day_chats(")
    handlers = _DAY_CHATS_SRC.count("except AgentUnreachable as e:")
    assert calls == 3, f"expected 3 proxy call sites, found {calls}"
    assert handlers == calls, (
        f"{calls} proxy call sites but {handlers} AgentUnreachable handlers — "
        "an unhandled one falls through to the empty platform DB"
    )
    assert _DAY_CHATS_SRC.count("except AgentSaidNo as e:") == calls


def test_sessions_routes_have_no_fall_through_either():
    calls = _SESSIONS_SRC.count("await _proxy_sessions(")
    handlers = _SESSIONS_SRC.count("except SessionsAgentUnreachable as e:")
    assert calls == 4, f"expected 4 proxy call sites, found {calls}"
    assert handlers == calls


def test_the_unreachable_answer_is_503_with_retry_after():
    """503, not 500 and certainly not 200: the client must be able to tell
    "we could not read your history" from "you have no history", because it
    CACHES the second one."""
    from app.api.day_chats import AgentUnreachable, _unreachable

    exc = _unreachable(AgentUnreachable("ReadTimeout()"))
    assert exc.status_code == 503
    assert exc.headers.get("Retry-After")
    assert exc.headers.get("X-Toup-Reason") == "agent_unreachable"
    assert "safe" in str(exc.detail).lower(), (
        "the sentence a user reads must say their data is intact — the whole "
        "report was 'my messages got deleted'"
    )


def test_sessions_list_limit_admits_the_shipped_clients_ask():
    """`getDayChats(n)` in the app falls back to `getSessions(n * 4)`, i.e.
    120 and 360, against a route that capped at 100. Every call on the
    recovery path 422'd — two of them are in the incident trail. Binaries
    already on phones cannot be clamped, so the ceiling moves here."""
    m = re.search(r"limit: int = Query\(20, ge=1, le=(\d+)\)", _SESSIONS_SRC)
    assert m, "list_sessions' limit Query() changed shape"
    assert int(m.group(1)) >= 360, (
        f"list_sessions caps limit at {m.group(1)}; the shipped app asks for 360"
    )


# ── 3. A pool member is never re-provisioned through the named path ──────

def test_provision_container_refuses_a_pool_member():
    """The named path binds `toup_agent_<user_prefix>`; a pool slot's data is
    in `toup_agent_feedNNNN`. Nothing migrates between them, so this recreate
    is silent, permanent data loss — and it was reachable from four callers."""
    from app.services import docker_host_service as dhs

    class _MC:
        container_name = "toup-agent-pool-17"
        status = "running"
        pin_image_tag = None
        user_id = "51d4ed2f-0000-0000-0000-000000000000"

    class _Res:
        def scalar_one_or_none(self):
            return _MC()

    class _DB:
        async def execute(self, *a, **k):
            return _Res()

    with pytest.raises(dhs.PoolMemberSwapRefused):
        asyncio.run(dhs.provision_container(
            _DB(), "51d4ed2f-0000-0000-0000-000000000000", recreate=True,
        ))


def test_the_refusal_is_the_default_and_the_override_is_explicit():
    assert "allow_pool_swap: bool = False" in _DHS_SRC
    assert "raise PoolMemberSwapRefused(" in _DHS_SRC
    # …and no caller quietly opts out of it. `code_only`, because the docstring
    # explaining the override names it.
    assert not has_code(_DHS_SRC, "allow_pool_swap=True"), (
        "something in this module opted out of the pool-swap guard"
    )


def test_update_container_env_never_recreates_on_a_bridge_hiccup():
    """The caller that actually fired on 2026-08-31. It answered ONE transient
    bridge 500 by moving the user to a different, empty database."""
    fn = _DHS_SRC.split("async def _update_container_env(")[1].split("\nasync def ")[0]
    # Only the POOL branch. The tail of this function is the NAMED-tenant env
    # push, which is what `provision_container(recreate=True)` is for and must
    # keep doing — the swap is only lethal for a container whose data lives in
    # a slot database.
    marker = 'startswith("toup-agent-pool-")'
    assert marker in fn, "the pool branch has gone"
    pool_branch = fn[fn.index(marker):]
    # The branch ends where the function's unindented `return` for the named
    # path begins.
    tail = "\n    return await provision_container("
    if tail in pool_branch:
        pool_branch = pool_branch[:pool_branch.index(tail)]
    block = pool_branch
    assert not has_code(block, "provision_container("), (
        "update_container_env's POOL branch can reach provision_container "
        "again — that is the pool→named database swap, from a transient "
        "bridge error"
    )
    raw = _DHS_SRC.split("async def _update_container_env(")[1].split("\nasync def ")[0]
    assert "[POOL-REFRESH-FAILED]" in raw and "[POOL-DRIFT]" in raw, (
        "both give-up paths must be greppable; they were silent, and the one "
        "log line they did emit was `%s` of an httpx timeout — the empty string"
    )
    assert has_code(block, "for attempt in range(1, 4):"), (
        "the refresh must retry before it gives up — a single transient bridge "
        "500 is what fired this path in production"
    )


def test_the_180s_reconciler_skips_pool_members():
    """Both arms of its predicate (`image_tag LIKE '%:latest'`, `container_id
    IS NULL`) fire on healthy pool rows in normal operation — a slot's image
    tag is stale bookkeeping by design. Unattended, every 180 s."""
    block = _DHS_SRC.split("async def backfill_sentinel_image_containers(")[1]
    block = block.split("\nasync def ")[0]
    assert 'startswith("toup-agent-pool-")' in block
    # …and the skip must sit ABOVE the sentinel branch that forces the
    # unconditional rebuild, or it is decoration.
    assert block.index('startswith("toup-agent-pool-")') < block.index("image_is_sentinel ="), (
        "the pool skip runs after the sentinel arm has already been decided"
    )


def test_restart_container_uses_the_pool_endpoint_for_a_pool_member():
    """`/v1/tenants/<prefix>/restart` 404s for a pool container, and the old
    code recorded that as `status='error'` — the state the pool reclaimer
    reads as "this slot is free", and a reclaimed slot is TRUNCATEd."""
    block = _DHS_SRC.split("async def restart_container(")[1]
    block = block.split("\nasync def ")[0]
    # `code_only`: the docstring names the endpoint too, and a guard that
    # matches its own rationale survives the mutation that deletes the call.
    # (It did — this assertion passed with the pool endpoint removed.)
    assert has_code(block, '"/v1/pool/restart-member"')
    assert has_code(block, "if not is_pool:"), (
        "a failed restart must not paint a pool member 'error' — that hands "
        "it to the reclaimer"
    )


def test_key_rotation_pushes_a_pool_members_key_in_place():
    assert _KR_SRC.count('startswith("toup-agent-pool-")') == 2, (
        "both the forward and the rollback path must route a pool member "
        "through update_container_env, not through a named recreate"
    )


def test_the_agent_never_proxies_to_itself():
    """`serving_locally()` is true inside an agent container: the AGENT_ONLY
    tables ARE this process's database. `tenant_proxy.agent_proxy_info` has
    always checked it; the three hand-rolled copies never did, so an agent
    could resolve its own `agent_configs` row and hop to itself over the public
    internet. That was harmless while a failed hop fell through to a local
    SELECT — and it is a 503 over a perfectly readable local database now that
    it does not."""
    for label, src in (
        ("day_chats", _DAY_CHATS_SRC),
        ("sessions", _SESSIONS_SRC),
        ("messages_recover", _RECOVER_SRC),
    ):
        fn = src.split("async def _get_agent_proxy_info(")[1].split("\nasync def ")[0]
        assert has_code(fn, "if serving_locally():"), (
            f"{label}._get_agent_proxy_info lost the serving_locally guard — the "
            f"agent will proxy its own reads to itself and 503 when that hop fails"
        )
        # ORDER: before the AgentConfig lookup, or the agent still pays for it.
        assert fn.index("serving_locally") < fn.index("AgentConfig"), (
            f"{label}: the guard runs after the config lookup it exists to skip"
        )


def test_a_wedged_pool_member_is_restarted_in_place_before_anything_else():
    """`_verify_and_heal_pool_claim` used to answer an unreachable claim with
    `provision_container(recreate=True)` unconditionally — the named path,
    which binds `toup_agent_<prefix>` while the member's data is in the slot's
    own database. A wedged member is usually just wedged, and
    `/v1/pool/restart-member` fixes that without touching the database."""
    fn = _POOL_SRC.split("async def _verify_and_heal_pool_claim(")[1].split("\nasync def ")[0]
    assert has_code(fn, "await restart_container(heal_db, user_id)"), (
        "the in-place restart is gone — the heal goes straight to the named "
        "recreate again"
    )
    assert code_index(fn, "restart_container(heal_db") < code_index(fn, "provision_container("), (
        "the named recreate runs before the in-place restart it exists to avoid"
    )


def test_the_only_allowed_pool_swap_is_gated_on_a_fresh_claim():
    """After the in-place restart, the named recreate is the last resort — and
    it is safe only while the claim is FRESH: at that point the user has never
    sent a message and the slot database holds nothing to strand. It is the one
    caller in the tree that may opt out of `PoolMemberSwapRefused`."""
    fn = _POOL_SRC.split("async def _verify_and_heal_pool_claim(")[1].split("\nasync def ")[0]
    assert has_code(fn, "allow_pool_swap=is_pool"), (
        "the opt-out moved out of the heal guard, or stopped being conditional "
        "on the container actually being a pool member"
    )
    assert has_code(fn, "if is_pool and not _claim_is_fresh(mc):"), (
        "the freshness gate is gone — an established pool member reaching this "
        "line is silently moved onto an empty database"
    )
    assert code_index(fn, "_claim_is_fresh(mc)") < code_index(fn, "allow_pool_swap=is_pool"), (
        "the gate runs after the swap it is supposed to prevent"
    )

    # UNKNOWN must answer FALSE. A failed lookup, a row with no timestamps and
    # a mock are none of them evidence that the slot holds nothing.
    from app.services.pool_service import _claim_is_fresh, POOL_CLAIM_FRESH_WINDOW
    from datetime import datetime, timedelta, timezone

    class _Bare:
        pass

    assert _claim_is_fresh(None) is False, "a missing row must not read as fresh"
    assert _claim_is_fresh(_Bare()) is False, "a row with no timestamp must not read as fresh"

    class _Old:
        started_at = datetime.now(timezone.utc) - timedelta(days=30)
    assert _claim_is_fresh(_Old()) is False

    class _New:
        started_at = datetime.now(timezone.utc) - timedelta(seconds=20)
    assert _claim_is_fresh(_New()) is True

    class _Naive:
        started_at = datetime.utcnow() - timedelta(seconds=20)
    assert _claim_is_fresh(_Naive()) is True, (
        "a naive timestamp (which is what SQLAlchemy hands back for these "
        "columns) must be read as UTC, not crash the comparison"
    )

    # …and the window is actually SHORT. A guard that admits every claim is not
    # a guard: widening it to ten years left every assertion above green.
    assert POOL_CLAIM_FRESH_WINDOW <= timedelta(hours=1), (
        f"POOL_CLAIM_FRESH_WINDOW is {POOL_CLAIM_FRESH_WINDOW} — the heal fires "
        f"within `budget_s` (30 s) of the claim, so anything approaching an hour "
        f"stops excluding the established members this exists to protect"
    )

    # Nothing ELSE in the tree opts out. `code_only` per file, because
    # `docker_host_service` necessarily NAMES the override in the docstring and
    # the comment that explain when it is allowed — a raw grep counts those and
    # a `__pycache__` blob besides.
    callers = []
    for path in sorted((BACKEND_DIR / "app").rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        src = path.read_text(encoding="utf-8")
        if "allow_pool_swap=" not in src:
            continue
        code = code_only(src).replace(" ", "")
        # An opt-out is any `allow_pool_swap=<something other than False>`.
        if re.search(r"allow_pool_swap=(?!False\b)\w", code):
            callers.append(str(path.relative_to(BACKEND_DIR)))
    assert callers == ["app/services/pool_service.py"], (
        "the set of callers that opt out of the pool-swap guard changed; every "
        "one of them silently moves a user onto an empty database unless it can "
        f"prove the slot holds nothing:\n  " + "\n  ".join(callers)
    )


def test_the_reconciler_still_backstops_a_wedged_pool_member():
    """Skipping pool rows outright would leave one stuck forever once the
    30 s claim guard is spent — this loop is their only durable backstop. It
    restarts them in place instead, and only when a health probe says so."""
    block = _DHS_SRC.split("async def backfill_sentinel_image_containers(")[1]
    block = block.split("\nasync def ")[0]
    assert has_code(block, "await restart_container(db, mc.user_id)"), (
        "the reconciler no longer heals wedged pool members at all"
    )
    assert has_code(block, "healthy = True"), (
        "a FAILED health probe must not trigger a restart of somebody's live "
        "container — absence of evidence is not evidence of a wedge"
    )
    # The pool arm must come BEFORE the sentinel arm that forces the rebuild.
    assert code_index(block, 'startswith("toup-agent-pool-")') < code_index(block, "image_is_sentinel ="), (
        "the pool arm runs after the sentinel arm has already decided to rebuild"
    )


def test_one_config_push_per_user_at_a_time():
    """Twelve call sites reach `update_container_env` and nothing coordinated
    them. The 2026-08-31 trail has `pool refresh failed for 51d4ed2f` logged
    TWICE in the same millisecond — two concurrent bridge calls for one user,
    which on the code as it then stood was two concurrent container recreates
    on a shared VPS host. That is the load that made a neighbouring tenant
    time out."""
    wrapper = _DHS_SRC.split("async def update_container_env(")[1].split("\nasync def ")[0]
    assert has_code(wrapper, "async with lock:"), (
        "the per-user push lock is gone — twelve callers can hit the bridge for "
        "one user at once again"
    )
    assert has_code(wrapper, "await _update_container_env(db, user_id, agent_config)")
    # PER USER, not global: two different users pushing at once is normal.
    assert has_code(wrapper, "_env_push_lock(user_id)"), (
        "the lock is no longer keyed by user — a global one serialises the whole "
        "fleet behind one slow bridge call"
    )
    # …and the dict cannot grow with every user the platform has ever pushed.
    assert has_code(wrapper, "_ENV_PUSH_LOCKS.pop(user_id, None)")


def test_a_redeploying_tenant_is_still_the_authority():
    """`deploy_status` is "deploying" for the whole of a redeploy and "error"
    the moment a stale-deploy sweep fires 15 minutes later — while the
    container is very often still up and holding the user's entire history.
    Gating the proxy on `== "active"` skipped it for exactly those users and
    served them the platform's own empty tables: the same defect through a
    second door, and one no retry can help with because nothing failed."""
    for label, src in (
        ("day_chats", _DAY_CHATS_SRC),
        ("sessions", _SESSIONS_SRC),
        ("messages_recover", _RECOVER_SRC),
    ):
        fn = src.split("async def _get_agent_proxy_info(")[1].split("\nasync def ")[0]
        assert not has_code(fn, 'deploy_status == "active"'), (
            f"{label}._get_agent_proxy_info gates the proxy on deploy_status again — "
            f"a tenant mid-redeploy is skipped and its user is told they have no history"
        )
        # …and it still requires BOTH halves of the credential, or the hop
        # would be attempted with no key and 401 for every user without an agent.
        assert has_code(fn, "if row and row.agent_url and row.agent_api_key:"), (
            f"{label}: the credential check is gone"
        )


# ── 4. The timeout hierarchy points the right way ────────────────────────

def test_a_read_fails_inside_the_mobile_clients_budget():
    """The phone aborts at 15 s. This proxy waited 30 s, so for the whole
    15–30 s band the client had already drawn "the server didn't answer"
    while the platform was still holding the connection — and the answer,
    when it came at 22.7 s, went nowhere.

    The number that matters is the WHOLE LADDER — `attempts × timeout +
    backoff` — not the per-attempt timeout. The first version of this
    assertion checked only the per-attempt value and passed while day-chats
    could run 2 × 10 s = 20 s, re-creating the inversion it exists to catch.
    """
    from app.api import automations_proxy as ap
    from app.api import day_chats
    from app.api import sessions
    from app.api import tenant_proxy as tp

    MOBILE_BUDGET_S = 15.0
    ladders = (
        ("automations read", 1, ap._READ_TIMEOUT_S, 0.0),
        ("day-chats", day_chats._PROXY_ATTEMPTS, day_chats._PROXY_TIMEOUT_S,
         day_chats._PROXY_BACKOFF_S),
        ("sessions", sessions._SESSIONS_PROXY_ATTEMPTS,
         sessions._SESSIONS_PROXY_TIMEOUT_S, sessions._SESSIONS_PROXY_BACKOFF_S),
        ("tenant_proxy read", 2, tp.READ_PROXY_TIMEOUT_S, tp.READ_PROXY_BACKOFF_S),
    )
    for name, attempts, per_attempt, backoff in ladders:
        worst = attempts * per_attempt + (attempts - 1) * backoff
        assert worst < MOBILE_BUDGET_S, (
            f"{name}'s worst case is {attempts}×{per_attempt}s + {backoff}s = "
            f"{worst}s, at or above the client's {MOBILE_BUDGET_S}s — the client "
            f"gives up first and the platform's honest 503 arrives after nobody "
            f"is listening"
        )
    assert ap._WRITE_TIMEOUT_S > ap._READ_TIMEOUT_S, (
        "a mutation must keep the long budget: abandoning it does not undo it"
    )
    # And a read really does take the SHORT budget in tenant_proxy — the
    # default there is 15 s, which two attempts of would run to 30 s.
    fn = _TP_SRC.split("async def proxy_to_agent(")[1].split("\n\n\n")[0]
    assert has_code(fn, "timeout = READ_PROXY_TIMEOUT_S"), (
        "tenant_proxy reads retry on DEFAULT_PROXY_TIMEOUT_S again — 2 × 15 s"
    )
    assert has_code(fn, "if read_only and timeout == DEFAULT_PROXY_TIMEOUT_S:"), (
        "the short read budget must not override a caller's EXPLICIT timeout — "
        "uploads pass their own 180 s and must keep it"
    )


def test_a_slow_tenant_is_504_and_an_unreachable_one_is_502():
    """Two different operational facts, and the trail could not tell them
    apart — `str()` of an httpx timeout is the empty string, so the old
    "failed: %s" logged nothing after the colon."""
    block = _AP_SRC.split("async def _proxy(")[1].split("\n@router")[0]
    # `has_code` and the trailing ` as`, not a bare substring: the first
    # version of this assertion was `"except httpx.TimeoutException" in block`,
    # and it SURVIVED the mutation that renamed the caught class to
    # `httpx.TimeoutException_UNUSED` — a prefix match cannot tell a live
    # handler from a disabled one.
    assert has_code(block, "except httpx.TimeoutException as e:"), (
        "the 504 arm is gone — a slow tenant is being reported as unreachable"
    )
    assert has_code(block, "except httpx.RequestError as e:")
    assert has_code(block, "status_code=504")
    assert has_code(block, "status_code=502")
    assert 'X-Toup-Reason": "agent_slow"' in block
    assert 'X-Toup-Reason": "agent_unreachable"' in block
    # The timeout arm must be BEFORE the generic RequestError arm, or it is
    # unreachable: TimeoutException is a subclass of RequestError.
    import httpx as _hx
    assert issubclass(_hx.TimeoutException, _hx.RequestError), (
        "the premise changed — re-check the ordering rule below"
    )
    assert block.index("except httpx.TimeoutException as e:") < block.index("except httpx.RequestError as e:"), (
        "httpx.TimeoutException is a RequestError — catching the general case "
        "first makes the 504 arm dead code"
    )


def test_no_active_agent_is_not_rendered_as_an_empty_library():
    """`_get_agent_target` requires `deploy_status == 'active'`. A container
    that is redeploying is not, while the user's automations sit intact
    inside it — and the client's rule is "404 = the feature is absent, render
    the empty state". So a redeploy painted "No automations yet"."""
    block = _AP_SRC.split("async def _proxy(")[1].split("\n@router")[0]
    assert "No active agent for this user" not in block
    assert 'X-Toup-Reason": "agent_provisioning"' in block
    assert "status_code=503" in block


# ── 5. Every log line in these paths can name its own cause ──────────────

def test_no_httpx_exception_is_logged_with_str():
    """`str(httpx.ReadTimeout())` is the EMPTY STRING. The production trail
    for this incident reads `Day-chats proxy … failed: ` and
    `pool refresh failed for 51d4ed2f: .` — neither names what happened, and
    that is why the outage took a log correlation to read at all."""
    assert str(httpx.ReadTimeout("")) == "", (
        "the premise changed: httpx timeouts now stringify to something"
    )
    for label, src in (
        ("day_chats", _DAY_CHATS_SRC),
        ("sessions", _SESSIONS_SRC),
        ("automations_proxy", _AP_SRC),
    ):
        for line in src.splitlines():
            if "failed" in line and "%s" in line and "logger." not in line:
                continue
        assert "repr(e)" in src or "%r" in src, (
            f"{label} still formats its transport exceptions with %s only"
        )


# ── 6. Teardown: a slot that is never released keeps its database ────────
#
# Four pool slots were found leaked on the production fleet on 2026-08-31 —
# 09, 17, 27 and 28, every container still running 23 hours later — and THREE
# of them belonged to accounts that had already been deleted. `destroy_container`
# chose pool-vs-named from `container_name`, which the R40 swap rewrites, and
# swallowed every bridge failure into a `True` return that `user_deletion`
# recorded as `container_destroyed: true`.
#
# These EXECUTE the real function. Source reading cannot tell you that a branch
# was taken; only calling it can.

def _teardown_case(container_name, *, release_ok=True, release_found=False,
                   delete_status=204, delete_raises=None, holder=None):
    """Run destroy_container against fakes. Returns (result, calls, row, db).

    `holder`, when given, is filled with the row and db BEFORE the call, so a
    test that expects an exception can still inspect what the call did to them.
    """
    from app.services import docker_host_service as dhs

    calls = {"released": None, "deleted": []}

    class _MC:
        def __init__(self):
            self.container_name = container_name
            self.status = "running"
            self.stopped_at = None
            self.user_id = "51d4ed2f-0000-0000-0000-000000000000"

    row = _MC()

    class _Res:
        def scalar_one_or_none(self):
            return row

    class _DB:
        def __init__(self):
            self.committed = False

        async def execute(self, *a, **k):
            return _Res()

        async def commit(self):
            self.committed = True

    from app.services import pool_service

    async def _fake_release(prefix=None, user_id=None):
        calls["released"] = {"prefix": prefix, "user_id": user_id}
        return pool_service.PoolRelease(ok=release_ok, found=release_found,
                                        detail="" if release_ok else "bridge 500")

    class _Resp:
        status_code = delete_status

        def raise_for_status(self):
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def delete(self, path):
            calls["deleted"].append(path)
            if delete_raises:
                raise delete_raises
            return _Resp()

    db = _DB()
    if holder is not None:
        holder["row"] = row
        holder["db"] = db
        holder["calls"] = calls
    orig_release = pool_service.release_pool_member
    orig_client = dhs._bridge_client
    pool_service.release_pool_member = _fake_release
    dhs._bridge_client = lambda *a, **k: _Client()
    try:
        result = asyncio.run(dhs.destroy_container(db, row.user_id))
    finally:
        pool_service.release_pool_member = orig_release
        dhs._bridge_client = orig_client
    return result, calls, row, db


def test_a_pool_member_is_released_and_not_named_deleted():
    ok, calls, row, db = _teardown_case("toup-agent-pool-17", release_found=True)
    assert ok is True
    assert calls["released"]["user_id"].startswith("51d4ed2f")
    assert calls["deleted"] == [], (
        "a pool member has no named container — DELETE /v1/tenants/<prefix> "
        "would be aimed at something that does not exist"
    )
    assert row.status == "deleted" and db.committed


def test_a_named_tenant_STILL_asks_the_bridge_to_release_a_slot():
    """The whole defect: `container_name` is the platform's bookkeeping, and
    the swap rewrites it while the bridge keeps the slot ASSIGNED. Asking is
    one idempotent round trip; not asking leaked four slots."""
    ok, calls, row, db = _teardown_case("toup-agent-51d4ed2f", release_found=False)
    assert ok is True
    assert calls["released"] is not None, (
        "a named row skipped the pool release — this is exactly how slots 09, "
        "27 and 28 outlived the accounts that owned them"
    )
    assert calls["deleted"] == ["/v1/tenants/51d4ed2f"]


def test_a_swapped_user_gets_BOTH_teardowns():
    """Alireza's shape on 2026-08-31: named locally, ASSIGNED on the bridge.
    An if/else can only ever clean up half of that."""
    ok, calls, row, db = _teardown_case("toup-agent-51d4ed2f", release_found=True)
    assert ok is True
    assert calls["released"] is not None and calls["deleted"] == ["/v1/tenants/51d4ed2f"]
    assert row.status == "deleted"


def test_a_failed_release_is_raised_and_the_row_is_NOT_marked_deleted():
    from app.services import docker_host_service as dhs
    with pytest.raises(dhs.ContainerTeardownIncomplete):
        _teardown_case("toup-agent-pool-17", release_ok=False)


def test_a_failed_named_delete_is_raised_and_the_row_is_NOT_marked_deleted():
    from app.services import docker_host_service as dhs
    with pytest.raises(dhs.ContainerTeardownIncomplete):
        _teardown_case("toup-agent-51d4ed2f",
                       delete_raises=httpx.ConnectError("bridge down"))


def test_the_row_survives_a_failed_teardown():
    """The row is the only thing still pointing at a container we could not
    remove. Marking it deleted is what orphans it — which is the state the
    fleet audit found four slots in."""
    from app.services import docker_host_service as dhs

    holder = {}
    with pytest.raises(dhs.ContainerTeardownIncomplete):
        _teardown_case("toup-agent-pool-17", release_ok=False, holder=holder)

    assert holder["row"].status == "running", (
        "the row was marked deleted despite the teardown failing"
    )
    assert holder["row"].stopped_at is None
    assert holder["db"].committed is False, (
        "a failed teardown committed anyway"
    )


def test_nothing_to_destroy_is_still_False_not_an_exception():
    """`managed_agents.destroy` maps False to 404. "There is no container" and
    "I could not remove the container" must not collapse into one answer."""
    from app.services import docker_host_service as dhs

    class _Res:
        def scalar_one_or_none(self):
            return None

    class _DB:
        async def execute(self, *a, **k):
            return _Res()

    assert asyncio.run(dhs.destroy_container(_DB(), "51d4ed2f")) is False


def test_the_deletion_cascade_treats_teardown_as_a_real_hard_fail():
    """`container_destroyed=True` is set on any non-exceptional return, so the
    honesty of the receipt rests entirely on destroy_container raising."""
    src = (BACKEND_DIR / "app/services/user_deletion.py").read_text(encoding="utf-8")
    block = src.split("# ── Container teardown via bridge (HARD-FAIL)")[1].split("# ── OpenAI")[0]
    assert has_code(block, "await destroy_container(db, user_id)")
    assert code_index(block, "await destroy_container(db, user_id)") < code_index(block, "container_destroyed = True"), (
        "the receipt is stamped before the teardown it describes"
    )
    assert has_code(block, "raise DeletionAbortedError(DeletionStep.CONTAINER"), (
        "a failed teardown no longer aborts the cascade — the user's row would "
        "be wiped while their database stayed on the VPS"
    )


def test_a_user_facing_destroy_reports_502_not_404_when_teardown_fails():
    src = (BACKEND_DIR / "app/api/managed_agents.py").read_text(encoding="utf-8")
    assert has_code(src, "except docker_host_service.ContainerTeardownIncomplete"), (
        "the destroy route still turns an incomplete teardown into a 500, or "
        "worse into the 404 that means 'there was nothing there'"
    )
    assert has_code(src, "HTTPException(502")
