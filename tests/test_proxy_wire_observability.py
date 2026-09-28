"""R48 — what the PROVIDER received, on the lines that already exist.

The agent fingerprints its tools array before the proxy ever sees it
(`[PERF] prefix_head tools=`), and the proxy then dedups it, caps it to 128
with a protected set derived from *this request's* `tool_choice`, and
prunes the allowlist to match. So the one array that heads the provider's
cached prefix has never been measured on either side of the hop.

PRODUCTION (supervisor extract E2, test account, 2026-09-19/20) — three
gpt-5.6-terra requests, identical agent-side fingerprints, identical
`cache_key_hash=3c275b30`, all `174 > 128, dropped 46 (namespace-fair)`,
all `cached=0`, one of them TEN SECONDS after a 45,347-token cache WRITE,
and a DIFFERENT dropped list every time.

Read that exactly: a wire-array fork was OBSERVED TO COINCIDE with three
misses on one account. It is not a demonstration that caching cannot work
over 128 tools, and not a claim that every over-cap turn must miss — a
partial-prefix hit up to the first differing tool, an unchanged protected
set across turns and early eviction are all still open, and the documented
30-minute retention is a MINIMUM lifetime, not a ceiling. What the three
requests do establish is that the agent's own instrument cannot see the
difference, because it hashes the PRE-cap array — and a stable pre-cap
hash says nothing either way about what the wire carried.

These tests pin the log-only instrument that makes that countable:

  (a) `tools_sha` DIFFERS when two requests differ only in
      `tool_choice.allowed_tools` over a >128 array, and is EQUAL when the
      forwarded arrays are equal. That is the fork, pinned as a visible fact
      rather than left as an inference.
  (b) `retention=` survives the agent's move from `prompt_cache_retention`
      to `prompt_cache_options.ttl` — the instrument that must verify that
      migration used to go dark exactly on it.
  (c) the provider's `x-request-id` reaches a log line at INFO, from inside
      a generator that is the only holder of the httpx response.
  (d) pre_ms / ttfb_ms / total_ms split a scalar that has always spanned
      auth-to-last-token.
  (e) with the flag off, every line is BYTE-IDENTICAL to R47 and the body
      sent upstream is byte-identical to the body sent with the flag on.

Lane: RUN_MODE=platform.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import types

import pytest

from app.api import llm_proxy as lp

CANARY_UID = "0a1b2c3d-aaaa-4bbb-8ccc-000000000001"
OTHER_UID = "0e9f8d7c-aaaa-4bbb-8ccc-000000000002"


# ── helpers ──────────────────────────────────────────────────────────


def _core(n: int, *, desc: str = "x") -> list:
    """`n` core tools — no `__`, i.e. one namespace, exactly like the
    shipped definitions (63 of them, counted 2026-09-20)."""
    return [{"name": f"core_{i:03d}", "description": desc} for i in range(n)]


def _connector(ns: str, n: int) -> list:
    return [{"name": f"{ns}__op_{i}", "description": "x"} for i in range(n)]


def _over_cap_tools() -> list:
    """174 tools shaped like the production sample: a large core namespace
    plus several small connector namespaces."""
    tools = _core(120)
    for ns, n in (("slack", 10), ("teams", 10), ("outlook", 10),
                  ("github", 10), ("notion", 8), ("automations", 6)):
        tools += _connector(ns, n)
    assert len(tools) == 174
    return tools


def _allowed(names: list[str]) -> dict:
    """Responses-shape allowed_tools tool_choice."""
    return {"type": "allowed_tools", "mode": "auto",
            "tools": [{"type": "function", "name": n} for n in names]}


class _FakeRequest:
    def __init__(self, body: dict, headers: dict | None = None,
                 body_delay: float = 0.0):
        self._body = body
        self.headers = headers or {}
        self._body_delay = body_delay

    async def json(self) -> dict:
        # `await request.json()` is a socket RECEIVE of the agent's body, not
        # a parse of something already in hand. `body_delay` stands in for
        # that upload so `body_ms` can be proved to hold it (R48 review, B2).
        if self._body_delay:
            await asyncio.sleep(self._body_delay)
        return self._body


def _cache_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("[CACHE] user=")]


@pytest.fixture
def responses_driver(monkeypatch):
    """Drive the REAL `proxy_responses` with auth/budget/persistence faked.

    Everything under test — dedup, `_cap_tools`, `_prune_tool_choice`, both
    [CACHE] lines and the timing stamps — runs unmodified. The test double
    for the backend keeps production's signature, checked below, because a
    double whose signature has drifted from the method it stands in for
    cannot notice the method changing.
    """
    state: dict = {"sent": [], "auth_delay": 0.0, "ttfb_delay": 0.0,
                   "body_delay": 0.0}

    async def fake_auth(request, db):
        if state["auth_delay"]:
            await asyncio.sleep(state["auth_delay"])
        return types.SimpleNamespace(user_id=state["user_id"])

    async def fake_budget(cfg, provider, db):
        return None

    async def fake_log_event(*a, **kw):
        return None

    monkeypatch.setattr(lp, "_auth_agent", fake_auth)
    monkeypatch.setattr(lp, "_check_budget", fake_budget)
    monkeypatch.setattr(lp, "_log_event", fake_log_event)

    # The stream finally opens its own session; keep it off any database.
    class _NullSession:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    import app.db.database as _dbmod
    monkeypatch.setattr(_dbmod, "async_session_maker", lambda: _NullSession())

    async def drive(body: dict, *, user_id: str = CANARY_UID,
                    sse: bytes = b"", request_id: str | None = "req_abc123",
                    auth_delay: float = 0.0, ttfb_delay: float = 0.0,
                    body_delay: float = 0.0,
                    json_payload: dict | None = None,
                    resp_headers: dict | None = None,
                    headers: dict | None = None):
        state["user_id"] = user_id
        state["auth_delay"] = auth_delay
        state["sent_headers"] = None

        async def fake_stream(self, body, api_key, meta=None):
            if ttfb_delay:
                await asyncio.sleep(ttfb_delay)
            state["sent"].append(json.loads(json.dumps(body)))
            if meta is not None and request_id is not None:
                meta["request_id"] = request_id
            yield sse

        async def fake_responses(self, body, api_key):
            """The JSON twin. Exercised so the non-stream usage line is
            covered by BEHAVIOUR and not by a source grep — a mutation that
            dumped every upstream header into it survived four test files
            (R48 review, N2)."""
            if ttfb_delay:
                await asyncio.sleep(ttfb_delay)
            state["sent"].append(json.loads(json.dumps(body)))
            headers = dict(resp_headers or {})
            if request_id is not None:
                headers["x-request-id"] = request_id
            payload = json_payload or {
                "usage": {"input_tokens": 44647, "output_tokens": 5,
                          "input_tokens_details": {"cached_tokens": 0}},
            }
            return types.SimpleNamespace(
                status_code=200, headers=headers, content=b"",
                json=lambda: payload)

        monkeypatch.setattr(lp.OpenAIBackend, "responses_stream", fake_stream)
        monkeypatch.setattr(lp.OpenAIBackend, "responses", fake_responses)
        monkeypatch.setattr(
            lp, "_route_chat", lambda model, cfg: (lp._openai, "sk-test"))
        resp = await lp.proxy_responses(
            _FakeRequest(body, headers=headers, body_delay=body_delay),
            db=None)
        if not body.get("stream"):
            return resp
        chunks = [c async for c in resp.body_iterator]
        return b"".join(chunks)

    drive.state = state  # type: ignore[attr-defined]
    return drive


def _body(tools: list | None = None, **extra) -> dict:
    out = {
        "model": "gpt-5.6-terra",
        "stream": True,
        "input": [{"role": "user", "content": "hi"}],
        "prompt_cache_key": "0a1b2c3d:all",
    }
    if tools is not None:
        out["tools"] = tools
    out.update(extra)
    return out


@pytest.mark.parametrize("wire", ["chat", "responses"])
@pytest.mark.parametrize("failure", ["exception", "status"])
@pytest.mark.parametrize("canary", [False, True])
async def test_non_stream_failures_keep_request_join_on_summary(
    monkeypatch, caplog, wire, failure, canary,
):
    """A failed JSON request still joins its request and event log lines."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    monkeypatch.setattr(lp.settings, "llm_proxy_wire_observability", False,
                        raising=False)
    monkeypatch.setattr(
        lp.settings, "llm_proxy_wire_observability_canary_user_ids",
        CANARY_UID if canary else "", raising=False,
    )

    async def fake_auth(request, db):
        return types.SimpleNamespace(user_id=CANARY_UID)

    async def fake_budget(config, provider, db):
        return None

    calls = []

    async def fake_log_event(*args, **kwargs):
        calls.append(kwargs)

    async def fake_provider(body, api_key):
        if failure == "exception":
            raise RuntimeError("upstream unavailable")
        return types.SimpleNamespace(status_code=429, content=b"rate limited")

    backend = types.SimpleNamespace(
        name="openai", chat=fake_provider, responses=fake_provider,
    )
    monkeypatch.setattr(lp, "_auth_agent", fake_auth)
    monkeypatch.setattr(lp, "_check_budget", fake_budget)
    monkeypatch.setattr(lp, "_log_event", fake_log_event)
    monkeypatch.setattr(lp, "_route_chat", lambda model, config: (backend, "test"))

    if wire == "responses":
        body = _body(_core(2), stream=False)
        handler = lp.proxy_responses
    else:
        body = {"model": "gpt-5.6-terra", "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _core(2)}
        handler = lp.proxy_chat

    with pytest.raises(lp.HTTPException):
        await handler(_FakeRequest(body, headers={
            "x-toup-trace": "a1b2c3d4.7",
        }), db=None)

    assert len(calls) == 1
    if canary:
        request_line = _cache_lines(caplog)[0]
        assert calls[0]["wire_suffix"] == (
            " rid=%s trace=a1b2c3d4.7" % _field(request_line, "rid")
        )
    else:
        assert calls[0]["wire_suffix"] == ""


# ── (b) retention survives the prompt_cache_options migration ────────


def test_retention_reports_prompt_cache_options_ttl():
    """The GPT-5.6+ spelling. Before R48 this printed retention=none."""
    has_key, _h, retention = lp._cache_log_fields(
        {"prompt_cache_key": "k", "prompt_cache_options": {"ttl": "30m"}})
    assert has_key is True
    assert retention == "opt:30m"


def test_legacy_retention_still_works_and_wins():
    assert lp._cache_log_fields(
        {"prompt_cache_retention": "24h"})[2] == "24h"
    # Both present: report what a pre-5.6 model will actually honour.
    assert lp._cache_log_fields({
        "prompt_cache_retention": "24h",
        "prompt_cache_options": {"ttl": "30m"},
    })[2] == "24h"


@pytest.mark.parametrize("opts", [None, {}, {"ttl": ""}, "30m", 30, [1]])
def test_malformed_cache_options_fall_back_to_none(opts):
    """A telemetry field may never be the thing that raises."""
    assert lp._cache_log_fields({"prompt_cache_options": opts})[2] == "none"


def test_absent_everything_is_still_none():
    assert lp._cache_log_fields({}) == (False, "none", "none")


# ── (FINAL round 2, BLOCKING-1) `retention=` is a CLIENT BODY STRING ──
#
# `prompt_cache_options.ttl` is the route THIS patch added, and it went
# onto both [CACHE] lines through an f-string with no allowlist, no bound
# and no type check — UNCONDITIONALLY, i.e. with the flag off. The
# reviewer's payloads, reproduced by the builder before the fix:
#
#   {"ttl": "5m\n[CACHE] user=deadbeef … tools_sha=forged00 tc=auto"}
#       -> 'opt:5m\n[CACHE] user=deadbeef …'   a well-formed FORGED G7 row
#   {"ttl": {"a": "b"}}
#       -> "opt:{'a': 'b'}"                    the wholesale str() shape
#                                              that made `tc=` blocking
#
# The legacy `prompt_cache_retention` route had the identical hole at
# `4f0e9fe1` and is closed by the same call. Two levels, because the unit
# alone was green while B1 shipped.


@pytest.mark.parametrize("value,expected", [
    # every real spelling, reported verbatim
    ("24h", "24h"),
    ("1h", "1h"),
    ("opt:30m", "opt:30m"),
    ("opt:5m", "opt:5m"),
    ("x" * 32, "x" * 32),          # the bound itself
    # the log-structure set
    ("x" * 33, "invalid"),
    ("24h\nFORGED", "invalid"),
    ("24h\n", "invalid"),          # the `\A…\Z` half — `$` would accept it
    ("24h\r\n[CACHE] user=deadbeef", "invalid"),
    ("24 h", "invalid"),           # a space splits a field
    ("24\th", "invalid"),
    ("24\x00h", "invalid"),
    ("24\x1bh", "invalid"),        # an ANSI escape into an operator's tty
    ("", "invalid"),
    (None, "invalid"),
    (7, "invalid"),
    ({"a": "b"}, "invalid"),
])
def test_log_atom_is_an_allowlist(value, expected):
    assert lp._log_atom(value, max_len=32) == expected


def test_log_atom_echoes_nothing_of_a_rejected_value():
    """Not a prefix, not a length, not a `%r`. Quoting the rejected string
    into the line is the bug the validation exists to prevent."""
    payload = "24h\n[CACHE] user=deadbeef model=x SECRET-MARKER"
    out = lp._log_atom(payload, max_len=32)
    assert out == "invalid"
    assert "deadbeef" not in out and "SECRET-MARKER" not in out
    assert "24" not in out and str(len(payload)) not in out


def test_the_log_atom_length_guard_short_circuits_before_the_regex():
    """Structural, and deliberately separate, for the reason
    `test_the_length_guard_short_circuits_before_the_regex` gives about
    `_TRACE_MAX_LEN`: dropping the guard changes NO behaviour, so only a
    named test keeps it. `_LOG_ATOM_RE` is unbounded by design — the bound
    is the caller's, because `retention` and a provider id want different
    ones — which makes the `len()` the only thing between a megabyte body
    field and the regex engine."""
    src = inspect.getsource(lp._log_atom)
    assert src.index("max_len") < src.index("_LOG_ATOM_RE.match")
    assert lp._log_atom("x" * 4096, max_len=32) == "invalid"


@pytest.mark.parametrize("body", [
    {"prompt_cache_options": {"ttl": "5m\n[CACHE] user=deadbeef model=x "
                              "has_cache_key=True cache_key_hash=forged00 "
                              "retention=24h tools_n=1 tools_sha=forged00 "
                              "tools_sent=1 dropped_n=0 tc=auto"}},
    {"prompt_cache_options": {"ttl": {"a": "b"}}},
    {"prompt_cache_options": {"ttl": "5m " * 40}},
    {"prompt_cache_retention": "24h\n[CACHE] user=deadbeef x=1"},
    {"prompt_cache_retention": "24h\r\nFORGED"},
])
def test_a_hostile_retention_renders_invalid_and_leaks_nothing(body):
    _k, _h, retention = lp._cache_log_fields(body)
    assert retention == "invalid"
    assert "deadbeef" not in retention
    assert "\n" not in retention and "\r" not in retention


async def test_a_hostile_ttl_reaches_no_log_line_with_the_flag_off(
    responses_driver, canary_off, caplog,
):
    """HANDLER level, FLAG OFF — the state the fleet deploys in, and the
    state the reviewer's probe failed in. `_cache_log_fields` is NOT behind
    the canary gate (NOTES §4 item 1), so the unit fix has to hold here."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    body = _body(_core(2), prompt_cache_options={
        "ttl": "5m\n[CACHE] user=deadbeef model=x has_cache_key=True "
               "cache_key_hash=forged00 retention=24h tools_n=1 "
               "tools_sha=forged00 tools_sent=1 dropped_n=0 tc=auto"})
    await responses_driver(body, sse=_SSE)
    everything = "\n".join(r.getMessage() for r in caplog.records)
    assert "deadbeef" not in everything, "FORGED TEXT REACHED THE LOG"
    assert "forged00" not in everything
    # exactly ONE request-side line, i.e. no line was manufactured
    # (`prompt_tokens=` is the usage-side line's discriminator — with the
    # flag ON `cache_key_hash=` appears on BOTH, inside the timing suffix)
    req = [ln for ln in _cache_lines(caplog) if "prompt_tokens=" not in ln]
    assert len(req) == 1, _cache_lines(caplog)
    assert _field(req[0], "retention") == "invalid"


async def test_a_hostile_ttl_reaches_no_log_line_with_the_flag_on(
    responses_driver, canary_on, caplog,
):
    """The canary state. Same payload, and the gated fields must still be
    the only thing appended — a forged line would carry a `tools_sha` of
    the sender's choosing straight into G7."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    body = _body(_core(2), prompt_cache_options={
        "ttl": "5m\n[CACHE] user=deadbeef tools_sha=forged00"})
    await responses_driver(body, sse=_SSE)
    everything = "\n".join(r.getMessage() for r in caplog.records)
    assert "deadbeef" not in everything and "forged00" not in everything
    req = [ln for ln in _cache_lines(caplog) if "prompt_tokens=" not in ln]
    assert len(req) == 1, _cache_lines(caplog)
    assert _field(req[0], "retention") == "invalid"
    assert _field(req[0], "tools_sha") != "forged00"
    # and the usage-side line, which repeats retention, is clean too
    usage = [ln for ln in _cache_lines(caplog) if "prompt_tokens=" in ln]
    assert len(usage) == 1
    assert _field(usage[0], "retention") == "invalid"


async def test_a_real_ttl_is_unchanged_end_to_end(
    responses_driver, canary_off, caplog,
):
    """The allowlist must not change a single line the fleet produces.
    R47 rendered `retention=opt:30m` for this body and so does R48."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(
        _body(_core(2), prompt_cache_options={"ttl": "30m"}), sse=_SSE)
    req = [ln for ln in _cache_lines(caplog) if "prompt_tokens=" not in ln]
    assert len(req) == 1
    assert _field(req[0], "retention") == "opt:30m"


# ── the residual this patch does NOT close, executed rather than claimed


async def test_the_model_field_is_still_caller_written_pinned(
    responses_driver, canary_off, caplog,
):
    """PINNING TEST — it asserts a KNOWN-OPEN door, on purpose.

    `model=` is printed raw on both [CACHE] lines, on the `llm_proxy`
    summary line, on the `[credits]` lines and on the two `[LLM-PROXY]`
    upstream WARNINGs. All PRE-EXISTING at `4f0e9fe1`; `/responses` only
    requires `str(model).lower().startswith(("gpt","o1","o3","o4"))`, which
    a newline-bearing value satisfies. This patch does not close it: every
    remaining render site is inside `_log_event`, shared with embeddings,
    images, kie and internal_llm, and changing their log text is a wider
    behaviour delta than a log-only patch's envelope — closing it at only
    the [CACHE] sites would leave the class open through the summary line
    and be exactly the gate-that-cannot-fail this test exists to prevent
    claiming.

    So: **G7/G8 rows are caller-influenced**, by the holder of that
    tenant's own agent token. This test executes that fact. The day someone
    closes it, this test FAILS — and NOTES §9, §11 G19 and the follow-up
    list must be updated in the same change rather than left stale.
    """
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    body = _body(_core(2))
    body["model"] = ("gpt-5.6-terra\n[CACHE] user=deadbeef model=x "
                     "has_cache_key=True cache_key_hash=forged00 "
                     "retention=24h tools_n=1 tools_sha=forged00")
    await responses_driver(body, sse=_SSE)
    everything = "\n".join(r.getMessage() for r in caplog.records)
    assert "deadbeef" in everything, (
        "model= is no longer caller-written — close the loop: update "
        "NOTES §9's enforcement table, §11 G19 and the follow-up list"
    )


# ── tools digest + tool_choice shape (pure) ──────────────────────────


def test_digest_is_stable_and_order_insensitive():
    a = [{"name": "t", "description": "d", "parameters": {"x": 1}}]
    b = [{"parameters": {"x": 1}, "description": "d", "name": "t"}]
    assert lp._tools_digest(a) == lp._tools_digest(b)
    assert len(lp._tools_digest(a)) == 8


def test_digest_moves_on_a_one_character_description_change():
    assert lp._tools_digest(_core(3)) != lp._tools_digest(_core(3, desc="y"))


def test_digest_moves_when_membership_changes():
    assert lp._tools_digest(_core(5)) != lp._tools_digest(_core(4))


def test_digest_reveals_nothing():
    """8 hex chars over a one-way hash — no name, no description."""
    d = lp._tools_digest([{"name": "slack__send_message",
                           "description": "TOP SECRET"}])
    assert "slack" not in d and "SECRET" not in d


def test_digest_of_a_non_list_and_of_undigestible_input():
    assert lp._tools_digest(None) == "none"
    assert lp._tools_digest("tools") == "none"
    # TypeError from json.dumps — our own assembly put a non-JSON value in.
    assert lp._tools_digest([{"name": object()}]) == "unserialisable"


def test_the_two_undigestible_causes_are_reported_separately():
    """`unhashable` collapsed two failures with different owners (FINAL
    review round 2, non-blocking 3). A TypeError is OUR tool-definition
    assembly; a ValueError is a third party's UTF-16 mis-serialisation
    reaching us as a lone surrogate. An operator reading `tools_sha=` has
    to be able to tell those apart, so they are two words."""
    assert lp._tools_digest([{"name": {1, 2}}]) == "unserialisable"
    assert lp._tools_digest([{"name": "t", "d": "\ud800"}]) == "unencodable"
    circular: list = []
    circular.append(circular)
    # ValueError("Circular reference detected") — raised by dumps, not encode.
    assert lp._tools_digest([{"name": "t", "d": circular}]) == "unencodable"


# ── (B1) a telemetry field must not be the thing that raises ─────────
#
# `ensure_ascii=False` leaves a lone surrogate in the dumped string and
# `str.encode("utf-8")` then raises `UnicodeEncodeError`. With the encode
# outside the try that raise left the HANDLER, before the upstream call,
# and lost the turn — a 500 caused by a log field, in the canary state
# this patch asks to enter. Two tests: the unit, and the handler, because
# the unit alone was already green when the defect shipped.


@pytest.mark.parametrize("payload", [
    "\ud800",                        # a lone high surrogate
    "\udfff",                        # a lone low surrogate
    "ok\ud83d",                      # half a surrogate PAIR (emoji)
])
def test_a_lone_surrogate_is_unencodable_and_never_raises(payload):
    assert lp._tools_digest(
        [{"name": "t", "description": payload}]) == "unencodable"


def test_a_lone_surrogate_survives_json_loads_so_the_route_in_is_real():
    """The premise, not the fix: a surrogate reaches a tools array by an
    ordinary route, so this is not a hand-built input."""
    assert json.loads('{"description": "\\ud800"}')["description"] == "\ud800"


async def test_a_surrogate_in_a_tool_does_not_kill_the_turn(
    responses_driver, canary_on, caplog,
):
    """HANDLER level. The real `proxy_responses`, canary on, one tool whose
    description carries a lone surrogate: the turn completes, the upstream
    call happens, and the line reports `tools_sha=unencodable`."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    body = _body([{"type": "function", "name": "t",
                   "description": "\ud800"}])
    out = await responses_driver(body, sse=_SSE)
    assert out == _SSE                             # the turn produced bytes
    assert responses_driver.state["sent"], "upstream was never called"
    line = _cache_lines(caplog)[0]
    assert _field(line, "tools_sha") == "unencodable"
    assert _field(line, "tools_n") == "1"


@pytest.mark.parametrize("tc,expected", [
    (None, "none"),
    ("auto", "auto"),
    ("required", "required"),
    ({"type": "function", "name": "grep"}, "function"),
    ({"type": "allowed_tools", "tools": [{"name": "a"}, {"name": "b"}]},
     "allowed:2"),
    ({"type": "allowed_tools",
      "allowed_tools": {"tools": [{"function": {"name": "a"}}]}}, "allowed:1"),
    ({"type": "allowed_tools"}, "allowed:0"),
    (42, "other"),
])
def test_tool_choice_kind(tc, expected):
    assert lp._tool_choice_kind({"tool_choice": tc} if tc is not None else {}) \
        == expected


def test_tool_choice_kind_never_logs_a_name():
    assert "grep" not in lp._tool_choice_kind(
        {"tool_choice": {"type": "function", "name": "grep"}})


# ── (B2) `tc=` is an ALLOWLIST, not a passthrough ────────────────────
#
# `tool_choice` is a CLIENT-SUPPLIED body field. Returning a string form
# verbatim, or `str()`-ing a dict's `type`, put an arbitrary-length,
# newline-bearing client string onto a [CACHE] line in a shared
# multi-tenant log stream — the same log-forging primitive the
# `x-toup-trace` allowlist exists to stop, one function away. The h11 wall
# that rejects a newline in a HEADER value does not apply to a body field.

_TC_VOCABULARY = {"none", "auto", "required", "function", "other"}


def _tc_is_in_vocabulary(out: str) -> bool:
    return out in _TC_VOCABULARY or (
        out.startswith("allowed:") and out[len("allowed:"):].isdigit())


_HOSTILE_TOOL_CHOICES = [
    "auto\n2026-09-20 12:00:00 WARNING [CACHE] user=deadbeef model=x",
    "auto\r\n[CACHE] user=deadbeef",
    "AUTO",                                    # case is not the vocabulary
    "auto ",                                   # a trailing space splits a field
    "none tools_sha=forged00",                 # field injection without \n
    "A" * 5000,                                # unbounded length
    "",                                        # empty string
    {"type": "x\nFORGED [CACHE] user=deadbeef"},
    {"type": {"secret_tool_names": ["slack__send_message"]}},
    {"type": ["a", "b"]},
    {"type": 7},
    {"type": None},
    {"type": "allowed_tools", "tools": "not-a-list"},
    {"no_type_at_all": 1},
    ["auto"],
    7,
    7.5,
    True,
]


@pytest.mark.parametrize("tc", _HOSTILE_TOOL_CHOICES)
def test_a_hostile_tool_choice_renders_only_a_vocabulary_word(tc):
    out = lp._tool_choice_kind({"tool_choice": tc})
    assert _tc_is_in_vocabulary(out), repr(out)
    assert "\n" not in out and "\r" not in out and " " not in out
    assert len(out) <= 32


@pytest.mark.parametrize("tc", _HOSTILE_TOOL_CHOICES)
def test_no_fragment_of_a_hostile_tool_choice_survives(tc):
    """Not just 'safe-looking' — nothing recognisable from the payload is
    carried through. `FORGED`, `deadbeef`, `secret_tool_names` and the
    5000-char run all have to be absent from the output."""
    out = lp._tool_choice_kind({"tool_choice": tc})
    for probe in ("FORGED", "deadbeef", "CACHE", "secret_tool_names",
                  "slack", "forged00", "AAAA"):
        assert probe not in out, (probe, out)


def test_the_whole_vocabulary_is_reachable():
    """A two-directional control: the allowlist would also pass the tests
    above if it returned "other" for everything, so each word must still be
    produced by the input that means it."""
    got = {
        lp._tool_choice_kind({}),
        lp._tool_choice_kind({"tool_choice": "auto"}),
        lp._tool_choice_kind({"tool_choice": "required"}),
        lp._tool_choice_kind({"tool_choice": "none"}),
        lp._tool_choice_kind({"tool_choice": {"type": "function"}}),
        lp._tool_choice_kind({"tool_choice": {"type": "nope"}}),
    }
    assert got == _TC_VOCABULARY
    assert lp._tool_choice_kind(
        {"tool_choice": {"type": "allowed_tools",
                         "tools": [{"name": "a"}]}}) == "allowed:1"


def test_allowed_n_is_an_integer_by_construction():
    """`allowed:<n>` is the one variable-content word. `<n>` is a `len()`,
    so no client string can ride it."""
    out = lp._tool_choice_kind(
        {"tool_choice": {"type": "allowed_tools",
                         "tools": [{"name": "x\nFORGED"}] * 3}})
    assert out == "allowed:3"


async def test_a_newline_in_tool_choice_cannot_forge_a_cache_line(
    responses_driver, canary_on, caplog,
):
    """HANDLER level, flag ON — the state in which the defect could fire.
    One request must produce exactly ONE request-side [CACHE] line, and no
    log record anywhere may contain the attacker's text."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    forged = ("auto\n2026-09-20 12:00:00 WARNING [CACHE] user=deadbeef "
              "model=gpt-5.6-terra has_cache_key=True "
              "cache_key_hash=00000000 retention=none rid=00000000 "
              "trace=- tools_n=1 tools_sha=forged00")
    await responses_driver(_body(_core(2), tool_choice=forged), sse=_SSE)
    lines = _cache_lines(caplog)
    req = [ln for ln in lines if "tools_n=" in ln]
    assert len(req) == 1, lines
    assert _field(req[0], "tc") == "other"
    everything = "\n".join(r.getMessage() for r in caplog.records)
    for probe in ("deadbeef", "forged00", "FORGED"):
        assert probe not in everything, probe


# ── (N5) the provider's own id gets the same discipline ──────────────


@pytest.mark.parametrize("raw,expected", [
    ("req_abc123", "req_abc123"),
    ("REQ-1.2:3_4", "REQ-1.2:3_4"),
    ("x" * 128, "x" * 128),
    ("x" * 129, "-"),                      # bounded
    ("req_a\nFORGED", "-"),
    # the `\A…\Z` half: Python's `$` also matches before a TRAILING newline,
    # so a `^…$` pattern would accept this one and end the [CACHE] line.
    ("req_ok\n", "-"),
    ("req_a\r\n[CACHE] user=deadbeef", "-"),
    ("req a", "-"),                        # a space splits a field
    ("req\ta", "-"),
    ("req\x00a", "-"),
    ("", "-"),
    (None, "-"),
    (7, "-"),
    ({"x": 1}, "-"),
])
def test_wire_req_id_is_allowlisted(raw, expected):
    assert lp._wire_req_id(raw) == expected


async def test_a_hostile_upstream_request_id_reaches_no_log_line(
    responses_driver, canary_on, caplog,
):
    """The provider's header is a string we did not author. httpx/h11
    reject a control character upstream, so this is defence in depth — but
    it is the same class as B2 and it costs one regex."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(
        _body(_core(2)), sse=_SSE,
        request_id="req_ok\n2026-09-20 WARNING [CACHE] user=deadbeef x=1")
    usage = [ln for ln in _cache_lines(caplog) if "req_id=" in ln]
    assert len(usage) == 1, _cache_lines(caplog)
    assert _field(usage[0], "req_id") == "-"
    everything = "\n".join(r.getMessage() for r in caplog.records)
    assert "deadbeef" not in everything


# ── the canary gate ──────────────────────────────────────────────────


def test_observability_is_off_by_default(monkeypatch):
    monkeypatch.setattr(lp.settings, "llm_proxy_wire_observability", False,
                        raising=False)
    monkeypatch.setattr(lp.settings,
                        "llm_proxy_wire_observability_canary_user_ids", "",
                        raising=False)
    assert lp._wire_observability_on(CANARY_UID) is False
    assert lp._wire_tools_suffix(
        {"tools": _core(3)}, 3, CANARY_UID,
        rid="deadbeef", trace="-") == ("", "")
    assert lp._wire_timing_suffix(
        CANARY_UID, rid="deadbeef", trace="-", req_id="r",
        cache_key_hash="h", tools_sha="s",
        body_ms=0, pre_ms=1, ttfb_ms=2, total_ms=3) == ""
    assert lp._wire_join_suffix(CANARY_UID, rid="deadbeef", trace="-") == ""


@pytest.mark.parametrize("global_flag", [True, False])
def test_the_fleet_wide_flag_is_an_enable_route_of_its_own(
    monkeypatch, global_flag,
):
    """The rollout's LAST step is `LLM_PROXY_WIRE_OBSERVABILITY=true`, with
    no canary list at all. Every other fixture here pins the global flag to
    False and exercises only the allowlist, so replacing that branch with
    `if False:` survived the whole suite (R48 review, N2/R-M2). Both enable
    routes are asserted now, and a user who is on neither stays off."""
    monkeypatch.setattr(lp.settings, "llm_proxy_wire_observability",
                        global_flag, raising=False)
    monkeypatch.setattr(
        lp.settings, "llm_proxy_wire_observability_canary_user_ids", "",
        raising=False)
    assert lp._wire_observability_on(OTHER_UID) is global_flag
    # And with no user id at all — a fleet-wide flag must not need one.
    assert lp._wire_observability_on(None) is global_flag
    suffix, sha = lp._wire_tools_suffix(
        {"tools": _core(3)}, 3, OTHER_UID, rid="deadbeef", trace="-")
    assert (suffix != "") is global_flag
    assert (sha != "") is global_flag
    assert (lp._wire_join_suffix(
        OTHER_UID, rid="deadbeef", trace="-") != "") is global_flag


async def test_the_fleet_wide_flag_reaches_the_real_handler(
    responses_driver, monkeypatch, caplog,
):
    """Same route, driven through `proxy_responses` — the unit test above
    proves the predicate, this proves the handler asks it."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    monkeypatch.setattr(lp.settings, "llm_proxy_wire_observability", True,
                        raising=False)
    monkeypatch.setattr(
        lp.settings, "llm_proxy_wire_observability_canary_user_ids", "",
        raising=False)
    await responses_driver(_body(_core(20)), sse=_SSE, user_id=OTHER_UID)
    lines = _cache_lines(caplog)
    assert _field(lines[0], "tools_n") == "20"
    assert _field(lines[1], "req_id") == "req_abc123"


def test_canary_list_admits_only_the_listed_user(monkeypatch):
    monkeypatch.setattr(lp.settings, "llm_proxy_wire_observability", False,
                        raising=False)
    monkeypatch.setattr(
        lp.settings, "llm_proxy_wire_observability_canary_user_ids",
        f" {CANARY_UID} , ", raising=False)
    assert lp._wire_observability_on(CANARY_UID) is True
    assert lp._wire_observability_on(OTHER_UID) is False
    assert lp._wire_observability_on(None) is False
    assert lp._wire_observability_on("") is False


def test_the_canary_list_is_not_memoised(monkeypatch):
    """`/admin/bind` can re-point this process at another tenant; a memo
    keyed to nothing would outlive the bind and answer for the wrong one."""
    monkeypatch.setattr(lp.settings, "llm_proxy_wire_observability", False,
                        raising=False)
    monkeypatch.setattr(
        lp.settings, "llm_proxy_wire_observability_canary_user_ids",
        CANARY_UID, raising=False)
    assert lp._wire_observability_on(CANARY_UID) is True
    monkeypatch.setattr(
        lp.settings, "llm_proxy_wire_observability_canary_user_ids", "",
        raising=False)
    assert lp._wire_observability_on(CANARY_UID) is False


# ── the test double tracks production ────────────────────────────────


def test_backend_stream_signature_carries_meta():
    """If `meta` is renamed or dropped, the doubles below stop modelling
    production and every req_id assertion becomes decoration."""
    params = inspect.signature(lp.OpenAIBackend.responses_stream).parameters
    assert "meta" in params
    assert params["meta"].default is None


async def test_real_backend_writes_the_request_id_into_meta(monkeypatch):
    """Executes the shipped generator (not a re-implementation) against a
    fake httpx client, so the meta write is proved where it lives."""
    class _Resp:
        status_code = 200
        headers = {"x-request-id": "req_from_headers", "openai-org": "org-x"}

        async def aiter_bytes(self):
            yield b"data: {}\n\n"

    class _Stream:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *a):
            return False

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, *a, **kw):
            return _Stream()

    monkeypatch.setattr(lp.httpx, "AsyncClient", _Client)
    meta: dict = {}
    chunks = [c async for c in lp._openai.responses_stream(
        {"model": "gpt-5.6-terra"}, "sk", meta=meta)]
    assert meta["request_id"] == "req_from_headers"
    assert chunks == [b"data: {}\n\n"]


async def test_real_backend_without_meta_is_unchanged(monkeypatch):
    """A caller passing nothing must behave exactly as it did in R47."""
    class _Resp:
        status_code = 200
        headers = {"x-request-id": "req_from_headers"}

        async def aiter_bytes(self):
            yield b"data: {}\n\n"

    class _Stream:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *a):
            return False

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, *a, **kw):
            return _Stream()

    monkeypatch.setattr(lp.httpx, "AsyncClient", _Client)
    chunks = [c async for c in lp._openai.responses_stream(
        {"model": "gpt-5.6-terra"}, "sk")]
    assert chunks == [b"data: {}\n\n"]


# ── (a) THE FORK: tool_choice alone moves the forwarded array ────────


_SSE = (b"event: response.completed\n"
        b'data: {"response": {"usage": {"input_tokens": 44647, '
        b'"output_tokens": 5, "input_tokens_details": {"cached_tokens": 0}}}}\n\n')


@pytest.fixture
def canary_on(monkeypatch):
    monkeypatch.setattr(lp.settings, "llm_proxy_wire_observability", False,
                        raising=False)
    monkeypatch.setattr(
        lp.settings, "llm_proxy_wire_observability_canary_user_ids",
        CANARY_UID, raising=False)


@pytest.fixture
def canary_off(monkeypatch):
    monkeypatch.setattr(lp.settings, "llm_proxy_wire_observability", False,
                        raising=False)
    monkeypatch.setattr(
        lp.settings, "llm_proxy_wire_observability_canary_user_ids", "",
        raising=False)


def _field(line: str, key: str) -> str:
    for part in line.split():
        if part.startswith(key + "="):
            return part[len(key) + 1:]
    raise AssertionError(f"{key}= not in {line!r}")


async def test_tool_choice_alone_forks_the_forwarded_tools_array(
    responses_driver, canary_on, caplog,
):
    """E2, pinned. Two requests with the SAME 174-tool array and the same
    cache key differ only in `tool_choice.allowed_tools` — and the provider
    is handed two different 128-tool arrays, because `_cap_tools` protects
    exactly the names `tool_choice` mentions.

    This is the mechanism behind E2's three PRODUCTION requests, executed
    locally. It shows the fork exists and that the agent cannot see it —
    its own fingerprint is taken pre-cap and is IDENTICAL across both
    requests, which this test also asserts. It does NOT show that the fork
    caused those misses; that is what the `tools_sha` series is for."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    tools = _over_cap_tools()
    pre_cap_sha = lp._tools_digest(tools)

    await responses_driver(
        _body(list(tools), tool_choice=_allowed(
            ["core_100", "core_101", "core_102"])), sse=_SSE)
    first = _cache_lines(caplog)[0]
    caplog.clear()

    await responses_driver(
        _body(list(tools), tool_choice=_allowed(
            ["core_110", "core_111", "core_112"])), sse=_SSE)
    second = _cache_lines(caplog)[0]

    # The agent's instrument: identical, both times.
    assert pre_cap_sha == lp._tools_digest(_over_cap_tools())
    # The wire: different.
    assert _field(first, "tools_sha") != _field(second, "tools_sha")
    assert _field(first, "tools_sent") == "174"
    assert _field(first, "tools_n") == "128"
    assert _field(first, "dropped_n") == "46"
    assert _field(first, "tc") == "allowed:3"
    # And neither equals the pre-cap array the agent believes it sent.
    assert _field(first, "tools_sha") != pre_cap_sha


async def test_identical_requests_produce_an_identical_tools_sha(
    responses_driver, canary_on, caplog,
):
    """The other direction — without this, a sha that always differs would
    'prove' the fork on every pair of requests."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    tc = _allowed(["core_100", "core_101"])
    tools = _over_cap_tools()

    await responses_driver(_body(list(tools), tool_choice=dict(tc)), sse=_SSE)
    first = _cache_lines(caplog)[0]
    caplog.clear()
    await responses_driver(_body(list(tools), tool_choice=dict(tc)), sse=_SSE)
    second = _cache_lines(caplog)[0]

    assert _field(first, "tools_sha") == _field(second, "tools_sha")


async def test_under_the_cap_nothing_is_dropped_and_tc_is_reported(
    responses_driver, canary_on, caplog,
):
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(_body(_core(20)), sse=_SSE)
    line = _cache_lines(caplog)[0]
    assert _field(line, "tools_n") == "20"
    assert _field(line, "tools_sent") == "20"
    assert _field(line, "dropped_n") == "0"
    assert _field(line, "tc") == "none"


async def test_dedup_casualties_are_counted_in_dropped_n(
    responses_driver, canary_on, caplog,
):
    """`dropped_n` is 'the proxy removed something', by whatever route —
    dedup changes the cached array exactly as the cap does."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    dupes = _core(5) + [{"name": "core_000", "description": "x"}]
    await responses_driver(_body(dupes), sse=_SSE)
    line = _cache_lines(caplog)[0]
    assert _field(line, "tools_sent") == "6"
    assert _field(line, "tools_n") == "5"
    assert _field(line, "dropped_n") == "1"


def test_dropped_n_is_the_raw_difference_and_may_go_negative(monkeypatch):
    """R48 review, N4. `max(diff, 0)` clamped the only case where the sign
    carries information — a converter that ADDS tools between `tools_sent`
    and the forwarded array. An anomaly that reads as 0 is an anomaly nobody
    sees; -2 is a number somebody asks about."""
    monkeypatch.setattr(lp.settings, "llm_proxy_wire_observability", False,
                        raising=False)
    monkeypatch.setattr(
        lp.settings, "llm_proxy_wire_observability_canary_user_ids",
        CANARY_UID, raising=False)
    suffix, _sha = lp._wire_tools_suffix(
        {"tools": _core(3)}, 1, CANARY_UID, rid="deadbeef", trace="-")
    assert _field(suffix, "dropped_n") == "-2"


def test_the_anthropic_fallback_shape_is_documented_not_a_cap_casualty():
    """R48 review, N3. `_anthropic_to_openai_request` builds its result with
    no `tools` key at all, while `_cap_tools` ran earlier against the
    PRE-fallback backend — so a daily-cap fallback prints
    `tools_n=0 tools_sent=N dropped_n=N tools_sha=none` in the very series
    built to count cap casualties. Asserted here: the converter really does
    strip tools (so the discovery is not folklore), and the field's docstring
    warns the reader."""
    converted = lp._anthropic_to_openai_request(
        {"messages": [{"role": "user", "content": "hi"}],
         "tools": [{"name": "grep"}], "max_tokens": 10})
    assert "tools" not in converted
    doc = lp._wire_tools_suffix.__doc__ or ""
    assert "daily-cap fallback" in doc


async def test_no_tool_name_or_schema_reaches_the_cache_lines(
    responses_driver, canary_on, caplog,
):
    """The cap's WARN already names what it dropped; these lines add counts
    and digests only.

    The round-1 objection was that "no substring of instructions appears" is
    unfalsifiable at one character. The specified window is **42 characters**
    — the length of the probe below — plus the full tool name and the raw
    cache key, all three of which must be absent from every [CACHE] line."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    secret = "A_VERY_DISTINCTIVE_TOOL_DESCRIPTION_42CHAR"
    assert len(secret) == 42  # the window this test claims to guard
    tools = _over_cap_tools()
    tools[-1] = {"name": "notion__op_secret", "description": secret}
    await responses_driver(
        _body(tools, tool_choice=_allowed(["core_000"])), sse=_SSE)
    for line in _cache_lines(caplog):
        assert secret not in line
        assert "notion__op_secret" not in line
        assert "0a1b2c3d:all" not in line  # the cache key itself, never


# ── (c)(d) req_id and the timing split ───────────────────────────────


async def test_usage_line_carries_req_id_and_the_timing_split(
    responses_driver, canary_on, caplog,
):
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(_body(), sse=_SSE, request_id="req_xyz789")
    usage = _cache_lines(caplog)[1]
    assert _field(usage, "req_id") == "req_xyz789"
    assert _field(usage, "prompt_tokens") == "44647"
    assert _field(usage, "cached_tokens") == "0"
    assert int(_field(usage, "pre_ms")) >= 0
    assert int(_field(usage, "ttfb_ms")) >= 0
    assert int(_field(usage, "total_ms")) >= int(_field(usage, "ttfb_ms"))


async def test_missing_upstream_request_id_is_a_dash_not_a_crash(
    responses_driver, canary_on, caplog,
):
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(_body(), sse=_SSE, request_id=None)
    assert _field(_cache_lines(caplog)[1], "req_id") == "-"


async def test_pre_ms_measures_work_before_start_ts(
    responses_driver, canary_on, caplog,
):
    """`start_ts` is stamped after auth, dedup, the cap and the budget
    check — 50 ms spent in auth must land in pre_ms and NOT in total_ms."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(_body(), sse=_SSE, auth_delay=0.05)
    usage = _cache_lines(caplog)[1]
    assert int(_field(usage, "pre_ms")) >= 40
    assert int(_field(usage, "ttfb_ms")) < 40


async def test_ttfb_ms_measures_the_wait_for_the_first_upstream_chunk(
    responses_driver, canary_on, caplog,
):
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(_body(), sse=_SSE, ttfb_delay=0.05)
    usage = _cache_lines(caplog)[1]
    assert int(_field(usage, "ttfb_ms")) >= 40
    assert int(_field(usage, "pre_ms")) < 40


def test_non_stream_path_reports_ttfb_minus_one():
    """There is no first-chunk boundary on the JSON path; a number there
    would mean 'the whole call' and poison the streaming series' p50."""
    src = inspect.getsource(lp.proxy_responses)
    assert "ttfb_ms=-1" in src


async def test_the_non_stream_usage_line_behaves(
    responses_driver, canary_on, caplog,
):
    """R48 review, N2. The source grep above cannot notice that branch being
    rewritten: a mutation replacing it with `req_id=str(dict(resp.headers)),
    pre_ms=0` survived 126 tests across four files, i.e. a full upstream
    header dump into an INFO line would have shipped. This drives the JSON
    path through the real handler and pins what the line may and may not
    contain."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    body = _body(_core(20))
    body["stream"] = False
    await responses_driver(
        body, sse=b"", request_id="req_json_1", auth_delay=0.05,
        resp_headers={"openai-organization": "org-SECRET-VALUE",
                      "set-cookie": "sess=NEVERLOGTHIS"})
    usage = _cache_lines(caplog)[1]
    assert _field(usage, "req_id") == "req_json_1"
    assert _field(usage, "ttfb_ms") == "-1"
    # pre_ms is real on this path too — 50 ms of auth has to land in it.
    assert int(_field(usage, "pre_ms")) >= 40
    # and nothing else from the upstream headers may ride along.
    assert "org-SECRET-VALUE" not in usage
    assert "NEVERLOGTHIS" not in usage
    assert "openai-organization" not in usage
    # the join fields are on this wire too
    assert _field(usage, "cache_key_hash") == _field(
        _cache_lines(caplog)[0], "cache_key_hash")
    assert _field(usage, "tools_sha") == _field(
        _cache_lines(caplog)[0], "tools_sha")


# ── (B1) the two lines can be JOINED without relying on adjacency ────


async def test_the_usage_line_repeats_its_own_requests_join_keys(
    responses_driver, canary_on, caplog,
):
    """`tools_sha` is emitted on the request line and `cached_tokens` on the
    usage line, so every gate that relates them needs a join. Same values,
    both lines."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(_body(_over_cap_tools(),
                                 tool_choice=_allowed(["core_100"])),
                           sse=_SSE)
    req, usage = _cache_lines(caplog)[:2]
    assert _field(usage, "tools_sha") == _field(req, "tools_sha")
    assert _field(usage, "cache_key_hash") == _field(req, "cache_key_hash")
    assert _field(usage, "tools_sha") != "none"


async def test_interleaved_requests_do_not_cross_their_join_keys(
    monkeypatch, canary_on, caplog,
):
    """The hazard B1 names, executed. `subagent.py:129` starts child runs
    with `asyncio.create_task`, `cache_warm` is a third producer, and
    platform-api runs two replicas into one Railway stream — so two
    `/responses` calls for the SAME user genuinely overlap and their four
    [CACHE] lines arrive in an order nobody controls.

    This test forces exactly that ordering (A's request line, B's request
    line, B's usage line, A's usage line) and asserts each usage line still
    carries ITS OWN request's `cache_key_hash` and `tools_sha`. Pairing by
    adjacency here would hand request A's tools_sha to request B's
    cached_tokens — a confident wrong answer to the gate that decides
    whether the next round touches `_cap_tools`.
    """
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")

    async def fake_auth(request, db):
        return types.SimpleNamespace(user_id=CANARY_UID)

    async def fake_budget(cfg, provider, db):
        return None

    async def fake_log_event(*a, **kw):
        return None

    monkeypatch.setattr(lp, "_auth_agent", fake_auth)
    monkeypatch.setattr(lp, "_check_budget", fake_budget)
    monkeypatch.setattr(lp, "_log_event", fake_log_event)
    monkeypatch.setattr(
        lp, "_route_chat", lambda model, cfg: (lp._openai, "sk-test"))

    class _NullSession:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    import app.db.database as _dbmod
    monkeypatch.setattr(_dbmod, "async_session_maker", lambda: _NullSession())

    # A is slow to first byte, B is fast — so B's whole turn completes
    # between A's request line and A's usage line.
    async def fake_stream(self, body, api_key, meta=None):
        delay = 0.05 if len(body.get("tools") or []) == 20 else 0.0
        if delay:
            await asyncio.sleep(delay)
        if meta is not None:
            meta["request_id"] = "req_A" if delay else "req_B"
        yield _SSE

    monkeypatch.setattr(lp.OpenAIBackend, "responses_stream", fake_stream)

    a_body = _body(_core(20))
    a_body["prompt_cache_key"] = "0a1b2c3d:AAA"
    b_body = _body(_core(31))
    b_body["prompt_cache_key"] = "0a1b2c3d:BBB"

    async def run(body):
        resp = await lp.proxy_responses(_FakeRequest(body), db=None)
        return b"".join([c async for c in resp.body_iterator])

    await asyncio.gather(run(a_body), run(b_body))

    lines = _cache_lines(caplog)
    assert len(lines) == 4
    req_lines = [ln for ln in lines if " tools_sent=" in ln]
    usage_lines = [ln for ln in lines if " req_id=" in ln]
    assert len(req_lines) == 2 and len(usage_lines) == 2

    by_key = {_field(ln, "cache_key_hash"): ln for ln in req_lines}
    assert len(by_key) == 2

    # The hazard is real in THIS run, not merely in principle: pair each
    # usage line with the nearest PRECEDING request line — the only join
    # adjacency offers — and at least one pair is wrong. If this assertion
    # ever stops holding, the test has stopped exercising what it was
    # written for and the join fields below are proving nothing.
    def _nearest_preceding_request(usage: str) -> str:
        i = lines.index(usage)
        for j in range(i - 1, -1, -1):
            if lines[j] in req_lines:
                return lines[j]
        raise AssertionError("no preceding request line")

    assert any(
        _field(_nearest_preceding_request(u), "cache_key_hash")
        != _field(u, "cache_key_hash")
        for u in usage_lines
    ), lines

    for usage in usage_lines:
        req = by_key[_field(usage, "cache_key_hash")]
        assert _field(usage, "tools_sha") == _field(req, "tools_sha")
        # and the pairing really distinguishes them
        assert _field(usage, "req_id") in ("req_A", "req_B")
        expected_n = "20" if _field(usage, "req_id") == "req_A" else "31"
        assert _field(req, "tools_n") == expected_n


# ── (B2) body_ms separates the upload from the proxy's own pre-work ──


async def test_body_ms_holds_the_request_upload_and_pre_ms_contains_it(
    responses_driver, canary_on, caplog,
):
    """`pre_ms` spans handler entry → `start_ts`, which INCLUDES
    `await request.json()` — the socket receive of ~200 KB over Contabo →
    Cloudflare → Railway. Read as "the proxy's own pre-work" it would send
    the next round into the platform DB while the real term was upload."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(_body(), sse=_SSE, body_delay=0.05)
    usage = _cache_lines(caplog)[1]
    assert int(_field(usage, "body_ms")) >= 40
    assert int(_field(usage, "pre_ms")) >= int(_field(usage, "body_ms"))
    assert int(_field(usage, "ttfb_ms")) < 40


async def test_body_ms_excludes_auth(responses_driver, canary_on, caplog):
    """The other direction: 50 ms in `_auth_agent` is proxy pre-work, and it
    must land in `pre_ms` WITHOUT inflating `body_ms`. Without this, a
    `body_ms` that simply aliased `pre_ms` would pass the test above."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(_body(), sse=_SSE, auth_delay=0.05)
    usage = _cache_lines(caplog)[1]
    assert int(_field(usage, "pre_ms")) >= 40
    assert int(_field(usage, "body_ms")) < 40


def test_the_timing_docstring_names_what_it_cannot_see():
    """The residual against `[req-timing]` is not "one of the clocks is
    wrong" — middleware, routing and `Depends(get_db)` all resolve before
    `entry_mono` and appear in no field here. A reader who does not know
    that will attribute the gap to the wrong subsystem."""
    doc = lp._wire_timing_suffix.__doc__ or ""
    assert "Depends(get_db)" in doc
    assert "upload" in doc.lower()
    assert "middleware" in doc.lower()


# ── (N6/N7) the request id is captured before anything can lose it ───


async def test_meta_is_written_before_any_body_byte(monkeypatch):
    """Position, not presence. Written after `aiter_bytes()` the id would be
    absent on a client disconnect — the exact turns worth investigating —
    and both unit tests here would still pass because they drain the
    generator. So: pull ONE chunk and assert the id is already there."""
    class _Resp:
        status_code = 200
        headers = {"x-request-id": "req_early"}

        async def aiter_bytes(self):
            yield b"one"
            raise AssertionError("must not be reached by this test")

    class _Stream:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *a):
            return False

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, *a, **kw):
            return _Stream()

    monkeypatch.setattr(lp.httpx, "AsyncClient", _Client)
    meta: dict = {}
    gen = lp._openai.responses_stream({"model": "gpt-5.6-terra"}, "sk",
                                      meta=meta)
    first = await gen.__anext__()
    assert first == b"one"
    assert meta["request_id"] == "req_early"
    await gen.aclose()


async def test_meta_is_written_even_on_an_upstream_error(monkeypatch):
    """R48 review, N6. A 4xx/5xx is the response an OpenAI escalation is
    about; writing the id after the status check left that WARNING with
    nothing to quote."""
    class _Resp:
        status_code = 429
        headers = {"x-request-id": "req_err"}

        async def aread(self):
            return b'{"error":"rate_limited"}'

    class _Stream:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *a):
            return False

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, *a, **kw):
            return _Stream()

    monkeypatch.setattr(lp.httpx, "AsyncClient", _Client)
    meta: dict = {}
    gen = lp._openai.responses_stream({"model": "gpt-5.6-terra"}, "sk",
                                      meta=meta)
    with pytest.raises(lp.UpstreamProviderError):
        await gen.__anext__()
    assert meta["request_id"] == "req_err"


async def test_the_upstream_error_warning_carries_the_req_id(
    monkeypatch, canary_on, caplog,
):
    """End of the same thread, through the real handler: the WARNING that
    fires on an upstream non-2xx now quotes the provider id. No usage-side
    [CACHE] line exists on this path at all, so this is the only place it
    can appear."""
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")

    async def fake_auth(request, db):
        return types.SimpleNamespace(user_id=CANARY_UID)

    async def fake_budget(cfg, provider, db):
        return None

    async def fake_log_event(*a, **kw):
        return None

    async def fake_stream(self, body, api_key, meta=None):
        if meta is not None:
            meta["request_id"] = "req_boom"
        raise lp.UpstreamProviderError(500, b"upstream exploded", "openai")
        yield b""  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(lp, "_auth_agent", fake_auth)
    monkeypatch.setattr(lp, "_check_budget", fake_budget)
    monkeypatch.setattr(lp, "_log_event", fake_log_event)
    monkeypatch.setattr(lp.OpenAIBackend, "responses_stream", fake_stream)
    monkeypatch.setattr(
        lp, "_route_chat", lambda model, cfg: (lp._openai, "sk-test"))

    with pytest.raises(lp.HTTPException):
        await lp.proxy_responses(_FakeRequest(_body()), db=None)

    warns = [r.getMessage() for r in caplog.records
             if "[LLM-PROXY]" in r.getMessage() and "upstream 500" in r.getMessage()]
    assert warns, [r.getMessage() for r in caplog.records]
    assert _field(warns[0], "req_id") == "req_boom"


# ── (e) flag-off byte identity, and the body is never touched ────────


_R47_REQUEST_LINE = (
    "[CACHE] user=0a1b2c3d model=gpt-5.6-terra has_cache_key=True "
    "cache_key_hash=%s retention=none"
)
_R47_USAGE_LINE = (
    "[CACHE] user=0a1b2c3d model=gpt-5.6-terra has_cache_key=True "
    "retention=none prompt_tokens=44647 cached_tokens=0 cache_write_tokens=0"
)


async def test_flag_off_lines_are_byte_identical_to_r47(
    responses_driver, canary_off, caplog,
):
    """Literal expected messages, not 'does not contain tools_'. If the
    suffix ever renders as anything but the empty string when off, this
    fails on the exact bytes."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(_body(_core(20), tool_choice="auto"), sse=_SSE)
    lines = _cache_lines(caplog)
    expected_hash = hashlib.sha256(b"0a1b2c3d:all").hexdigest()[:8]
    assert lines[0] == _R47_REQUEST_LINE % expected_hash
    assert lines[1] == _R47_USAGE_LINE


async def test_flag_on_only_appends(responses_driver, canary_on, caplog):
    """Same two lines, same order, same prefix — the R47 text is a strict
    prefix of the R48 text, so every existing grep still matches."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(_body(_core(20), tool_choice="auto"), sse=_SSE)
    lines = _cache_lines(caplog)
    expected_hash = hashlib.sha256(b"0a1b2c3d:all").hexdigest()[:8]
    assert lines[0].startswith(_R47_REQUEST_LINE % expected_hash)
    assert lines[1].startswith(_R47_USAGE_LINE)
    assert lines[0] != _R47_REQUEST_LINE % expected_hash


async def test_upstream_body_and_sse_are_identical_with_the_flag_on_or_off(
    responses_driver, monkeypatch, caplog,
):
    """The instrument must not move the thing it measures: the dict handed
    to the backend, and the bytes handed back to the agent, are compared
    across a flag-off and a flag-on run of the same request."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    tools = _over_cap_tools()
    tc = _allowed(["core_100", "core_101", "core_102"])

    monkeypatch.setattr(lp.settings, "llm_proxy_wire_observability", False,
                        raising=False)
    monkeypatch.setattr(
        lp.settings, "llm_proxy_wire_observability_canary_user_ids", "",
        raising=False)
    out_off = await responses_driver(
        _body(list(tools), tool_choice=dict(tc)), sse=_SSE)
    body_off = responses_driver.state["sent"][-1]

    monkeypatch.setattr(
        lp.settings, "llm_proxy_wire_observability_canary_user_ids",
        CANARY_UID, raising=False)
    out_on = await responses_driver(
        _body(list(tools), tool_choice=dict(tc)), sse=_SSE)
    body_on = responses_driver.state["sent"][-1]

    assert json.dumps(body_off, sort_keys=True) == \
        json.dumps(body_on, sort_keys=True)
    assert out_off == out_on == _SSE
    # And the capped body really is what was measured.
    assert len(body_on["tools"]) == 128


async def test_a_user_off_the_canary_gets_the_r47_lines(
    responses_driver, canary_on, caplog,
):
    """One tenant on, the rest untouched — the whole point of the list."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(_body(_core(20)), sse=_SSE, user_id=OTHER_UID)
    for line in _cache_lines(caplog):
        assert " tools_sha=" not in line
        assert " req_id=" not in line


# ── the chat wire carries the same request-side fields ───────────────


async def test_chat_completions_request_line_carries_the_tools_fields(
    monkeypatch, canary_on, caplog,
):
    """`/openai/v1/chat/completions` runs the same cap through a different
    handler; fixing one wire and leaving the other is this file's own
    recorded failure mode (see the cap's docstring)."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")

    async def fake_auth(request, db):
        return types.SimpleNamespace(user_id=CANARY_UID)

    async def fake_budget(cfg, provider, db):
        return None

    async def fake_log_event(*a, **kw):
        return None

    async def fake_chat(b, api_key):
        return types.SimpleNamespace(
            status_code=200, headers={}, content=b"",
            json=lambda: {"usage": {"prompt_tokens": 10,
                                    "completion_tokens": 2}})

    monkeypatch.setattr(lp, "_auth_agent", fake_auth)
    monkeypatch.setattr(lp, "_check_budget", fake_budget)
    monkeypatch.setattr(lp, "_log_event", fake_log_event)
    monkeypatch.setattr(
        lp, "_route_chat",
        lambda model, cfg: (types.SimpleNamespace(name="openai",
                                                  chat=fake_chat), "k"))

    await lp.proxy_chat(_FakeRequest({
        "model": "gpt-5.5",
        "stream": False,
        "messages": [],
        "tools": _over_cap_tools(),
        "tool_choice": {"type": "allowed_tools",
                        "allowed_tools": {"tools": [
                            {"function": {"name": "core_100"}}]}},
    }), db=None)

    line = _cache_lines(caplog)[0]
    assert _field(line, "tools_sent") == "174"
    assert _field(line, "tools_n") == "128"
    assert _field(line, "dropped_n") == "46"
    assert _field(line, "tc") == "allowed:1"
    assert len(_field(line, "tools_sha")) == 8


# ── the corrected cap comment ────────────────────────────────────────


def test_the_false_core_namespace_comment_is_gone():
    """The comment claimed un-namespaced tools are 'never the biggest by
    construction'. They are ONE namespace and the largest by a wide margin,
    which inverts the rule it was explaining."""
    src = inspect.getsource(lp._cap_tools)
    assert "never the biggest by construction" not in src
    assert "LARGEST namespace" in src  # rule 3 still says what it does


def _cap_comment_text() -> str:
    """`_cap_tools`' source with comment markers and line wrapping removed,
    so an assertion about a SENTENCE is not an assertion about where the
    author happened to break the line."""
    src = inspect.getsource(lp._cap_tools)
    flat = src.replace("\n", " ").replace("#", " ")
    return " ".join(flat.split())


def test_the_cap_comment_does_not_overclaim_the_production_extract():
    """B3. The replacement comment must not restate the claim the cited
    extract refutes. Nine of the 51 distinct names dropped across the three
    production turns are NOT in the 63 static core definitions — they are
    un-namespaced first-party MCP tools — so 'every dropped name was a core
    tool' is false, and 63 is the STATIC SUBSET of the `""` namespace, not
    its size. The supportable claim, which the extract does carry, is the
    un-namespaced one."""
    text = _cap_comment_text()
    assert "every dropped name was a core tool" not in text
    assert "Core is ONE namespace of 63" not in text
    assert "every dropped name was UN-NAMESPACED" in text
    assert "not one namespaced connector tool was dropped" in text
    # 63 may still appear, but only as the static subset of a larger set
    assert ">= 72, not 63" in text
    assert "static subset alone is 63" in text
    # d3ebda17: Voice's floor made "nothing holds specific un-namespaced
    # names" untrue, so the comment has to account for it — and date the
    # production extract to before the floor existed.
    # Rule 1 is named BEFORE the floor: on a request that names anything,
    # the floor holds nothing rule 1 does not already hold (review B-1).
    assert "What holds SPECIFIC un-namespaced names is (rule 1) whatever THIS request's tool_choice names" in text
    assert "so it holds nothing rule 1 does not already hold" in text
    assert "The one thing that holds SPECIFIC" not in text
    assert "before the floor existed" in text
    assert "changes NO selection logic" in text


def test_the_namespace_the_cap_groups_by_is_the_un_namespaced_one():
    """Not a comment test: the mechanism the comment now describes.
    `_namespace_of` collapses every name without `__` into one bucket, so
    core definitions and un-namespaced first-party MCP tools compete for
    the same namespace budget — which is why 63 could never have been the
    size of that bucket."""
    assert lp._namespace_of("generate_image") == ""
    assert lp._namespace_of("graph_traverse") == ""       # MCP, un-namespaced
    assert lp._namespace_of("slack__send_message") == "slack"


def test_the_cap_still_drops_un_namespaced_first_on_a_realistic_array():
    """Not a comment test: the behaviour the corrected comment describes.
    Pins today's victim profile so a future change to the selection logic
    has to be deliberate. Named for what it ASSERTS — every dropped name is
    un-namespaced — rather than for 'core', which is only part of that
    namespace (B3).

    Re-pinned on d3ebda17, deliberately: the selection change is the Voice
    programme's floor (PR 756), not this patch's. The array now carries the
    five floor names at the TAIL of the un-namespaced block — exactly where
    tail-first would take them first — so the new profile is stated, not
    merely tolerated: still un-namespaced-first, still no connector tool,
    and never a floor name."""
    floor = sorted(lp._PROTECTED_CORE_DEFAULT)
    assert lp.PROTECTED_CORE_TOOLS == frozenset(floor)   # env not overriding
    tools = _over_cap_tools()
    tools[115:120] = [{"name": n, "description": "x"} for n in floor]
    assert len(tools) == 174
    kept, dropped = lp._cap_tools(tools)
    assert len(kept) == 128
    assert len(dropped) == 46
    assert all("__" not in n for n in dropped), dropped
    assert not set(floor) & set(dropped), dropped
    kept_names = {t["name"] for t in kept}
    assert set(floor) <= kept_names


def test_rule_one_not_the_floor_holds_named_un_namespaced_tail_names():
    """Review B-1, executed: what holds specific un-namespaced names on a
    request that NAMES something is rule 1, not the floor. Same 174-tool
    array as the re-pin. An allow-list naming two NON-floor tail names plus
    one floor name keeps all three; the floor names it omits are ordinary
    candidates and, sitting at the tail of the largest namespace, go. A
    forced function behaves the same way: the floor shrinks to what it
    names."""
    floor = sorted(lp._PROTECTED_CORE_DEFAULT)
    tools = _over_cap_tools()
    tools[115:120] = [{"name": n, "description": "x"} for n in floor]
    named = {"core_113", "core_114", "play_media"}
    kept, dropped = lp._cap_tools(tools, protected=named)
    kept_names = {t["name"] for t in kept}
    assert len(kept) == 128
    assert named <= kept_names                      # rule 1 held them
    assert not {"core_113", "core_114"} & set(lp._PROTECTED_CORE_DEFAULT)
    assert (set(floor) - named) <= set(dropped), dropped
    assert all("__" not in n for n in dropped), dropped

    kept, dropped = lp._cap_tools(tools, protected={"core_114"})
    kept_names = {t["name"] for t in kept}
    assert "core_114" in kept_names
    assert set(floor) <= set(dropped), dropped      # floor ∩ {name} = ∅


# ── (F6-1) `rid` — the join the cohort key cannot make ───────────────
#
# `cache_key_hash` + `tools_sha` identifies a COHORT: every request in it
# shares those values by construction. Two concurrent identical requests
# from one user therefore produce four [CACHE] lines whose cohort keys are
# all equal, and the only remaining "join" is adjacency, which interleaving
# destroys. `rid` is a per-request `secrets.token_hex(4)`.


def test_rid_is_eight_lowercase_hex_and_does_not_repeat():
    rids = {lp._new_request_rid() for _ in range(200)}
    assert len(rids) == 200                      # 32 bits, 200 draws
    for r in rids:
        assert len(r) == 8
        assert all(c in "0123456789abcdef" for c in r)


async def test_two_sequential_identical_requests_get_different_rids(
    responses_driver, canary_on, caplog,
):
    """The cohort key is IDENTICAL across the pair (same user, same body,
    same tools) — which is exactly why it cannot be the join. `rid` is the
    field that differs, and it differs without any input differing, so it
    cannot be a function of the body."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    body = _body(_core(20), tool_choice="auto")

    await responses_driver(dict(body), sse=_SSE)
    first_req, first_usage = _cache_lines(caplog)[:2]
    caplog.clear()
    await responses_driver(dict(body), sse=_SSE)
    second_req, second_usage = _cache_lines(caplog)[:2]

    # cohort key: collides
    assert _field(first_req, "cache_key_hash") == _field(
        second_req, "cache_key_hash")
    assert _field(first_req, "tools_sha") == _field(second_req, "tools_sha")
    # rid: does not
    assert _field(first_req, "rid") != _field(second_req, "rid")
    # and each request's two lines agree
    assert _field(first_usage, "rid") == _field(first_req, "rid")
    assert _field(second_usage, "rid") == _field(second_req, "rid")


async def test_interleaved_identical_requests_are_separated_only_by_rid(
    monkeypatch, canary_on, caplog,
):
    """F6, executed. TWO CONCURRENT IDENTICAL REQUESTS — same user, same
    body, byte-for-byte the same tools array — so `cache_key_hash` and
    `tools_sha` are equal on all four lines and the cohort key joins
    nothing.

    The timeline is forced, not hoped for. Request P uploads instantly and
    then waits 200 ms for its first upstream chunk; request Q spends 50 ms
    in `await request.json()` (a property of the socket, not of the body —
    the two bodies are equal) and is served instantly. The [CACHE] lines
    therefore land

        P-request , Q-request , Q-usage , P-usage

    so the nearest-preceding-request-line join — the only one adjacency
    offers — hands P's usage line Q's request line. The assertion below
    proves that mis-join HAPPENS in this run, so the rid assertions are
    not proving a property of an ordering that never occurs.
    """
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")

    async def fake_auth(request, db):
        return types.SimpleNamespace(user_id=CANARY_UID)

    monkeypatch.setattr(lp, "_auth_agent", fake_auth)
    monkeypatch.setattr(lp, "_check_budget",
                        lambda cfg, provider, db: asyncio.sleep(0))

    async def fake_log_event(*a, **kw):
        return None

    monkeypatch.setattr(lp, "_log_event", fake_log_event)
    monkeypatch.setattr(
        lp, "_route_chat", lambda model, cfg: (lp._openai, "sk-test"))

    class _NullSession:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    import app.db.database as _dbmod
    monkeypatch.setattr(_dbmod, "async_session_maker", lambda: _NullSession())

    # The upstream cannot tell the two apart either — the bodies are equal.
    # Which call is which is decided by ARRIVAL ORDER, and arrival order is
    # decided by the upload delay, not by gather's scheduling: P has none,
    # so P is always the first to reach the backend.
    calls: list[int] = []

    async def fake_stream(self, body, api_key, meta=None):
        n = len(calls)
        calls.append(n)
        if n == 0:                       # P: slow first byte
            if meta is not None:
                meta["request_id"] = "req_P"
            await asyncio.sleep(0.20)
        elif meta is not None:           # Q: immediate
            meta["request_id"] = "req_Q"
        yield _SSE

    monkeypatch.setattr(lp.OpenAIBackend, "responses_stream", fake_stream)

    shared = _body(_core(20), tool_choice="auto")

    async def run(body_delay: float):
        resp = await lp.proxy_responses(
            _FakeRequest(dict(shared), body_delay=body_delay), db=None)
        return b"".join([c async for c in resp.body_iterator])

    await asyncio.gather(run(0.0), run(0.05))

    lines = _cache_lines(caplog)
    assert len(lines) == 4, lines
    req_lines = [ln for ln in lines if " tools_sent=" in ln]
    usage_lines = [ln for ln in lines if " req_id=" in ln]
    assert len(req_lines) == 2 and len(usage_lines) == 2

    # 1. THE COHORT KEY COLLIDES — all four lines carry the same pair, so
    #    it identifies the cohort and cannot identify a request.
    assert len({_field(ln, "cache_key_hash") for ln in lines}) == 1
    assert len({_field(ln, "tools_sha") for ln in lines}) == 1

    # 2. The interleaving really is the hazardous one: P-req, Q-req, Q-use,
    #    P-use.
    assert [ln in req_lines for ln in lines] == [True, True, False, False], \
        lines

    def _nearest_preceding_request(usage: str) -> str:
        i = lines.index(usage)
        for j in range(i - 1, -1, -1):
            if lines[j] in req_lines:
                return lines[j]
        raise AssertionError("no preceding request line")

    # 3. THE OLD (adjacency) JOIN IS WRONG for at least one of the two.
    assert any(
        _field(_nearest_preceding_request(u), "rid") != _field(u, "rid")
        for u in usage_lines
    ), lines

    # 4. THE RID JOIN IS CORRECT FOR BOTH — checked against independent
    #    ground truth, not against itself. P's request line is logged first
    #    (no upload delay); P is the call that waited 200 ms for its first
    #    chunk and whose upstream id is req_P.
    assert len({_field(ln, "rid") for ln in req_lines}) == 2
    p_req, q_req = req_lines[0], req_lines[1]
    by_rid = {_field(u, "rid"): u for u in usage_lines}
    assert set(by_rid) == {_field(p_req, "rid"), _field(q_req, "rid")}

    p_usage = by_rid[_field(p_req, "rid")]
    q_usage = by_rid[_field(q_req, "rid")]
    assert _field(p_usage, "req_id") == "req_P"
    assert int(_field(p_usage, "ttfb_ms")) >= 150
    assert int(_field(p_usage, "body_ms")) < 40
    assert _field(q_usage, "req_id") == "req_Q"
    assert int(_field(q_usage, "body_ms")) >= 40
    assert int(_field(q_usage, "ttfb_ms")) < 40


async def test_the_summary_line_joins_the_cache_lines_by_rid(
    responses_driver, canary_on, caplog, monkeypatch,
):
    """`llm_proxy user=… latency=…` is the line an operator greps first and
    the only one carrying `latency`. Before this it could be attached to a
    [CACHE] pair by nothing at all."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    seen: list[dict] = []

    async def spy_log_event(*a, **kw):
        seen.append(kw)

    monkeypatch.setattr(lp, "_log_event", spy_log_event)
    await responses_driver(_body(_core(20)), sse=_SSE,
                           headers={"x-toup-trace": "a1b2c3d4.7"})
    req = _cache_lines(caplog)[0]
    assert seen, "the handler never logged a usage event"
    suffix = seen[-1]["wire_suffix"]
    assert suffix == " rid=%s trace=a1b2c3d4.7" % _field(req, "rid")


async def test_the_summary_suffix_is_empty_off_the_canary(
    responses_driver, canary_off, caplog, monkeypatch,
):
    seen: list[dict] = []

    async def spy_log_event(*a, **kw):
        seen.append(kw)

    monkeypatch.setattr(lp, "_log_event", spy_log_event)
    await responses_driver(_body(_core(20)), sse=_SSE,
                           headers={"x-toup-trace": "a1b2c3d4.7"})
    assert seen[-1]["wire_suffix"] == ""


class _StubDB:
    """Enough of an AsyncSession for `_log_event`'s no-credit path."""

    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        return None


async def test_log_event_renders_the_suffix_and_defaults_to_r47(caplog):
    """The format string itself, both ways. Zero tokens keeps the credit
    branch out of it — this test is about one line of text."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    uid = "00000000-aaaa-4bbb-8ccc-000000000001"

    await lp._log_event(_StubDB(), uid, "openai", "gpt-5.6-terra",
                        "responses", 0, 0, 0, 12)
    plain = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("llm_proxy user=")][-1]
    assert plain.endswith("op=user")

    caplog.clear()
    await lp._log_event(_StubDB(), uid, "openai", "gpt-5.6-terra",
                        "responses", 0, 0, 0, 12,
                        wire_suffix=" rid=deadbeef trace=-")
    withsuf = [r.getMessage() for r in caplog.records
               if r.getMessage().startswith("llm_proxy user=")][-1]
    assert withsuf == plain + " rid=deadbeef trace=-"


# ── (F6-2) the `x-toup-trace` contract, platform half ────────────────
#
# OPTIONAL inbound header. Allowlisted shape only; anything else is
# dropped and never echoed. The AGENT half that would send it is a
# separate patch (`g-llm-trace-header`) and does not exist in the
# deployed agent, so on the fleet as it stands every line reads `trace=-`
# and there is NO agent↔platform join.


@pytest.mark.parametrize("value", [
    "a1b2c3d4.7",
    "00000000.0",
    "ffffffff.999",
    "0f1e2d3c.12",
])
def test_valid_trace_values_pass_through_verbatim(value):
    assert lp._wire_trace_header({"x-toup-trace": value}) == value


@pytest.mark.parametrize("value", [
    "",                       # present but empty
    "a1b2c3d4",               # no iteration
    "a1b2c3d4.",              # empty iteration
    "a1b2c3d4.1234",          # iteration too long
    "a1b2c3d47",              # no separator
    "A1B2C3D4.7",             # uppercase hex — the agent emits lowercase
    "g1b2c3d4.7",             # not hex
    "a1b2c3d.7",              # 7 hex digits
    "a1b2c3d4e.7",            # 9 hex digits
    " a1b2c3d4.7",            # leading space
    "a1b2c3d4.7 ",            # trailing space
    "a1b2c3d4.7.2",           # extra field
    "a1b2c3d4.-1",            # negative
    "a1b2c3d4.٧",             # non-ASCII digit
])
def test_malformed_trace_values_become_a_dash(value):
    assert lp._wire_trace_header({"x-toup-trace": value}) == "-"


def test_absent_and_unreadable_headers_are_a_dash():
    assert lp._wire_trace_header({}) == "-"

    class _Boom:
        def get(self, k):
            raise RuntimeError("header store exploded")

    # A telemetry field may never be the thing that raises.
    assert lp._wire_trace_header(_Boom()) == "-"


def test_only_the_x_toup_trace_header_is_consulted():
    """One header name, by name. A reader that also accepted some other
    header would let a value the agent never chose ride the join field —
    and `x-request-id` in particular is a value an intermediary sets, so
    `trace=` would silently start reporting Cloudflare's or Railway's id
    as though it were the agent's turn marker."""
    valid = "a1b2c3d4.7"
    for other in ("x-request-id", "x-toup-channel", "traceparent",
                  "x-trace", "toup-trace", "x_toup_trace", "trace"):
        assert lp._wire_trace_header({other: valid}) == "-", other
    assert lp._wire_trace_header(
        {"x-request-id": valid, "x-toup-trace": "00000000.1"}) == "00000000.1"


def test_a_trailing_newline_is_rejected():
    """The one that a `^…$` pattern would ACCEPT: in Python `$` matches
    before a trailing newline, so `^[0-9a-f]{8}\\.[0-9]{1,3}$` admits
    "a1b2c3d4.7\\n" — and a newline inside a logged value ends the [CACHE]
    line and starts a line of the sender's own composition in the shared
    platform log stream. `\\A…\\Z` is the whole fix."""
    assert lp._wire_trace_header({"x-toup-trace": "a1b2c3d4.7\n"}) == "-"
    assert lp._wire_trace_header(
        {"x-toup-trace": "a1b2c3d4.7\n[CACHE] user=00000000 forged=1"}) == "-"
    assert lp._wire_trace_header({"x-toup-trace": "a1b2c3d4.7\r\n"}) == "-"
    # and the pattern in use really is anchored at both ends
    assert "\\A" in lp._TRACE_RE.pattern and "\\Z" in lp._TRACE_RE.pattern


def test_a_non_string_header_value_is_a_dash():
    for v in (None, 7, ["a1b2c3d4.7"], {"a": 1}, b"a1b2c3d4.7"):
        assert lp._wire_trace_header({"x-toup-trace": v}) == "-"


def test_an_absurdly_long_value_is_rejected():
    """The RESULT, which the anchored pattern alone already guarantees."""
    assert lp._wire_trace_header({"x-toup-trace": "a" * 100_000}) == "-"
    assert lp._wire_trace_header(
        {"x-toup-trace": "a1b2c3d4.7" + "0" * 100_000}) == "-"


def test_the_length_guard_short_circuits_before_the_regex():
    """Structural, and deliberately separate from the test above, because
    removing the guard changes NO behaviour — mutation M20 dropped it and
    all 108 tests stayed green. It is a cost short-circuit on an
    unauthenticated field: `_TRACE_MAX_LEN` is the longest string the
    pattern can accept, so a 100 KB header is rejected by a `len()` instead
    of by the regex engine (LOCAL: 0.046 µs vs 0.14 µs for a valid value).
    Asserting it here says the guard is intentional; a future reader who
    deletes it breaks a named test rather than a silent property."""
    src = inspect.getsource(lp._wire_trace_header)
    assert "_TRACE_MAX_LEN" in src
    assert src.index("_TRACE_MAX_LEN") < src.index("_TRACE_RE.match")
    assert lp._TRACE_MAX_LEN == 12
    # 12 = 8 hex + "." + 3 digits, i.e. exactly the pattern's maximum
    assert lp._TRACE_RE.match("f" * 8 + "." + "9" * 3)
    assert len("f" * 8 + "." + "9" * 3) == lp._TRACE_MAX_LEN


def test_the_header_lookup_is_case_insensitive_on_a_real_request():
    """Production sends headers through Starlette, whose `Headers.get`
    lowercases the key. A plain dict in a test does not, so this asserts
    the real datastructure rather than the fixture's stand-in."""
    from starlette.datastructures import Headers
    # `Headers(headers=...)` normalises exactly as an ASGI server does.
    h = Headers(headers={"X-Toup-Trace": "a1b2c3d4.7"})
    assert lp._wire_trace_header(h) == "a1b2c3d4.7"
    assert lp._wire_trace_header(Headers(headers={})) == "-"


async def test_a_valid_trace_rides_every_line_of_the_request(
    responses_driver, canary_on, caplog,
):
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(_body(_core(20)), sse=_SSE,
                           headers={"x-toup-trace": "a1b2c3d4.7"})
    req, usage = _cache_lines(caplog)[:2]
    assert _field(req, "trace") == "a1b2c3d4.7"
    assert _field(usage, "trace") == "a1b2c3d4.7"


async def test_no_trace_header_reads_as_a_dash_on_every_line(
    responses_driver, canary_on, caplog,
):
    """The state of the whole fleet today: no agent sends the header."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(_body(_core(20)), sse=_SSE)
    for line in _cache_lines(caplog)[:2]:
        assert _field(line, "trace") == "-"


@pytest.mark.parametrize("hostile,tell", [
    # log forgery: a newline would end the [CACHE] line and start one of
    # the sender's own composition in the shared platform log stream
    ("a1b2c3d4.7\n[CACHE] user=00000000 model=evil cached_tokens=99999",
     "cached_tokens=99999"),
    # logging-format injection (the value is an ARG, not the format — but a
    # future "malformed %r" branch would make it one)
    ("%s%s%s%(asctime)s", "%(asctime)s"),
    ("\' OR 1=1 --", "OR 1=1"),
    ("../../etc/passwd", "passwd"),
    ("<script>alert(1)</script>", "script"),
    ("\x00a1b2c3d4.7", "\x00"),
    # someone else's identifier, offered as a "trace"
    ("00000000-aaaa-4bbb-8ccc-000000000009", "000000000009"),
    ("a1b2c3d4.7; DROP TABLE llm_proxy_events", "DROP TABLE"),
])
async def test_a_hostile_trace_value_reaches_no_log_line(
    responses_driver, canary_on, caplog, hostile, tell,
):
    """Every rejection is silent: not the value, not a prefix of it, not
    its length, not a "malformed %r". A quoted attacker string IS the
    log-forging bug the allowlist exists to prevent — and the header is
    unauthenticated by construction (the bearer token authenticates the
    CALLER, not this field's contents)."""
    caplog.set_level(logging.DEBUG, logger="app.api.llm_proxy")
    await responses_driver(_body(_core(20)), sse=_SSE,
                           headers={"x-toup-trace": hostile})
    whole_log = "\n".join(r.getMessage() for r in caplog.records)
    assert hostile not in whole_log
    assert tell not in whole_log, whole_log
    for line in _cache_lines(caplog)[:2]:
        assert _field(line, "trace") == "-"


async def test_the_trace_header_never_reaches_the_upstream_body(
    responses_driver, canary_on, caplog,
):
    """It is a platform-local log field. Forwarding it would hand the
    provider a per-turn correlator it has no business holding."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(_body(_core(3)), sse=_SSE,
                           headers={"x-toup-trace": "a1b2c3d4.7"})
    sent = responses_driver.state["sent"][-1]
    assert "a1b2c3d4.7" not in json.dumps(sent)
    assert "x-toup-trace" not in json.dumps(sent).lower()
    assert "trace" not in sent


async def test_the_trace_header_does_not_change_the_body_sent_upstream(
    responses_driver, canary_on, caplog,
):
    """The instrument must not move the thing it measures."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    await responses_driver(_body(_core(3)), sse=_SSE)
    without = json.dumps(responses_driver.state["sent"][-1], sort_keys=True)
    await responses_driver(_body(_core(3)), sse=_SSE,
                           headers={"x-toup-trace": "a1b2c3d4.7"})
    with_hdr = json.dumps(responses_driver.state["sent"][-1], sort_keys=True)
    assert without == with_hdr


async def test_no_inbound_header_is_forwarded_upstream_at_all(monkeypatch):
    """Structural, on the SHIPPED generator: the outbound header dict is
    built by name, so there is no passthrough for `x-toup-trace` or for
    anything else the agent chooses to send. A future refactor that starts
    copying `request.headers` upstream fails here."""
    captured: dict = {}

    class _Resp:
        status_code = 200
        headers = {"x-request-id": "req_x"}

        async def aiter_bytes(self):
            yield b"data: {}\n\n"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url, json=None, headers=None):
            captured["headers"] = dict(headers or {})
            return _Resp()

    monkeypatch.setattr(lp.httpx, "AsyncClient", _Client)
    chunks = [c async for c in lp._openai.responses_stream(
        {"model": "gpt-5.6-terra"}, "sk")]
    assert chunks == [b"data: {}\n\n"]
    assert set(k.lower() for k in captured["headers"]) == {
        "authorization", "content-type"}


# ── (F6-3) the blind areas are named where the fields are defined ────


def test_the_timing_docstring_names_all_three_blind_areas():
    """A split that does not say what it cannot see gets reconciled
    against `[req-timing]` as though the residual were clock error."""
    doc = lp._wire_timing_suffix.__doc__ or ""
    assert "BLIND AREAS" in doc
    # 1. pre-handler work, and why there is no dep_ms
    assert "Depends(get_db)" in doc
    assert "dep_ms" in doc
    assert "platform_main.py:1018" in doc
    # 2. the chat wire has no timing split
    assert "chat/completions" in doc
    # 3. [req-timing] only fires on the slow tail — and the gate is quoted
    #    in FULL. A partial quote of a gate reads as the whole gate (N1).
    assert "500 ms" in doc
    assert "/api/auth/" in doc
    assert "/agent-setup/config" in doc
    assert "THREE" in doc
    # and the upload is still separated from proxy work
    assert "upload" in doc.lower()


def test_the_middleware_start_stamp_really_is_unreachable():
    """The documented reason for having no `dep_ms`, asserted against the
    source rather than asserted in prose. `_t0` is a local of
    `_RequestTimingMiddleware.__call__`; if a future change publishes it
    into `scope`/`scope["state"]`/`request.state`, this test fails and the
    `dep_ms` field becomes available — which is the outcome we want to be
    told about."""
    import pathlib
    main_py = pathlib.Path(lp.__file__).parents[2] / "platform_main.py"
    src = main_py.read_text()
    start = src.index("class _RequestTimingMiddleware")
    end = src.index("app.add_middleware(_RequestTimingMiddleware)")
    mw = src[start:end]
    assert "_t0 = " in mw
    for publish in ('scope["state"]', "scope['state']", "request.state",
                    "scope.setdefault", 'scope["_'):
        assert publish not in mw, mw
    # and the stamp is not ASGI entry even so: something is mounted outside
    # it. `add_middleware` inserts at index 0, so the LAST add is outermost.
    assert src.index("app.add_middleware(_RequestTimingMiddleware)") < \
        src.index("app.add_middleware(AttachmentBodyLimitMiddleware)")
    # The emission gate this patch documents, asserted as the WHOLE line
    # and not as a leading substring — the earlier version asserted only
    # the first two disjuncts, so it could not notice the docstring
    # omitting the third (N1). The conclusion for /api/llm/ is unchanged
    # and is asserted separately below: none of the three matches it.
    gate = [ln.strip() for ln in mw.splitlines()
            if "_dur_ms > 500" in ln]
    assert len(gate) == 1, gate
    assert gate[0] == (
        'if _dur_ms > 500 or _p.startswith("/api/auth/") '
        'or "/agent-setup/config" in _p:'
    ), gate[0]
    _p = "/api/llm/openai/v1/responses"
    assert not _p.startswith("/api/auth/")
    assert "/agent-setup/config" not in _p


# ── the chat wire's three lines share one rid ────────────────────────


async def test_chat_wire_lines_are_joined_by_rid(
    monkeypatch, canary_on, caplog,
):
    """`/chat` gets the join but NOT the timing split (blind area 2). All
    three of its lines — request [CACHE], usage [CACHE], summary — carry
    the same rid, or the wire has a half-built instrument."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    seen: list[dict] = []

    async def fake_auth(request, db):
        return types.SimpleNamespace(user_id=CANARY_UID)

    async def spy_log_event(*a, **kw):
        seen.append(kw)

    async def fake_chat(b, api_key):
        return types.SimpleNamespace(
            status_code=200, headers={}, content=b"",
            json=lambda: {"usage": {"prompt_tokens": 10,
                                    "completion_tokens": 2,
                                    "prompt_tokens_details":
                                        {"cached_tokens": 4}}})

    monkeypatch.setattr(lp, "_auth_agent", fake_auth)
    monkeypatch.setattr(lp, "_check_budget",
                        lambda cfg, provider, db: asyncio.sleep(0))
    monkeypatch.setattr(lp, "_log_event", spy_log_event)
    monkeypatch.setattr(
        lp, "_route_chat",
        lambda model, cfg: (types.SimpleNamespace(name="openai",
                                                  chat=fake_chat), "k"))

    await lp.proxy_chat(_FakeRequest(
        {"model": "gpt-5.5", "stream": False, "messages": [],
         "tools": _core(10)},
        headers={"x-toup-trace": "a1b2c3d4.7"}), db=None)

    lines = _cache_lines(caplog)
    assert len(lines) == 2, lines
    rid = _field(lines[0], "rid")
    assert len(rid) == 8
    assert _field(lines[1], "rid") == rid
    assert _field(lines[1], "trace") == "a1b2c3d4.7"
    assert seen[-1]["wire_suffix"] == " rid=%s trace=a1b2c3d4.7" % rid
    # and no timing fields leaked onto this wire
    assert " ttfb_ms=" not in lines[1]


async def test_chat_stream_usage_line_carries_the_join(
    monkeypatch, canary_on, caplog,
):
    """The STREAMING `/chat` usage line — a separate `logger.info` inside
    `stream_and_log`, and the one mutation M28 proved was uncovered: with
    its suffix replaced by `""` all 108 tests stayed green, because every
    other `/chat` test drove the JSON path. "Fixing one wire and leaving
    the other" is this file's recorded failure mode; so is fixing one
    BRANCH of one wire."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    seen: list[dict] = []

    async def fake_auth(request, db):
        return types.SimpleNamespace(user_id=CANARY_UID)

    async def spy_log_event(*a, **kw):
        seen.append(kw)

    async def fake_chat_stream(body, api_key):
        yield (b'data: {"usage": {"prompt_tokens": 100, '
               b'"completion_tokens": 7, '
               b'"prompt_tokens_details": {"cached_tokens": 40}}}\n\n')

    monkeypatch.setattr(lp, "_auth_agent", fake_auth)
    monkeypatch.setattr(lp, "_check_budget",
                        lambda cfg, provider, db: asyncio.sleep(0))
    monkeypatch.setattr(lp, "_log_event", spy_log_event)
    monkeypatch.setattr(
        lp, "_route_chat",
        lambda model, cfg: (types.SimpleNamespace(
            name="openai", chat_stream=fake_chat_stream), "k"))

    class _NullSession:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    import app.db.database as _dbmod
    monkeypatch.setattr(_dbmod, "async_session_maker", lambda: _NullSession())

    resp = await lp.proxy_chat(_FakeRequest(
        {"model": "gpt-5.5", "stream": True, "messages": [],
         "tools": _core(10)},
        headers={"x-toup-trace": "a1b2c3d4.7"}), db=None)
    async for _ in resp.body_iterator:
        pass

    lines = _cache_lines(caplog)
    assert len(lines) == 2, lines
    rid = _field(lines[0], "rid")
    assert len(rid) == 8
    assert _field(lines[1], "rid") == rid
    assert _field(lines[1], "trace") == "a1b2c3d4.7"
    assert _field(lines[1], "cached_tokens") == "40"
    assert seen[-1]["wire_suffix"] == " rid=%s trace=a1b2c3d4.7" % rid


@pytest.mark.parametrize("canary,expected", [(True, "rid"), (False, "")])
async def test_the_chat_error_path_summary_carries_the_join(
    monkeypatch, caplog, canary, expected,
):
    """The `/chat` twin of the `/responses` error-path fix (FINAL round 2,
    non-blocking 2). "Fixing one wire and leaving the other" is this file's
    recorded failure mode, so the error path gets both."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    monkeypatch.setattr(lp.settings, "llm_proxy_wire_observability", False,
                        raising=False)
    monkeypatch.setattr(
        lp.settings, "llm_proxy_wire_observability_canary_user_ids",
        CANARY_UID if canary else "", raising=False)
    seen: list[dict] = []

    async def fake_auth(request, db):
        return types.SimpleNamespace(user_id=CANARY_UID)

    async def spy_log_event(*a, **kw):
        seen.append(kw)

    async def fake_chat_stream(b, api_key):
        raise lp.UpstreamProviderError(500, b"boom", "openai")
        yield b""  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(lp, "_auth_agent", fake_auth)
    monkeypatch.setattr(lp, "_check_budget",
                        lambda cfg, provider, db: asyncio.sleep(0))
    monkeypatch.setattr(lp, "_log_event", spy_log_event)
    monkeypatch.setattr(
        lp, "_route_chat",
        lambda model, cfg: (types.SimpleNamespace(
            name="openai", chat_stream=fake_chat_stream), "k"))

    with pytest.raises(lp.HTTPException):
        await lp.proxy_chat(_FakeRequest(
            {"model": "gpt-5.5", "stream": True, "messages": [],
             "tools": _core(10)},
            headers={"x-toup-trace": "a1b2c3d4.7"}), db=None)

    assert seen, "_log_event was never called on the /chat error path"
    suffix = seen[0].get("wire_suffix")
    if expected:
        req = _cache_lines(caplog)[0]
        assert suffix == " rid=%s trace=a1b2c3d4.7" % _field(req, "rid")
    else:
        assert suffix == ""


async def test_chat_stream_usage_line_is_r47_off_the_canary(
    monkeypatch, canary_off, caplog,
):
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")

    async def fake_auth(request, db):
        return types.SimpleNamespace(user_id=CANARY_UID)

    async def fake_log_event(*a, **kw):
        return None

    async def fake_chat_stream(body, api_key):
        yield (b'data: {"usage": {"prompt_tokens": 100, '
               b'"completion_tokens": 7}}\n\n')

    monkeypatch.setattr(lp, "_auth_agent", fake_auth)
    monkeypatch.setattr(lp, "_check_budget",
                        lambda cfg, provider, db: asyncio.sleep(0))
    monkeypatch.setattr(lp, "_log_event", fake_log_event)
    monkeypatch.setattr(
        lp, "_route_chat",
        lambda model, cfg: (types.SimpleNamespace(
            name="openai", chat_stream=fake_chat_stream), "k"))

    class _NullSession:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    import app.db.database as _dbmod
    monkeypatch.setattr(_dbmod, "async_session_maker", lambda: _NullSession())

    resp = await lp.proxy_chat(_FakeRequest(
        {"model": "gpt-5.5", "stream": True, "messages": [],
         "tools": _core(10)},
        headers={"x-toup-trace": "a1b2c3d4.7"}), db=None)
    async for _ in resp.body_iterator:
        pass

    assert _cache_lines(caplog)[1] == (
        "[CACHE] user=0a1b2c3d model=gpt-5.5 has_cache_key=False "
        "retention=none prompt_tokens=100 cached_tokens=0 "
        "cache_write_tokens=0")


async def test_chat_wire_is_byte_identical_off_the_canary(
    monkeypatch, canary_off, caplog,
):
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    seen: list[dict] = []

    async def fake_auth(request, db):
        return types.SimpleNamespace(user_id=CANARY_UID)

    async def spy_log_event(*a, **kw):
        seen.append(kw)

    async def fake_chat(b, api_key):
        return types.SimpleNamespace(
            status_code=200, headers={}, content=b"",
            json=lambda: {"usage": {"prompt_tokens": 10,
                                    "completion_tokens": 2}})

    monkeypatch.setattr(lp, "_auth_agent", fake_auth)
    monkeypatch.setattr(lp, "_check_budget",
                        lambda cfg, provider, db: asyncio.sleep(0))
    monkeypatch.setattr(lp, "_log_event", spy_log_event)
    monkeypatch.setattr(
        lp, "_route_chat",
        lambda model, cfg: (types.SimpleNamespace(name="openai",
                                                  chat=fake_chat), "k"))

    await lp.proxy_chat(_FakeRequest(
        {"model": "gpt-5.5", "stream": False, "messages": [],
         "tools": _core(10)},
        headers={"x-toup-trace": "a1b2c3d4.7"}), db=None)

    assert _cache_lines(caplog) == [
        "[CACHE] user=0a1b2c3d model=gpt-5.5 has_cache_key=False "
        "cache_key_hash=none retention=none",
        "[CACHE] user=0a1b2c3d model=gpt-5.5 has_cache_key=False "
        "retention=none prompt_tokens=10 cached_tokens=0 "
        "cache_write_tokens=0",
    ]
    assert seen[-1]["wire_suffix"] == ""


async def test_the_upstream_error_warning_is_joined_to_its_request_line(
    monkeypatch, canary_on, caplog,
):
    """A failed turn's trail is exactly three lines — the request-side
    [CACHE], this WARNING, and the `llm_proxy … status=error` summary
    (there is no usage-side line). Before the rid nothing paired them.
    `req_id` alone cannot: it comes from the provider and is absent on
    every failure that never reached one.

    The summary line's half was missing until the round-2 FINAL pass
    (non-blocking 2): the error-path `_log_event` calls passed no
    `wire_suffix`, so a FAILED turn was the one case where the summary
    could not be tied to its own request — and those are exactly the turns
    G7/G8 have to exclude."""
    caplog.set_level(logging.INFO, logger="app.api.llm_proxy")
    seen: list = []

    async def fake_auth(request, db):
        return types.SimpleNamespace(user_id=CANARY_UID)

    async def fake_log_event(*a, **kw):
        seen.append(kw)
        return None

    async def fake_stream(self, body, api_key, meta=None):
        if meta is not None:
            meta["request_id"] = "req_boom"
        raise lp.UpstreamProviderError(500, b"upstream exploded", "openai")
        yield b""  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(lp, "_auth_agent", fake_auth)
    monkeypatch.setattr(lp, "_check_budget",
                        lambda cfg, provider, db: asyncio.sleep(0))
    monkeypatch.setattr(lp, "_log_event", fake_log_event)
    monkeypatch.setattr(lp.OpenAIBackend, "responses_stream", fake_stream)
    monkeypatch.setattr(
        lp, "_route_chat", lambda model, cfg: (lp._openai, "sk-test"))

    with pytest.raises(lp.HTTPException):
        await lp.proxy_responses(_FakeRequest(
            _body(_core(3)), headers={"x-toup-trace": "a1b2c3d4.7"}), db=None)

    req = _cache_lines(caplog)[0]
    warns = [r.getMessage() for r in caplog.records
             if "[LLM-PROXY]" in r.getMessage()
             and "upstream 500" in r.getMessage()]
    assert warns, [r.getMessage() for r in caplog.records]
    assert _field(warns[0], "rid") == _field(req, "rid")
    assert _field(warns[0], "trace") == "a1b2c3d4.7"
    assert _field(warns[0], "req_id") == "req_boom"
    # the third line of the trail: the error-path summary
    assert seen, "_log_event was never called on the error path"
    assert seen[0].get("wire_suffix") == " rid=%s trace=a1b2c3d4.7" % (
        _field(req, "rid"),)


async def test_the_upstream_error_warning_is_r47_off_the_canary(
    monkeypatch, canary_off, caplog,
):
    """The `req_id=` field on this WARNING is UNCONDITIONAL (round 2, N6);
    the rid/trace pair is not. Off the canary the line must carry the
    provider id and nothing else new — and the error-path `wire_suffix`
    added in the round-2 FINAL pass must render `""`, so the failed turn's
    summary line stays byte-identical to R47."""
    caplog.set_level(logging.WARNING, logger="app.api.llm_proxy")
    seen: list = []

    async def fake_auth(request, db):
        return types.SimpleNamespace(user_id=CANARY_UID)

    async def fake_log_event(*a, **kw):
        seen.append(kw)
        return None

    async def fake_stream(self, body, api_key, meta=None):
        if meta is not None:
            meta["request_id"] = "req_boom"
        raise lp.UpstreamProviderError(500, b"upstream exploded", "openai")
        yield b""  # pragma: no cover

    monkeypatch.setattr(lp, "_auth_agent", fake_auth)
    monkeypatch.setattr(lp, "_check_budget",
                        lambda cfg, provider, db: asyncio.sleep(0))
    monkeypatch.setattr(lp, "_log_event", fake_log_event)
    monkeypatch.setattr(lp.OpenAIBackend, "responses_stream", fake_stream)
    monkeypatch.setattr(
        lp, "_route_chat", lambda model, cfg: (lp._openai, "sk-test"))

    with pytest.raises(lp.HTTPException):
        await lp.proxy_responses(_FakeRequest(
            _body(_core(3)), headers={"x-toup-trace": "a1b2c3d4.7"}), db=None)

    warns = [r.getMessage() for r in caplog.records
             if "[LLM-PROXY]" in r.getMessage()
             and "upstream 500" in r.getMessage()]
    assert warns
    assert " rid=" not in warns[0]
    assert " trace=" not in warns[0]
    assert "a1b2c3d4.7" not in warns[0]
    assert _field(warns[0], "req_id") == "req_boom"
    assert seen and seen[0].get("wire_suffix") == ""
