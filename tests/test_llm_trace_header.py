"""`x-toup-trace` — the agent half of the agent<->platform join key (R48-G).

WHAT THESE TESTS ESTABLISH (and what they do not)

They establish the WIRE BEHAVIOUR of the agent's Responses call: whether the
header is present, what its value is, and — the one that matters most for a
default-off patch — that with the flag off the captured request is byte-for-
byte what it is on an unpatched tree, headers AND body.

They establish NOTHING about latency, about the platform half, or about
production. The join this header creates is only usable once the platform
logs the validated header (patch B); until then the agent is talking to a
listener that is not there, which is harmless and also useless.

LANE: agent. Run ONE file per process:

    cd backend && RUN_MODE=agent PYTHONDONTWRITEBYTECODE=1 \
      python -m pytest -q -p no:cacheprovider tests/test_llm_trace_header.py
"""
from __future__ import annotations

import copy
import re

import pytest

from app.config import settings


SYSTEM = "you are a test"

#: An obviously synthetic id. Never a real user id, here or anywhere.
CANARY_USER = "00000000-aaaa-4bbb-8ccc-000000000001"
OTHER_USER = "00000000-aaaa-4bbb-8ccc-000000000002"
CMID = "11111111-2222-4333-8444-555555555555"


class _RecordingClient:
    """Stands in for the OpenAI SDK client — records the request kwargs.

    Deliberately the same shape as the one in
    `tests/test_cache_warm_on_connect.py`: it sits UNDER the service, so the
    real `_create_responses_stream` runs and what lands in the sink is the
    actual request the SDK would have been handed.
    """

    def __init__(self, sink, *, base_url=None):
        self.responses = self
        self._sink = sink
        self.base_url = base_url or f"{settings.platform_api_url.rstrip('/')}/llm/openai/v1/"

    async def create(self, **kwargs):
        # Deep-copied: the service mutates `kwargs` between retries, and a
        # sink holding live references would quietly compare a request to
        # itself.
        self._sink.append(copy.deepcopy(kwargs))

        class _Empty:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        return _Empty()


async def _capture(**call_kwargs) -> dict:
    """Drive the real Responses path once and return the request kwargs."""
    from app.services.openai_agent_service import OpenAIAgentService

    svc = OpenAIAgentService.__new__(OpenAIAgentService)
    svc._responses_reasoning = {}
    sink: list = []
    svc.client = _RecordingClient(sink)
    async for _ in svc._create_responses_stream(
        messages=[{"role": "user", "content": "."}],
        system=SYSTEM,
        tools=None,
        model="gpt-5.6-terra",
        max_tokens=16,
        tool_choice="none",
        **call_kwargs,
    ):
        pass
    assert len(sink) == 1, f"expected one request, captured {len(sink)}"
    return sink[0]


@pytest.fixture(autouse=True)
def _flags_off(monkeypatch):
    """Every test states its own flag state; nothing inherits one.

    Also clears the module's two latches (one ERROR per malformed-entry
    LENGTH, one WARNING per process for the no-proxy destination) so neither
    test of them can be made vacuous by an earlier test having already fired.

    And it puts the process on the BUNDLE path, because that is the only
    configuration in which the header exists at all and `Settings`' defaults
    are `llm_mode="manual"` / `toup_token=""` (SOURCE `app/config.py:456,
    :2051`) — i.e. the defaults are the BYOK-direct case. Tests that care
    about the destination gate set it themselves; this makes every OTHER test
    state the gate it is not about, instead of passing by accident.
    """
    from app.agent import llm_trace

    monkeypatch.setattr(settings, "llm_trace_header", False, raising=False)
    monkeypatch.setattr(
        settings, "llm_trace_header_canary_user_ids", "", raising=False)
    monkeypatch.setattr(settings, "llm_mode", "bundle", raising=False)
    monkeypatch.setattr(settings, "toup_token", "synthetic-token", raising=False)
    llm_trace._warned_lengths.clear()
    llm_trace._warned_destination = False
    yield
    llm_trace._warned_lengths.clear()
    llm_trace._warned_destination = False


# ── the wire ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_flag_off_the_request_is_byte_identical():
    """The whole safety claim for a default-off patch, tested rather than
    asserted in prose.

    Not "no trace header" — IDENTICAL. The two requests are compared whole, so
    a stray `extra_headers: {}` (which `setdefault` would create if the guard
    were `if llm_trace is not None`), an extra body key, or a reordered value
    all fail here. The `channel` is set so the comparison runs on a request
    that already HAS an extra_headers dict, which is the case where an
    in-place mutation would be easiest to miss.
    """
    baseline = await _capture(channel="app")
    with_param_none = await _capture(channel="app", llm_trace=None)
    assert with_param_none == baseline

    # …and the sensitivity half: the comparison can fail.
    with_header = await _capture(channel="app", llm_trace="deadbeef.0")
    assert with_header != baseline


@pytest.mark.asyncio
async def test_flag_off_creates_no_extra_headers_dict_at_all():
    """The narrower version of the above, for the no-channel case: the header
    block must not be the thing that first materialises `extra_headers`."""
    req = await _capture(llm_trace=None)
    assert "extra_headers" not in req


@pytest.mark.asyncio
async def test_the_header_is_present_and_well_formed_when_a_value_is_passed():
    from app.services.openai_agent_service import (
        LLM_TRACE_HEADER, _TRACE_VALUE_RE,
    )

    req = await _capture(llm_trace="a1b2c3d4.0")
    headers = req.get("extra_headers") or {}
    assert headers[LLM_TRACE_HEADER] == "a1b2c3d4.0"
    assert _TRACE_VALUE_RE.match(headers[LLM_TRACE_HEADER])


@pytest.mark.asyncio
async def test_the_public_entrypoint_forwards_the_value_to_the_responses_wire(
    monkeypatch,
):
    """`create_message_stream` is what `agent_runner` actually calls; the
    Responses path is reached through it, not directly.

    Written because the first version of this file only drove
    `_create_responses_stream`, so DELETING the `llm_trace=llm_trace`
    forwarding line in `create_message_stream` left every test green. That is
    exactly the seam a helper-only test cannot see.
    """
    from app.services.openai_agent_service import (
        LLM_TRACE_HEADER, OpenAIAgentService,
    )

    svc = OpenAIAgentService.__new__(OpenAIAgentService)
    svc._responses_reasoning = {}
    sink: list = []
    svc.client = _RecordingClient(sink)
    monkeypatch.setattr(OpenAIAgentService, "_ensure_client", lambda self: None)
    svc.default_model = "gpt-5.6-terra"
    svc.default_max_tokens = 16

    import app.services.credit_reporter as reporter

    async def _no_gate():
        return None

    monkeypatch.setattr(reporter, "raise_if_exhausted_async", _no_gate)

    async for _ in svc.create_message_stream(
        messages=[{"role": "user", "content": "."}],
        system=SYSTEM,
        model="gpt-5.6-terra",
        max_tokens=16,
        channel="app",
        llm_trace="a1b2c3d4.3",
    ):
        pass

    assert len(sink) == 1
    assert (sink[0].get("extra_headers") or {})[LLM_TRACE_HEADER] == "a1b2c3d4.3"


@pytest.mark.asyncio
async def test_stale_direct_client_after_bind_refresh_cannot_receive_trace(
    monkeypatch, caplog,
):
    """Settings may say bundle while a failed refresh leaves a direct client.

    The actual Responses destination decides whether the header is sent, and
    the completion event reports the same decision to the runner's log.
    """
    import logging

    from app.services.openai_agent_service import OpenAIAgentService, LLM_TRACE_HEADER

    svc = OpenAIAgentService.__new__(OpenAIAgentService)
    svc._responses_reasoning = {}
    svc.default_model = "gpt-5.6-terra"
    svc.default_max_tokens = 16
    sink: list = []
    svc.client = _RecordingClient(sink, base_url="https://api.openai.com/v1/")
    monkeypatch.setattr(OpenAIAgentService, "_ensure_client", lambda self: None)

    import app.services.credit_reporter as reporter

    async def _no_gate():
        return None

    monkeypatch.setattr(reporter, "raise_if_exhausted_async", _no_gate)
    assert settings.llm_mode == "bundle" and bool(settings.toup_token)

    events = []
    with caplog.at_level(logging.WARNING, logger="app.services.openai_agent_service"):
        async for event in svc.create_message_stream(
            messages=[{"role": "user", "content": "."}],
            system=SYSTEM,
            model="gpt-5.6-terra",
            max_tokens=16,
            llm_trace="a1b2c3d4.3",
        ):
            events.append(event)

    assert len(sink) == 1
    assert LLM_TRACE_HEADER not in (sink[0].get("extra_headers") or {})
    assert events[-1].type == "message_end"
    assert events[-1].llm_trace_sent is None
    assert any("trace withheld" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_destination_guard_matches_real_factory_clients(monkeypatch):
    """The SDK normalizes base URLs; exercise its actual factory, not only
    the recording stub used for request assertions."""
    from app.services.bundle_client import make_openai_client
    from app.services.openai_agent_service import _client_targets_platform_proxy

    bundle = make_openai_client()
    assert bundle is not None
    try:
        assert _client_targets_platform_proxy(bundle)
    finally:
        await bundle.close()

    monkeypatch.setattr(settings, "llm_mode", "manual", raising=False)
    direct = make_openai_client(byok_key="synthetic-key")
    assert direct is not None
    try:
        assert not _client_targets_platform_proxy(direct)
    finally:
        await direct.close()


@pytest.mark.asyncio
async def test_the_header_rides_alongside_the_existing_ones():
    """X-Toup-Channel and the operation-type marker must survive: this is a
    copy into a dict another block may already own."""
    import app.agent.cache_warm as cw
    from app.services.openai_agent_service import (
        LLM_TRACE_HEADER, OPERATION_TYPE_HEADER,
    )

    req = await _capture(
        channel="app",
        operation_type=cw.SYSTEM_OPERATION,
        llm_trace="a1b2c3d4.2",
    )
    headers = req["extra_headers"]
    assert headers["X-Toup-Channel"] == "app"
    assert headers[OPERATION_TYPE_HEADER] == cw.SYSTEM_OPERATION
    assert headers[LLM_TRACE_HEADER] == "a1b2c3d4.2"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    "",                      # empty
    "A1B2C3D4.0",            # upper-case hex — the platform regex is lower
    "a1b2c3d.0",             # 7 hex
    "a1b2c3d4e.0",           # 9 hex
    "a1b2c3d4.",             # no iteration
    "a1b2c3d4.1000",         # 4 digits — outside the platform grammar
    "a1b2c3d4.0\r\nx: y",    # header injection attempt
    "a1b2c3d4.0\n",          # LONE LF — see the anchoring test below
    "a1b2c3d4.0\r",          # lone CR
    "a1b2c3d4.0\n\n",
    "a1b2c3d4.0 ",           # trailing space
    "not-a-trace",
])
async def test_the_transport_refuses_a_value_outside_the_grammar(bad):
    """The transport re-validates rather than trusting its caller.

    `agent_runner` only ever passes what `llm_trace.trace_value` minted, but a
    header value is the wrong place to discover that a future caller changed.
    A refused value must leave the request exactly as it was — so this asserts
    equality with the no-header baseline, not merely absence of the key.
    """
    baseline = await _capture(channel="app")
    assert await _capture(channel="app", llm_trace=bad) == baseline


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    7,                    # re.match: "expected string or bytes-like object"
    ["a1b2c3d4.0"],       # same
    b"a1b2c3d4.0",        # "cannot use a string pattern on a bytes-like object"
    object(),
])
async def test_the_transport_refuses_a_value_of_the_wrong_TYPE(bad):
    """A guard that refuses must REFUSE, not raise.

    `re.match` raises `TypeError` on anything that is not str — so a guard
    spelled `if llm_trace and _RE.match(llm_trace)` turns the wrong-type case
    into a turn-ending exception instead of a request without the header. The
    runner's attempt loop would catch it, which makes the outcome a retried
    (slow) turn rather than a crash — the exact class this investigation is
    about. Unreachable from `trace_for_turn`, which returns Optional[str];
    "last line of defence" is only true if the last line holds for the values
    the line exists for.
    """
    baseline = await _capture(channel="app")
    assert await _capture(channel="app", llm_trace=bad) == baseline


# ── the value: hash, gate, iteration ───────────────────────────────


def test_the_hash_is_the_existing_one_not_a_second_implementation():
    """The join only works because all three hops compute the same 8 hex
    digits. If `llm_trace` ever grew its own hash this test is the only thing
    in the repo that would notice."""
    from app.agent import llm_trace
    from app.api._turn_trace import cmid_hash

    for probe in (CMID, "abc", "", "a" * 200, "sürrogate-é"):
        assert llm_trace._hash()(probe) == cmid_hash(probe)

    # …and the half that matters: the value ACTUALLY SENT is built from that
    # function. Asserting only on `_hash()` leaves `trace_value` free to
    # compute its own digest — a mutation that swapped it for md5 passed the
    # line above untouched.
    for probe in (CMID, "abc", "a" * 200, "sürrogate-é"):
        assert llm_trace.trace_value(probe, 4) == f"{cmid_hash(probe)}.4"


def test_the_transport_constants_do_not_drift_from_the_contract():
    """`openai_agent_service` restates the header NAME and the GRAMMAR so it
    need not import `app/agent/`. Restating is only safe while the two agree."""
    from app.agent import llm_trace
    from app.services import openai_agent_service as svc

    assert svc.LLM_TRACE_HEADER.lower() == llm_trace.LLM_TRACE_HEADER.lower()
    assert svc._TRACE_VALUE_RE.pattern == llm_trace.TRACE_VALUE_RE.pattern


@pytest.mark.parametrize("pattern_owner", ["contract", "transport"])
def test_the_grammar_is_anchored_against_a_trailing_newline(pattern_owner):
    """`^…$` and `\\A…\\Z` are NOT the same rule in Python.

    `$` also matches immediately before a final newline, so the contract as
    written — `^[0-9a-f]{8}\\.[0-9]{1,3}$` — ACCEPTS `"a1b2c3d4.0\\n"`. That
    was measured through the real transport, not argued: the round-1 final
    review of this patch caught it, and the test list had been advertising
    "CRLF injection" coverage while the lone-LF case went through.

    Nothing the shipped producer mints can end in a newline, so this is not a
    live defect — it is a defect in the thing this regex exists for, which is
    refusing a value that did NOT come from `trace_value`. Both spellings are
    asserted, because the drift test above only catches a ONE-sided revert.
    """
    from app.agent import llm_trace
    from app.services import openai_agent_service as svc

    rx = (llm_trace.TRACE_VALUE_RE if pattern_owner == "contract"
          else svc._TRACE_VALUE_RE)

    assert rx.match("a1b2c3d4.0"), "the well-formed value must still pass"
    for bad in ("a1b2c3d4.0\n", "a1b2c3d4.0\r", "a1b2c3d4.0\r\n",
                "a1b2c3d4.0\n\n", "a1b2c3d4.0\nx: y"):
        assert not rx.match(bad), repr(bad)
    # The property, not just the cases: `$` would pass the first of those.
    assert r"\A" in rx.pattern and r"\Z" in rx.pattern, rx.pattern


def test_no_client_msg_id_means_no_header_value():
    """Routines, channel turns and sub-agent runs have no client_msg_id.
    `cmid_hash(None)` answers the sentinel `00000000`, so without this guard
    every id-less turn in the fleet would share one trace value — a join key
    that joins strangers."""
    from app.agent import llm_trace

    assert llm_trace.trace_value(None, 0) is None
    assert llm_trace.trace_value("", 0) is None
    # The sentinel is real, which is what makes the guard necessary.
    from app.api._turn_trace import cmid_hash
    assert cmid_hash(None) == "00000000"


def test_the_iteration_index_increments_and_is_zero_based():
    from app.agent import llm_trace

    head = llm_trace.trace_value(CMID, 0).split(".")[0]
    assert [llm_trace.trace_value(CMID, i) for i in range(3)] == [
        f"{head}.0", f"{head}.1", f"{head}.2",
    ]


def test_an_iteration_past_the_platform_grammar_sends_nothing():
    """1000 cannot be expressed in `[0-9]{1,3}`. Sending a value the platform
    will reject is worse than sending none: on the platform side it reads as a
    broken contract rather than a long turn."""
    from app.agent import llm_trace

    assert llm_trace.trace_value(CMID, llm_trace.MAX_TRACE_ITERATION) is not None
    assert llm_trace.trace_value(CMID, llm_trace.MAX_TRACE_ITERATION + 1) is None
    assert llm_trace.trace_value(CMID, -1) is None
    assert llm_trace.trace_value(CMID, "two") is None  # type: ignore[arg-type]


@pytest.mark.parametrize("hostile", [
    "a\r\nX-Injected: 1",
    "b\nc",
    "\x00null",
    "'; DROP TABLE users;--",
    "x" * 4096,
    "emoji-\U0001f600-é",
    "../../etc/passwd",
])
def test_a_hostile_client_msg_id_cannot_break_the_header(hostile):
    """The value is hex by CONSTRUCTION — `cmid_hash` formats `%08x` — so a
    malformed id can only change WHICH eight hex digits come out. Asserted
    rather than argued, because "by construction" is exactly the kind of claim
    that survives a refactor it should not have."""
    from app.agent import llm_trace
    from app.services.openai_agent_service import _TRACE_VALUE_RE

    value = llm_trace.trace_value(hostile, 7)
    assert value is not None
    assert _TRACE_VALUE_RE.match(value), value
    assert re.fullmatch(r"[0-9a-f]{8}\.7", value)
    # Nothing of the input survives into the value.
    assert hostile[:8] not in value or re.fullmatch(r"[0-9a-f]{8}", hostile[:8])


# ── the gate ───────────────────────────────────────────────────────


def test_default_off_for_everyone(monkeypatch):
    from app.agent import llm_trace

    assert llm_trace.trace_enabled(CANARY_USER) is False
    assert llm_trace.trace_for_turn(CANARY_USER, CMID, 0) is None


def test_the_global_flag_enables_everyone(monkeypatch):
    from app.agent import llm_trace

    monkeypatch.setattr(settings, "llm_trace_header", True, raising=False)
    assert llm_trace.trace_enabled(CANARY_USER) is True
    assert llm_trace.trace_enabled(None) is True
    assert llm_trace.trace_for_turn(OTHER_USER, CMID, 1).endswith(".1")


def test_the_canary_list_enables_exactly_its_members(monkeypatch):
    from app.agent import llm_trace

    monkeypatch.setattr(
        settings, "llm_trace_header_canary_user_ids", CANARY_USER,
        raising=False)
    assert llm_trace.trace_enabled(CANARY_USER) is True
    assert llm_trace.trace_enabled(OTHER_USER) is False
    assert llm_trace.trace_enabled(None) is False
    assert llm_trace.trace_for_turn(OTHER_USER, CMID, 0) is None
    assert llm_trace.trace_for_turn(CANARY_USER, CMID, 0) is not None


def test_a_truncated_canary_id_is_an_ERROR_not_a_silent_no_op(monkeypatch, caplog):
    """F5, generalised. A rollout note in this very investigation carried an
    8-CHARACTER PREFIX as the canary value, because that is what log lines
    print. Matching is exact on the full id, so the flag looks set, the canary
    is dark, and nothing is red anywhere."""
    import logging

    from app.agent import llm_trace

    monkeypatch.setattr(
        settings, "llm_trace_header_canary_user_ids", "00000000",
        raising=False)
    with caplog.at_level(logging.ERROR, logger="app.agent.llm_trace"):
        assert llm_trace.trace_enabled("00000000-aaaa-4bbb-8ccc-000000000001") is False
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "a truncated canary id passed without an ERROR"
    text = errors[0].getMessage()
    assert "never match" in text
    # The id itself is never logged — only its length.
    assert "00000000" not in text


def test_a_well_formed_canary_id_logs_no_error(monkeypatch, caplog):
    """The sensitivity half of the test above: the ERROR must not fire for a
    correct list, or it is noise and will be ignored."""
    import logging

    from app.agent import llm_trace

    monkeypatch.setattr(
        settings, "llm_trace_header_canary_user_ids",
        f"{CANARY_USER},{OTHER_USER}", raising=False)
    with caplog.at_level(logging.ERROR, logger="app.agent.llm_trace"):
        assert llm_trace.trace_enabled(CANARY_USER) is True
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


# ── the DESTINATION gate: where the header would actually go ───────
#
# The gate above answers "did someone ask for this header". It does not
# answer "where is this request going", and those are different questions on
# this fleet: `bundle_client.make_openai_client` points the SDK at the Toup
# proxy only on the bundle path and otherwise returns a client on the SDK's
# DEFAULT base url — api.openai.com. `LLM_MODE=manual` + empty `TOUP_TOKEN`
# is the documented boot state of free-tier / pre-activation tenants, so a
# fleet-wide flip of this flag reaches containers whose next request leaves
# our infrastructure entirely. These tests are what makes the notes' sentence
# "x-toup-trace terminates at the platform proxy" true rather than hopeful.


@pytest.mark.parametrize("llm_mode,token,expected", [
    ("bundle", "t", True),
    ("bundle", "", False),     # provisioned, not yet bound — the free-tier boot state
    ("manual", "t", False),    # BYOK with a stale token lying around
    ("manual", "", False),     # the Settings defaults
])
def test_the_destination_gate_is_bundle_clients_own_predicate(
    monkeypatch, llm_mode, token, expected,
):
    """Not a paraphrase of the condition — the condition itself.

    `destination_is_platform_proxy` must agree with the predicate
    `make_openai_client` actually branches on, for every combination, or the
    header can be minted for a request that is not going to our proxy. A
    second copy of a condition is a condition that silently stops agreeing;
    this is the test that would notice.
    """
    from app.agent import llm_trace
    from app.services.bundle_client import _bundle_active

    monkeypatch.setattr(settings, "llm_mode", llm_mode, raising=False)
    monkeypatch.setattr(settings, "toup_token", token, raising=False)

    assert bool(_bundle_active()) is expected
    assert llm_trace.destination_is_platform_proxy() is expected


@pytest.mark.parametrize("llm_mode,token,why", [
    ("manual", "", "BYOK / free-tier boot state"),
    ("manual", "byok", "BYOK with a key"),
    ("bundle", "", "bundle claimed but unbound"),
])
def test_a_container_that_talks_straight_to_the_provider_sends_no_header(
    monkeypatch, llm_mode, token, why,
):
    """Flag fully ON, global not canary — and still no header.

    The payload is a hash and an integer, so the absolute sensitivity is low;
    what is not low is shipping an operator-facing claim that a third party
    does not receive it while a reachable configuration hands it to one. The
    gate, not the paragraph, is what makes that claim hold.
    """
    from app.agent import llm_trace

    monkeypatch.setattr(settings, "llm_trace_header", True, raising=False)
    monkeypatch.setattr(settings, "llm_mode", llm_mode, raising=False)
    monkeypatch.setattr(settings, "toup_token", token, raising=False)

    assert llm_trace.trace_enabled(CANARY_USER) is False, why
    assert llm_trace.trace_for_turn(CANARY_USER, CMID, 0) is None, why

    # …and the sensitivity half, so this cannot pass because the flag never
    # took effect: the same call on the bundle path DOES mint a value.
    monkeypatch.setattr(settings, "llm_mode", "bundle", raising=False)
    monkeypatch.setattr(settings, "toup_token", "t", raising=False)
    assert llm_trace.trace_for_turn(CANARY_USER, CMID, 0) is not None


def test_the_canary_list_also_obeys_the_destination_gate(monkeypatch):
    """The canary is the path a single tenant is lit on, and a canary tenant
    is exactly the kind that might be mid-activation. Both doors, same key."""
    from app.agent import llm_trace

    monkeypatch.setattr(
        settings, "llm_trace_header_canary_user_ids", CANARY_USER,
        raising=False)
    monkeypatch.setattr(settings, "llm_mode", "manual", raising=False)
    assert llm_trace.trace_enabled(CANARY_USER) is False

    monkeypatch.setattr(settings, "llm_mode", "bundle", raising=False)
    assert llm_trace.trace_enabled(CANARY_USER) is True


def test_enabled_but_withheld_is_logged_once_and_names_no_tenant_data(
    monkeypatch, caplog,
):
    """"Enabled and nothing on the wire" must not be silent.

    §7 step 5 of the notes tells an operator to read "channel header present,
    trace header absent" as an intermediary STRIPPING it. On a non-bundle
    container that reading is wrong, and this line is the only thing that
    would say so. Latched to one per process: the alternative is a line per
    LLM iteration for the life of the container.
    """
    import logging

    from app.agent import llm_trace

    monkeypatch.setattr(settings, "llm_trace_header", True, raising=False)
    monkeypatch.setattr(settings, "llm_mode", "manual", raising=False)
    monkeypatch.setattr(settings, "toup_token", "", raising=False)

    with caplog.at_level(logging.WARNING, logger="app.agent.llm_trace"):
        assert llm_trace.trace_enabled(CANARY_USER) is False
        assert llm_trace.trace_enabled(CANARY_USER) is False
        assert llm_trace.trace_enabled(OTHER_USER) is False

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one WARNING, got {len(warnings)}"
    text = warnings[0].getMessage()
    assert "x-toup-trace" in text and "third party" in text
    # Never a tenant identifier and never the token itself.
    assert CANARY_USER not in text and OTHER_USER not in text
    assert CMID not in text


def test_the_bundle_path_logs_no_destination_warning(monkeypatch, caplog):
    """Sensitivity half: a warning that fires on the healthy path is noise,
    and noise on a per-container log is how a real line gets ignored."""
    import logging

    from app.agent import llm_trace

    monkeypatch.setattr(settings, "llm_trace_header", True, raising=False)
    with caplog.at_level(logging.WARNING, logger="app.agent.llm_trace"):
        assert llm_trace.trace_enabled(CANARY_USER) is True
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_the_off_path_never_asks_where_the_request_is_going(monkeypatch):
    """Ordering, which is cost and not correctness — but it is the ordering
    §3 of the notes claims, so it is asserted rather than described.

    With the flag off, `trace_enabled` must answer before it reaches the
    destination check, so the module that is imported on every container adds
    no import of its own on the path every container is on.
    """
    from app.agent import llm_trace

    calls: list = []
    monkeypatch.setattr(
        llm_trace, "destination_is_platform_proxy",
        lambda: calls.append(1) or True)

    assert llm_trace.trace_enabled(CANARY_USER) is False
    assert calls == [], "the destination was resolved on the flag-off path"

    monkeypatch.setattr(settings, "llm_trace_header", True, raising=False)
    assert llm_trace.trace_enabled(CANARY_USER) is True
    assert calls == [1], "the destination was not resolved on the flag-on path"


# ── settings; bridge delivery is a separate, gated release ────────


def test_the_settings_fields_exist_and_default_off():
    """A flag whose field name does not match the env var reaches nothing.
    pydantic-settings maps LLM_TRACE_HEADER -> settings.llm_trace_header."""
    from app.config import Settings

    assert Settings.model_fields["llm_trace_header"].default is False
    assert Settings.model_fields["llm_trace_header_canary_user_ids"].default == ""


# ── the RUNNER path ────────────────────────────────────────────────
#
# Everything above tests the helper and the transport. The defect class this
# investigation keeps rediscovering lives in the CALLER: a guard whose
# precondition something above it destroys, a value computed once outside a
# loop that had to move per iteration. So these drive the real
# `AgentRunner.run()` over a real tool loop and read what the LLM layer was
# actually handed. Harness copied from
# `tests/test_agent_runner_voice_context_isolation.py`.
#
# agent-mode: `run()` writes `users` / `conversations` / `messages`.


class _TwoRoundLLM:
    """Tool call on the first LLM call, text on the second — the minimum
    shape that makes "iteration increments" observable at all."""

    def __init__(self, *, send_trace: bool = True) -> None:
        self.calls: list = []
        self.send_trace = send_trace

    async def create_message_stream(self, **kwargs):
        from app.services.openai_agent_service import StreamEvent

        self.calls.append(kwargs)
        if len(self.calls) == 1:
            yield StreamEvent(
                type="tool_use_start", tool_name="web_search", tool_id="s1")
            yield StreamEvent(
                type="tool_use_end", tool_name="web_search", tool_id="s1",
                tool_input={"query": "q"})
            yield StreamEvent(
                type="message_end", stop_reason="tool_use",
                usage={"input_tokens": 5, "output_tokens": 2},
                llm_trace_sent=(kwargs.get("llm_trace") if self.send_trace else None))
            return
        yield StreamEvent(type="text", text="done")
        yield StreamEvent(
            type="message_end", stop_reason="end_turn",
            usage={"input_tokens": 5, "output_tokens": 2},
            llm_trace_sent=(kwargs.get("llm_trace") if self.send_trace else None))


async def _seed_user() -> str:
    import uuid as _uuid

    from app.db import User, async_session_maker

    user_id = str(_uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=user_id, email=f"trace-{user_id[:8]}@example.test",
                    hashed_password="x" * 60, name="Trace"))
        await db.commit()
    return user_id


async def _drive_two_round_turn(
    monkeypatch, tmp_path, *, client_msg_id, user_id=None,
    model_override="gpt-5.5-mini",
    send_trace=True,
):
    """One real turn with one tool call. Returns (llm stub, user_id).

    `model_override` exists so the Claude branch can be driven too. The runner
    picks `self.anthropic` for a `claude-` model (`agent_runner.py`
    `active_llm = self.anthropic if _is_claude_model(...)`), and that attribute
    is a REAL `AnthropicService()` built in `__init__` — so the stub is bound
    onto it as well, or a claude-model test would reach for a network call
    instead of recording what it was handed.
    """
    import uuid as _uuid

    import app.agent.agent_runner as ar
    from app.agent.tool_executor import ToolExecutor

    async def no_history(self, db, session_id, max_messages=50, client_tz=None):
        return []

    async def fixed_prompt(self, *args, **kwargs):
        return "You are Toup."

    monkeypatch.setattr(ar.AgentRunner, "_load_history", no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", fixed_prompt)
    monkeypatch.setattr(ar, "_spawn_background", lambda coro: coro.close())
    monkeypatch.setattr(ar, "_spawn_bg", lambda coro, **kwargs: coro.close())
    monkeypatch.setattr(ar.settings, "citation_gate_enabled", False, raising=False)

    async def fake_execute(name, payload):
        return "result"

    tools = ToolExecutor(workspace=str(tmp_path))
    monkeypatch.setattr(tools, "execute", fake_execute)

    user_id = user_id or await _seed_user()

    llm = _TwoRoundLLM(send_trace=send_trace)
    runner = ar.AgentRunner(llm_service=llm, tool_executor=tools)
    runner.anthropic = llm
    await runner.run(
        user_message="search please",
        user_id=user_id,
        session_id=str(_uuid.uuid4()),
        channel="app",
        model_override=model_override,
        client_msg_id=client_msg_id,
        save_user_message=False,
        save_assistant_message=False,
        disable_post_processing=True,
    )
    return llm, user_id


@pytest.mark.asyncio
async def test_runner_path_flag_off_passes_no_llm_trace_kwarg(
    monkeypatch, tmp_path,
):
    """Default state of every container today."""
    llm, _ = await _drive_two_round_turn(
        monkeypatch, tmp_path, client_msg_id=CMID)
    assert len(llm.calls) == 2, "the tool loop did not run twice"
    for call in llm.calls:
        assert "llm_trace" not in call


@pytest.mark.asyncio
async def test_runner_path_a_byok_container_sends_nothing_with_the_flag_ON(
    monkeypatch, tmp_path, caplog,
):
    """The fleet-flip case, driven through a real turn rather than the helper.

    `LLM_TRACE_HEADER=true` is a fleet-wide env, so it reaches tenants whose
    container talks straight to the upstream provider. On those the whole
    feature must be inert end to end: no kwarg into the LLM layer, `trace=-`
    on the [PERF] line, and one WARNING saying why — not a header on a request
    leaving our infrastructure.
    """
    import logging

    monkeypatch.setattr(settings, "llm_trace_header", True, raising=False)
    monkeypatch.setattr(settings, "llm_mode", "manual", raising=False)
    monkeypatch.setattr(settings, "toup_token", "", raising=False)

    with caplog.at_level(logging.INFO):
        llm, _ = await _drive_two_round_turn(
            monkeypatch, tmp_path, client_msg_id=CMID)

    assert len(llm.calls) == 2, "the tool loop did not run twice"
    for call in llm.calls:
        assert "llm_trace" not in call

    lines = [r.getMessage() for r in caplog.records
             if "[PERF] llm_total" in r.getMessage()]
    assert len(lines) == 2, f"expected two llm_total lines, got {len(lines)}"
    for line in lines:
        assert "trace=-" in line, line

    withheld = [r.getMessage() for r in caplog.records
                if r.name == "app.agent.llm_trace" and r.levelno >= logging.WARNING]
    assert len(withheld) == 1, withheld


@pytest.mark.asyncio
async def test_runner_path_iteration_increments_across_a_tool_loop(
    monkeypatch, tmp_path,
):
    """The header value must change between the two LLM calls of ONE turn.

    This is the check a helper test cannot make: `_llm_extra_kwargs` is built
    ONCE above the iteration loop, so folding the trace into it — the obvious
    edit — yields the same value on every iteration and a join key that cannot
    tell a turn's first call from its fifth.
    """
    from app.agent import llm_trace

    monkeypatch.setattr(settings, "llm_trace_header", True, raising=False)
    llm, _ = await _drive_two_round_turn(
        monkeypatch, tmp_path, client_msg_id=CMID)

    assert len(llm.calls) == 2, "the tool loop did not run twice"
    values = [c.get("llm_trace") for c in llm.calls]
    head = llm_trace.trace_value(CMID, 0).split(".")[0]
    assert values == [f"{head}.0", f"{head}.1"], values


@pytest.mark.asyncio
async def test_runner_path_the_perf_line_carries_the_same_value_it_sent(
    monkeypatch, tmp_path, caplog,
):
    """Half a join key is not a join key.

    The header alone gives the PLATFORM a value with nothing on the agent side
    to match it against. `[PERF] llm_total` must print the value that rode
    THAT call — same iteration, same string — or the two halves cannot be
    joined at all and the patch buys nothing.
    """
    import logging

    monkeypatch.setattr(settings, "llm_trace_header", True, raising=False)
    with caplog.at_level(logging.INFO, logger="app.agent.agent_runner"):
        llm, _ = await _drive_two_round_turn(
            monkeypatch, tmp_path, client_msg_id=CMID)

    sent = [c.get("llm_trace") for c in llm.calls]
    assert sent == [llm_trace_value(CMID, 0), llm_trace_value(CMID, 1)], sent

    lines = [r.getMessage() for r in caplog.records
             if "[PERF] llm_total" in r.getMessage()]
    assert len(lines) == 2, f"expected two llm_total lines, got {len(lines)}"
    for value, line in zip(sent, lines):
        assert f"trace={value}" in line, line


@pytest.mark.asyncio
async def test_runner_path_logs_no_join_key_when_transport_withholds_it(
    monkeypatch, tmp_path, caplog,
):
    """A stale direct client can reject a minted trace after the runner gate.
    The PERF line must describe the actual wire decision, not the kwarg."""
    import logging

    monkeypatch.setattr(settings, "llm_trace_header", True, raising=False)
    with caplog.at_level(logging.INFO, logger="app.agent.agent_runner"):
        llm, _ = await _drive_two_round_turn(
            monkeypatch, tmp_path, client_msg_id=CMID, send_trace=False)

    assert all(c.get("llm_trace") for c in llm.calls)
    lines = [r.getMessage() for r in caplog.records
             if "[PERF] llm_total" in r.getMessage()]
    assert len(lines) == 2
    assert all("trace=-" in line for line in lines), lines


@pytest.mark.asyncio
async def test_runner_path_the_perf_line_says_dash_when_the_flag_is_off(
    monkeypatch, tmp_path, caplog,
):
    """Sensitivity half — and the default every container is on today. A `-`
    is what makes "this turn had no trace" distinguishable from "the field was
    dropped", which is the distinction a silent-by-design feature needs."""
    import logging

    with caplog.at_level(logging.INFO, logger="app.agent.agent_runner"):
        await _drive_two_round_turn(monkeypatch, tmp_path, client_msg_id=CMID)
    lines = [r.getMessage() for r in caplog.records
             if "[PERF] llm_total" in r.getMessage()]
    assert len(lines) == 2
    assert all("trace=-" in ln for ln in lines), lines


@pytest.mark.asyncio
async def test_runner_path_no_client_msg_id_sends_nothing_even_with_the_flag_on(
    monkeypatch, tmp_path,
):
    """Routines, channel turns and sub-agent runs take this branch."""
    monkeypatch.setattr(settings, "llm_trace_header", True, raising=False)
    llm, _ = await _drive_two_round_turn(
        monkeypatch, tmp_path, client_msg_id=None)
    assert len(llm.calls) == 2
    for call in llm.calls:
        assert "llm_trace" not in call


@pytest.mark.asyncio
async def test_runner_path_canary_gates_on_the_full_user_id(
    monkeypatch, tmp_path,
):
    """Three turns for the SAME user, through the real runner: canary holding
    somebody else's id (dark), canary holding the 8-char PREFIX of this user's
    own id (dark — the operational defect F5 records), canary holding the full
    id (lit). Without the third case the first two would pass on a gate that
    is simply broken."""
    uid = await _seed_user()

    monkeypatch.setattr(
        settings, "llm_trace_header_canary_user_ids", OTHER_USER, raising=False)
    llm, _ = await _drive_two_round_turn(
        monkeypatch, tmp_path, client_msg_id=CMID, user_id=uid)
    assert all("llm_trace" not in c for c in llm.calls), "somebody else's id lit this turn"

    monkeypatch.setattr(
        settings, "llm_trace_header_canary_user_ids", uid[:8], raising=False)
    llm, _ = await _drive_two_round_turn(
        monkeypatch, tmp_path, client_msg_id=CMID, user_id=uid)
    assert all("llm_trace" not in c for c in llm.calls), (
        "an 8-char PREFIX matched — the gate is not exact on the full id"
    )

    monkeypatch.setattr(
        settings, "llm_trace_header_canary_user_ids", uid, raising=False)
    llm, _ = await _drive_two_round_turn(
        monkeypatch, tmp_path, client_msg_id=CMID, user_id=uid)
    assert [c.get("llm_trace") for c in llm.calls] == [
        llm_trace_value(CMID, 0), llm_trace_value(CMID, 1),
    ]


@pytest.mark.asyncio
async def test_runner_path_a_claude_model_is_never_handed_the_kwarg(
    monkeypatch, tmp_path,
):
    """The Claude gate, flag GLOBALLY ON — the one state in which it can fail.

    `AnthropicService.create_message_stream` has no `llm_trace` parameter and
    no `**kwargs` (SOURCE: `app/services/anthropic_service.py`), so handing it
    one is a `TypeError` on EVERY Claude turn while the flag is on. And it
    would not be loud: that exception is swallowed into the retry and
    cross-provider-fallback path, so the symptom is a degraded, slower turn —
    the failure class this whole investigation keeps rediscovering.

    This test exists because the round-1 final review mutated the gate
    (`if not _is_claude_model(active_model)` -> `if True`) and all 39 tests
    stayed green: the gate was the one guard in the patch with no test. The
    non-claude half of the assertion is what keeps it from passing on a gate
    that has simply stopped emitting anything at all.

    `anthropic_enabled` must be forced ON. It defaults to False
    (`app/config.py:500`, SOURCE at 4f0e9fe1), and with it off the runner
    COERCES an explicit `claude-…` override to the OpenAI default before
    `active_llm` is chosen (`agent_runner.py:2903-2911`) — so without this
    line the test drives an OpenAI turn under a Claude-looking name, the
    kwarg is legitimately present, and the test asserts nothing about the
    gate. Found by writing the test and watching it fail for the wrong reason.
    """
    monkeypatch.setattr(settings, "llm_trace_header", True, raising=False)
    monkeypatch.setattr(settings, "anthropic_enabled", True, raising=False)

    llm, _ = await _drive_two_round_turn(
        monkeypatch, tmp_path, client_msg_id=CMID,
        model_override="claude-opus-4-7")
    assert len(llm.calls) == 2, "the tool loop did not run twice"
    assert all("llm_trace" not in c for c in llm.calls), (
        "a Claude turn was handed llm_trace — AnthropicService has no such "
        "parameter, so this is a TypeError on every Claude turn with the "
        "flag on"
    )
    # Sensitivity: the same flag state DOES emit on the non-claude path, so a
    # gate that emits nothing at all cannot pass this test.
    llm, _ = await _drive_two_round_turn(
        monkeypatch, tmp_path, client_msg_id=CMID)
    assert [c.get("llm_trace") for c in llm.calls] == [
        llm_trace_value(CMID, 0), llm_trace_value(CMID, 1),
    ]


def llm_trace_value(cmid, i):
    from app.agent import llm_trace

    return llm_trace.trace_value(cmid, i)
