"""R44 I2 — the connect-time prompt-cache warm.

Measured 2026-09-13/14 (76 iteration-1 LLM calls, 26 misses): 7 misses had a
byte-identical head hash to the container's previous call after 4-6 h idle,
and the longest observed HIT gap was 1,616 s. A miss costs ttft p50 5,586 ms
against a hit's 2,611 ms, and the cacheable head (instructions + tools,
40,192 tok in that sample) needs no day chat and no history — so it can be
paid for while the user is still typing.

The one way this feature can lose money is a warm whose bytes differ from the
next turn's: the model still answers, the log still says the warm ran, and
nothing ever reports the waste. So the first test here is byte-equality, and
the rest is the gate that stops a warm firing when it cannot pay off.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from app.agent import cache_warm as cw
from app.config import settings


TOOLS = [
    {"name": "web_search", "description": "search", "input_schema": {}},
    {"name": "memory_search", "description": "recall", "input_schema": {}},
]
SYSTEM = "# Core Identity\nYou are an agent.\n\n# Runtime Context\n- Today: X\n"


class _Ev:
    type = "message_end"
    usage = {"cache_read_input_tokens": 40192, "input_tokens": 40210,
             "output_tokens": 1}


class _RecordingClient:
    """Stands in for the OpenAI SDK client — records the request kwargs.

    Separate from `_RecordingLLM` above: that one replaces the whole service,
    this one sits UNDER it, so the tests that assert on headers and metering
    exercise the real `_create_responses_stream`.
    """

    def __init__(self, sink):
        self.responses = self
        self._sink = sink

    async def create(self, **kwargs):
        self._sink.append(kwargs)

        class _Empty:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        return _Empty()


class _RecordingLLM:
    """Stands in for OpenAIAgentService — records the one call it receives."""

    def __init__(self):
        self.calls: list[dict] = []

    def create_message_stream(self, **kwargs):
        self.calls.append(kwargs)

        async def _gen():
            yield _Ev()

        return _gen()


def _reset():
    cw._HEADS.clear()
    cw._LAST_CALL.clear()
    cw._LAST_WARM.clear()
    cw._WARMS_TODAY.clear()
    cw._IN_FLIGHT.clear()


@pytest.fixture(autouse=True)
def _clean_module_state(monkeypatch):
    _reset()
    # The shipped default is OFF (see config). Every test here is about what
    # happens once it is turned on, so it is turned on explicitly — and one
    # test below pins the default itself.
    monkeypatch.setattr(settings, "llm_cache_warm_on_connect", True)
    monkeypatch.setattr(settings, "llm_cache_warm_idle_s", 240)
    monkeypatch.setattr(settings, "llm_cache_warm_min_gap_s", 300)
    monkeypatch.setattr(settings, "llm_cache_warm_max_per_day", 12)
    yield
    _reset()


def _record(user="u1", *, llm=None, tools=None, system=SYSTEM,
            model="gpt-5.6-terra", channel="app", day=None, tz="UTC",
            is_main_turn=True):
    llm = llm or _RecordingLLM()
    cw.record_head(
        user, llm=llm, system_prompt=system,
        tools=TOOLS if tools is None else tools, model=model,
        prompt_cache_key=None, safety_identifier=user,
        stable_prefix_active=True, channel=channel,
        local_date=day or cw._today_in(tz), tz_name=tz,
        is_main_turn=is_main_turn,
    )
    cw.set_cache_key(user, f"{user}:all")
    return llm


# ── byte-equality: the whole point ────────────────────────────────────

@pytest.mark.asyncio
async def test_the_warm_sends_the_recorded_head_byte_for_byte():
    llm = _record()
    await cw.warm_prefix("u1")
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["system"] == SYSTEM
    assert call["tools"] == TOOLS
    assert call["model"] == "gpt-5.6-terra"
    assert call["prompt_cache_key"] == "u1:all"
    assert call["stable_prefix_active"] is True


@pytest.mark.asyncio
async def test_a_later_turn_mutating_its_tools_list_cannot_edit_the_record():
    """`run()` keeps adding names to the executor's disabled set and rebuilds
    arrays in place; a shared list would silently rewrite the bytes we
    promised to replay."""
    live = list(TOOLS)
    llm = _record(tools=live)
    live.append({"name": "exec", "description": "shell", "input_schema": {}})
    await cw.warm_prefix("u1")
    assert llm.calls[0]["tools"] == TOOLS
    assert len(llm.calls[0]["tools"]) == 2


@pytest.mark.asyncio
async def test_the_warm_request_is_minimal_and_writes_nothing():
    llm = _record()
    await cw.warm_prefix("u1")
    call = llm.calls[0]
    assert call["messages"] == [{"role": "user", "content": "."}]
    assert call["max_tokens"] == 16
    assert call["tool_choice"] == "none"
    # The billing idempotency key is a TURN key; sharing it would make one of
    # the two disappear from metering.
    assert "idempotency_key" not in call


@pytest.mark.asyncio
async def test_the_warm_touches_no_database(monkeypatch):
    import app.db.database as _db

    def _boom(*a, **k):
        raise AssertionError("the warm opened a DB session")

    monkeypatch.setattr(_db, "async_session_maker", _boom)
    llm = _record()
    await cw.warm_prefix("u1")
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_a_warm_is_not_a_turn():
    from app.services import health_signals as hs

    before = hs.get("turns_completed")
    warms = hs.get("llm_cache_warms")
    _record()
    await cw.warm_prefix("u1")
    assert hs.get("turns_completed") == before
    assert hs.get("llm_cache_warms") == warms + 1


@pytest.mark.asyncio
async def test_a_provider_failure_is_swallowed():
    class _Boom:
        def create_message_stream(self, **kwargs):
            async def _gen():
                raise RuntimeError("upstream 500")
                yield  # pragma: no cover

            return _gen()

    _record(llm=_Boom())
    await cw.warm_prefix("u1")  # must not raise


# ── the gate ──────────────────────────────────────────────────────────

def test_gate_lets_an_idle_recorded_user_through():
    _record()
    assert cw.gate("u1", None) == ""


def test_gate_refuses_when_the_flag_is_off(monkeypatch):
    _record()
    monkeypatch.setattr(settings, "llm_cache_warm_on_connect", False)
    assert cw.gate("u1", None) == "flag_off"


def test_gate_refuses_without_a_recorded_head():
    assert cw.gate("nobody", None) == "no_recorded_head"


@pytest.mark.parametrize("channel", sorted(cw.WARM_DENY_CHANNELS))
def test_gate_refuses_the_denied_channels(channel):
    _record()
    assert cw.gate("u1", channel).startswith("denied_channel")


def test_gate_refuses_a_user_mid_conversation():
    _record()
    cw.note_llm_call("u1")
    cw.note_llm_done("u1")
    assert cw.gate("u1", None).startswith("not_idle")


def test_gate_refuses_for_the_whole_length_of_a_turn():
    """R44 F4. `note_llm_call` stamps at call START, so an iteration longer
    than the 240 s idle window used to look idle while it was still
    streaming — and a socket attaching then fired a warm alongside the very
    turn about to read the cache."""
    _record()
    cw.note_llm_call("u1")
    long_after = cw._LAST_CALL["u1"] + settings.llm_cache_warm_idle_s + 60
    assert cw.gate("u1", None, now=long_after) == "turn_in_flight"


def test_a_turn_that_never_reports_done_does_not_disable_warms_forever():
    """An iteration that raises never clears its stamp. Failing safe is
    right; failing safe FOREVER is a feature silently switched off for the
    life of the process."""
    _record()
    cw.note_llm_call("u1")
    stale = cw._IN_FLIGHT["u1"] + cw._IN_FLIGHT_MAX_S + 1
    assert cw.gate("u1", None, now=stale) == ""


def test_gate_allows_once_the_turn_is_done_and_the_idle_window_has_passed():
    _record()
    cw.note_llm_call("u1")
    cw.note_llm_done("u1")
    later = cw._LAST_CALL["u1"] + settings.llm_cache_warm_idle_s + 1
    assert cw.gate("u1", None, now=later) == ""


def test_note_llm_done_restamps_the_idle_clock_from_the_end():
    _record()
    cw.note_llm_call("u1")
    start = cw._LAST_CALL["u1"]
    cw.note_llm_done("u1")
    assert cw._LAST_CALL["u1"] >= start
    assert "u1" not in cw._IN_FLIGHT


def test_gate_enforces_the_min_gap_between_warms():
    _record()
    cw._LAST_WARM["u1"] = 1000.0
    assert cw.gate("u1", None, now=1000.0 + 10).startswith("min_gap")
    assert cw.gate("u1", None, now=1000.0 + settings.llm_cache_warm_min_gap_s + 1) == ""


def test_a_head_from_a_previous_local_date_is_still_warmable():
    """R44 inverted this. The gate used to refuse across local midnight
    because the system prompt carried a `Today's date:` line, which made
    yesterday's head a different head — and that refusal excluded the
    largest miss bucket there is, the first message of a day. The date now
    rides the per-turn <turn_context> tail, so the head outlives midnight
    and the warm may replay it."""
    _record(day=date(2026, 9, 14) - timedelta(days=400))
    assert cw.gate("u1", None) == ""


def test_the_head_the_runner_records_carries_no_date():
    """The reason the gate above can be permissive. If a date ever returns
    to the head this fails here rather than as a mystery warm that never
    lands."""
    from datetime import datetime

    from app.agent.prefix_stability import render_time_lines

    lines = render_time_lines(
        datetime(2026, 7, 23, 14, 5), "Europe/Berlin", "afternoon", stable=True
    )
    assert "2026" not in lines["runtime"]
    assert "2026" not in lines["about_you"]
    assert "Today's date: Thursday, July 23, 2026 (Europe/Berlin)." in lines["turn_context"]


def test_gate_refuses_an_anthropic_head():
    _record(model="claude-opus-4-7")
    assert cw.gate("u1", None) == "not_openai"


@pytest.mark.asyncio
async def test_a_refused_warm_issues_no_request():
    llm = _record()
    cw.note_llm_call("u1")          # not idle
    await cw.warm_prefix("u1")
    assert llm.calls == []


# ── the caller ────────────────────────────────────────────────────────

def test_runner_records_the_same_bytes_it_sends():
    """The record and the wire call must name the SAME variables. A runtime
    test cannot see this: it would pass just as well against a head built a
    second time, which is the drift the design refuses."""
    import app.agent.agent_runner as runner

    src = Path(runner.__file__).read_text(encoding="utf-8")
    assert "_cw.record_head(" in src, "run() no longer records a head"
    i = src.index("_cw.record_head(")
    window = src[i : i + 400]
    assert "system_prompt=system_prompt" in window
    assert "tools=current_tools" in window
    # …and those are the identifiers the real call uses.
    assert "system=system_prompt," in src
    assert "tools=current_tools or None," in src


def test_runner_marks_every_llm_call_for_the_idle_gate():
    import app.agent.agent_runner as runner

    src = Path(runner.__file__).read_text(encoding="utf-8")
    assert "_cw_call.note_llm_call(" in src, (
        "without this the idle gate never fires and a warm can land on a "
        "user who is mid-conversation — pure cost"
    )
    assert "_cw_call.set_cache_key(" in src


# ── the hook ──────────────────────────────────────────────────────────

def test_the_ws_accept_path_schedules_a_warm():
    """One caller, at the accept point, after auth. A runtime test cannot
    reach here without standing the whole agent app up; what matters is that
    the call exists at the right anchor and passes the right channel."""
    import app.api.ws_chat as ws_chat

    src = Path(ws_chat.__file__).read_text(encoding="utf-8")
    assert "from app.agent.cache_warm import schedule_warm as _schedule_warm" in src
    assert "_schedule_warm(user_id)" in src
    anchor = 'logger.info(f"[WS] Authenticated user: {user_id}")'
    assert anchor in src
    after = src[src.index(anchor):]
    call_at = after.index("_schedule_warm(user_id)")
    # Post-auth, and before the handler starts doing real work — a warm
    # scheduled before `user_id` exists would warm nobody.
    assert call_at < after.index("_register_ws_queue("), (
        "the warm must be scheduled at the accept point, not later in the "
        "message loop where it would fire once per turn"
    )


@pytest.mark.asyncio
async def test_schedule_warm_runs_the_warm_when_the_gate_passes():
    llm = _record()
    cw.schedule_warm("u1")
    # One turn of the loop is enough for a task that only awaits the stub.
    import asyncio

    for _ in range(4):
        await asyncio.sleep(0)
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_schedule_warm_starts_nothing_when_the_gate_refuses():
    llm = _record()
    cw.note_llm_call("u1")          # mid-conversation
    cw.schedule_warm("u1")
    import asyncio

    for _ in range(4):
        await asyncio.sleep(0)
    assert llm.calls == []


@pytest.mark.asyncio
async def test_schedule_warm_never_raises_into_the_handler(monkeypatch):
    """Every failure mode the accept path could see. A socket must not fail
    because a cache optimisation did."""
    monkeypatch.setattr(cw, "gate", lambda *a, **k: 1 / 0)
    cw.schedule_warm("u1")

    monkeypatch.undo()
    _record()
    monkeypatch.setattr(cw, "warm_prefix", lambda *a, **k: 1 / 0)
    cw.schedule_warm("u1")

    monkeypatch.undo()
    _record()
    # The bookkeeping itself failing (the task is already created here, so
    # this is the narrow window between scheduling and tracking).
    monkeypatch.setattr(cw, "_PENDING", object())
    cw.schedule_warm("u1")


def test_schedule_warm_outside_an_event_loop_is_a_no_op():
    """A sync caller (a probe, a test, a future non-async accept path) must
    get silence, not a RuntimeError about the missing loop."""
    _record()
    cw.schedule_warm("u1")


# ══════════════════════════════════════════════════════════════════════
# R44 F1 — the warm is NEVER billed to the user
# ══════════════════════════════════════════════════════════════════════
#
# Review of bc7c669b caught this as a blocker and it is the reason the
# feature ships dark. `_create_responses_stream` reported usage
# unconditionally, and on BOTH deployment shapes that reaches the user's
# money:
#
#   bundle/proxy (the fleet norm) — llm_proxy._log_event writes an
#     LLMProxyEvent with operation_type=None, which it treats as
#     user-attributable and charges via credit_service.try_charge(
#     already_incurred=True). A charge downstream of the provider call
#     cannot be denied; it lands.
#   manual/BYOK — credit_reporter POSTs /credits/agent-deduct with
#     idempotency key `oaireq:{completion_id}`, unique per warm, so there is
#     not even a dedupe to save it.
#
# A missed warm is ~10 credits for latency the user never asked for. These
# tests are the ones that would have caught it.

def test_the_warm_declares_itself_a_system_operation():
    assert cw.SYSTEM_OPERATION == "system.cache_warm"
    assert cw.SYSTEM_OPERATION.startswith("system."), (
        "the platform keys its exemption off the `system.` prefix in three "
        "places (llm_proxy._log_event, _get_spend, credits.agent-deduct)"
    )


@pytest.mark.asyncio
async def test_the_warm_tags_the_request_and_files_it_as_system_channel():
    llm = _record()
    await cw.warm_prefix("u1")
    call = llm.calls[0]
    assert call["operation_type"] == "system.cache_warm"
    # F5: filed as `system`, not as the socket's channel — otherwise
    # /cache/daily counts the warm's miss and the turn's hit as two turns.
    assert call["channel"] == "system"


@pytest.mark.asyncio
async def test_a_warm_reaches_report_llm_usage_bg_zero_times(monkeypatch):
    """F6 — the test that was missing. The agent's own metering call is the
    manual/BYOK charge, and it must not happen at all for a warm."""
    import app.services.credit_reporter as reporter
    from app.services.openai_agent_service import OpenAIAgentService

    reported: list = []
    monkeypatch.setattr(reporter, "report_llm_usage_bg",
                        lambda **kw: reported.append(kw))
    monkeypatch.setattr(settings, "user_id", "u1", raising=False)

    svc = OpenAIAgentService.__new__(OpenAIAgentService)
    svc._responses_reasoning = {}
    sink: list = []
    svc.client = _RecordingClient(sink)

    async def _drive(operation_type):
        async for _ in svc._create_responses_stream(
            messages=[{"role": "user", "content": "."}], system=SYSTEM,
            tools=None, model="gpt-5.6-terra", max_tokens=16,
            tool_choice="none", operation_type=operation_type,
        ):
            pass

    await _drive(cw.SYSTEM_OPERATION)
    assert reported == [], (
        "the warm reported usage — on manual/BYOK that is a real charge with "
        "a per-call idempotency key, so nothing dedupes it away"
    )
    # …and the sensitivity half: an ordinary turn still reports.
    await _drive(None)
    assert len(reported) == 1


@pytest.mark.asyncio
async def test_the_proxy_header_is_set_only_for_a_system_operation():
    """The bundle/proxy path's only defence. Without this header the row is
    operation_type=None and llm_proxy charges it."""
    from app.services.openai_agent_service import (
        OPERATION_TYPE_HEADER, OpenAIAgentService,
    )

    async def _headers(operation_type):
        svc = OpenAIAgentService.__new__(OpenAIAgentService)
        svc._responses_reasoning = {}
        sink: list = []
        svc.client = _RecordingClient(sink)
        async for _ in svc._create_responses_stream(
            messages=[{"role": "user", "content": "."}], system=SYSTEM,
            tools=None, model="gpt-5.6-terra", max_tokens=16,
            tool_choice="none", channel="system",
            operation_type=operation_type,
        ):
            pass
        return sink[0].get("extra_headers") or {}

    assert (await _headers(cw.SYSTEM_OPERATION))[OPERATION_TYPE_HEADER] == \
        "system.cache_warm"
    assert OPERATION_TYPE_HEADER not in await _headers(None)
    # A caller cannot smuggle a non-system value through the agent either.
    assert OPERATION_TYPE_HEADER not in await _headers("user.free_please")


class TestProxyHonoursTheMarker:
    """The platform half. `_system_operation_for` is what stands between a
    client-supplied header and the billing decision."""

    WARM_BODY = {
        "model": "gpt-5.6-terra", "max_output_tokens": 16,
        "tool_choice": "none", "input": [{"role": "user", "content": "."}],
    }

    def _call(self, raw, body=None):
        from app.api.llm_proxy import _system_operation_for

        return _system_operation_for(raw, body or self.WARM_BODY)

    def test_the_warm_marker_is_honoured(self):
        assert self._call("system.cache_warm") == "system.cache_warm"

    def test_an_absent_header_is_a_user_request(self):
        assert self._call(None) is None
        assert self._call("") is None

    @pytest.mark.parametrize("raw", [
        "user.chat", "chat", "system", "System.Cache_Warm ",
        "system.free_chat", "system.", "systemcache_warm",
        "system.cache_warm; drop table", "SYSTEM.EVERYTHING",
    ])
    def test_anything_outside_the_allowlist_is_billed_normally(self, raw):
        # Note "System.Cache_Warm " normalises to the allowed value — the
        # comparison is case/space-insensitive on purpose. Everything else
        # that merely LOOKS systemish is refused.
        assert self._call(raw) in (None, "system.cache_warm")
        if raw.strip().lower() != "system.cache_warm":
            assert self._call(raw) is None, (
                f"{raw!r} was accepted — a prefix test alone lets any agent "
                f"key invent an exemption and stop paying for chat"
            )

    @pytest.mark.parametrize("override", [
        {"max_output_tokens": 4096},
        {"max_output_tokens": 0},
        {"tool_choice": "auto"},
        {"input": [{"role": "user", "content": "a"}] * 40},
        {"input": "not a list"},
    ])
    def test_a_real_turn_wearing_the_header_is_still_billed(self, override):
        """Shape gate: a cache warm asks for ~16 tokens, forbids tools and
        sends one input item. A chat turn cannot satisfy that and still be
        useful, so the worst a leaked agent key buys is a free 16-token
        completion — not free chat."""
        body = {**self.WARM_BODY, **override}
        assert self._call("system.cache_warm", body) is None

    def test_every_log_site_in_proxy_responses_carries_the_operation_type(self):
        """The interesting branches are the ERROR ones. A warm that 502s and
        is filed with operation_type=None is a user-attributable row for a
        request the user never made — and it is billed if it carried tokens.
        Same completeness rule the channel header already has."""
        import inspect
        import re

        from app.api import llm_proxy

        src = inspect.getsource(llm_proxy.proxy_responses)
        n_calls = len(re.findall(r"await _log_event\(", src))
        n_tagged = len(re.findall(r"operation_type=req_operation", src))
        assert n_calls > 0
        assert n_tagged == n_calls, (
            f"{n_calls} _log_event calls in proxy_responses but only "
            f"{n_tagged} carry operation_type"
        )
        assert "req_operation = _system_operation_for(" in src

    def test_proxy_chat_does_not_pretend_to_support_the_marker(self):
        """It does not read the header, which is exactly why the warm gate
        refuses any model that would route to the chat wire. If this ever
        becomes false, relax `not_responses_wire` deliberately."""
        import inspect

        from app.api import llm_proxy

        assert "req_operation" not in inspect.getsource(llm_proxy.proxy_chat)

    def test_the_exemption_the_proxy_relies_on_still_exists(self):
        """If `_log_event`'s rule ever stops keying off the prefix, every
        warm becomes a charge again and nothing here would notice."""
        from pathlib import Path

        import app.api.llm_proxy as proxy

        src = Path(proxy.__file__).read_text(encoding="utf-8")
        assert 'operation_type.startswith("system.")' in src
        assert "is_system_op = bool(" in src
        assert "if status == \"ok\" and not is_system_op" in src


# ── F2: the cap ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_daily_cap_stops_the_warms(monkeypatch):
    monkeypatch.setattr(settings, "llm_cache_warm_max_per_day", 2)
    monkeypatch.setattr(settings, "llm_cache_warm_min_gap_s", 0)
    llm = _record()
    for _ in range(5):
        cw._LAST_CALL.pop("u1", None)
        await cw.warm_prefix("u1")
    assert len(llm.calls) == 2, (
        "the per-day cap is the blast-radius bound if any other gate is "
        "wrong — without it a reconnect loop is unbounded spend"
    )
    assert cw.gate("u1", None).startswith("daily_cap")


def test_the_cap_has_a_default():
    from app.config import Settings

    assert Settings.model_fields["llm_cache_warm_max_per_day"].default == 12


# ── F3: a sub-agent head is not the user's head ───────────────────────

def test_a_subagent_turn_records_no_head():
    """A SUBAGENT run carries the sub-agent system prompt and a
    `subagent:{job_id}` cache scope. Warming that heats a prefix the next
    chat turn will never ask for — a paid miss that logs as a success."""
    _record(is_main_turn=False)
    assert cw._HEADS == {}
    assert cw.gate("u1", None) == "no_recorded_head"


def test_the_runner_only_records_on_a_main_turn():
    import app.agent.agent_runner as runner

    src = Path(runner.__file__).read_text(encoding="utf-8")
    i = src.index("_cw.record_head(")
    window = src[i : i + 1200]
    assert "is_main_turn=(prompt_profile == PromptProfile.FULL)" in window


# ── F1 rollout posture ────────────────────────────────────────────────

def test_the_feature_ships_dark():
    from app.config import Settings

    assert Settings.model_fields["llm_cache_warm_on_connect"].default is False, (
        "the no-charge path has a platform half that must be deployed and "
        "observed first; an agent sending warms against an older platform "
        "bills every one of them to a user"
    )


# ── F5: the wire channel ──────────────────────────────────────────────

def test_the_deny_list_reads_the_channel_of_the_last_real_turn():
    """The socket cannot supply one (it learns the channel per message, and
    the extension dials the same endpoint), so a caller-passed literal made
    WARM_DENY_CHANNELS inert."""
    _record(channel="whatsapp")
    assert cw.gate("u1", None).startswith("denied_channel:whatsapp")
    _record(channel="app")
    assert cw.gate("u1", None) == ""


# ── F1: only the wire where the exemption is complete ─────────────────

def test_a_chat_wire_model_is_not_warmed():
    """proxy_chat does not read the operation-type header, so a gpt-4o
    recorded head would be billed on the bundle path."""
    _record(model="gpt-4o")
    assert cw.gate("u1", None).startswith("not_responses_wire")
