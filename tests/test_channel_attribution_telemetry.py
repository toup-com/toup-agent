"""Channel attribution on llm_proxy_events (alembic 082).

WHY THIS EXISTS
---------------
Prompt caching is prefix-exact and the wire TOOLS ARRAY heads the prefix, so a
channel that strips a tool starts a separate provider cache lineage and
re-bills the whole request — measured 2026-08-08: `app` diverges from web at
byte 10 of a 46,517-byte tools array, and there is no partial credit (a cold
voice turn read 0 cached while a simultaneous web control read 18,070).

"Which surface is burning uncached tokens" was therefore a first-order cost
question that could not be answered. The value existed on the agent and was
dropped at the agent→proxy boundary; it is not recoverable from anything else
on the row (`operation_type` is NULL for every user-facing chat call *by
design* — that is what makes a call count toward the user's cap).

The tests below concentrate on the failure modes, because this code runs in
the metering write that happens AFTER the user's LLM call already succeeded.
Anything that raises there converts a served request into a 500. The sanitizer
must therefore be total: every input maps to a storable value or to None.
"""

from __future__ import annotations

import pytest

from app.api.llm_proxy import CHANNEL_HEADER, _CHANNEL_MAX, _sanitize_channel


# ── The sanitizer is total ───────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("web", "web"),
    ("voice", "voice"),
    ("vibecoding", "vibecoding"),
    ("  WEB  ", "web"),          # trimmed + lowercased
    ("Telegram", "telegram"),
    ("extension", "extension"),
])
def test_normal_channels_pass_through(raw, expected):
    assert _sanitize_channel(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", 123, [], {}, object()])
def test_absent_or_non_string_becomes_none_and_never_raises(raw):
    """`None` is the honest value for 'not reported'. Every non-agent caller
    (embeddings, images, internal_llm system ops) hits this path on every
    call, so it must be the cheap, silent one."""
    assert _sanitize_channel(raw) is None


def test_an_over_long_value_is_truncated_not_rejected():
    """The column is VARCHAR(20). A longer value would raise on INSERT —
    inside the metering write, after the user's call already succeeded — so it
    is truncated here. Rejecting would trade a telemetry gap for a 500."""
    out = _sanitize_channel("w" * 500)
    assert out is not None and len(out) == _CHANNEL_MAX


def test_punctuation_and_control_characters_are_stripped():
    """A header is caller-supplied. Strip it to the characters a channel name
    can contain so nothing exotic reaches the column or a log line."""
    assert _sanitize_channel("we b\n\t;drop") == "webdrop"
    assert _sanitize_channel("../../etc/passwd") == "etcpasswd"
    assert _sanitize_channel("'; DELETE FROM users --") == "deletefromusers--"


def test_a_value_that_sanitizes_to_nothing_becomes_none():
    """Not an empty string — the column means 'not reported', and '' would be
    a third state that every aggregate query would have to know about."""
    assert _sanitize_channel("!!!") is None
    assert _sanitize_channel("...") is None


def test_an_unknown_channel_is_recorded_not_dropped():
    """Deliberately NOT validated against channel_util.KNOWN_CHANNELS.

    An allowlist would silently drop a newly-added surface: its traffic would
    appear to vanish rather than appear under a new label, which is the worse
    failure and the drift this repo has been bitten by before. It would also
    put `app.agent` on the platform image's import path.
    """
    assert _sanitize_channel("some_new_surface") == "some_new_surface"


# ── The column exists and is optional ────────────────────────────────────

def test_the_model_carries_a_nullable_channel_column():
    from app.db.models.platform import LLMProxyEvent

    col = LLMProxyEvent.__table__.columns["channel"]
    assert col.nullable is True, "existing rows predate the header; NULL is honest"
    assert col.type.length == _CHANNEL_MAX, (
        "sanitizer ceiling and column width must agree, or a truncated value "
        "still overflows"
    )


def test_the_query_this_column_exists_for_is_indexed():
    from app.db.models.platform import LLMProxyEvent

    names = {i.name for i in LLMProxyEvent.__table__.indexes}
    assert "ix_llm_proxy_created_channel" in names


def test_an_event_can_be_built_without_a_channel():
    """Every non-agent caller constructs one of these with no channel at all."""
    from app.db.models.platform import LLMProxyEvent

    ev = LLMProxyEvent(
        id="x", user_id="u", provider="openai", model="m", endpoint="chat",
        input_tokens=1, output_tokens=1, cost_cents=0, latency_ms=1,
    )
    assert getattr(ev, "channel", None) is None


# ── The plumbing is connected end to end ─────────────────────────────────

def test_both_proxy_handlers_read_the_header_and_pass_it_to_every_log_site():
    """The interesting calls are the ERROR branches — a channel recorded only
    on success would show every surface as healthy. Assert the header is read
    once per handler and that no _log_event inside them is left unattributed.
    """
    import inspect
    import re

    from app.api import llm_proxy

    for fn in (llm_proxy.proxy_chat, llm_proxy.proxy_responses):
        src = inspect.getsource(fn)
        assert "req_channel = _sanitize_channel(request.headers.get(CHANNEL_HEADER))" in src, (
            f"{fn.__name__} does not read the channel header"
        )
        n_calls = len(re.findall(r"await _log_event\(", src))
        n_tagged = len(re.findall(r"channel=req_channel", src))
        assert n_calls > 0, f"{fn.__name__} logs nothing — test is vacuous"
        assert n_tagged == n_calls, (
            f"{fn.__name__}: {n_calls} _log_event calls but only {n_tagged} "
            f"carry channel — an unattributed branch is a hole exactly where "
            f"the interesting calls are"
        )


def test_the_agent_sends_the_channel_as_a_header_on_both_wires():
    """A header, not a body field: the proxy forwards the body verbatim to
    OpenAI, where an unknown key is a 400 on the whole turn."""
    import inspect

    from app.services.openai_agent_service import OpenAIAgentService

    for meth in (OpenAIAgentService.create_message_stream,
                 OpenAIAgentService._create_responses_stream):
        assert "channel" in inspect.signature(meth).parameters, meth.__name__

    src = inspect.getsource(OpenAIAgentService)
    assert src.count('kwargs["extra_headers"] = {"X-Toup-Channel"') == 2, (
        "both the chat and responses wires must send it"
    )
    assert '"channel"' not in src.replace('"channel": ', ''), (
        "channel must not be added to the request BODY"
    )


def test_the_anthropic_service_accepts_channel_for_signature_parity():
    """agent_runner calls both services with the same kwargs; a missing
    parameter here is a TypeError on the cross-provider fallback path — which
    only fires when the primary provider is already failing."""
    import inspect

    from app.services.anthropic_service import AnthropicService

    assert "channel" in inspect.signature(
        AnthropicService.create_message_stream
    ).parameters


def test_agent_runner_passes_channel_on_both_the_primary_and_fallback_calls():
    """Anchored on the LLM call sites, not on a count.

    `channel=channel,` appears six times in `run()` — session creation, day
    context, the system-prompt builder — so counting occurrences measures
    mostly noise and stays green when an LLM call loses it. Each site is
    identified by the `stable_prefix_active` kwarg that only the two LLM calls
    carry, and the assertion is that channel is passed in the SAME call.
    """
    import inspect
    import re

    from app.agent.agent_runner import AgentRunner

    src = inspect.getsource(AgentRunner.run)
    sites = re.findall(
        r"stable_prefix_active=_stable_prefix,\s*\n\s*(\w+)=", src
    )
    assert len(sites) == 2, (
        f"expected the primary and fallback LLM calls, found {len(sites)}"
    )
    assert all(kw == "channel" for kw in sites), (
        f"an LLM call does not pass channel (next kwargs: {sites}) — the "
        "fallback path matters most: a provider outage would otherwise "
        "silently blank the attribution for the calls you most want to see"
    )
