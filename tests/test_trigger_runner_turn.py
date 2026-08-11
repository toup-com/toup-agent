"""G-19b pins — email-trigger turns through AgentRunner, behind a flag.

Gmail-trigger turns historically bypassed AgentRunner entirely: the
`summarize_and_post` action called the bare `call_system_llm` (no day
context, no memory retrieval, no persona, unattributable metering).
`EmailReceivedHandler._do_agent_turn` now routes the turn through
`AgentRunner.run` when `settings.trigger_turns_via_runner` is on AND a
runner ref is wired — with the runner forbidden from persisting
(`save_user_message=False`, `save_assistant_message=False`,
`disable_post_processing=True`) so `write_trigger_message` stays the
single persistence path.

Contract pins:

  1. Flag ships OFF (field default False) — ops flips it via env
     TRIGGER_TURNS_VIA_RUNNER, not a deploy.
  2. Flag OFF → the bare summarize path runs; the runner is NOT called.
  3. Flag ON → runner called with EXACTLY the proven kwarg set
     (autopilot_handler / agent_task_handler lineage), and the runner's
     text reaches the message writer.
  4. ANY runner failure → fall back to summarize (fail-open — a trigger
     must never go silent because the new path broke).
  5. Channel policy: "trigger" is vault-blocked, has CHANNEL_GUIDANCE,
     loses the background-scheduling tools, and is a KNOWN_CHANNEL.

Sqlite-safe: no DB is touched — the writer/broadcaster/llm are fakes
and `db` passes through them untouched.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


# ── Fakes (FakeRunner style, per test_autopilot_engine.py) ──────────


class FakeRunner:
    def __init__(self, replies=("runner text",)):
        self.replies = list(replies)
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            text=reply, tokens_input=1000, tokens_output=500,
            model="gpt-4o-mini", asst_message_id="am-1",
        )


def _trigger(config_json=None):
    class _T:
        id = "trig-1"
        user_id = "user-1"
        name = "Inbox"
        kind = "email_received"
        action = "summarize_and_post"
    _T.config_json = config_json if config_json is not None else {
        "delivery_channels": ["website"],
    }
    return _T()


def _email(event_id="ev-1"):
    from app.agent.triggers.email_received_handler import _FetchedEmail
    return _FetchedEmail(
        event_id=event_id,
        gmail_id="gm-1",
        headers={"From": "Boss <boss@example.com>", "Subject": "Q3 numbers"},
        snippet="Please review the Q3 numbers",
        body="Please review the Q3 numbers before Friday.",
        labels=["INBOX"],
        raw_message={},
    )


def _handler(runner=None, llm_calls=None):
    """Handler with fake writer/broadcaster/llm; returns
    (handler, writes, broadcasts)."""
    from app.agent.triggers.email_received_handler import EmailReceivedHandler

    writes: list = []
    broadcasts: list = []

    async def _writer(db, **kw):
        writes.append(kw)
        return "msg-1", "day-1"

    async def _broadcaster(user_id, **kw):
        broadcasts.append(kw)
        return 1

    async def _llm(**kw):
        if llm_calls is not None:
            llm_calls.append(kw)
        return "summarize text"

    h = EmailReceivedHandler(
        llm_fn=_llm, writer=_writer, broadcaster=_broadcaster,
        agent_runner=runner,
    )
    return h, writes, broadcasts


# ── 1. Flag default pin ─────────────────────────────────────────────


def test_flag_ships_default_off():
    from app.config import settings

    assert settings.model_fields["trigger_turns_via_runner"].default is False, (
        "trigger_turns_via_runner must SHIP off — the runner path rolls "
        "out via env TRIGGER_TURNS_VIA_RUNNER on a canary first, not via "
        "a code default the whole fleet inherits on the next deploy"
    )


# ── 2. Flag OFF → summarize path, runner untouched ──────────────────


async def test_flag_off_uses_summarize_path_runner_not_called(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "trigger_turns_via_runner", False)
    runner = FakeRunner()
    llm_calls: list = []
    h, writes, _ = _handler(runner=runner, llm_calls=llm_calls)

    msg_id, model = await h._do_agent_turn(_trigger(), [_email()], db=None)

    assert runner.calls == [], (
        "flag OFF but AgentRunner.run was called — the default must stay "
        "the bare summarize path until ops flips the flag"
    )
    assert len(llm_calls) == 1, "summarize path (call_system_llm shim) did not run"
    assert msg_id == "msg-1"
    assert len(writes) == 1 and "summarize text" in writes[0]["content"]


# ── 3. Flag ON → runner called with the exact contract kwargs ───────


async def test_flag_on_runner_called_with_exact_kwargs(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "trigger_turns_via_runner", True)
    monkeypatch.setattr(settings, "trigger_turn_credit_budget", 7.5)
    runner = FakeRunner(replies=("the runner wrote this",))
    llm_calls: list = []
    h, writes, broadcasts = _handler(runner=runner, llm_calls=llm_calls)

    msg_id, model = await h._do_agent_turn(_trigger(), [_email("ev-42")], db=None)

    assert len(runner.calls) == 1, "flag ON with a wired ref must call the runner"
    kw = runner.calls[0]
    # The proven kwarg set (autopilot_handler:361-390 lineage). Each of
    # the three False/True flags has a documented incident behind it —
    # the runner must NOT persist; the trigger writer is the single
    # persistence path.
    assert kw["channel"] == "trigger"
    assert kw["save_user_message"] is False
    assert kw["save_assistant_message"] is False
    assert kw["disable_post_processing"] is True
    assert kw["model_override"] == "gpt-4o-mini"  # no cfg.model → default-cheap
    assert kw["credit_budget"] == 7.5
    assert kw["user_id"] == "user-1"
    assert kw["current_job_id"] == "ev-42"  # the claimed BuildJob row
    # The synthesized prompt carries the formatted email.
    assert "Q3 numbers" in kw["user_message"]

    # The summarize LLM must NOT also run — one turn, one bill.
    assert llm_calls == []

    # The runner's text reaches the message writer (inside the Gmail
    # chat-card wrapper) and the broadcast fan-out still fires.
    assert msg_id == "msg-1" and model == "gpt-4o-mini"
    assert len(writes) == 1 and "the runner wrote this" in writes[0]["content"]
    assert len(broadcasts) == 1 and "the runner wrote this" in broadcasts[0]["content"]


async def test_flag_on_honors_trigger_configured_model(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "trigger_turns_via_runner", True)
    runner = FakeRunner()
    h, _, _ = _handler(runner=runner)

    await h._do_agent_turn(
        _trigger(config_json={"model": "gpt-5.5"}), [_email()], db=None,
    )

    assert runner.calls[0]["model_override"] == "gpt-5.5"


# ── 4. Fail-open: runner failure → summarize fallback ───────────────


async def test_runner_failure_falls_back_to_summarize(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "trigger_turns_via_runner", True)
    runner = FakeRunner(replies=(RuntimeError("runner exploded"),))
    llm_calls: list = []
    h, writes, _ = _handler(runner=runner, llm_calls=llm_calls)

    msg_id, model = await h._do_agent_turn(_trigger(), [_email()], db=None)

    assert len(runner.calls) == 1  # it tried
    assert len(llm_calls) == 1, (
        "runner raised but the summarize fallback did not run — a broken "
        "runner path must never make a trigger go silent"
    )
    assert msg_id == "msg-1"
    assert len(writes) == 1 and "summarize text" in writes[0]["content"]


async def test_runner_empty_response_falls_back_to_summarize(monkeypatch):
    """Empty text is a failure too — a blank chat card is silence with
    extra steps."""
    from app.config import settings

    monkeypatch.setattr(settings, "trigger_turns_via_runner", True)
    runner = FakeRunner(replies=("   ",))
    llm_calls: list = []
    h, writes, _ = _handler(runner=runner, llm_calls=llm_calls)

    await h._do_agent_turn(_trigger(), [_email()], db=None)

    assert len(llm_calls) == 1
    assert len(writes) == 1 and "summarize text" in writes[0]["content"]


async def test_flag_on_without_wired_runner_uses_summarize(monkeypatch):
    """Pool/boot ordering safety: flag flipped but no ref wired yet →
    bare summarize, no crash."""
    from app.config import settings

    monkeypatch.setattr(settings, "trigger_turns_via_runner", True)
    llm_calls: list = []
    h, writes, _ = _handler(runner=None, llm_calls=llm_calls)

    msg_id, _ = await h._do_agent_turn(_trigger(), [_email()], db=None)

    assert msg_id == "msg-1" and len(llm_calls) == 1


# ── 5. Channel policy pins ──────────────────────────────────────────


def test_trigger_is_vault_blocked():
    from app.agent.agent_runner import VAULT_TOOL_CHANNEL_BLOCK

    assert "trigger" in VAULT_TOOL_CHANNEL_BLOCK, (
        "a trigger turn cannot render the CredentialConfirmCard — no "
        "user is present to confirm it"
    )


def test_trigger_has_channel_guidance():
    from app.agent.agent_runner import CHANNEL_GUIDANCE

    guidance = CHANNEL_GUIDANCE.get("trigger", "")
    assert guidance, "channel='trigger' fell through to the unknown-channel default"
    # The load-bearing behavioural line for an unattended surface.
    assert "NEVER claim" in guidance


def test_trigger_loses_background_scheduling_tools():
    from app.agent.prompt_profile import disabled_tools_for_channel

    denied = disabled_tools_for_channel("trigger")
    assert {
        "create_job", "update_job", "spawn", "start_mission",
        "save_streaming_credential",
    } <= denied, (
        "an unattended background turn must not be able to schedule MORE "
        "background work (or write credentials nobody confirmed)"
    )


def test_trigger_is_a_known_channel():
    from app.agent.channel_util import KNOWN_CHANNELS

    for ch in ("trigger", "routine", "autopilot", "subagent"):
        assert ch in KNOWN_CHANNELS, (
            f"{ch!r} is an established policy channel (connector_dispatcher "
            "deny sets key on it) but resolve_channel still warns "
            "unknown_value on every turn"
        )


def test_runner_wiring_exists_on_trigger_runner():
    """TriggerRunner.set_agent_runner pushes the ref into registered
    handlers — same pattern as set_mcp_client."""
    from app.agent.triggers.runner import TriggerRunner
    from app.agent.triggers.email_received_handler import get_handler

    tr = TriggerRunner(session_maker=object())  # never started
    sentinel = object()
    tr.set_agent_runner(sentinel)
    assert get_handler()._agent_runner is sentinel


# ── 8. B-3 (closeout run): the shadow + the per-tenant canary ────────
#
# The two written flip prerequisites, built: a W-6-style shadow that
# runs the runner leg BESIDE the served summarize path (tools
# suppressed, output discarded, one fingerprint log line), and a
# per-tenant canary list so one tenant can flip alone. The outputs are
# generative prose, so the shadow's criterion is availability — a
# non-empty body within budget on consecutive real fires — not hash
# equality, which two LLM calls will essentially never achieve.


def test_shadow_and_canary_ship_default_off():
    from app.config import Settings

    assert Settings.model_fields["trigger_turn_shadow"].default is False, (
        "the shadow doubles LLM cost per fire — it ships OFF and is "
        "enabled via bridge env TRIGGER_TURN_SHADOW for the evidence run"
    )
    assert Settings.model_fields["trigger_turns_via_runner_user_ids"].default == "", (
        "the canary list ships empty — nobody is flipped by a merge"
    )


async def test_shadow_serves_legacy_and_runs_runner_leg_suppressed(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "trigger_turns_via_runner", False)
    monkeypatch.setattr(settings, "trigger_turns_via_runner_user_ids", "")
    monkeypatch.setattr(settings, "trigger_turn_shadow", True)
    runner = FakeRunner(replies=("shadow runner body",))
    llm_calls: list = []
    h, writes, broadcasts = _handler(runner=runner, llm_calls=llm_calls)

    msg_id, model = await h._do_agent_turn(_trigger(), [_email()], db=None)

    # Served result is the LEGACY summarize text — the shadow never
    # touches what the user sees.
    assert msg_id == "msg-1"
    assert len(writes) == 1 and "summarize text" in writes[0]["content"]
    assert "shadow runner body" not in writes[0]["content"]
    assert len(broadcasts) == 1 and "shadow runner body" not in broadcasts[0]["content"]
    assert len(llm_calls) == 1, "the summarize leg must serve"

    # The runner leg ran exactly once, tools suppressed, no persistence.
    assert len(runner.calls) == 1
    kw = runner.calls[0]
    assert kw["suppress_tools"] is True, (
        "a DISCARDED turn that can fire tools is a user-visible ghost — "
        "the shadow must suppress the entire tool surface"
    )
    assert kw["save_user_message"] is False
    assert kw["save_assistant_message"] is False
    assert kw["disable_post_processing"] is True


async def test_shadow_failure_never_reaches_the_trigger_result(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "trigger_turns_via_runner", False)
    monkeypatch.setattr(settings, "trigger_turn_shadow", True)
    runner = FakeRunner(replies=(RuntimeError("shadow leg exploded"),))
    h, writes, _ = _handler(runner=runner)

    msg_id, model = await h._do_agent_turn(_trigger(), [_email()], db=None)

    assert msg_id == "msg-1", (
        "a shadow failure cost the user their notification — the served "
        "path must complete before and independent of the shadow"
    )
    assert len(writes) == 1 and "summarize text" in writes[0]["content"]


async def test_shadow_log_carries_fingerprints_not_content(monkeypatch, caplog):
    import logging as _logging

    from app.config import settings

    monkeypatch.setattr(settings, "trigger_turns_via_runner", False)
    monkeypatch.setattr(settings, "trigger_turn_shadow", True)
    runner = FakeRunner(replies=("the runner's secret-ish body",))
    h, _, _ = _handler(runner=runner)

    with caplog.at_level(_logging.INFO, logger="app.agent.triggers.email_received_handler"):
        await h._do_agent_turn(_trigger(), [_email()], db=None)

    shadow_lines = [r.getMessage() for r in caplog.records if "turn_shadow" in r.getMessage()]
    assert shadow_lines, "the shadow ran but logged no turn_shadow line"
    line = shadow_lines[0]
    assert "runner_chars=" in line and "legacy_chars=" in line and "runner_fp=" in line
    assert "secret-ish" not in line and "summarize text" not in line, (
        "the shadow log leaked a body — the bodies carry the user's email "
        "substance; fingerprints only"
    )


async def test_canary_list_serves_runner_for_listed_user_only(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "trigger_turns_via_runner", False)
    monkeypatch.setattr(settings, "trigger_turn_shadow", True)

    # Listed tenant → runner path SERVES (no shadow, no summarize call).
    monkeypatch.setattr(settings, "trigger_turns_via_runner_user_ids", " user-1 , other")
    runner = FakeRunner(replies=("canary runner body",))
    llm_calls: list = []
    h, writes, _ = _handler(runner=runner, llm_calls=llm_calls)
    msg_id, _ = await h._do_agent_turn(_trigger(), [_email()], db=None)
    assert "canary runner body" in writes[0]["content"]
    assert llm_calls == [], "listed tenant must not also bill the summarize leg"
    assert "suppress_tools" not in runner.calls[0], (
        "the SERVED canary turn runs with its real tool surface — "
        "suppression is the shadow's rail, not the product's"
    )

    # Unlisted tenant → legacy serves, shadow compares.
    monkeypatch.setattr(settings, "trigger_turns_via_runner_user_ids", "someone-else")
    runner2 = FakeRunner(replies=("shadow body",))
    llm_calls2: list = []
    h2, writes2, _ = _handler(runner=runner2, llm_calls=llm_calls2)
    await h2._do_agent_turn(_trigger(), [_email()], db=None)
    assert "summarize text" in writes2[0]["content"]
    assert len(llm_calls2) == 1
    assert runner2.calls and runner2.calls[0].get("suppress_tools") is True


def test_fp_is_content_free():
    from app.agent.triggers.email_received_handler import _fp

    chars, digest = _fp("a private email body")
    assert chars == 20 and len(digest) == 8
    assert "private" not in repr((chars, digest))


def test_suppress_tools_and_ephemeral_trigger_session_exist_in_runner():
    """The two runner-side rails the shadow depends on. Source pins in
    the house style (test_voice_context_relay.py): a rename shows up
    here, not as a silently tool-firing shadow in production."""
    import inspect

    from app.agent.agent_runner import AgentRunner

    sig = inspect.signature(AgentRunner._run_inner)
    assert "suppress_tools" in sig.parameters, (
        "_run_inner lost the suppress_tools kwarg — the trigger shadow "
        "would fire real tools on discarded turns"
    )
    src = inspect.getsource(AgentRunner._run_inner)
    assert "suppress_tools" in src and "shadow turn" in src
    assert '== "trigger"' in src and "_ephemeral_session" in src, (
        "trigger turns lost the ephemeral-session rail — one litter "
        "Conversation row per fire (and per shadow fire) returns"
    )


def test_trigger_flip_envs_are_bridge_forwardable():
    """G-19b's own config comment said 'ops flips via env
    TRIGGER_TURNS_VIA_RUNNER' — but the name was never in the bridge's
    `_FEATURE_FLAG_ENVS`, so a bridge env set reached no container. The
    flip machinery must stay forwardable, all three names."""
    import pathlib

    bridge = pathlib.Path(__file__).resolve().parents[2] / "bridge" / "pool_addon.py"
    src = bridge.read_text()
    # The tuple's comments contain ')' — cut at the closing line, not the
    # first close-paren.
    tuple_src = src.split("_FEATURE_FLAG_ENVS = (")[1].split("\n)")[0]
    for name in (
        "TRIGGER_TURNS_VIA_RUNNER",
        "TRIGGER_TURN_SHADOW",
        "TRIGGER_TURNS_VIA_RUNNER_USER_IDS",
    ):
        assert f'"{name}"' in tuple_src, f"{name} not bridge-forwardable"
