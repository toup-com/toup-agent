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
