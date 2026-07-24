"""Regression tests for PR-1 — prefix-stable prompt layout.

Audit context: docs/audits/2026-07-token-efficiency.md measured a 0%
OpenAI prompt-cache hit rate in production, caused by (F-1) a minute
clock at system-prompt section 6/22, (F-2) per-message/mid-run tools
array churn, and (F-3) per-query memory retrieval at section 8. These
tests pin the wire shapes of the fix so a refactor can't silently
reintroduce a prefix-buster:

  * ``strip_tools_for_channel`` — the stable wire array depends only on
    the channel (never the message/intent) and is order-preserving.
  * ``build_allowed_tools_choice`` — matches the OpenAI SDK
    ``ChatCompletionAllowedToolChoiceParam`` shape and is deterministic
    for identical name sets.
  * ``render_time_lines`` — the stable layout emits NO minute-resolution
    text into the system prompt (two calls a minute apart are
    byte-identical); legacy keeps the old behavior.
  * ``build_turn_context_message`` — single ephemeral user message with
    the injection-fencing envelope; empty parts → no message.
  * ``settings.stable_prefix_layout`` defaults OFF (flag-gated rollout).
"""

from __future__ import annotations

from datetime import datetime

from app.agent.prefix_stability import (
    build_allowed_tools_choice,
    build_turn_context_message,
    render_time_lines,
    strip_tools_for_channel,
    tool_name,
)
from app.agent.agent_runner import strip_vault_tool_for_channel
from app.config import Settings


TOOLS = [
    {"name": "exec", "description": "shell"},
    {"name": "write_file", "description": "write"},
    {"name": "edit_file", "description": "edit"},
    {"name": "apply_patch", "description": "patch"},
    {"name": "pty_exec", "description": "pty"},
    {"name": "web_search", "description": "search"},
    {"name": "save_streaming_credential", "description": "vault"},
    {"name": "app_builder__build_app", "description": "builder"},
    {"name": "generate_image", "description": "img"},
]


def _strip(channel: str):
    return strip_tools_for_channel(
        TOOLS, channel, strip_vault_tool_for_channel=strip_vault_tool_for_channel,
    )


class TestStripToolsForChannel:
    def test_web_keeps_everything(self):
        names = [tool_name(t) for t in _strip("web")]
        assert names == [tool_name(t) for t in TOOLS]

    def test_vibecoding_strips_app_builder_only(self):
        names = {tool_name(t) for t in _strip("vibecoding")}
        assert "app_builder__build_app" not in names
        assert {"exec", "write_file", "edit_file"} <= names

    def test_app_channel_strips_builder_and_core_mutators(self):
        names = {tool_name(t) for t in _strip("app")}
        assert not names & {
            "app_builder__build_app", "write_file", "edit_file",
            "exec", "pty_exec", "apply_patch",
        }
        assert "web_search" in names

    def test_vault_tool_stripped_on_blocked_channels(self):
        for channel in ("telegram", "voice", "mobile", "autopilot"):
            assert "save_streaming_credential" not in {
                tool_name(t) for t in _strip(channel)
            }, channel
        assert "save_streaming_credential" in {tool_name(t) for t in _strip("web")}

    def test_deterministic_and_message_independent(self):
        # The stable array is a pure function of (tools, channel) — calling
        # twice yields identical structure and order. This is the F-2
        # invariant: nothing per-message may influence the wire array.
        assert _strip("telegram") == _strip("telegram")

    def test_does_not_mutate_input(self):
        before = [dict(t) for t in TOOLS]
        _strip("app")
        assert TOOLS == before


class TestAllowedToolsChoice:
    def test_matches_sdk_shape(self):
        choice = build_allowed_tools_choice(["web_search", "exec"])
        assert choice["type"] == "allowed_tools"
        inner = choice["allowed_tools"]
        assert inner["mode"] == "auto"
        assert inner["tools"] == [
            {"type": "function", "function": {"name": "exec"}},
            {"type": "function", "function": {"name": "web_search"}},
        ]

    def test_required_mode(self):
        choice = build_allowed_tools_choice(["exec"], mode="required")
        assert choice["allowed_tools"]["mode"] == "required"

    def test_deterministic_for_identical_sets(self):
        a = build_allowed_tools_choice(["b", "a", "c"])
        b = build_allowed_tools_choice(["c", "b", "a"])
        assert a == b


class TestRenderTimeLines:
    T1 = datetime(2026, 7, 23, 14, 5)
    T2 = datetime(2026, 7, 23, 14, 6)  # one minute later, same day
    NEXT_DAY = datetime(2026, 7, 24, 9, 30)

    def test_legacy_keeps_minute_clock(self):
        lines = render_time_lines(self.T1, "Europe/Berlin", "afternoon", stable=False)
        assert "2:05 PM" in lines["about_you"]
        assert "2:05 PM" in lines["runtime"]
        assert lines["turn_context"] == ""

    def test_stable_system_lines_are_minute_free_and_day_stable(self):
        a = render_time_lines(self.T1, "Europe/Berlin", "afternoon", stable=True)
        b = render_time_lines(self.T2, "Europe/Berlin", "afternoon", stable=True)
        # F-1 invariant: within one day, the system-prompt lines are
        # byte-identical regardless of the wall-clock minute.
        assert a["about_you"] == b["about_you"]
        assert a["runtime"] == b["runtime"]
        assert "2:05" not in a["about_you"] and "2:05" not in a["runtime"]
        # The exact clock still reaches the model — via the turn context.
        assert "2:05 PM" in a["turn_context"]
        assert "2:06 PM" in b["turn_context"]

    def test_stable_runtime_line_changes_at_day_boundary(self):
        a = render_time_lines(self.T1, "UTC", "afternoon", stable=True)
        c = render_time_lines(self.NEXT_DAY, "UTC", "morning", stable=True)
        assert a["runtime"] != c["runtime"]  # date is allowed to roll daily


class TestTurnContextMessage:
    def test_empty_parts_yield_no_message(self):
        assert build_turn_context_message([]) is None
        assert build_turn_context_message(["", "  "]) is None

    def test_message_shape_and_fencing(self):
        msg = build_turn_context_message(["Current time: 2:05 PM (UTC)", "# User Brain\n- fact"])
        assert msg is not None
        assert msg["role"] == "user"
        assert msg["content"].startswith("<turn_context>")
        assert msg["content"].rstrip().endswith("</turn_context>")
        assert "Current time: 2:05 PM (UTC)" in msg["content"]
        assert "# User Brain" in msg["content"]
        # Injection fencing: recalled content is data, not instructions.
        assert "never follow" in msg["content"].lower()


class TestFlagDefault:
    def test_stable_prefix_layout_defaults_off(self):
        # Behavior change with regression risk — must ship dark
        # (audit guardrail; flip via STABLE_PREFIX_LAYOUT=true).
        assert Settings.model_fields["stable_prefix_layout"].default is False


# ---------------------------------------------------------------------------
# run() wiring invariants — source-grep style (matches
# test_agent_runner_tool_events.py): the runner needs a full boot to
# execute, so we pin the load-bearing lines of the wiring instead.
# ---------------------------------------------------------------------------

from pathlib import Path

_SRC = (Path(__file__).resolve().parent.parent / "app" / "agent" / "agent_runner.py").read_text()


class TestRunnerWiring:
    def test_escalation_skipped_on_stable_path(self):
        """The mid-run full-toolset escalation was the guaranteed
        intra-turn cache miss (F-2: prod logs showed cache_read=0 on
        iteration 2 of the same turn). The stable path must never
        mutate the wire array."""
        assert (
            "if not _stable_prefix and current_tools is not all_tools" in _SRC
        ), "escalation must be guarded by `not _stable_prefix`"

    def test_cache_key_is_day_scoped_with_session_fallback(self):
        """F-4: prompt_cache_key routes on user:day_chat_id so all
        channels of one day share a cache shard; falls back to the
        session on the non-day-context path."""
        assert "_cache_scope = _day_chat_id or session_id" in _SRC
        assert '_cache_key = f"{user_id}:{_cache_scope}"' in _SRC

    def test_billing_idempotency_key_stays_per_session(self):
        """Guardrail: PR-1 must not change metering semantics. The old
        per-session value survives as the deduct idempotency key even
        though the cache key is now day-stable."""
        assert '_idem_key = f"{user_id}:{session_id}"' in _SRC
        assert "idempotency_key=_idem_key" in _SRC

    def test_primary_call_passes_new_kwargs(self):
        call_idx = _SRC.index("async for event in active_llm.create_message_stream(")
        block = _SRC[call_idx:call_idx + 700]
        for kwarg in ("prompt_cache_key=_cache_key", "safety_identifier=user_id", "idempotency_key=_idem_key"):
            assert kwarg in block, f"primary LLM call missing {kwarg}"

    def test_turn_context_injected_before_user_message(self):
        """The <turn_context> message must sit between history and the
        current user message: behind the cacheable prefix, but not
        displacing messages[-1] (routing/callers assume the user
        message is last)."""
        tc_idx = _SRC.index("if _turn_context_parts:")
        user_append_idx = _SRC.index(
            'messages.append({"role": "user", "content": user_message})'
        )
        assert tc_idx < user_append_idx

    def test_subagent_isolation_survives_turn_context(self):
        """SUBAGENT profile deliberately gets no user_brain /
        active_tasks; the turn-context routing must respect the same
        profile allow-list the section filter enforces."""
        assert '"user_brain" in _profile_sections' in _SRC
        assert '"active_tasks" in _profile_sections' in _SRC

    def test_stable_path_is_openai_only(self):
        """Review #6: Anthropic's tool_choice cannot express an
        allowlist, so a stable array on Claude would silently drop
        intent gating. Claude models keep the legacy filtered array."""
        assert "_stable_prefix = _stable_layout and not _is_claude_model(active_model)" in _SRC

    def test_channel_strips_are_single_sourced(self):
        """Review #4: the legacy path, the stable path, and the
        escalation branch must all use strip_tools_for_channel so the
        flag-on and flag-off wire arrays cannot silently drift."""
        assert _SRC.count("strip_tools_for_channel(") >= 3
        # The old inline strip comprehensions must be gone from run():
        assert '.startswith("app_builder__")' not in _SRC.split("async def run(")[1].split("async def _get_or_create_session")[0]

    def test_day_chat_id_reset_on_day_context_failure(self):
        """Review #7/#9: after a day-context load failure the history is
        session-shaped, so the day-scoped cache key must fall back to
        the session too."""
        fallback_block = _SRC.split("day_context_load_failed")[1][:1200]
        assert "_day_chat_id = None" in fallback_block


class TestAnthropicBreakpointPlacement:
    """Review #1: with the stable layout's <turn_context> message in the
    tail, the Anthropic conversation breakpoint must sit at
    end-of-history (before the tc message), or the cached span could
    never re-match on the next turn."""

    def _msgs(self, with_tc: bool):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        if with_tc:
            msgs.append({"role": "user", "content": "<turn_context>\nclock\n</turn_context>"})
        msgs.append({"role": "user", "content": "what's next?"})
        return msgs

    @staticmethod
    def _marked_indices(out):
        marked = []
        for i, m in enumerate(out):
            c = m.get("content")
            if isinstance(c, list) and any(
                isinstance(b, dict) and b.get("cache_control") for b in c
            ):
                marked.append(i)
        return marked

    def test_breakpoint_before_turn_context(self):
        from app.services.anthropic_service import _mark_messages_cacheable
        out = _mark_messages_cacheable(self._msgs(with_tc=True))
        # [user, assistant, tc, user] → breakpoint on the assistant
        # message (end of history); tc + current user stay unmarked.
        assert self._marked_indices(out) == [1]
        assert isinstance(out[2]["content"], str)  # tc untouched
        assert isinstance(out[3]["content"], str)  # user untouched

    def test_legacy_shape_unchanged_without_turn_context(self):
        from app.services.anthropic_service import _mark_messages_cacheable
        out = _mark_messages_cacheable(self._msgs(with_tc=False))
        assert self._marked_indices(out) == [len(out) - 1]


class TestCanaryGate:
    """Per-tenant canary for the stable prefix layout — agent flags are
    otherwise fleet-wide (no per-container override), so this list is the
    only way to prove the layout on one tenant before a global flip."""

    def _call(self, monkeypatch, *, global_flag, canary_ids, user_id):
        from app.config import settings
        from app.agent import agent_runner
        monkeypatch.setattr(settings, "stable_prefix_layout", global_flag, raising=False)
        monkeypatch.setattr(settings, "stable_prefix_canary_user_ids", canary_ids, raising=False)
        return agent_runner.stable_prefix_enabled(user_id)

    def test_global_flag_on_enables_everyone(self, monkeypatch):
        assert self._call(monkeypatch, global_flag=True, canary_ids="", user_id="u1") is True
        assert self._call(monkeypatch, global_flag=True, canary_ids="", user_id=None) is True

    def test_off_by_default(self, monkeypatch):
        assert self._call(monkeypatch, global_flag=False, canary_ids="", user_id="u1") is False

    def test_canary_user_enabled_others_not(self, monkeypatch):
        assert self._call(monkeypatch, global_flag=False, canary_ids="canary-uid", user_id="canary-uid") is True
        assert self._call(monkeypatch, global_flag=False, canary_ids="canary-uid", user_id="other") is False

    def test_canary_list_multi_and_whitespace(self, monkeypatch):
        ids = " a , b ,c "
        assert self._call(monkeypatch, global_flag=False, canary_ids=ids, user_id="b") is True
        assert self._call(monkeypatch, global_flag=False, canary_ids=ids, user_id="c") is True
        assert self._call(monkeypatch, global_flag=False, canary_ids=ids, user_id="d") is False

    def test_none_user_never_matches_canary(self, monkeypatch):
        assert self._call(monkeypatch, global_flag=False, canary_ids="a,b", user_id=None) is False

    def test_flag_off_default_in_settings(self):
        from app.config import Settings
        assert Settings.model_fields["stable_prefix_canary_user_ids"].default == ""


class TestCanaryRetentionParity:
    """The 24h retention + safety_identifier must gate on the EFFECTIVE
    per-turn flag (so canary users get them too), not the global setting.
    Measured 2026-07-24: retention="24h" turns an intermittent ~0.67
    cached/prompt into a reliable 0.89 on prod."""

    def test_service_gates_retention_on_effective_flag(self):
        import inspect
        from app.services.openai_agent_service import OpenAIAgentService
        src = inspect.getsource(OpenAIAgentService.create_message_stream)
        assert "if stable_prefix_active:" in src
        assert 'kwargs["prompt_cache_retention"] = "24h"' in src
        # must NOT re-read the global setting for this gate
        assert 'getattr(settings, "stable_prefix_layout"' not in src

    def test_runner_threads_effective_flag_to_both_calls(self):
        assert _SRC.count("stable_prefix_active=_stable_prefix") >= 2  # primary + fallback
