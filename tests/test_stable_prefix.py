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
    text AND no time-of-day word into the system prompt (two calls a
    minute apart — or straddling a 5/12/17/22 tod boundary — are
    byte-identical); legacy keeps the old behavior byte-for-byte.
  * ``build_turn_context_message`` — single ephemeral user message with
    the injection-fencing envelope; empty parts → no message.
  * ``settings.stable_prefix_layout`` defaults ON since 2026-08-05 —
    off-by-default turned a missing env var into a silent, permanent
    regression for any tenant provisioned before the flag existed.
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
        # W1.1: nor the time-of-day word — it flipped at 5/12/17/22 local
        # and was the last scheduled intra-day prefix bust.
        assert "afternoon" not in a["about_you"]
        # The exact clock still reaches the model — via the turn context.
        assert "2:05 PM" in a["turn_context"]
        assert "2:06 PM" in b["turn_context"]

    def test_stable_system_lines_survive_tod_boundary(self):
        """W1.1: two calls straddling the 12:00 boundary (morning →
        afternoon) yield byte-identical system-prompt lines; only the
        turn-context clock line differs and carries the tod word."""
        before = render_time_lines(
            datetime(2026, 7, 23, 11, 59), "Europe/Berlin", "morning", stable=True
        )
        after = render_time_lines(
            datetime(2026, 7, 23, 12, 1), "Europe/Berlin", "afternoon", stable=True
        )
        assert before["about_you"] == after["about_you"]
        assert before["runtime"] == after["runtime"]
        assert before["turn_context"] != after["turn_context"]
        # Tone calibration is preserved — the tod word rides the clock line.
        assert "morning" in before["turn_context"]
        assert "afternoon" in after["turn_context"]

    def test_stable_runtime_line_changes_at_day_boundary(self):
        a = render_time_lines(self.T1, "UTC", "afternoon", stable=True)
        c = render_time_lines(self.NEXT_DAY, "UTC", "morning", stable=True)
        assert a["runtime"] != c["runtime"]  # date is allowed to roll daily

    def test_legacy_lines_byte_identical_regression(self):
        """W1.1 guardrail: the legacy (flag-off) path must not move a
        byte — these are the exact pre-W1.1 strings."""
        lines = render_time_lines(self.T1, "Europe/Berlin", "afternoon", stable=False)
        assert lines["about_you"] == (
            "- Local time for them right now: **afternoon** (2:05 PM). "
            "Let it inform tone subtly — late at night, be quieter and "
            "lower-energy; morning, be fresh. Don't announce the time of "
            "day; just feel it."
        )
        assert lines["runtime"] == (
            "- Current date/time: Thursday, July 23, 2026 at 2:05 PM (Europe/Berlin)"
        )
        assert lines["turn_context"] == ""


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
    def test_stable_prefix_layout_defaults_on(self):
        """Flipped 2026-08-05 after soaking on 59 of 61 fleet containers.

        It shipped dark, which was right. What was NOT right is that OFF was
        the default for so long that a missing env var became a silent,
        permanent regression: assigned tenants take env from a per-tenant
        .env written at provision time and never receive later fleet flags,
        so the canary and the founder's own account ran without it for weeks
        at 9.9% cached where the flag delivers 92-94%. A default that only
        works when every deploy path remembers to set it is not a safe
        default.
        """
        assert Settings.model_fields["stable_prefix_layout"].default is True

    def test_stable_prefix_layout_can_still_be_turned_off(self):
        """The kill switch has to survive the default flip, or there is no way
        back without a code change."""
        assert Settings(
            _env_file=None, stable_prefix_layout=False
        ).stable_prefix_layout is False


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


# ── W2.4 — prefix edge-path hardening ─────────────────────────────────

from app.agent.prefix_stability import tools_array_change, tools_wire_hash


class TestToolsWireHash:
    """W2.4(c): the fingerprint that turns tools-array churn from a
    mystery cache miss into a logged [PERF] event."""

    OPENAI_TOOL = {
        "type": "function",
        "function": {"name": "exec", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}},
    }

    def test_identical_arrays_hash_identically(self):
        assert tools_wire_hash(TOOLS) == tools_wire_hash([dict(t) for t in TOOLS])

    def test_rename_reorder_add_remove_all_change_hash(self):
        base = tools_wire_hash(TOOLS)
        renamed = [dict(t) for t in TOOLS]
        renamed[0] = {**renamed[0], "name": "exec2"}
        assert tools_wire_hash(renamed) != base
        assert tools_wire_hash(list(reversed(TOOLS))) != base
        assert tools_wire_hash(TOOLS[:-1]) != base
        assert tools_wire_hash(TOOLS + [{"name": "extra"}]) != base

    def test_schema_growth_changes_hash(self):
        with_schema = [{**TOOLS[0], "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}}}]
        grown = [{**TOOLS[0], "input_schema": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}}]
        assert tools_wire_hash(with_schema) != tools_wire_hash(grown)

    def test_description_only_change_does_not_change_hash(self):
        # Descriptions aren't part of the fingerprint (by design — the
        # hash serializes names + schema lengths only, per the spec).
        a = [{**TOOLS[0], "description": "one"}]
        b = [{**TOOLS[0], "description": "two"}]
        assert tools_wire_hash(a) == tools_wire_hash(b)

    def test_openai_and_anthropic_shapes_both_fingerprint(self):
        anthropic = {"name": "exec", "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}}}
        # Same name, same schema → same fingerprint across wire dialects.
        assert tools_wire_hash([self.OPENAI_TOOL]) == tools_wire_hash([anthropic])


class TestToolsArrayChange:
    def test_first_turn_records_but_never_fires(self):
        seen: dict = {}
        assert tools_array_change(seen, "u1", TOOLS) is None
        assert "u1" in seen

    def test_identical_turn_does_not_fire(self):
        seen: dict = {}
        tools_array_change(seen, "u1", TOOLS)
        assert tools_array_change(seen, "u1", [dict(t) for t in TOOLS]) is None

    def test_changed_turn_fires_with_counts(self):
        seen: dict = {}
        tools_array_change(seen, "u1", TOOLS)
        assert tools_array_change(seen, "u1", TOOLS[:-2]) == (len(TOOLS), len(TOOLS) - 2)

    def test_per_user_isolation(self):
        seen: dict = {}
        tools_array_change(seen, "u1", TOOLS)
        # u2's first sighting must not fire even though u1 has state
        assert tools_array_change(seen, "u2", TOOLS[:-1]) is None
        # and u1's unchanged next turn stays quiet
        assert tools_array_change(seen, "u1", TOOLS) is None

    def test_growth_guard_clears_without_breaking(self):
        seen = {f"u{i}": ("h", 1) for i in range(1025)}
        assert tools_array_change(seen, "fresh", TOOLS) is None
        assert len(seen) == 1  # guard fired, fresh entry recorded

    def test_runner_fingerprints_after_stable_tools_finalized(self):
        # Source-pin: the fingerprint must run on the FINAL wire array
        # (after the stable-tools branch may swap current_tools).
        gate = _SRC.find("_tac = tools_array_change(self._last_tools_hash, user_id, current_tools)")
        stable_swap = _SRC.find("current_tools = _stable_tools")
        assert gate != -1, "runner no longer fingerprints the wire tools array"
        assert stable_swap != -1 and stable_swap < gate
        assert "[PERF] tools_array_changed old_n=%d new_n=%d" in _SRC


class TestMcpRefetchDeterminism:
    """W2.4(b): MCP tool defs serialize ahead of system+history — refresh
    must sort by name and leave held references untouched on a no-op."""

    class _FakeClient:
        def __init__(self, tools):
            self._tools = tools

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def list_tools(self):
            return self._tools

    class _FakeTool:
        def __init__(self, name, description="d", schema=None):
            self.name = name
            self.description = description
            self.inputSchema = schema or {"type": "object"}

    def _cache(self, tools):
        from app.agent.mcp_tools_cache import MCPToolsCache
        return MCPToolsCache(self._FakeClient(tools))

    def test_refresh_sorts_by_name(self):
        import asyncio
        cache = self._cache([self._FakeTool("zeta"), self._FakeTool("alpha"), self._FakeTool("mid")])
        asyncio.run(cache.refresh())
        assert cache.tools == ["alpha", "mid", "zeta"]
        assert [d["name"] for d in cache.tool_defs] == ["alpha", "mid", "zeta"]

    def test_noop_refresh_leaves_held_lists_untouched(self):
        import asyncio

        async def scenario():
            cache = self._cache([self._FakeTool("b"), self._FakeTool("a")])
            await cache.refresh()
            held_names, held_defs = cache.tools, cache.tool_defs
            inner_def_ids = [id(d) for d in cache.tool_defs]
            cache.invalidate()
            await cache.refresh()
            assert cache.tools is held_names and cache.tool_defs is held_defs
            # deep no-op: the inner def dicts were not rebuilt either
            assert [id(d) for d in cache.tool_defs] == inner_def_ids

        asyncio.run(scenario())

    def test_reordered_upstream_same_content_is_noop(self):
        import asyncio

        async def scenario():
            cache = self._cache([self._FakeTool("b"), self._FakeTool("a")])
            await cache.refresh()
            inner_def_ids = [id(d) for d in cache.tool_defs]
            # upstream flips its list order — the sort makes it a no-op
            cache._client = self._FakeClient([self._FakeTool("a"), self._FakeTool("b")])
            cache.invalidate()
            await cache.refresh()
            assert [id(d) for d in cache.tool_defs] == inner_def_ids

        asyncio.run(scenario())

    def test_real_change_swaps_contents_in_place(self):
        import asyncio

        async def scenario():
            cache = self._cache([self._FakeTool("a")])
            await cache.refresh()
            held = cache.tools
            cache._client = self._FakeClient([self._FakeTool("a"), self._FakeTool("c")])
            cache.invalidate()
            await cache.refresh()
            assert cache.tools is held  # in-place mutation preserved the reference
            assert cache.tools == ["a", "c"]

        asyncio.run(scenario())


class TestEdgePathOrderingAndFallbacks:
    """W2.4(a)+(d): source pins for the loader tiebreaker and the two
    loud prefix-lineage fallbacks (behavior is exercised in
    test_day_context.py's sqlite harness; these pin the code shape)."""

    _DCL = (Path(__file__).resolve().parent.parent / "app" / "agent" / "day_context_loader.py").read_text()

    def test_loader_orders_with_id_tiebreaker(self):
        assert ".order_by(Message.created_at.asc(), Message.id.asc())" in self._DCL
        assert ".order_by(Message.created_at.asc())\n" not in self._DCL

    def test_day_ctx_check_failure_warns(self):
        assert "day_chat_context_check_failed" in _SRC

    def test_empty_gated_set_fallback_warns(self):
        assert "stable_tools empty gated set" in _SRC


class TestHeadHashes:
    """Prefix-head attribution (pair-probe follow-up 2026-07-28)."""

    def test_stable_inputs_stable_hashes(self):
        from app.agent.prefix_stability import head_hashes
        a = head_hashes(TOOLS, "sys", [{"role": "user", "content": "hi"}])
        b = head_hashes([dict(t) for t in TOOLS], "sys", [{"role": "user", "content": "hi"}])
        assert a == b
        assert all(len(h) == 8 for h in a)

    def test_each_tier_isolated(self):
        from app.agent.prefix_stability import head_hashes
        base = head_hashes(TOOLS, "sys", [{"role": "user", "content": "hi"}])
        t2 = head_hashes(TOOLS[:-1], "sys", [{"role": "user", "content": "hi"}])
        s2 = head_hashes(TOOLS, "sys2", [{"role": "user", "content": "hi"}])
        h2 = head_hashes(TOOLS, "sys", [{"role": "user", "content": "hi2"}])
        assert t2[0] != base[0] and t2[1] == base[1] and t2[2] == base[2]
        assert s2[1] != base[1] and s2[0] == base[0] and s2[2] == base[2]
        assert h2[2] != base[2] and h2[0] == base[0] and h2[1] == base[1]
        # full-byte sensitivity: a description-only tool change moves the
        # tools hash here (unlike tools_wire_hash, by design)
        d2 = head_hashes([{**TOOLS[0], "description": "different"}] + list(TOOLS[1:]), "sys", [])
        assert d2[0] != base[0]

    def test_empty_inputs_do_not_raise(self):
        from app.agent.prefix_stability import head_hashes
        a, b, c = head_hashes([], "", [])
        assert len(a) == len(b) == len(c) == 8

    def test_runner_logs_prefix_head_before_stream(self):
        assert '"[PERF] prefix_head tools=%s sys=%s hist=%s n_hist=%d"' in _SRC
        # hashed from `history` (pre-tail), not `messages`
        assert "head_hashes(\n                current_tools, system_prompt, history\n            )" in _SRC


# ── W2.3a — channel convergence ────────────────────────────────────────

from app.agent.prefix_stability import channel_banned_names


class TestChannelConverge:
    def test_flag_defaults_off(self):
        assert Settings.model_fields["channel_converge"].default is False

    def test_bridge_ships_the_flag(self):
        bridge = (Path(__file__).resolve().parent.parent.parent / "bridge" / "pool_addon.py").read_text()
        assert '"CHANNEL_CONVERGE"' in bridge

    def _banned(self, channel):
        return channel_banned_names(
            TOOLS, channel, strip_vault_tool_for_channel=strip_vault_tool_for_channel,
        )

    def test_banned_mirrors_strip_rules_exactly(self):
        # The definitional property: banned ∪ kept == all, disjoint — the
        # ban list can never drift from the strips it replaces.
        for channel in ("web", "telegram", "vibecoding", "app", "voice", "mobile"):
            kept = {tool_name(t) for t in _strip(channel)}
            banned = self._banned(channel)
            assert banned | kept == {tool_name(t) for t in TOOLS}
            assert not (banned & kept)

    def test_channel_policies(self):
        assert self._banned("vibecoding") >= {"app_builder__build_app"}
        assert self._banned("app") >= {"app_builder__build_app", "write_file", "exec"}
        # web keeps everything (vault card renders there)
        assert self._banned("web") == frozenset()

    def test_runner_converge_wiring(self):
        # wire array is the full set; policy moves to allowed_tools +
        # executor disabled-set; gated names subtract the banned set.
        #
        # This used to pin the literal line `_stable_tools = list(all_tools)`.
        # That line was the BUG: `all_tools` is `self.tool_defs`, which has
        # already had the per-SURFACE disable set filtered out of it, so a
        # voice turn converged an array that had already lost
        # VOICE_DISABLED_TOOLS (measured 2026-08-05: 49 defs/8,233 tok vs
        # web's 52/9,116 — a separate cache lineage). Pinning the exact line
        # made the correct fix read as a regression, so assert the mechanism
        # instead of one spelling of it. The behavioural contract — voice and
        # web emitting byte-identical arrays — lives in
        # tests/test_channel_converge_voice_array.py, which executes it.
        assert "_stable_tools = list(self.tool_defs_ignoring(" in _SRC, (
            "the converge branch must build from the array with the surface "
            "disable set exempted, or voice never converges"
        )
        assert "_channel_banned = channel_banned_names(" in _SRC
        assert "| frozenset(_surface_disabled)" in _SRC, (
            "surface-disabled names must still be banned via allowed_tools + "
            "the executor — exposing a definition is not permitting a call"
        )
        assert "self.tools.user_disabled_tools = (" in _SRC
        assert ") - _channel_banned" in _SRC
        # ContextVar union keeps the W2.2 race fix intact
        assert "_RUN_DISABLED_TOOLS_CTX.set(\n                        (_RUN_DISABLED_TOOLS_CTX.get() or frozenset())\n                        | _channel_banned" in _SRC

    def test_runner_cache_scope_user_wide(self):
        # both the primary and fallback call sites collapse the scope
        assert '_cache_scope = "all"' in _SRC
        assert '"all" if (_channel_converge and prompt_profile != PromptProfile.SUBAGENT)' in _SRC
        # subagent isolation survives converge
        assert _SRC.index("if prompt_profile == PromptProfile.SUBAGENT:\n                        _cache_scope = session_id") < _SRC.index('_cache_scope = "all"')

    def test_flag_off_keeps_legacy_paths(self):
        # legacy per-channel strip + day-scoped key both still present
        assert "_stable_tools = strip_tools_for_channel(" in _SRC
        assert "_cache_scope = _day_chat_id or session_id" in _SRC
