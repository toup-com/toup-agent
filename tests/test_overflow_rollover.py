"""Regression tests for PR-3 — overflow rollover (finding F-6 / A8-2..A8-7).

Audit context: docs/audits/2026-07-token-efficiency.md. Before this PR:

  * A8-2 — a context_length_exceeded 400 had ZERO handling repo-wide: the
    identical oversized payload was retried twice, then streamed to the
    SMALLER-window fallback model (gpt-4o, 128k) — guaranteed failure.
  * A8-3 — the mid-loop compaction window was keyed to
    ``settings.agent_model`` even when ``active_model`` differed, and the
    canonical Anthropic default was missing from MODEL_CONTEXT_WINDOWS.
  * A8-4 — ``compact_messages`` cut at a raw ``messages[-10:]`` boundary
    and could orphan a tool_result from its tool_use (OpenAI 400); the
    mid-loop continuation marker claimed the summary was "above" while
    being inserted above it.
  * A8-5/A8-6 (flag ``cache_aware_overflow``, default OFF) — compaction
    rewrote messages[0] with a fresh nondeterministic non-persisted
    summary (full prefix-cache bust) and promoted nothing to durable
    memory at drop time.

These tests cover the pure helpers directly and pin the run() wiring
source-grep style (matches test_stable_prefix.py — the runner needs a
full boot to execute).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.context_manager import (
    KEEP_RECENT_MESSAGES,
    compact_messages,
    find_safe_cut,
    get_context_window,
    is_context_overflow_error,
    is_plain_text_message,
)
from app.config import Settings, settings


# ---------------------------------------------------------------------------
# Message fixtures
# ---------------------------------------------------------------------------

def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _tool_use(tool_id: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "calling"},
            {"type": "tool_use", "id": tool_id, "name": "web_search", "input": {}},
        ],
    }


def _tool_result(tool_id: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}],
    }


def _plain_history(n: int) -> list:
    return [_user(f"u{i}") if i % 2 == 0 else _assistant(f"a{i}") for i in range(n)]


# ---------------------------------------------------------------------------
# A8-2 — overflow-error classification
# ---------------------------------------------------------------------------

class _FakeOpenAIError(Exception):
    def __init__(self, message: str, code=None, body=None):
        super().__init__(message)
        self.code = code
        self.body = body


class TestOverflowClassification:
    def test_openai_message_phrasing(self):
        err = _FakeOpenAIError(
            "Error code: 400 - This model's maximum context length is "
            "128000 tokens. However, your messages resulted in 191234 tokens."
        )
        assert is_context_overflow_error(err)

    def test_openai_code_attribute(self):
        err = _FakeOpenAIError("Bad request", code="context_length_exceeded")
        assert is_context_overflow_error(err)

    def test_openai_body_error_code(self):
        err = _FakeOpenAIError(
            "Bad request",
            body={"error": {"code": "context_length_exceeded", "message": "too big"}},
        )
        assert is_context_overflow_error(err)

    def test_anthropic_phrasing(self):
        assert is_context_overflow_error(
            _FakeOpenAIError("400: prompt is too long: 210000 tokens > 200000 maximum")
        )

    def test_negatives(self):
        # The generic retry/failover ladder must keep owning these.
        for msg in (
            "Error code: 429 - rate_limit_exceeded",
            "Error code: 401 - invalid api key",
            "APIConnectionError: connection reset",
            "Error code: 400 - invalid tool schema",
        ):
            assert not is_context_overflow_error(_FakeOpenAIError(msg)), msg


# ---------------------------------------------------------------------------
# A8-4 — tool-pair-safe cut
# ---------------------------------------------------------------------------

class TestFindSafeCut:
    def test_plain_history_keeps_desired_cut(self):
        msgs = _plain_history(20)
        assert find_safe_cut(msgs, 10) == 10

    def test_cut_on_tool_result_walks_forward_past_it(self):
        # ... [8]=tool_use, [9]=tool_result, [10]=plain — a cut at 9 would
        # orphan the tool_result from its parent tool_use.
        msgs = _plain_history(8) + [_tool_use("t1"), _tool_result("t1")] + _plain_history(4)
        cut = find_safe_cut(msgs, 9)
        assert cut == 10
        assert is_plain_text_message(msgs[cut])

    def test_cut_on_tool_use_walks_to_plain_boundary(self):
        # A cut at the tool_use itself is pair-intact but not a plain
        # message; the strict pass walks to the next plain one.
        msgs = _plain_history(8) + [_tool_use("t1"), _tool_result("t1")] + _plain_history(4)
        assert find_safe_cut(msgs, 8) == 10

    def test_all_tool_tail_falls_back_to_pair_intact_start(self):
        # No plain message at or after the cut — fall back to the
        # tool_use start (its result follows: pair intact).
        msgs = _plain_history(6) + [
            _tool_use("t1"), _tool_result("t1"),
            _tool_use("t2"), _tool_result("t2"),
        ]
        assert find_safe_cut(msgs, 7) == 8  # msgs[8] is tool_use t2

    def test_bounds(self):
        msgs = _plain_history(4)
        assert find_safe_cut(msgs, 0) == 0
        assert find_safe_cut(msgs, -3) == 0
        assert find_safe_cut(msgs, 99) == len(msgs)

    def test_is_plain_text_message(self):
        assert is_plain_text_message(_user("hi"))
        assert is_plain_text_message(_assistant("yo"))
        assert not is_plain_text_message(_tool_use("t"))
        assert not is_plain_text_message(_tool_result("t"))
        assert not is_plain_text_message({"role": "system", "content": "x"})


class TestCompactToolPairSafety:
    async def test_compaction_never_orphans_tool_result(self):
        # 16 messages; the raw -10 cut lands exactly on the tool_result
        # at index 6 (pair at 5/6) — pre-fix this produced a kept window
        # starting with an orphaned tool_result (OpenAI 400).
        msgs = _plain_history(5) + [_tool_use("t1"), _tool_result("t1")] + _plain_history(9)
        assert len(msgs) - KEEP_RECENT_MESSAGES == 6
        out = await compact_messages(msgs, "gpt-4o")
        assert out[0]["content"].startswith("[Conversation summary of")
        kept = out[1:]
        # The kept window must not start with (or ever contain) a
        # tool_result whose tool_use was summarized away.
        first = kept[0]
        assert is_plain_text_message(first)
        for m in kept:
            for block in (m.get("content") if isinstance(m.get("content"), list) else []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    ids = [
                        b.get("id")
                        for km in kept
                        for b in (km.get("content") if isinstance(km.get("content"), list) else [])
                        if isinstance(b, dict) and b.get("type") == "tool_use"
                    ]
                    assert block.get("tool_use_id") in ids

    async def test_short_history_unchanged(self):
        msgs = _plain_history(KEEP_RECENT_MESSAGES)
        assert await compact_messages(msgs, "gpt-4o") is msgs


# ---------------------------------------------------------------------------
# A8-5 / A8-6 — cache-aware compaction (flag-gated, default OFF)
# ---------------------------------------------------------------------------

class TestCacheAwareCompaction:
    @pytest.fixture(autouse=True)
    def _no_db(self, monkeypatch):
        # Persistence is exercised via monkeypatched seams — no DB here.
        self._store: dict = {}

        async def _fake_load(conversation_id, span_key):
            return self._store.get((conversation_id, span_key))

        async def _fake_persist(conversation_id, span_key, summary_text):
            self._store[(conversation_id, span_key)] = summary_text

        import app.agent.context_manager as cm
        monkeypatch.setattr(cm, "_load_persisted_summary", _fake_load)
        monkeypatch.setattr(cm, "_persist_summary", _fake_persist)

    async def test_flag_off_legacy_shape(self, monkeypatch):
        monkeypatch.setattr(settings, "cache_aware_overflow", False)
        msgs = _plain_history(16)
        out = await compact_messages(msgs, "gpt-4o")
        # Legacy: summary replaces the front — messages[0] IS rewritten.
        assert out[0]["content"].startswith("[Conversation summary of")
        assert out[1:] == msgs[-KEEP_RECENT_MESSAGES:]

    async def test_flag_on_head_is_byte_untouched(self, monkeypatch):
        monkeypatch.setattr(settings, "cache_aware_overflow", True)
        msgs = _plain_history(16)
        out = await compact_messages(msgs, "gpt-4o", conversation_id="conv-1")
        # F-6: [head..., summary, recent...] — the cached prefix head
        # keeps the SAME objects (byte-identical serialization).
        assert out[0] is msgs[0] and out[1] is msgs[1]
        assert out[2]["content"].startswith("[Conversation summary of")
        assert out[3:] == msgs[-KEEP_RECENT_MESSAGES:]

    async def test_flag_on_summary_is_persisted_and_reused(self, monkeypatch):
        monkeypatch.setattr(settings, "cache_aware_overflow", True)
        msgs = _plain_history(16)
        out1 = await compact_messages(msgs, "gpt-4o", conversation_id="conv-1")
        assert len(self._store) == 1  # persisted under the span hash
        out2 = await compact_messages(msgs, "gpt-4o", conversation_id="conv-1")
        # Idempotent: identical span → identical summary message, no
        # regeneration drift.
        assert out1[2] == out2[2]

    async def test_flag_on_on_drop_receives_the_middle_span(self, monkeypatch):
        monkeypatch.setattr(settings, "cache_aware_overflow", True)
        msgs = _plain_history(16)
        dropped: list = []
        await compact_messages(
            msgs, "gpt-4o", conversation_id="conv-1", on_drop=dropped.extend,
        )
        # A8-6: exactly the summarized middle — not the head, not recent.
        assert dropped == msgs[2:-KEEP_RECENT_MESSAGES]

    async def test_flag_off_on_drop_not_invoked(self, monkeypatch):
        monkeypatch.setattr(settings, "cache_aware_overflow", False)
        called: list = []
        await compact_messages(
            _plain_history(16), "gpt-4o", conversation_id="conv-1",
            on_drop=called.extend,
        )
        assert called == []  # promotion is flag-gated (A8-6)

    async def test_flag_on_head_boundary_is_tool_pair_safe(self, monkeypatch):
        monkeypatch.setattr(settings, "cache_aware_overflow", True)
        # Head boundary (2) lands on the tool_result at index 2 — the head
        # must extend past the pair rather than strand a dangling tool_use.
        msgs = [_user("u0"), _tool_use("t1"), _tool_result("t1")] + _plain_history(14)
        out = await compact_messages(msgs, "gpt-4o", conversation_id="conv-1")
        summary_idx = next(
            i for i, m in enumerate(out)
            if isinstance(m.get("content"), str)
            and m["content"].startswith("[Conversation summary of")
        )
        head = out[:summary_idx]
        assert head == msgs[:3]  # pair pulled into the head intact

    def test_flag_defaults_off(self):
        # Behavior change with regression risk — must ship dark (F-6;
        # flip via CACHE_AWARE_OVERFLOW=true).
        assert Settings.model_fields["cache_aware_overflow"].default is False


# ---------------------------------------------------------------------------
# A8-3 — window registry
# ---------------------------------------------------------------------------

class TestContextWindows:
    def test_canonical_anthropic_default_is_registered(self):
        # model_resolver._CANONICAL_ANTHROPIC_MODEL — the failover path's
        # Anthropic pick was silently budgeted at the 128k default.
        assert get_context_window("claude-opus-4-7") == 200_000

    def test_fallback_model_registered(self):
        # The A8-3 invariant is REGISTRATION (window math must never fall
        # through to DEFAULT_CONTEXT_WINDOW for a model the failover path
        # can actually pick), not any specific size — agent_fallback_model
        # is env-configurable (gpt-4o default, claude-opus-4-6 in some
        # deployments), so a hard-coded size is environment-brittle.
        from app.agent.context_manager import MODEL_CONTEXT_WINDOWS
        assert settings.agent_fallback_model in MODEL_CONTEXT_WINDOWS


# ---------------------------------------------------------------------------
# run() wiring invariants — source-grep style (matches
# test_stable_prefix.py): the runner needs a full boot to execute, so we
# pin the load-bearing lines of the wiring instead.
# ---------------------------------------------------------------------------

_SRC = (Path(__file__).resolve().parent.parent / "app" / "agent" / "agent_runner.py").read_text()
_CTX_SRC = (Path(__file__).resolve().parent.parent / "app" / "agent" / "context_manager.py").read_text()
_RUN_SRC = _SRC.split("async def run(")[1].split("async def _get_or_create_session")[0]


class TestRunnerWiring:
    def test_overflow_branch_exists_before_generic_ladder(self):
        """A8-2: the overflow branch must classify BEFORE the auth/429
        classification so a deterministic 400 never reaches the
        identical-payload retry or the smaller-window fallback."""
        overflow_idx = _RUN_SRC.index("if is_context_overflow_error(e):")
        ladder_idx = _RUN_SRC.index("_should_cross_provider = _is_auth_error or _is_rate_limit")
        assert overflow_idx < ladder_idx

    def test_overflow_compacts_once_then_raises_clear_error(self):
        assert "_overflow_compacted = False" in _RUN_SRC
        assert "Context window exceeded for {active_model}" in _RUN_SRC

    def test_overflow_retry_uses_shrunk_payload_not_identical_bytes(self):
        block = _RUN_SRC[_RUN_SRC.index("if is_context_overflow_error(e):"):][:1600]
        assert "messages = await compact_messages(" in block
        assert "continue" in block

    def test_window_math_rekeyed_to_active_model(self):
        """A8-3: after routing/override/coercion resolves active_model,
        the mid-loop compaction threshold must use ITS window."""
        assert "_context_window = get_context_window(active_model)" in _RUN_SRC

    def test_midloop_compaction_uses_active_model(self):
        # The old settings.agent_model single-line call must be gone.
        assert "compact_messages(messages, settings.agent_model)" not in _SRC

    def test_continuation_marker_inserted_after_summary(self):
        """A8-4b: the marker claims the summary is above — it must be
        inserted AFTER the summary message, never at index 0 (which also
        rewrote messages[0], busting the preserved head)."""
        assert "messages.insert(_sum_idx + 1, {" in _RUN_SRC
        assert "messages.insert(0, {" not in _RUN_SRC

    def test_drop_promotion_is_fire_and_forget(self):
        """A8-6: the dropped span rides the existing background memory
        extractor via a scheduled task — never inline on the turn."""
        assert "def _promote_dropped_span(" in _RUN_SRC
        assert "asyncio.create_task(_promote())" in _RUN_SRC
        assert "on_drop=_promote_dropped_span," in _RUN_SRC

    def test_compaction_sites_pass_conversation_for_persistence(self):
        """A8-5: both compaction call sites hand the Conversation id to
        compact_messages so the flag-on path can persist/reuse the
        span summary."""
        assert _RUN_SRC.count("conversation_id=session_id,") >= 3  # pre-loop, overflow, mid-loop


class TestReviewFixes:
    """Pins for the SHIP_WITH_FIXES review round (pr3-#1, #3, #4)."""

    def test_on_drop_gated_on_new_summary_and_cursor(self):
        """pr3-#1: promotion must not re-fire per turn as the span
        drifts — on_drop only runs when a NEW summary was generated, and
        only over messages beyond the persisted promoted-through cursor."""
        src = _CTX_SRC
        assert "if on_drop is not None and cached_summary is None" in src
        assert "async def _filter_unpromoted(" in src
        assert '"compaction_promoted_through"' in src
        # The cursor advances to the end of the span every compaction.
        assert "_span_hash([span[-1]])" in src

    def test_overflow_noop_compaction_never_retries_identical_bytes(self):
        """pr3-#3: when compaction fails to shrink the payload, retrying
        is a guaranteed second 400 — the runner must raise instead."""
        assert "if _after_tokens < _before_tokens:" in _RUN_SRC
        assert "raising without an identical-bytes retry" in _RUN_SRC

    def test_overflow_error_message_reflects_compaction_attempt(self):
        """pr3-#4: the raise says whether compaction was attempted."""
        assert "overflow surfaced before compaction could be attempted" in _RUN_SRC
