"""Regression test for TKT-LAT-019 — context trim for trivial queries.

Drops the whole memory-files block when
the message is a greeting, acknowledgment, or simple time/date
question. Motivated by 24 548-token "what time is it?" turn taking
11.6 s — that's portrait + memories + entities + tasks all loaded
for a question the agent can answer from its system prompt alone.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_qc():
    """Load query_classifier without triggering app.config validation."""
    p = Path(__file__).resolve().parents[1] / "app" / "services" / "query_classifier.py"
    spec = importlib.util.spec_from_file_location("qc_under_test", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── is_trivial_query: positive cases ───────────────────────────────────────

def test_trivial_one_word_acknowledgments():
    qc = _load_qc()
    for w in ["ok", "okay", "k", "yep", "thanks", "thx", "cool", "nice"]:
        assert qc.is_trivial_query(w), f"{w!r} should be trivial"


def test_trivial_greetings():
    qc = _load_qc()
    for g in ["hi", "hello", "hey", "good morning", "good night", "gm", "gn"]:
        assert qc.is_trivial_query(g), f"{g!r} should be trivial"


def test_trivial_time_date_questions():
    qc = _load_qc()
    for q in [
        "what time is it",
        "what time is it?",
        "what's the time",
        "what's the date",
        "what day is it",
        "what's today",
        "What Time Is It?",  # case-insensitive
        "tell me the time",
    ]:
        assert qc.is_trivial_query(q), f"{q!r} should be trivial"


def test_trivial_handles_punctuation_and_whitespace():
    qc = _load_qc()
    assert qc.is_trivial_query("  thanks!  ")
    assert qc.is_trivial_query("OK.")
    assert qc.is_trivial_query("yes!!")


# ── is_trivial_query: negative cases (must remain non-trivial) ────────────

def test_nontrivial_factual_questions_about_user():
    qc = _load_qc()
    for q in [
        "what did i do last week",
        "where do i live",
        "tell me about my project",
        "what's my favorite restaurant",
        "remind me to call mom",
        "what's the weather in paris",
    ]:
        assert not qc.is_trivial_query(q), f"{q!r} should NOT be trivial"


def test_nontrivial_long_messages_with_trivial_words():
    """'thanks for the help with the auth migration last week' contains
    'thanks' but is not trivial — the rest of the sentence carries
    real intent. Word-count gate must catch this."""
    qc = _load_qc()
    long = "thanks for the help with the auth migration last week — could you double check the rollback?"
    assert not qc.is_trivial_query(long)


def test_nontrivial_empty_and_none():
    qc = _load_qc()
    assert not qc.is_trivial_query("")
    assert not qc.is_trivial_query(None)
    assert not qc.is_trivial_query("   ")


# ── Source-level invariants on the gate wiring ────────────────────────────

def test_config_flag_defaults_to_on():
    src = (Path(__file__).resolve().parents[1] / "app" / "config.py").read_text()
    assert "context_trim_for_trivial_queries: bool = True" in src


def test_agent_runner_gates_the_memory_block_on_skip_flag():
    src = (Path(__file__).resolve().parents[1] / "app" / "agent" / "agent_runner.py").read_text()
    # The flag is computed once at the top of _build_system_prompt.
    assert "_skip_deep_context = bool(" in src
    # And the memory block is gated on it. Memory v3 (§3.1) replaced the
    # retrieval fan-out with ONE memory-files load, and retired the
    # active_tasks block with the `active_task` rows it rendered — so
    # there is one gated block here now, not two.
    assert 'logger.info("[PERF] memory_files_skipped reason=trivial_query")' in src
    assert "active_tasks skipped" not in src


def test_agent_runner_emits_perf_log_with_depth():
    """Every turn must record context_depth so we can measure trim-rate
    in production logs without sampling individual messages."""
    src = (Path(__file__).resolve().parents[1] / "app" / "agent" / "agent_runner.py").read_text()
    assert "[PERF] context_depth=%s user_message_len=%d" in src


def test_blocking_fallback_when_flag_off():
    """When CONTEXT_TRIM_FOR_TRIVIAL_QUERIES is disabled, _skip_deep_context
    must always be False — the gate falls through to the original
    non-trimmed path. Verified via the `and` boolean: getattr first,
    then is_trivial_query."""
    src = (Path(__file__).resolve().parents[1] / "app" / "agent" / "agent_runner.py").read_text()
    assert 'getattr(settings, "context_trim_for_trivial_queries", True)' in src
