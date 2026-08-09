"""The first request of a small-window route had no budget guard (G-7).

Two halves, both from the 2026-08-09 audit (docs/GAP_ANALYSIS.md G-7):

1. No output headroom anywhere: the context window bounds input+output
   TOGETHER, but needs_compaction budgeted input against the full window —
   a prompt that "fits" with 0 tokens spare 400s the moment the model
   answers. `reserve_output_tokens` subtracts the caller's max-output.

2. The pre-loop needs_compaction runs against settings.agent_model (1M)
   BEFORE routing; a session/model override or router preference can then
   select a 128k/200k model. A8-3 re-keyed the MID-loop math to the active
   model, which applies from iteration 2 — iteration 1 sailed straight into
   a deterministic 400 rescued only by the reactive overflow handler, one
   full round-trip late. The runner now re-checks (with headroom) against
   the ACTIVE model when its window is smaller, before the first send.
"""

from __future__ import annotations

import inspect
import re

from app.agent.context_manager import (
    COMPACTION_THRESHOLD,
    get_context_window,
    needs_compaction,
)


# ── 1. reserve_output_tokens semantics ───────────────────────────────────

def _messages_of_tokens(n_tokens: int):
    # estimator is chars/4
    return [{"role": "user", "content": "x" * (n_tokens * 4)}]


def test_headroom_flips_the_verdict_exactly_when_it_should():
    """~90k tokens against gpt-4o (128k window): fine with no reserve
    (0.70 < 0.75), over threshold once 16k output headroom is held back
    (90/112 = 0.80). One case, both sides — pins the subtraction, not just
    'bigger input compacts more'."""
    window = get_context_window("gpt-4o")
    assert window == 128_000, "gpt-4o window moved; rebalance this test"

    msgs = _messages_of_tokens(90_000)
    assert needs_compaction("", msgs, "gpt-4o") is False
    assert needs_compaction("", msgs, "gpt-4o", reserve_output_tokens=16_000) is True


def test_reserve_never_produces_a_nonpositive_window():
    assert needs_compaction("", _messages_of_tokens(10), "gpt-4o",
                            reserve_output_tokens=10**9) is True


def test_default_reserve_is_zero_for_existing_callers():
    import inspect as _i
    sig = _i.signature(needs_compaction)
    assert sig.parameters["reserve_output_tokens"].default == 0


# ── 2. the runner guards the first send of a routed model ────────────────

def test_pre_loop_check_reserves_output_headroom():
    from app.agent.agent_runner import AgentRunner

    src = inspect.getsource(AgentRunner._run_inner)
    pre_loop = src.split("A8-3:")[0]
    m = re.search(
        r"needs_compaction\(\s*system_prompt, messages, settings\.agent_model,"
        r"\s*reserve_output_tokens=", pre_loop,
    )
    assert m, "initial needs_compaction no longer reserves output headroom"


def test_routed_model_first_request_is_guarded_before_the_send():
    """The gate must sit AFTER the A8-3 re-key (so it sees the active
    model) and BEFORE the agent loop's first create_message_stream."""
    from app.agent.agent_runner import AgentRunner

    src = inspect.getsource(AgentRunner._run_inner)
    rekey = src.find("_context_window = get_context_window(active_model)")
    first_send = src.find("create_message_stream(")
    assert rekey != -1 and first_send != -1

    between = src[rekey:first_send]
    m = re.search(
        r"needs_compaction\(\s*system_prompt, messages, active_model,"
        r"\s*reserve_output_tokens=", between,
    )
    assert m, (
        "no first-request budget check against the ACTIVE model between "
        "the A8-3 re-key and the first send — a 200k route with a 600k "
        "day history is a deterministic 400 again"
    )
    assert "compact_messages(" in between[m.start():], (
        "the first-request check no longer compacts when it trips"
    )


def test_the_day_budget_still_cannot_reach_the_compaction_threshold():
    """Guard the guard: the day loader bounds history at window×0.60; the
    compaction trigger is (window − headroom)×0.75. If someone raises the
    day ratio past the trigger, every under-budget day turn would compact —
    silently rewriting history bytes and busting the prompt cache."""
    from app.agent.day_context_loader import HISTORY_BUDGET_RATIO
    from app.config import settings

    window = get_context_window(settings.agent_model)
    day_budget = window * HISTORY_BUDGET_RATIO
    trigger = (window - settings.agent_max_tokens) * COMPACTION_THRESHOLD
    assert day_budget < trigger, (
        f"day budget {day_budget:.0f} ≥ compaction trigger {trigger:.0f} — "
        "under-budget day turns would now compact on every send"
    )
