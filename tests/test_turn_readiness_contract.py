"""Readiness means A TURN CAN RUN — and an old image still gets in.

Round 46, incident 2 (2026-09-15). Two probes gate onboarding and they
disagreed by construction: `/ready` read `boot_progress` plus the bind,
`test-connection` read `boot_progress` alone over an UNAUTHENTICATED GET and
defaulted the field to `True` when it was absent. Both answered "ready" at
14:47:52 for container pool-82, which was up, bound and booted — and which
could not serve a turn until 14:50:44.

`agent_turn_readiness()` is the ONE predicate both callers now execute. Two
properties decide whether it is safe to ship:

  * it must REFUSE a container that is booted and cannot serve a turn, naming
    which part is missing (a reason token, never a sentence, never an id); and
  * it must ACCEPT an image that reports none of the new fields. The fleet runs
    `1cd801aacb11` for a full rollout cycle, so a probe that requires
    `turn_ready_detail` fails closed against every container and wedges
    onboarding for every new user — the same defect class as the
    `boot.get("ready", True)` it replaces, in the other direction.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. pytest tests/test_turn_readiness_contract.py \
      -q -p no:cacheprovider
"""
from __future__ import annotations

import pytest

# Synthetic. Never a production tenant id in a fixture (IMPL_RULES §5).
USER = "00000000-0000-4000-8000-000000000001"
OTHER = "00000000-0000-4000-8000-000000000002"


def predicate():
    # A plain import, never `importorskip`: this is FIRST-PARTY code. A module
    # that fails to import in the sweep venv (a new third-party import added
    # to agent_setup, a dep missing from the CI install list) must be a RED
    # sweep, not 15 silent skips reading green — the precedent is in
    # .github/workflows/test-backend.yml's own install-step comment.
    from app.api import agent_setup as mod

    assert hasattr(mod, "agent_turn_readiness"), (
        "the shared predicate is gone — two probes with two copies of the rule "
        "is how they disagreed in the first place"
    )
    return mod.agent_turn_readiness


# ── The old image: every new field absent ────────────────────────────────

def test_old_image_boot_ready_is_accepted():
    """THE FLEET-LAG RULE. No `turn_ready_detail`, no `serving` — the deployed
    image. It must answer ready, or onboarding wedges fleet-wide."""
    ready, reason = predicate()(
        {"boot_progress": {"ready": True, "phase": "ready", "percent": 100},
         "is_bound": True, "bound_user_id": USER},
        USER,
    )
    assert ready is True, reason
    assert reason == ""


def test_image_with_no_boot_progress_at_all_is_accepted():
    """A 200 from /agent/health on an image that predates boot_progress IS its
    readiness signal. Refusing it would wedge every pre-rollout tenant."""
    ready, _ = predicate()({"status": "healthy"}, USER)
    assert ready is True


def test_unreadable_body_is_refused_but_named():
    ready, reason = predicate()(None, USER)
    assert ready is False and reason == "unreadable"


# ── Bind is authoritative wherever it is reported ────────────────────────

def test_generic_pool_container_is_not_ready_for_this_user():
    ready, reason = predicate()(
        {"boot_progress": {"ready": True}, "is_bound": False, "bound_user_id": None},
        USER,
    )
    assert ready is False and reason == "not_bound"


def test_a_container_bound_to_someone_else_is_not_ready():
    ready, reason = predicate()(
        {"boot_progress": {"ready": True}, "is_bound": True, "bound_user_id": OTHER},
        USER,
    )
    assert ready is False and reason == "not_bound"


# ── The new image: the parts decide, and they name themselves ────────────

def _detail(**over):
    d = {"runner": True, "db": True, "bound_user": True,
         "llm_wire_warm": True, "channels_settled": True}
    d.update(over)
    return {
        "boot_progress": {"ready": True, "phase": "ready", "percent": 100},
        "is_bound": True, "bound_user_id": USER,
        "turn_ready_detail": d,
        # The producer's own AND, mirrored: the GATING parts only. D1 demoted
        # `llm_wire_warm` and `channels_settled` to diagnostics, and a helper
        # that kept ANDing them would smuggle the old gate back in through
        # `serving` — which the predicate trusts when present.
        "serving": bool(d["runner"] and d["db"] and d["bound_user"]),
    }


def test_a_cold_llm_wire_is_a_DIAGNOSTIC_and_never_a_refusal():
    """The 15 Sep container's LLM wire was rebuilt 111 s late, inside the
    user's first message — which is why the warm exists and why the flag is
    reported. It is NOT a gate (D1): its only producer is a fire-and-forget
    task whose two schedulers both wrap the call in an outer `except` that
    logs and forgets, so one swallowed failure would refuse this container
    for the life of the process while it answers turns perfectly well. Same
    for `channels_settled`, which is false for the whole of every WhatsApp
    adapter restart. `not_ready_because` may never name either."""
    ready, reason = predicate()(_detail(llm_wire_warm=False), USER)
    assert ready is True, reason
    assert reason == ""

    ready, reason = predicate()(_detail(channels_settled=False), USER)
    assert ready is True, reason

    ready, reason = predicate()(
        _detail(llm_wire_warm=False, channels_settled=False), USER)
    assert ready is True, reason

    # …and a real gate still refuses while they are false, so this is not
    # "nothing refuses any more".
    ready, reason = predicate()(
        _detail(db=False, llm_wire_warm=False, channels_settled=False), USER)
    assert ready is False and reason == "db"


def test_a_dead_database_is_named(monkeypatch=None):
    """pool-38 answered /agent/health 200 four times during an eleven-minute
    total database outage."""
    ready, reason = predicate()(_detail(db=False), USER)
    assert ready is False and reason == "db"


def test_no_runner_is_named_first():
    """`runner` outranks everything: it is the only part that means "this
    process cannot answer at all"."""
    ready, reason = predicate()(_detail(runner=False, db=False), USER)
    assert ready is False and reason == "runner"


def test_all_parts_true_is_ready():
    ready, reason = predicate()(_detail(), USER)
    assert ready is True and reason == ""


def test_a_part_this_build_does_not_report_is_not_a_refusal():
    """A field present as `None` is UNKNOWN. A mid-rollout image that reports
    the object but not every key must not be refused for the gap."""
    body = _detail()
    body["turn_ready_detail"]["channels_settled"] = None
    body["serving"] = True
    ready, reason = predicate()(body, USER)
    assert ready is True, reason


def test_serving_false_is_honoured_even_when_the_parts_look_fine():
    """The server's own AND wins when it disagrees, so the two sides can never
    drift into answering differently for the same body."""
    body = _detail()
    body["serving"] = False
    ready, reason = predicate()(body, USER)
    assert ready is False and reason == "serving"


def test_boot_not_finished_is_named_by_phase():
    body = _detail()
    body["boot_progress"] = {"ready": False, "phase": "embeddings", "percent": 25}
    ready, reason = predicate()(body, USER)
    assert ready is False and reason == "boot_embeddings"


# ── The reason token is telemetry, not prose ─────────────────────────────

@pytest.mark.parametrize("body", [
    _detail(llm_wire_warm=False),
    _detail(db=False),
    {"boot_progress": {"ready": True}, "is_bound": False},
])
def test_reason_carries_no_identity_and_no_prose(body):
    _, reason = predicate()(body, USER)
    assert " " not in reason, f"a reason token is a token, not a sentence: {reason!r}"
    assert USER not in reason and USER[:8] not in reason
    assert reason == reason.lower()


# ── The two halves must be the SAME rule, not two copies of one ──────────
#
# Every test above hands the predicate a dict the test author wrote. Nothing
# in them notices that the PRODUCER emits different keys: `_READINESS_PARTS`
# skips a key it does not find (`if val is None: continue` — an unrecognised
# part is "unknown", i.e. always ready), so renaming `bound_user` on either
# side leaves both files green while every container reports ready.


def _producer_detail() -> dict:
    """The real `_turn_readiness()`, executed. It is fail-open by design (each
    part is probed inside its own try), so it runs anywhere."""
    import asyncio

    import agent_main

    return asyncio.run(agent_main._turn_readiness())["turn_ready_detail"]


def test_every_part_the_consumer_checks_is_a_part_the_producer_emits():
    from app.api.agent_setup import _READINESS_PARTS

    emitted = set(_producer_detail())
    missing = sorted(set(_READINESS_PARTS) - emitted)
    assert not missing, (
        f"the readiness gate checks {missing}, which /agent/health never "
        "emits — every one of them is skipped as 'unknown' and the gate is inert"
    )


def test_the_producers_own_output_is_accepted_by_the_consumer():
    """End to end, with no hand-written dict anywhere: what the container
    actually says, fed to the rule that actually judges it."""
    from app.api.agent_setup import _READINESS_PARTS

    detail = _producer_detail()
    body = {
        "boot_progress": {"ready": True, "phase": "ready", "percent": 100},
        "is_bound": True,
        "bound_user_id": USER,
        "turn_ready_detail": {**detail, "runner": True, "db": True,
                              "bound_user": True, "llm_wire_warm": True,
                              "channels_settled": True},
    }
    ready, reason = predicate()(body, USER)
    assert ready is True, reason
    # And the refusal side, one part at a time, on the producer's own keys.
    for part in _READINESS_PARTS:
        body2 = {**body, "turn_ready_detail": {**body["turn_ready_detail"], part: False}}
        ready2, reason2 = predicate()(body2, USER)
        assert ready2 is False and reason2 == part, (part, reason2)


def test_the_producer_serves_on_the_three_gating_parts_only():
    """D1, executed on the producer. `serving` is `runner ∧ db ∧ bound_user`.
    Re-adding `llm_wire_warm` or `channels_settled` to that AND puts a single
    silent producer back in charge of every onboarding — the consumer trusts
    `serving` when it is present, so the demotion has to hold on BOTH sides."""
    import asyncio
    import inspect

    import agent_main

    src = inspect.getsource(agent_main._turn_readiness)
    line = [ln for ln in src.splitlines() if ln.strip().startswith("serving =")]
    assert len(line) == 1, line
    assert "llm_wire_warm" not in line[0], line[0]
    assert "channels_settled" not in line[0], line[0]

    # …and executed: a cold wire and an unsettled sidecar, with the three
    # gating parts green, still serves.
    out = asyncio.run(agent_main._turn_readiness())
    detail = out["turn_ready_detail"]
    expected = bool(detail["runner"] and detail["db"] and detail["bound_user"])
    assert out["serving"] == expected, out
    # Both diagnostics are still REPORTED — demoted, not deleted.
    assert "llm_wire_warm" in detail and "channels_settled" in detail
