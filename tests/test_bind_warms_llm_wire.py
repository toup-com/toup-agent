"""The LLM wire is warmed at bind, off the loop — and the warm can never fail a bind.

Round 46, A1. `/admin/bind` refreshes the key cache and deliberately defers the
client rebuild to "the next chat call". On a freshly claimed container that
call is the user's FIRST MESSAGE: on pool-82 the refresh logged at 14:47:46.904
and `[OPENAI] Client rebuilt` at 14:49:37.705 — 111 s later, inside the turn —
and that line is the last one before a 9.47 s whole-process freeze.

What actually runs there is the SDK's once-per-process lazy work. Measured on
2026-09-15 with the pinned `openai==2.53.0` / `pydantic==2.13.5`, fresh process,
three runs, unthrottled M-series:

    client.responses (the @cached_property that imports the resource)  134-141 ms
    construct_type(ResponseStreamEvent, <response.completed>)          202-214 ms
      …the same call, warm                                            0.42-0.54 ms

The ~400x ratio on the second is pydantic building the core schema for the
Response/output-item union, not the payload. 343 ms total, unthrottled — which
does NOT account for 9.47 s and is not claimed to. This test pins the two
properties that make the warm safe to run at bind at all.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. pytest tests/test_bind_warms_llm_wire.py \
      -q -p no:cacheprovider
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]


def svc():
    # A plain import, never `importorskip`: FIRST-PARTY code that fails to
    # import must be a RED sweep, not silent skips reading green.
    from app.services import openai_agent_service as mod

    return mod


def test_warm_is_a_method_on_the_service():
    """Not a caller poking at the client's internals: the service owns the fact
    that its wire is warm, and `wire_warm()` is what /agent/health reports."""
    m = svc()
    assert hasattr(m.OpenAIAgentService, "warm")
    assert callable(m.wire_warm) and callable(m.mark_wire_warm)


def test_warm_pays_the_first_use_cost_once():
    m = svc()
    s = m.get_openai_agent_service()
    t0 = time.perf_counter()
    first = s.warm()
    cold_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    s.warm()
    warm_ms = (time.perf_counter() - t0) * 1000.0
    assert first >= 0.0
    # The point of the whole exercise: the second call is free. A generous
    # bound — this runs on shared CI hardware — but the ratio is ~1000x locally.
    # SECONDARY SIGNAL ONLY: a `warm()` whose body was deleted satisfies it
    # too (both timings ≈ 0), which is exactly what the two post-conditions
    # below exist to catch.
    assert warm_ms < max(20.0, cold_ms / 2), f"cold={cold_ms:.1f}ms warm={warm_ms:.1f}ms"
    # `mark_wire_warm()` lives in a `finally`, so this is true even for a warm
    # that did nothing. It pins the flag's producer, never the work.
    assert m.wire_warm() is True

    # ── The two halves actually ran. ──────────────────────────────────────
    # (a) the lazy resource import: `client.responses` is a @cached_property
    #     whose body is `from .resources.responses import AsyncResponses`
    #     (134-141 ms on first touch). Nothing else in this process imports it.
    import sys

    assert "openai.resources.responses" in sys.modules, (
        "warm() did not touch client.responses — the 134 ms import is still "
        "waiting for the user's first turn"
    )
    # (b) the pydantic core schema for the Responses stream union. Cold it is
    #     202-214 ms; warm, 0.42-0.54 ms — a ~400x ratio, which is what makes
    #     this a DETERMINISTIC assertion rather than a timing one. This is the
    #     larger half of the cost the whole A1 mitigation exists to pay, and
    #     deleting its call from warm() left all 19 tests in this file green.
    from openai._models import construct_type
    from openai.types.responses import ResponseStreamEvent

    t0 = time.perf_counter()
    construct_type(type_=ResponseStreamEvent, value=m._WARM_RESPONSE_COMPLETED)
    bare_ms = (time.perf_counter() - t0) * 1000.0
    assert bare_ms < 20.0, (
        f"a bare construct_type took {bare_ms:.1f}ms after warm() — the core "
        "schema was NOT built by the warm (cold is ~200ms, warm ~0.5ms)"
    )


def test_warm_never_raises_even_when_the_client_cannot_be_built():
    """A bind must never fail because a warm did — that is the whole reason it
    is scheduled rather than awaited. Two ways this can go wrong on a real
    container: the key cache raises, or there is no client to touch."""
    m = svc()

    boom = m.OpenAIAgentService.__new__(m.OpenAIAgentService)
    boom._ensure_client = lambda: (_ for _ in ()).throw(RuntimeError("no keys"))
    assert boom.warm() == 0.0

    none = m.OpenAIAgentService.__new__(m.OpenAIAgentService)
    none._ensure_client = lambda: None
    none.client = None
    assert none.warm() == 0.0


def test_the_canned_payload_is_a_real_response_completed():
    """If the canned event stops parsing, the warm silently stops warming the
    expensive half — and nothing else in the repo would notice."""
    m = svc()
    from openai._models import construct_type
    from openai.types.responses import ResponseStreamEvent

    ev = construct_type(type_=ResponseStreamEvent, value=m._WARM_RESPONSE_COMPLETED)
    assert type(ev).__name__ == "ResponseCompletedEvent", type(ev).__name__


def test_anthropic_has_the_mirror():
    from app.services import anthropic_service as a

    assert hasattr(a.AnthropicService, "warm")


# ── The bind wiring: OFF THE LOOP, and never awaited ─────────────────────

def _bind_warm_block() -> str:
    src = (BACKEND / "app/api/admin_pool.py").read_text()
    i = src.index("2c-1.")
    return src[i:i + 2600]


def test_the_bind_warm_runs_on_a_worker_thread():
    """Both halves hold the import lock / GIL. Running them on the loop is the
    defect, not the fix — a bind that warmed inline would freeze the container
    exactly the way the first turn did."""
    block = _bind_warm_block()
    assert "to_thread" in block, block[:400]
    assert re.search(r"to_thread\(get_openai_agent_service\(\)\.warm\)", block), block[:600]


def test_the_bind_does_not_wait_for_the_warm():
    block = _bind_warm_block()
    assert "create_task(_warm_llm_wire())" in block, block[:600]
    # A bare `await _warm_llm_wire()` would put the bind behind it.
    assert not re.search(r"^\s*await _warm_llm_wire\(\)", block, re.M)


def test_the_warm_failure_is_non_fatal():
    block = _bind_warm_block()
    assert "non-fatal" in block
    assert block.count("except Exception") >= 2, "both the schedule and the body must be guarded"


def test_it_emits_the_measurement_line():
    """This line is the only evidence the round will have that the warm ran,
    and it carries milliseconds — never a key, a model, or a user."""
    block = _bind_warm_block()
    assert "[PERF] llm_wire_warm_ms=" in block
    assert "api_key" not in block and "user_id" not in block


def test_the_boot_path_warms_an_already_bound_container():
    """A container that boots ALREADY bound (a restart, a blue-green promote)
    never sees /admin/bind, and would otherwise pay the cost inside its owner's
    first message."""
    src = (BACKEND / "agent_main.py").read_text()
    i = src.index("_boot_warm_llm_wire")
    block = src[i:i + 1400]
    assert "is_bound()" in block, "an UNBOUND lobby container must not warm a wire it has no key for"
    assert "to_thread" in block
    assert "src=boot" in block


def test_health_reports_the_flag():
    src = (BACKEND / "agent_main.py").read_text()
    assert '"llm_wire_warm": llm_wire_warm' in src
    assert "from app.services.openai_agent_service import wire_warm" in src


def test_warm_is_scheduled_after_the_key_refresh_not_before():
    """A warm that ran BEFORE `_llm_keys.refresh()` would build the client on
    the lobby's empty key and mark the wire warm — the exact 401 class the
    refresh exists to prevent, with a flag on top saying everything is fine."""
    src = (BACKEND / "app/api/admin_pool.py").read_text()
    assert src.index("_llm_keys.refresh()") < src.index("2c-1.")


def test_the_warm_does_not_run_on_every_turn():
    """`_ensure_client` is the hot path. The warm must be a separate method,
    not something the stream creation calls."""
    src = (BACKEND / "app/services/openai_agent_service.py").read_text()
    i = src.index("async def create_message_stream")
    assert ".warm()" not in src[i:i + 4000]


def test_measurement_is_recorded_where_it_can_be_found():
    """The round's only evidence for A1 is a measurement, and a measurement
    nobody can locate later is not evidence. It lives in the warm's docstring,
    beside the code it justifies."""
    doc = svc().OpenAIAgentService.warm.__doc__ or ""
    # Self-checking against the PIN rather than against a literal: nothing
    # updates a docstring when `requirements.agent.txt` is bumped, so a
    # hard-coded "openai==2.53.0" here would keep passing while certifying a
    # false statement about which SDK the measurement came from. Now a bump
    # turns the stale measurement red, which is the point of recording it.
    req = (BACKEND / "requirements.agent.txt").read_text()
    m = re.search(r"^openai==([0-9][0-9A-Za-z.\-]*)", req, re.M)
    assert m, "openai is no longer pinned in requirements.agent.txt"
    pinned = f"openai=={m.group(1)}"
    assert pinned in doc, (
        f"the warm's measurement names a different SDK than the pin ({pinned}) — "
        "re-measure and update the docstring, or the number is fiction"
    )
    assert "ms" in doc and "9.47" in doc, (
        "the docstring must also say what the number does NOT explain"
    )


def test_asyncio_is_available_for_the_bind_block():
    """Sanity: the block imports asyncio locally, so a module-level rename
    cannot silently disable it."""
    block = _bind_warm_block()
    assert "import asyncio as _asyncio" in block
    assert asyncio is not None


# ── Readiness must be able to RECOVER from a failed warm ─────────────────
#
# `llm_wire_warm` is a hard term of `serving`, `serving` is what both
# onboarding probes now gate on, and `warm()` is its ONE producer. Marked only
# on the success path, a single swallowed failure — a private import the SDK
# moves (`openai._models.construct_type`, `openai.types.responses`), a client
# that could not be built, a key cache that raised — made `serving` false for
# the LIFE OF THE PROCESS, for a container that answers turns perfectly well.
# The fact the readiness term needs is "the once-per-process work has been
# attempted here", not "it succeeded".


def _fresh_flag(m):
    m._wire_warm = False
    m._wire_warm_at = None


def test_a_failed_warm_still_marks_the_wire_attempted():
    m = svc()
    _fresh_flag(m)
    assert m.wire_warm() is False

    boom = m.OpenAIAgentService.__new__(m.OpenAIAgentService)
    boom._ensure_client = lambda: (_ for _ in ()).throw(RuntimeError("no keys"))
    assert boom.warm() == 0.0
    assert m.wire_warm() is True, (
        "one swallowed warm failure refuses readiness forever"
    )


def test_a_clientless_warm_still_marks_the_wire_attempted():
    m = svc()
    _fresh_flag(m)
    none = m.OpenAIAgentService.__new__(m.OpenAIAgentService)
    none._ensure_client = lambda: None
    none.client = None
    assert none.warm() == 0.0
    assert m.wire_warm() is True


def test_the_warm_tasks_are_referenced_so_they_cannot_be_collected():
    """`asyncio.create_task` keeps only a WEAK reference; the docs say such a
    task may be garbage-collected mid-execution. Both schedulers of the ONLY
    producer of `llm_wire_warm` were fire-and-forget."""
    import re

    bind = (BACKEND / "app/api/admin_pool.py").read_text()
    i = bind.index("create_task(_warm_llm_wire())")
    block = bind[max(0, i - 200):i + 300]
    assert re.search(r"_BACKGROUND_TASKS\.add\(", block), block

    boot = (BACKEND / "agent_main.py").read_text()
    j = boot.index("create_task(_boot_warm_llm_wire())")
    bblock = boot[max(0, j - 300):j + 400]
    assert re.search(r"_BOOT_BACKGROUND_TASKS\.add\(", bblock), bblock


def test_the_warm_is_a_DIAGNOSTIC_and_can_never_refuse_a_container():
    """Supervisor decision D1. `llm_wire_warm` has exactly ONE producer, and
    both of its schedulers wrap the call in an outer `except` that logs and
    forgets — so an import the SDK moves, or a `get_openai_agent_service()`
    that raises, leaves the flag false for the LIFE OF THE PROCESS. As a term
    of `serving` that pinned both onboarding probes at "warming up" forever
    for a container answering turns perfectly well; the app's poll is deadline
    -bounded, so every onboarding against it burned the full 20 s.

    It stays in `turn_ready_detail` (that is where a cold wire is diagnosed)
    and it may never be NAMED as the reason a container is not serving."""
    from app.api.agent_setup import _READINESS_PARTS, agent_turn_readiness

    assert "llm_wire_warm" not in _READINESS_PARTS
    assert "channels_settled" not in _READINESS_PARTS
    assert callable(svc().wire_warm)

    ready, reason = agent_turn_readiness(
        {
            "boot_progress": {"ready": True},
            "is_bound": True,
            "bound_user_id": "00000000-0000-4000-8000-000000000001",
            "turn_ready_detail": {
                "runner": True, "db": True, "bound_user": True,
                "llm_wire_warm": False, "channels_settled": False,
            },
        },
        "00000000-0000-4000-8000-000000000001",
    )
    assert ready is True, reason
    assert reason == ""


def test_the_client_swap_is_atomic_for_a_concurrent_turn():
    """`warm()` calls `_ensure_client` from `asyncio.to_thread`; every other
    caller reaches it from the loop. The old body stamped `_key_version`
    BEFORE building the client, so a turn landing inside the ~500 ms bind
    window saw the new version beside the OLD client and early-returned on it
    — and on a freshly claimed container the old client is
    `AsyncOpenAI(api_key="missing")`, i.e. the first turn 401s.

    Executed, not grepped: the fake builder observes the service's own state
    at the instant the new client is being constructed."""
    import threading

    m = svc()
    s = m.OpenAIAgentService.__new__(m.OpenAIAgentService)
    s._client_lock = threading.RLock()
    s.client = "OLD"
    s._key_version = 1

    class _Keys:
        version = 2
        openai = "k"

    s._keys = _Keys()
    seen = {}

    import app.services.bundle_client as bc

    real = bc.make_openai_client

    def _fake(*a, **kw):
        # What a concurrent reader would see right now.
        seen["version"] = s._key_version
        seen["client"] = s.client
        return "NEW"

    bc.make_openai_client = _fake
    try:
        s._ensure_client()
    finally:
        bc.make_openai_client = real

    assert seen["version"] == 1, (
        "the version was stamped before the client existed — a concurrent "
        "turn early-returns onto the pre-bind client"
    )
    assert seen["client"] == "OLD"
    assert s.client == "NEW" and s._key_version == 2


def test_anthropic_swaps_atomically_too():
    from app.services import anthropic_service as a
    import threading

    s = a.AnthropicService.__new__(a.AnthropicService)
    s._client_lock = threading.RLock()
    s.client = "OLD"
    s._key_version = 1
    s.is_oauth = False

    class _Keys:
        version = 2
        anthropic = "k"

    s._keys = _Keys()
    seen = {}

    import app.services.bundle_client as bc

    real = bc.make_anthropic_client

    def _fake(*a_, **kw):
        seen["version"] = s._key_version
        return "NEW"

    bc.make_anthropic_client = _fake
    try:
        s._ensure_client()
    finally:
        bc.make_anthropic_client = real

    assert seen["version"] == 1
    assert s.client == "NEW" and s._key_version == 2
