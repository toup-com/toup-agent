"""`[PERF] turn_host` — the cgroup's own account of the turn, from inside it.

WHY THIS FILE EXISTS

Round 1 could not answer "was this container throttled during the turn?", and
could not even agree on its CPU limit: the same container NAME is created by
four different paths in `bridge/` with three different limits (0.5 CPU /
1500m, 1.0 / 768m, 1.5 / 1024m), and the host copy of the bridge is
hand-deployed, so the repo does not settle it either. Every starvation
argument in the round therefore rests on an unread number.

A cgroup publishes its own throttling INSIDE the namespace, and PSI publishes
how long runnable tasks waited. Two file reads at each end of a turn answer
both, with no host access, no ssh, and no new endpoint.

The parsers are pure functions because the reads are not testable here — this
is macOS, there is no `/sys/fs/cgroup`, and a parser that is only exercised in
production is a parser that is wrong in production. The fixtures below are
real kernel output shapes.

The mutation these tests are written against: drop the nanosecond→microsecond
conversion in the cgroup-v1 branch. `test_v1_nanoseconds_become_microseconds`
goes RED; without it a host on cgroup v1 reports 1000x its real throttling and
the number reads as a catastrophe that is not happening.
"""
from __future__ import annotations

import asyncio
import logging
import os

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

#: Synthetic, and a FULL uuid on purpose: `span_enabled` matches exactly and
#: rejects anything too short to be a user id, so a test that used the
#: 8-character prefix from a container name would be testing the mistake.
CANARY = "00000000-aaaa-4bbb-8ccc-000000000001"
OTHER_USER = "00000000-aaaa-4bbb-8ccc-000000000002"

# cgroup v2 (`/sys/fs/cgroup/cpu.stat`) — microseconds.
V2_CPU_STAT = """usage_usec 1234567
user_usec 800000
system_usec 434567
nr_periods 4210
nr_throttled 137
throttled_usec 2048000
"""

# cgroup v1 (`/sys/fs/cgroup/cpu/cpu.stat`) — NANOseconds, different key.
V1_CPU_STAT = """nr_periods 4210
nr_throttled 137
throttled_time 2048000000
"""

# PSI (`/sys/fs/cgroup/cpu.pressure`). `full` is absent from
# `/proc/pressure/cpu` on many kernels — that is a different fact from zero.
PSI_CPU = """some avg10=0.31 avg60=0.12 avg300=0.04 total=987654
"""
PSI_MEM = """some avg10=0.00 avg60=0.00 avg300=0.00 total=12
full avg10=0.00 avg60=0.00 avg300=0.00 total=7
"""


def test_v2_cpu_stat_parses_the_three_fields_that_matter():
    from app.db.db_span import parse_cpu_stat

    got = parse_cpu_stat(V2_CPU_STAT)
    assert got["nr_periods"] == 4210
    assert got["nr_throttled"] == 137
    assert got["throttled_usec"] == 2048000


def test_v1_nanoseconds_become_microseconds():
    """v1 publishes `throttled_time` in nanoseconds and v2 `throttled_usec` in
    microseconds. Reporting them in the same field without converting would
    read as a 1000x throttling cliff the day a host moved between them."""
    from app.db.db_span import parse_cpu_stat

    got = parse_cpu_stat(V1_CPU_STAT)
    assert got["throttled_usec"] == 2048000, got
    assert got["nr_throttled"] == 137


def test_a_file_that_publishes_both_keys_prefers_the_native_unit():
    from app.db.db_span import parse_cpu_stat

    both = V2_CPU_STAT + "throttled_time 999999999999\n"
    assert parse_cpu_stat(both)["throttled_usec"] == 2048000


def test_garbage_never_raises_and_never_invents_a_field():
    from app.db.db_span import parse_cpu_stat, parse_pressure

    for junk in ("", "nr_throttled\n", "nr_throttled notanumber\n",
                 "\x00\x01\x02", "some avg10=x total=notanint\n"):
        assert isinstance(parse_cpu_stat(junk), dict)
        assert isinstance(parse_pressure(junk), dict)
    assert parse_cpu_stat("nr_throttled notanumber\n") == {}
    assert parse_pressure("some avg10=x total=notanint\n") == {}


def test_psi_reports_totals_only_and_keeps_absent_distinct_from_zero():
    """Only the cumulative `total=` fields: the `avgN` values are decaying
    averages over windows nobody chose, and only a DELTA of a monotonic total
    can be attributed to one turn. A `full` line that the kernel does not
    publish must stay MISSING, not become 0 — "not published" and "no
    pressure" have opposite meanings."""
    from app.db.db_span import parse_pressure

    cpu = parse_pressure(PSI_CPU)
    assert cpu == {"some_usec": 987654}
    assert "full_usec" not in cpu

    mem = parse_pressure(PSI_MEM)
    assert mem == {"some_usec": 12, "full_usec": 7}


def test_the_delta_drops_a_counter_that_appeared_vanished_or_reset():
    """A key present at only one end would report the whole cumulative counter
    as this turn's cost; a negative delta means a new cgroup, not negative
    pressure. Both are dropped rather than rendered."""
    from app.db.db_span import host_delta

    before = {"nr_throttled": 10, "throttled_usec": 500, "psi_cpu_some_usec": 7}
    after = {"nr_throttled": 13, "throttled_usec": 400, "psi_io_some_usec": 9}
    got = host_delta(before, after)
    assert got == {"nr_throttled": 3}, got
    assert host_delta({}, after) == {}
    assert host_delta(before, {}) == {}


def test_reading_the_counters_is_silent_on_a_host_that_has_none():
    """macOS here, and a cgroup-v1 host without PSI in production. Neither may
    raise, and neither may make the turn slower."""
    from app.db.db_span import read_host_counters

    got = read_host_counters()
    assert isinstance(got, dict)
    assert all(isinstance(v, int) for v in got.values())


@pytest.mark.asyncio
async def test_the_turn_host_line_says_it_read_nothing_rather_than_zero(
    caplog, monkeypatch,
):
    """Zeros are ambiguous: an unthrottled container, a cgroup-v1 host with no
    PSI, and a dev Mac all print the same zeros. `src=` is the only field that
    separates "measured, and it was fine" from "there was nothing to read"."""
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    monkeypatch.setattr(ds, "read_host_counters", lambda: {})
    ds._reset_for_tests()
    ds.begin_turn(user_id="u1", client_msg_id="c", channel="web", replace=True)
    with caplog.at_level(logging.INFO, logger="app.db.db_span"):
        line = ds.emit_turn_host()
    ds._reset_for_tests()
    assert line is not None and "src=none" in line, line
    assert "cpu_throttled_ms=0" in line


@pytest.mark.asyncio
async def test_the_turn_host_line_reports_the_delta_not_the_counter(
    caplog, monkeypatch,
):
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    readings = iter([
        {"nr_periods": 100, "nr_throttled": 10, "throttled_usec": 1_000_000,
         "psi_cpu_some_usec": 5_000_000},
        {"nr_periods": 140, "nr_throttled": 17, "throttled_usec": 1_350_000,
         "psi_cpu_some_usec": 5_420_000},
    ])
    monkeypatch.setattr(ds, "read_host_counters", lambda: next(readings))
    ds._reset_for_tests()
    ds.begin_turn(user_id="u1", client_msg_id="cmid", channel="web",
                  replace=True)          # takes the first reading
    with caplog.at_level(logging.INFO, logger="app.db.db_span"):
        line = ds.emit_turn_host()       # takes the second
    ds._reset_for_tests()

    assert "cpu_throttled_ms=350" in line, line
    assert "nr_throttled=7" in line and "nr_periods=40" in line, line
    assert "psi_cpu_some_ms=420" in line, line
    assert "src=cgroup" in line


@pytest.mark.asyncio
async def test_the_line_is_emitted_once_per_turn(caplog, monkeypatch):
    """It is called from `run()`'s `finally`, which is reached once — but a
    nested runner call or a retry must not double-count a turn."""
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    ds._reset_for_tests()
    ds.begin_turn(user_id="u1", client_msg_id="c", channel="web", replace=True)
    assert ds.emit_turn_host() is not None
    assert ds.emit_turn_host() is None
    ds._reset_for_tests()


@pytest.mark.asyncio
async def test_an_unarmed_turn_emits_nothing(monkeypatch):
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", False, raising=False)
    monkeypatch.setattr("app.config.settings.turn_db_span_canary_user_ids", "",
                        raising=False)
    ds._reset_for_tests()
    ds.begin_turn(user_id="u1", client_msg_id="c", channel="web", replace=True)
    assert ds.emit_turn_host() is None
    ds._reset_for_tests()


# ── one turn per run(), not one turn per context (R48 review, B-2) ───

class _TwoTurnRunner:
    """The real `AgentRunner.run`, bound to a stub with a stub `_run_inner`.

    `run()` is a five-line wrapper — `try: await self._run_inner(...)` and a
    `finally` — so binding the REAL function here exercises the real finally
    (the sweeps, `emit_turn_host()`, `end_turn()`) without a database, an LLM
    or a socket. The stub `_run_inner` makes exactly the arming call the real
    one makes, with no `replace` (pinned separately by
    `test_the_runner_never_overwrites_the_handlers_arming`)."""

    def __init__(self) -> None:
        self.seen: list = []

    def _sweep_unclosed_created_jobs(self, user_id):  # noqa: ANN001
        pass

    async def _run_inner(self, *, user_id, client_msg_id, channel):  # noqa: ANN001
        import app.db.db_span as ds

        ds.begin_turn(user_id=user_id, client_msg_id=client_msg_id,
                      channel=channel)
        self.seen.append((ds.turn_cmid_h(), ds.turn_enabled()))
        return "ok"


def _bound_run():
    from app.agent.agent_runner import AgentRunner

    return AgentRunner.run


@pytest.mark.asyncio
async def test_two_sequential_runs_in_one_context_are_two_turns(
    caplog, monkeypatch,
):
    """Every caller except the WS handler awaits `run()` repeatedly on ONE
    task context: `ws_router`'s `while True`, several `_think`s per voice
    call, the heartbeat's `for user_id, chat_id in user_chats`, cron. Without
    `end_turn()` in `run()`'s finally, `begin_turn`'s "keep the handler's
    identity" rule made turn 1 own all of them — same `cmid_h`, and
    `host_emitted=True` meant no `[PERF] turn_host` at all after the first.
    Reproduced against the shipped code at R48 review."""
    import app.agent.agent_runner as ar
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    monkeypatch.setattr(ar, "sweep_current_voice_job", lambda: None)
    ds._reset_for_tests()

    r = _TwoTurnRunner()
    r.run = _bound_run().__get__(r, _TwoTurnRunner)
    with caplog.at_level(logging.INFO, logger="app.db.db_span"):
        for n in (1, 2, 3):
            await r.run(user_id=CANARY, client_msg_id=f"turn-{n}",
                        channel="web")
    ds._reset_for_tests()

    hashes = [h for h, _en in r.seen]
    assert len(set(hashes)) == 3, r.seen
    assert all(en for _h, en in r.seen), r.seen
    host = [m.getMessage() for m in caplog.records
            if m.getMessage().startswith("[PERF] turn_host")]
    assert len(host) == 3, host
    assert len({h.split("cmid_h=")[1] for h in host}) == 3, host
    # …and the context is left clean, so a later turn in the same context is
    # not silently armed by this one.
    assert ds.turn_cmid_h() == "00000000"


@pytest.mark.asyncio
async def test_the_canary_allowlist_applies_per_turn_not_per_context(
    caplog, monkeypatch,
):
    """The cross-user half of the same defect, and the worse one: a heartbeat
    cycle that visited a non-canary user FIRST left the canary's own turn
    unarmed under the other user's `cmid_h` — so the one tenant the flag was
    turned on for produced no measurement at all, while acceptance gate 1
    passed on the WS path and reported the instrument healthy."""
    import app.agent.agent_runner as ar
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", False, raising=False)
    monkeypatch.setattr("app.config.settings.turn_db_span_canary_user_ids",
                        CANARY, raising=False)
    monkeypatch.setattr(ar, "sweep_current_voice_job", lambda: None)
    ds._reset_for_tests()

    r = _TwoTurnRunner()
    r.run = _bound_run().__get__(r, _TwoTurnRunner)
    with caplog.at_level(logging.INFO, logger="app.db.db_span"):
        await r.run(user_id=OTHER_USER, client_msg_id="turn-1",
                    channel="web")
        await r.run(user_id=CANARY, client_msg_id="turn-2", channel="web")
    ds._reset_for_tests()

    assert [en for _h, en in r.seen] == [False, True], r.seen
    host = [m.getMessage() for m in caplog.records
            if m.getMessage().startswith("[PERF] turn_host")]
    assert len(host) == 1, host
    from app.services.cmid import cmid_hash
    assert f"cmid_h={cmid_hash('turn-2')}" in host[0], host


# ── a CHILD run must not eat the parent's line (R48 review, B1) ──────

class _ParentChildRunner:
    """A parent turn that spawns a sub-agent, the way the real one does.

    `subagent_orchestrator.py:238` does `_spawn_bg(_run_child(...))` and
    `_run_child` awaits `agent_runner.run(channel="subagent",
    prompt_profile=SUBAGENT, save_assistant_message=False, ...)`;
    `subagent.py:129` does `create_task(self._execute(...))`. Both are a
    `create_task` from INSIDE a live turn, so the child's context is a COPY
    that holds the parent's `_Turn` object BY REFERENCE. That is the whole
    mechanism of B1 and it is what this stub reproduces — the real `run()`
    wrapper is bound in, so the real `enter_run`/`emit_turn_host`/`end_turn`/
    `exit_run` sequence is the one under test."""

    def __init__(self, children: int = 1) -> None:
        self.children = children
        self.seen: list = []

    def _sweep_unclosed_created_jobs(self, user_id):  # noqa: ANN001
        pass

    async def _run_inner(self, *, user_id, client_msg_id, channel,
                         child=False):  # noqa: ANN001
        import app.db.db_span as ds

        ds.begin_turn(user_id=user_id, client_msg_id=client_msg_id,
                      channel=channel)
        self.seen.append(("child" if child else "parent", ds.turn_cmid_h(),
                          ds.turn_enabled()))
        if not child:
            for _ in range(self.children):
                await asyncio.create_task(self.run(
                    user_id=user_id, client_msg_id=None, channel="subagent",
                    child=True,
                ))
        return "ok"


def _host_lines(caplog) -> list:
    return [m.getMessage() for m in caplog.records
            if m.getMessage().startswith("[PERF] turn_host")]


@pytest.mark.asyncio
async def test_a_child_run_does_not_consume_the_parents_host_line(
    caplog, monkeypatch,
):
    """B1. Before the fix there was exactly ONE `[PERF] turn_host` line for a
    turn that spawned a sub-agent — the CHILD's, carrying the parent's
    `cmid_h` and the parent's channel, covering parent-start → child-end —
    and the parent, still armed, emitted none. Acceptance gate 1 ("exactly one
    turn_host, all sharing one cmid_h") read GREEN on that output, so the one
    gate positioned to catch it could not.

    Reproduced LOCAL against the shipped module before the fix: parent
    `cmid_h=…`, child sees the SAME, child prints `ch=web`, parent prints
    `None`."""
    import app.agent.agent_runner as ar
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    monkeypatch.setattr(ar, "sweep_current_voice_job", lambda: None)
    ds._reset_for_tests()

    r = _ParentChildRunner()
    r.run = _bound_run().__get__(r, _ParentChildRunner)
    with caplog.at_level(logging.INFO, logger="app.db.db_span"):
        await r.run(user_id=CANARY, client_msg_id="parent-msg", channel="web")
    ds._reset_for_tests()

    host = _host_lines(caplog)
    assert len(host) == 2, host
    parent = [h for h in host if "nest=0" in h]
    child = [h for h in host if "nest=1" in h]
    assert len(parent) == 1, host
    assert len(child) == 1, host
    # The parent's line is the parent's: its own channel, not the child's.
    assert "ch=web" in parent[0], parent
    # …and the child's is the child's, which is what makes it discardable.
    assert "ch=subagent" in child[0], child
    # Both under one `cmid_h` — a child's DB work is part of the same
    # user-visible turn, and that join is what `cmid_h` is for.
    from app.services.cmid import cmid_hash
    want = cmid_hash("parent-msg")
    assert all(f"cmid_h={want}" in h for h in host), host


@pytest.mark.asyncio
async def test_a_child_spans_lines_are_tagged_and_the_parents_are_not(
    caplog, monkeypatch,
):
    """The second half of B1: the child's `db_span` phases rode the parent's
    `cmid_h` with nothing on the line to say so, so `phase=phase1` could
    appear twice per `cmid_h` and the two were indistinguishable. `nest=` is
    what separates them; without it no consumer of this log can tell a
    parent's save from a sub-agent's."""
    import app.agent.agent_runner as ar
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    monkeypatch.setattr(ar, "sweep_current_voice_job", lambda: None)
    ds._reset_for_tests()

    class _R(_ParentChildRunner):
        async def _run_inner(self, *, user_id, client_msg_id, channel,
                             child=False):  # noqa: ANN001
            import app.db.db_span as _ds

            _ds.begin_turn(user_id=user_id, client_msg_id=client_msg_id,
                           channel=channel)
            async with _ds.db_span("phase1"):
                pass
            if not child:
                await asyncio.create_task(self.run(
                    user_id=user_id, client_msg_id=None, channel="subagent",
                    child=True,
                ))
            return "ok"

    r = _R()
    r.run = _bound_run().__get__(r, _R)
    with caplog.at_level(logging.INFO, logger="app.db.db_span"):
        await r.run(user_id=CANARY, client_msg_id="parent-msg", channel="web")
    ds._reset_for_tests()

    spans = [m.getMessage() for m in caplog.records
             if m.getMessage().startswith("[PERF] db_span")]
    assert len(spans) == 2, spans
    assert len([s for s in spans if "phase=phase1" in s and "nest=0" in s]) == 1, spans
    assert len([s for s in spans if "phase=phase1" in s and "nest=1" in s]) == 1, spans
    assert len([s for s in spans if "ch=subagent" in s]) == 1, spans


@pytest.mark.asyncio
async def test_two_sibling_children_are_each_their_own_run_not_the_parents(
    caplog, monkeypatch,
):
    """Absent direction plus the documented LIMIT. Two sub-agents spawned by
    one turn each emit their own host line, so nothing is lost — and they BOTH
    print `nest=1`, because `nest` is a depth and not an identity. The module
    docstring's CANNOT list says exactly that; this test is what keeps the two
    honest together."""
    import app.agent.agent_runner as ar
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    monkeypatch.setattr(ar, "sweep_current_voice_job", lambda: None)
    ds._reset_for_tests()

    r = _ParentChildRunner(children=2)
    r.run = _bound_run().__get__(r, _ParentChildRunner)
    with caplog.at_level(logging.INFO, logger="app.db.db_span"):
        await r.run(user_id=CANARY, client_msg_id="parent-msg", channel="web")
    ds._reset_for_tests()

    host = _host_lines(caplog)
    assert len(host) == 3, host
    assert len([h for h in host if "nest=0" in h]) == 1, host
    assert len([h for h in host if "nest=1" in h]) == 2, host


@pytest.mark.asyncio
async def test_a_child_that_never_arms_its_own_turn_still_emits_nothing(
    caplog, monkeypatch,
):
    """The belt-and-braces half, and it has to be asserted on WHEN, not on
    WHAT. `begin_turn` is what mints the child's own turn, and it can be
    skipped — it raised, or a future child entry path does not call it. Then
    `_TURN` in the child's frame is still the PARENT's object, and an
    `emit_turn_host` that read `_TURN` directly would spend the parent's line
    from the child's frame: one host line, `nest=0`, `ch=web`, the parent's
    `cmid_h` — byte-identical to the correct output and emitted at the wrong
    moment, covering the wrong window. That is B1's whole failure mode
    (a confounded reading every gate reads as green), so the discriminator is
    a marker logged BETWEEN the child's exit and the parent's `finally`: the
    correct line lands after it, the confounded one before."""
    import app.agent.agent_runner as ar
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    monkeypatch.setattr(ar, "sweep_current_voice_job", lambda: None)
    ds._reset_for_tests()

    class _R(_ParentChildRunner):
        async def _run_inner(self, *, user_id, client_msg_id, channel,
                             child=False):  # noqa: ANN001
            import app.db.db_span as _ds

            if not child:
                _ds.begin_turn(user_id=user_id, client_msg_id=client_msg_id,
                               channel=channel)
                await asyncio.create_task(self.run(
                    user_id=user_id, client_msg_id=None, channel="subagent",
                    child=True,
                ))
                logging.getLogger("app.db.db_span").info("MARK child returned")
            # the child arms NOTHING
            return "ok"

    r = _R()
    r.run = _bound_run().__get__(r, _R)
    with caplog.at_level(logging.INFO, logger="app.db.db_span"):
        await r.run(user_id=CANARY, client_msg_id="parent-msg", channel="web")
    ds._reset_for_tests()

    msgs = [m.getMessage() for m in caplog.records]
    host = [m for m in msgs if m.startswith("[PERF] turn_host")]
    assert len(host) == 1, msgs
    assert "nest=0" in host[0] and "ch=web" in host[0], host
    # The line belongs to the PARENT's frame, so it is emitted after the child
    # has already returned.
    assert msgs.index(host[0]) > msgs.index("MARK child returned"), msgs


@pytest.mark.asyncio
async def test_a_nested_run_in_the_SAME_context_does_not_steal_the_parents_turn(
    caplog, monkeypatch,
):
    """Defence in depth, and honestly labelled as such: NO path read at
    4f0e9fe1 awaits `run()` from inside `run()` — both sub-agent entry points
    are `create_task`, and telegram/cron/heartbeat/voice/routines all await it
    at top level. But `begin_turn`'s nested mint writes `_TURN`, and in a
    directly-awaited nested frame that write is NOT private: the parent would
    come back to the CHILD's turn and emit the child's line as its own. So the
    frame restores what it found (`enter_run`/`exit_run`), and this is the
    test that can see it."""
    import app.agent.agent_runner as ar
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    monkeypatch.setattr(ar, "sweep_current_voice_job", lambda: None)
    ds._reset_for_tests()

    class _R(_ParentChildRunner):
        async def _run_inner(self, *, user_id, client_msg_id, channel,
                             child=False):  # noqa: ANN001
            import app.db.db_span as _ds

            _ds.begin_turn(user_id=user_id, client_msg_id=client_msg_id,
                           channel=channel)
            if not child:
                # awaited DIRECTLY — no create_task, so no context copy
                await self.run(user_id=user_id, client_msg_id=None,
                               channel="subagent", child=True)
            return "ok"

    r = _R()
    r.run = _bound_run().__get__(r, _R)
    with caplog.at_level(logging.INFO, logger="app.db.db_span"):
        await r.run(user_id=CANARY, client_msg_id="parent-msg", channel="web")
    ds._reset_for_tests()

    host = _host_lines(caplog)
    assert len(host) == 2, host
    assert len([h for h in host if "nest=0" in h and "ch=web" in h]) == 1, host
    assert len([h for h in host if "nest=1" in h and "ch=subagent" in h]) == 1, host


def test_the_runner_marks_its_frame_and_restores_it_last():
    """Source ORDER again. `emit_turn_host()` and `end_turn()` both ask
    `_RUN_NEST` whether this frame owns the turn; restoring the depth before
    them would make a nested run look like the outer one and hand B1 back."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "app/agent/agent_runner.py").read_text()
    i_run = src.index("    async def run(self, *args, **kwargs)")
    i_inner = src.index("    async def _run_inner(")
    block = src[i_run:i_inner]
    assert "_db_span_mod.enter_run()" in block, (
        "run() does not mark its frame — a sub-agent run is indistinguishable "
        "from the turn that spawned it"
    )
    assert block.index("_db_span_mod.enter_run()") < block.index("try:")
    assert block.index("_db_span_mod.end_turn()") < block.index(
        "_db_span_mod.exit_run("
    )
    assert block.index("_db_span_mod.emit_turn_host()") < block.index(
        "_db_span_mod.exit_run("
    )


def test_the_runner_disarms_the_turn_after_emitting_its_host_line():
    """Source ORDER, not presence: `emit_turn_host()` READS the turn that
    `end_turn()` clears, so a reordering here would silently delete the host
    line for every turn. A guard whose precondition something above it
    destroys is invisible to every other check in this repo."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "app/agent/agent_runner.py").read_text()
    i_run = src.index("    async def run(self, *args, **kwargs)")
    i_inner = src.index("    async def _run_inner(")
    block = src[i_run:i_inner]
    assert "_db_span_mod.end_turn()" in block, (
        "run() does not disarm the turn — turns 2..N in one context inherit "
        "turn 1's identity and emit no host line"
    )
    assert block.index("_db_span_mod.emit_turn_host()") < block.index(
        "_db_span_mod.end_turn()"
    )


def test_the_runner_emits_it_from_the_finally_that_survives_a_cancellation():
    """A cancelled voice turn and a turn that raised are the turns most likely
    to have been throttled. `run()`'s `finally` is the only try/finally on the
    path — `_run_inner` has none, by its own docstring."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "app/agent/agent_runner.py").read_text()
    i_run = src.index("    async def run(self, *args, **kwargs)")
    i_inner = src.index("    async def _run_inner(")
    block = src[i_run:i_inner]
    assert "_db_span_mod.emit_turn_host()" in block, (
        "the host line is not emitted from run()"
    )
    assert block.index("finally:") < block.index("_db_span_mod.emit_turn_host()")
    # …and it is the LAST thing, after the job sweeps: a sweep that raised
    # would otherwise take the measurement with it.
    assert block.index("sweep_current_voice_job()") < block.index(
        "_db_span_mod.emit_turn_host()"
    )
