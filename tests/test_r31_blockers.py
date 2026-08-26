"""R31-40…46 — the R30 rollout blockers, pinned.

Each of these was a row in ROLLOUT-READINESS with a description and no
test. The descriptions were mostly right and two of them were wrong in
ways that mattered, so every test here says what was MEASURED as well
as what is asserted.
"""

import pytest

from app.api.llm_proxy import (
    _OPENAI_MAX_TOOLS, _cap_tools, _requested_tool_names,
)


# ── R31-41: the 128-tool cap ─────────────────────────────────────────


def _named(prefix: str, n: int) -> list:
    return [{"name": f"{prefix}__t{i}", "description": "x"} for i in range(n)]


def test_the_cap_never_empties_a_connector():
    """The blocker, in one assertion.

    The trim was `tools[:limit]` — a head slice — and the tail is not
    arbitrary: the agent assembles core → skills → MCP, and the MCP
    block is SORTED BY NAME. So the casualties were always the same
    alphabetical tail — `outlook__send_message`, `session_*`, then every
    `slack__*` and every `teams__*`, reads and writes alike.

    An automation that reads Slack and posts to Slack lost BOTH on a
    ten-connector account, and the only signal was a log line — so the
    run failed with "it did not answer" and the user was told to
    reconnect a connector that was perfectly healthy.
    """
    # The SMALL connector sits at the tail, which is where the real
    # assembly order puts the alphabetically-last one — so a head slice
    # removes every one of its tools. Written this way deliberately: a
    # first draft put the big namespaces last, and a head-slice mutation
    # passed it, because dropping 2 of 25 leaves the connector present.
    # The overflow (2) must be >= the small namespace's size, or the
    # head slice does not empty it and the test proves nothing.
    tools = (
        _named("core", 60)
        + _named("gmail", 68)
        + _named("slack", 2)
    )
    assert len(tools) > _OPENAI_MAX_TOOLS
    kept, dropped = _cap_tools(tools)
    assert len(kept) == _OPENAI_MAX_TOOLS

    spaces = {n.split("__")[0] for n in
              (t["name"] for t in kept)}
    for connector in ("core", "gmail", "slack"):
        assert connector in spaces, (
            f"the cap removed every {connector} tool — the model cannot "
            f"discover a connector that has vanished"
        )


def test_the_cap_drops_from_the_biggest_namespace_first():
    """Degradation spreads across who can afford it.

    Alphabetical order is not importance order, and losing all five of a
    small connector's tools is categorically worse than losing five of a
    large one's.
    """
    # Big namespace FIRST, small one at the tail. A head slice would
    # take the whole tail — every `aaa` tool — which is the behaviour
    # under test; only a size-aware trim takes all seven from `zzz`.
    tools = _named("zzz", 130) + _named("aaa", 5)
    _kept, dropped = _cap_tools(tools)
    assert len(dropped) == 7, dropped
    assert all(n.startswith("zzz__") for n in dropped), dropped


def test_a_tool_the_request_names_is_never_dropped():
    """ND-22's guaranteed 400, made unreachable.

    `_prune_tool_choice` repairs the ALLOWLIST after the fact, which
    keeps the request valid while silently removing a capability the
    caller asked for — and for a forced `{"type":"function"}` it cannot
    repair anything at all, so the request is a certain 400. Observed
    live on the founder's voice turn: 141 tools, `slack__list_channels`
    capped away and still named, three 400s, then a silent fallback to a
    weaker model — 20.7 s for two output tokens.
    """
    tools = _named("zzz", 130) + _named("aaa", 5)
    body = {
        "tool_choice": {
            "type": "allowed_tools",
            "allowed_tools": {"tools": [
                {"function": {"name": "zzz__t129"}},
            ]},
        },
    }
    protected = _requested_tool_names(body)
    assert protected == {"zzz__t129"}
    _kept, dropped = _cap_tools(tools, protected=protected)
    assert "zzz__t129" not in dropped


def test_a_forced_choice_is_protected_too():
    body = {"tool_choice": {"type": "function",
                            "function": {"name": "zzz__t99"}}}
    assert _requested_tool_names(body) == {"zzz__t99"}
    _kept, dropped = _cap_tools(
        _named("zzz", 200), protected={"zzz__t99"})
    assert "zzz__t99" not in dropped


def test_an_array_under_the_cap_is_untouched():
    """The trim must be a no-op below the cliff — a reorder here would
    invalidate every tenant's prompt-cache prefix for nothing."""
    tools = _named("core", 50) + _named("slack", 20)
    kept, dropped = _cap_tools(tools)
    assert dropped == []
    assert kept == tools


# ── R31-42: the drain that only counted chat sockets ─────────────────


def test_a_run_holds_the_drain_open():
    """The counter had ONE incrementer and ONE decrementer, both inside
    the chat WebSocket handler. Nothing about a run touched it, and the
    ASGI drain gate lets HTTP through — so with no chat client attached
    a drain SIGTERMed within about a second however many runs were
    mid-flight."""
    from app.services import drain_state

    drain_state.run_started("run-a")
    drain_state.run_started("run-b")
    assert drain_state.active_runs() == {"run-a", "run-b"}
    assert "run-a" in drain_state.status()["active_runs"]

    drain_state.run_finished("run-a")
    # Idempotent: a double release is free, a MISSED one wedges a
    # deploy forever — which is why the set is keyed by id and logged.
    drain_state.run_finished("run-a")
    assert drain_state.active_runs() == {"run-b"}
    drain_state.run_finished("run-b")
    assert drain_state.active_runs() == set()


def test_a_run_is_refused_while_draining(monkeypatch):
    """§4.8: a deploy "never starts a run it will kill".

    R31-D measured the cost of the opposite: two of the founder's 26
    August runs ran 362 s and 413 s against a 180 s cap and ended
    `error_class: "interrupted"`. A cap cannot fire at 2× its own value
    — those runs were killed, not capped, and the same work took 58 s
    uninterrupted.
    """
    from types import SimpleNamespace
    from app.agent.automations import executor_v2
    from app.services import drain_state

    automation = SimpleNamespace(id="a-1", name="Brief")
    assert executor_v2._refuse_during_drain(automation, "scheduled") is False

    monkeypatch.setattr(drain_state, "_draining", True)
    assert executor_v2._refuse_during_drain(automation, "scheduled") is True
    assert drain_state.should_refuse_new_run() is True


def test_the_run_release_is_at_the_one_terminal_gate():
    """`on_terminal` is the obvious place and the wrong one.

    It runs only when the finalize's rowcount won AND the run carries a
    v3 thread — so a pre-v3 or thread-less run would hold a drain open
    forever. `_finalize_job` is the gate every terminal passes.
    """
    import inspect
    from app.agent.automations import executor

    src = inspect.getsource(executor._finalize_job)
    assert "run_finished" in src, (
        "the drain release left the one gate every terminal passes"
    )


# ── R31-44: started_at, and the two tag formats ──────────────────────


def test_the_image_tag_comparison_is_format_blind():
    """Two writers, two formats, one column.

    `reconcile_managed_rows` writes the bare digest; `upgrade_tenant
    _image` writes the fully-qualified reference. R31-D watched one row
    flip between them 50 seconds apart. A census that groups by
    `image_tag` then reads ONE image as two, so "the fleet is on X" is
    false for every value of X.
    """
    from app.services.docker_host_service import _normalise_image_tag as n

    assert n("0b3926cbe809") == "0b3926cbe809"
    assert n("ghcr.io/toup-com/toup-agent:0b3926cbe809") == "0b3926cbe809"
    assert n("") == "" and n(None) == ""


def test_a_missing_start_time_is_not_guessed():
    """`started_at` had four writers and all four were platform-side
    wall clocks — none of them the container's start — and the rollout
    path did not write it at all, so every recreate left it pointing at
    the previous provision.

    The fix reads `.State.StartedAt`. Where the bridge does not send it
    (it is hand-deployed and outside this repo), the column is LEFT
    ALONE: a wrong timestamp is the defect, and a null one at least
    reads as unknown.
    """
    from app.services.docker_host_service import _parse_started_at as p

    assert p("2026-08-26T18:24:19.123456789Z") is not None
    assert p("2026-08-26T18:24:19Z") is not None
    assert p("") is None
    assert p(None) is None
    assert p("not a time") is None


# ── The census instrument (ND-23's family) ───────────────────────────


def test_the_overview_omits_docker_when_the_bridge_is_unreachable():
    """An instrument that cannot be misread beats one that must be read
    carefully.

    R31-D characterised the flap on 2026-08-26 — about 90 s down on a
    ~2-minute cycle, every tenant serving 200 throughout, so the outage
    is in the CENSUS PATH and not the fleet. During it the payload
    returned `containers_running: []`, and a reader that does not check
    `bridge_reachable` first concludes the fleet has vanished. D nearly
    published "5 tenants disagree with docker" off one such sample.

    A missing key raises where an empty list computes.
    """
    import inspect
    from app.api.admin import infrastructure

    src = inspect.getsource(infrastructure)
    assert 'if reachable:' in src
    assert 'out["docker"]' in src
    # The key must be set INSIDE the guard, never before it.
    guard = src.index("if reachable:")
    assign = src.index('out["docker"]')
    assert guard < assign, (
        "the docker key is written before the reachability check"
    )


# ── The day reader that truncated by session count ───────────────────


def test_the_day_reader_is_bounded_by_messages_not_sessions():
    """R31-D, 2026-08-26: 30 rows at 17:35Z, 19 at 18:47Z, same date,
    same offset, 11 dropped and 0 added.

    `GET /api/sessions/by-date/{date}/messages` took the 10 most
    recently-UPDATED conversations and returned their messages. A day is
    a fixed set of rows, so that is a moving truncation: which rows you
    can see depends on how many conversations have been touched since,
    and nothing in the response says a session was dropped.

    The eleven were all three `automation_notification` cards and eight
    rows of the founder talking to his agent about reminders — his own
    morning, invisible in his own day. Nothing was deleted;
    `/api/day-chats/{date}/messages` returned all 51 throughout.

    Automations made it visible rather than caused it: each thread mints
    a conversation, so an active day now has more than ten and
    `updated_at DESC` puts the fresh automation rows on top.

    Driving the SOURCE rather than the route: the route needs a live DB,
    a user and a day, and the property under test is one line of query
    shape.
    """
    import inspect
    from app.api import sessions

    src = inspect.getsource(sessions.get_messages_by_date)

    # The session cap is gone.
    assert ".limit(10)" not in src, (
        "the day reader caps SESSIONS again — a day is bounded by its "
        "messages"
    )
    # And the message window takes the NEWEST rows, not the oldest:
    # with the cap gone, `ASC LIMIT n` would make the user's most recent
    # messages the ones that vanish, which is the worse half of the same
    # defect.
    assert "Message.created_at.desc()" in src
    assert "reversed(" in src


# ── Every agent route needs a door ───────────────────────────────────


def test_every_agent_automations_route_has_a_proxy_forwarder():
    """A route the app cannot reach is a route that does not exist.

    R31-D found three of this round's new routes answering 405 on the
    platform — `cleanup-day-chat`, `accounts/{id}/probe` and
    `workflow/commit` — because the proxy is a hand-written list of
    forwarders and nothing checks it against the agent's surface. The
    405 is the giveaway shape: the PATH prefix matches, the verb is not
    registered.

    This is the repo's "a handler for a frame nobody sends" class,
    pointed the other way: an endpoint with no caller. It fails silently
    at the seam between two files that no single test exercises.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "app" / "api"
    agent = (root / "automations.py").read_text()
    proxy = (root / "automations_proxy.py").read_text()

    def routes(text: str, decorator: str) -> set:
        out = set()
        for m in re.finditer(
            rf'@{decorator}\.(get|post|put|patch|delete)\(\s*"([^"]*)"',
            text,
        ):
            verb, path = m.group(1), m.group(2)
            # Normalise the parameter names — the proxy is free to call
            # its path variable something else.
            path = re.sub(r"\{[a-z_]+\}", "{}", path)
            out.add(f"{verb.upper()} {path}")
        return out

    agent_routes = routes(agent, "router") | routes(agent, "accounts_router")
    proxy_routes = (routes(proxy, "router")
                    | routes(proxy, "accounts_proxy_router"))

    # Platform→agent hooks are called BY the platform with the tenant's
    # key; they are deliberately not user-reachable.
    exempt = {
        "POST /_connector_connected",
        "POST /_grant_decided",
    }
    missing = {r for r in agent_routes if r not in proxy_routes} - exempt
    assert not missing, (
        "agent routes with no proxy forwarder — the app cannot reach "
        f"them and gets a 405: {sorted(missing)}"
    )


def test_the_accounts_proxy_forwards_the_body():
    """It never passed `content=`.

    Harmless while the two accounts routes were a GET and a bodyless
    POST; a silent data loss the moment `reconnect` grew `add_scopes`
    and `probe` grew `force`. A dropped body does not error — the agent
    sees an empty request and answers plausibly, so `Grant access`
    would run an ordinary reconnect and report success.
    """
    import inspect
    from app.api import automations_proxy

    src = inspect.getsource(automations_proxy._proxy_accounts)
    assert "await request.body()" in src
    assert "content=" in src
