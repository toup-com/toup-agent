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


# ── R31-46: a dry run that predicted a different run ─────────────────


@pytest.mark.asyncio
async def test_the_migration_report_mirrors_the_migration():
    """A preview whose classes differ from the run it previews is worse
    than no preview: it is read as permission.

    `migration_report`'s docstring claimed it mirrored
    `migrate_email_briefings` "exactly" and did not. It detected twins
    by EXACT normalised name equality while the migration uses the
    SUBSET test `_same_intent` — and ND-13 records that exact equality
    never fires, which is precisely why `_same_intent` exists. So the
    report's `would_supersede` branch was effectively dead and it
    predicted `would_migrate` for routines the run supersedes.

    It also tested the twin BEFORE `enabled` while the migration tests
    `enabled` first, so a disabled routine with a twin came back
    `would_supersede` in the preview and `needs_review` in the run.
    """
    import inspect
    from app.agent.automations import routine_migration as rm

    report = inspect.getsource(rm.migration_report)
    run = inspect.getsource(rm.migrate_email_briefings)

    # One twin test, shared.
    assert "_find_twin(" in report, (
        "the report has its own twin test again"
    )
    assert "by_name.get(" not in report

    # And the same branch ORDER: enabled before twin, in both.
    #
    # Anchored on the BRANCH text, not on the words. A first draft used
    # `src.index("enabled")` and matched the prose in a comment forty
    # lines above the branch — a probe that reads its own explanation
    # and calls it evidence.
    assert (report.index("elif not r.enabled:")
            < report.index("elif twin is not None:")), (
        "the report tests the twin before enabled"
    )
    assert (run.index("if not routine.enabled and not selected:")
            < run.index("twin = _find_twin(")), (
        "the migration tests the twin before enabled"
    )


# ── R31-47: a use with no binding, anywhere in app/ ──────────────────


def _unbound_names(root=None) -> list:
    """Every runtime name read in `app/` that nothing binds.

    Built on `symtable`, which IS CPython's own scope pass, so
    comprehension scopes, class scopes, `global`/`nonlocal`,
    `except ... as`, match captures and the walrus all come out right
    without being modelled by hand. The first version of this guard DID
    model them by hand and was wrong in both directions — silent on a
    `TYPE_CHECKING`-only import (the exact class it exists to catch) and
    noisy on comprehension targets and `global` writes.

    Two things `symtable` cannot know, added here:
      - a binding whose ONLY site is under `if TYPE_CHECKING:` does not
        exist at runtime, though the compiler sees an import;
      - `global NAME` + an assignment inside any function DOES create
        the module global.

    Deliberately blind to string annotations: `Mapped[List["Memory"]]`
    is a string, not a name read, so SQLAlchemy's forward-ref idiom
    produces nothing here. `ruff --select F821` reports 21 of those and
    they are all noise for this question.
    """
    import ast
    import builtins
    import pathlib
    import symtable

    BUILTINS = set(dir(builtins)) | {
        "__file__", "__name__", "__doc__", "__package__", "__spec__",
        "__loader__", "__builtins__", "__debug__", "__annotations__",
    }
    SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
              ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    COMPS = {ast.ListComp: "listcomp", ast.SetComp: "setcomp",
             ast.DictComp: "dictcomp", ast.GeneratorExp: "genexpr"}

    root = root or (pathlib.Path(__file__).resolve().parent.parent / "app")

    def is_tc(test) -> bool:
        return ((isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
                or (isinstance(test, ast.Attribute)
                    and test.attr == "TYPE_CHECKING"))

    def kind(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return "function", node.name
        if isinstance(node, ast.Lambda):
            return "function", "lambda"
        if isinstance(node, ast.ClassDef):
            return "class", node.name
        return "function", COMPS[type(node)]

    def tables(st, out):
        out.setdefault((st.get_type(), st.get_name(), st.get_lineno()), st)
        for c in st.get_children():
            tables(c, out)
        return out

    findings = []
    for path in sorted(root.rglob("*.py")):
        src = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
            st = symtable.symtable(src, str(path), "exec")
        except (SyntaxError, ValueError):        # pragma: no cover
            continue

        for n in ast.walk(tree):
            if isinstance(n, ast.If) and is_tc(n.test):
                for c in ast.walk(n):
                    c._tc = True

        tc, real = set(), set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                names = {(a.asname or a.name.split(".")[0]) for a in n.names}
                (tc if getattr(n, "_tc", False) else real).update(names)
        tc_only = tc - real

        bound = {y.get_name() for y in st.get_symbols()
                 if y.is_assigned() or y.is_imported()} - tc_only

        def global_writes(t):
            for y in t.get_symbols():
                if y.is_global() and y.is_assigned():
                    bound.add(y.get_name())
            for c in t.get_children():
                global_writes(c)
        global_writes(st)

        tbl = tables(st, {})

        def visit(node, chain):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, SCOPES):
                    visit(child, chain + [tbl.get((*kind(child), child.lineno))])
                    continue
                if (isinstance(child, ast.Name)
                        and isinstance(child.ctx, ast.Load)
                        and not getattr(child, "_tc", False)
                        and child.id not in BUILTINS
                        and child.id not in bound):
                    # Walk OUTWARD. A name can be missing from the
                    # innermost table entirely — the outer iterable of a
                    # genexpr is evaluated in the ENCLOSING scope, so
                    # `sum(x for x in self.items)` puts `self` in the
                    # method's table and not the genexpr's. Reporting on
                    # that KeyError instead of looking outward produced
                    # 446 false positives, `self` first.
                    for here in reversed(chain):
                        if here is None:
                            continue
                        try:
                            sym = here.lookup(child.id)
                        except KeyError:
                            continue
                        if not (sym.is_local() or sym.is_free()
                                or sym.is_assigned()):
                            findings.append(
                                f"{path.relative_to(root.parent)}:"
                                f"{child.lineno} {child.id}")
                        break
                visit(child, chain)

        visit(tree, [st])

    return sorted(set(findings))


def test_nothing_in_app_reads_a_name_that_nothing_binds():
    """The defect, and the whole class it belongs to.

    R31 (`b28d2cee`) swapped three day readers off the literal
    `Conversation.channel != "autopilot"` and onto the shared
    HIDDEN_DAY_CHANNELS tuple. Two gained the import;
    `app/api/messages_recover.py` did not — so `GET /api/messages/since`
    raised `NameError` and answered 500 on EVERY WebSocket reconnect,
    which is the one route whose whole job is recovering the assistant
    reply a dropped socket lost.

    Nothing caught it: the module imports fine (the name is read when
    the query is built, not at import), the R31 pin that mentions the
    tuple reads a DIFFERENT module's source, and the CI step that does
    execute the line runs only after the platform sweep is green, which
    it was not.

    Widened from that one constant to every name in `app/` after the
    narrow version was audited: three MORE live instances of the same
    class were sitting in the tree, none of them R31's, and the narrow
    guard could not see any of them —

      - `toup_code.py` used `Optional` in two Pydantic models and never
        imported it. `from __future__ import annotations` defers the
        annotation, so the module imports clean and the model fails when
        pydantic first builds it: **500 on POST /api/code/save-to-workspace**.
        "It imports fine" is not a proof, which is the same sentence
        this docstring already contained.
      - `ws_realtime.py::_finalize_onboarding` returned an f-string over
        `agent_memories` / `user_memories` that the memory-v3 rewiring
        had deleted, so every SUCCESSFUL onboarding raised, was caught by
        a bare `except Exception`, and reported
        "finalized with some issues". The success path had never been
        reached.
      - `apps.py` passed `label=title` where every sibling line reads
        `req.title`, inside a `try/except Exception: logger.debug`, so
        the waiting-on-user Live Activity never fired.

    All three are fixed. This asserts ZERO, with no baseline, so the
    next one fails the build instead of hiding behind a swallow.
    """
    assert _unbound_names() == []


def test_the_unbound_name_check_gets_python_scoping_right():
    """The guard's own falsification, kept in the suite.

    The first version of this check hand-rolled the scope rules and was
    wrong in BOTH directions — it treated a `TYPE_CHECKING`-only import
    as a real binding (silent on the very class it exists to catch) and
    it treated a comprehension target and a `global` write as unbound.
    A guard that models its subject has to prove the model.
    """
    import pathlib
    import tempfile

    cases = {
        # name              source                                       reported?
        "typechecking": ("from typing import TYPE_CHECKING\n"
                         "if TYPE_CHECKING:\n    from foo import T\n"
                         "def q():\n    return T\n", True),
        "local_import": ("def q():\n    from foo import T\n    return T\n", False),
        "comprehension": ("def q(xs):\n    return [T for T in xs]\n", False),
        "class_body":   ("class K:\n    T = 1\n"
                         "    def m(self):\n        return T\n", True),
        "global_write": ("def s():\n    global T\n    T = 1\n"
                         "def u():\n    return T\n", False),
        "except_as":    ("def q():\n    try:\n        pass\n"
                         "    except Exception as T:\n        return T\n", False),
        "docstring":    ('def q():\n    """mentions T only in prose."""\n'
                         "    return 1\n", False),
        "closure":      ("def outer():\n    from foo import T\n"
                         "    def inner():\n        return T\n"
                         "    return inner\n", False),
        "truly_absent": ("def q():\n    return T\n", True),
    }
    with tempfile.TemporaryDirectory() as d:
        pkg = pathlib.Path(d) / "app"
        pkg.mkdir()
        for name, (src, _) in cases.items():
            (pkg / f"{name}.py").write_text(src)
        got = _unbound_names(pkg)

    hit = {f.split("/")[-1].split(".py")[0] for f in got}
    for name, (_, should) in cases.items():
        assert (name in hit) is should, (
            f"{name}: expected {'a report' if should else 'silence'}, "
            f"got {sorted(hit)}"
        )


def _unbound_annotation_names(root=None) -> list:
    """Names used in an ANNOTATION that the module never binds.

    A second instrument, because the first one structurally cannot see
    this: under `from __future__ import annotations` the interpreter
    never evaluates an annotation, so `symtable` records nothing and a
    missing import is invisible to any scope analysis. Pydantic DOES
    evaluate them — when it builds the model, on first use — which is
    how `toup_code.py` used `Optional` in two models without importing
    it, imported clean, registered its route clean, and answered **500
    on POST /api/code/save-to-workspace**.

    Reads only real `ast.Name` nodes, so a STRING forward reference is
    invisible by design. That is the whole difference between this and
    `ruff --select F821`, which resolves strings and therefore reports
    20 `Mapped[List["Memory"]]`-shaped findings that are all correct
    SQLAlchemy and none of them defects.
    """
    import ast
    import builtins
    import pathlib

    KNOWN = set(dir(builtins)) | {"None", "True", "False"}
    root = root or (pathlib.Path(__file__).resolve().parent.parent / "app")
    out = []

    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:                              # pragma: no cover
            continue
        bound = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                bound |= {(a.asname or a.name.split(".")[0]) for a in n.names}
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
                bound.add(n.name)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                bound.add(n.id)
        for n in ast.walk(tree):
            if isinstance(n, ast.AnnAssign):
                anns = [n.annotation]
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = n.args
                anns = [x.annotation for x in
                        (a.posonlyargs + a.args + a.kwonlyargs)] + [n.returns]
            else:
                continue
            for ann in anns:
                if ann is None:
                    continue
                for nm in ast.walk(ann):
                    if (isinstance(nm, ast.Name)
                            and isinstance(nm.ctx, ast.Load)
                            and nm.id not in KNOWN and nm.id not in bound):
                        out.append(
                            f"{path.relative_to(root.parent)}:"
                            f"{nm.lineno} {nm.id}")
    return sorted(set(out))


def test_no_annotation_names_a_module_never_imported():
    """`toup_code.py`'s 500, and the reason the other guard missed it.

    `project: Optional[str] = None` in a pydantic model, with
    `Optional` absent from the line-40 `from typing import ...` and
    `from __future__ import annotations` at the top. The module imports.
    The router registers. `TypeAdapter` fails when pydantic first builds
    the model, and the route answers 500 forever.

    Found by the audit of the FIRST version of this guard, which had
    been written for exactly this class of defect and could not see it —
    the same shape as the pin that watched the wrong module. Asserts
    zero, with no baseline.
    """
    assert _unbound_annotation_names() == []


# ── R31-48: two holes an audit found in R31's own pins ───────────────


def test_every_promotion_to_db_ready_syncs_the_template_catalog():
    """"No suggestions" for everyone, with nothing logged.

    `automation_templates` is populated at boot by `sync_template_catalog`,
    under `if app.state.db_ready:`. When the boot path finds the database
    down it hands off to `_heal_db_schema`, which retries `init_db()` and,
    on success, sets `db_ready = True` — and used to return. It never
    synced. A replica that booted degraded and healed therefore served an
    EMPTY catalog for its whole life, and `GET /api/automations/catalog`
    returns every enabled template to every caller, so empty means every
    user sees no suggestions at all. Nothing raises, nothing logs: the
    heal prints a success line.

    So the rule is about the STATE, not the function: every place that
    promotes the app to ready must sync.
    """
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "platform_main.py").read_text()
    tree = ast.parse(src)

    def sets_ready(node) -> bool:
        for n in ast.walk(node):
            if not isinstance(n, ast.Assign):
                continue
            for t in n.targets:
                if (isinstance(t, ast.Attribute) and t.attr == "db_ready"
                        and isinstance(t.value, ast.Attribute)
                        and t.value.attr == "state"):
                    return True
        return False

    def calls_sync(node) -> bool:
        return any(
            isinstance(n, ast.Call) and (
                (isinstance(n.func, ast.Name)
                 and n.func.id == "_sync_template_catalog")
                or (isinstance(n.func, ast.Attribute)
                    and n.func.attr == "_sync_template_catalog"))
            for n in ast.walk(node)
        )

    promoters = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and sets_ready(n)]
    assert promoters, "nothing sets app.state.db_ready — did it move?"

    missing = [n.name for n in promoters if not calls_sync(n)]
    assert missing == [], (
        f"these promote the app to db_ready without syncing the template "
        f"catalog, so their replica serves an empty catalog: {missing}"
    )


def test_the_hidden_channel_readers_bind_the_canonical_tuple():
    """The re-anchored pin's own hole, found by auditing it.

    `test_read_paths_hide_autopilot_channel` matches the source TEXT
    `Conversation.channel.notin_(HIDDEN_DAY_CHANNELS)` and checks
    membership on the tuple the TEST imports. Mutation testing against
    the real sources showed two ways to satisfy it while autopilot rows
    are returned, neither of which existed when the predicate was an
    inline literal:

      - a reader rebinds `HIDDEN_DAY_CHANNELS = ("automation",)` locally
      - a reader imports the name from a DIFFERENT module

    Both are indirection, which is what the re-anchor introduced. Note
    the shape is not hypothetical here: R31 shipped the opposite half of
    it — a reader that used the name and imported it from nowhere.

    So: every reader must import it from `app.db.models.conversation`
    and none may rebind it.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "app"
    CANON = "app.db.models.conversation"
    NAME = "HIDDEN_DAY_CHANNELS"
    bad = []

    for path in sorted(root.rglob("*.py")):
        src = path.read_text(encoding="utf-8", errors="replace")
        if NAME not in src or path.name == "conversation.py":
            continue
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                if any((a.asname or a.name) == NAME for a in n.names):
                    if n.module != CANON:
                        bad.append(f"{path.name}:{n.lineno} imports {NAME} "
                                   f"from {n.module!r}, not {CANON!r}")
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id == NAME:
                        bad.append(f"{path.name}:{n.lineno} rebinds {NAME}")
            elif isinstance(n, ast.AnnAssign):
                if isinstance(n.target, ast.Name) and n.target.id == NAME:
                    bad.append(f"{path.name}:{n.lineno} rebinds {NAME}")

    assert bad == [], "\n".join(bad)
