"""A session must not be used after its `async with` block has closed.

THE DEFECT THIS EXISTS FOR

Python does not unbind a `with` target at the end of the block, so this
compiles and runs:

    async with async_session_maker() as db:
        ...                      # 400 lines
    # block closed here
    await db.execute(...)        # 150 lines further down

and it is not an error, because **a closed AsyncSession is still usable**. It
quietly opens a new transaction and CHECKS OUT A FRESH CONNECTION that nothing
will ever return — the `async with` already ran its `__aexit__` and will not
run again. The connection is then reclaimed by the garbage collector, which is
where `SAWarning: ... non-checked-in connection` comes from.

This was the production connection leak. `agent_runner._run_inner` opened its
Phase 1 session at line 907, closed it at 1335, and read
`AgentConfig.preferred_provider` off it at 1485 — inside `try/except: pass`,
so the failure mode was silence. Measured at ~0.5 leaked connections per turn
on the canary 2026-08-03; that branch runs once per turn whenever the model is
auto-routed, which is the default.

Three earlier fixes (#407, #408, #418) each removed a real defect and none of
them was this one, because the GC emits the warning far from the cause.

    cd backend && ENVIRONMENT=development python -m pytest tests/test_session_use_after_close.py
"""
from __future__ import annotations

import ast
import asyncio
import gc
import os
import pathlib
import sys
import tempfile

os.environ.setdefault("ENVIRONMENT", "development")
BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

SESSION_FACTORIES = ("async_session_maker", "AsyncSessionLocal", "async_sessionmaker")


# ── the behaviour, proven rather than asserted ────────────────────────

def test_using_a_closed_session_leaks_a_connection():
    """Pin the mechanism itself, so the guard below has a reason to exist."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from app.db import pool_debug

    with tempfile.TemporaryDirectory() as tmp:
        # NullPool mirrors the agent: one fresh connection per session.
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp}/p.db", poolclass=NullPool)
        assert pool_debug.install(engine)
        maker = async_sessionmaker(engine)
        before = pool_debug.stats()

        async def _main():
            async with maker() as s:
                await s.execute(text("SELECT 1"))
            # The block has exited. The NAME is still bound, and the session
            # is still usable — that is the whole trap.
            assert s.is_active, "a closed AsyncSession is still usable; that is why this is silent"
            await s.execute(text("SELECT 1"))

        asyncio.run(_main())
        for _ in range(3):
            gc.collect()
        after = pool_debug.stats()

    assert after["checkouts"] - before["checkouts"] == 2, (
        "expected the closed session to take a SECOND connection"
    )
    assert after["checkins"] - before["checkins"] == 1, (
        "expected only one of the two to be returned"
    )
    assert after["abandoned"] > before["abandoned"], (
        "the second connection should have been reported abandoned"
    )


# ── the repo-wide guard ───────────────────────────────────────────────

def _offenders() -> list[str]:
    """Every load of a name that an `async with <session factory>() as N`
    bound, occurring AFTER that block ends and before N is rebound."""
    found: list[str] = []
    for path in sorted((BACKEND / "app").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            continue

        # Resolve per-file aliases. `from app.db.database import
        # async_session_maker as _asm` is common, and without this the alias's
        # `async with` is not recognised as a binding at all — which made the
        # first version of this guard report four false positives in
        # ws_realtime.py, blaming a block 700 lines earlier.
        factories = set(SESSION_FACTORIES)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.name in SESSION_FACTORIES and a.asname:
                        factories.add(a.asname)
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
                if node.value.id in factories:
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            factories.add(t.id)

        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue

            # name -> list of (kind, start, end). kind 'with' means the name is
            # only valid inside [start, end]; 'bind' means valid from start on.
            events: dict[str, list[tuple[str, int, int]]] = {}
            for node in ast.walk(fn):
                if isinstance(node, (ast.AsyncWith, ast.With)):
                    for item in node.items:
                        call = item.context_expr
                        if not isinstance(call, ast.Call):
                            continue
                        f = call.func
                        fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                        if not isinstance(item.optional_vars, ast.Name):
                            continue
                        # Record EVERY with-binding so a later rebinding (even
                        # by a name this guard does not recognise) still counts
                        # as the nearest preceding binding. Only session
                        # factories are flaggable.
                        kind = "session-with" if fname in factories else "bind"
                        events.setdefault(item.optional_vars.id, []).append(
                            (kind, node.lineno, node.end_lineno or node.lineno)
                        )
                elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for t in targets:
                        if isinstance(t, ast.Name):
                            events.setdefault(t.id, []).append(("bind", node.lineno, node.lineno))
                elif isinstance(node, ast.arg):
                    events.setdefault(node.arg, []).append(("bind", fn.lineno, fn.lineno))

            for name, evs in events.items():
                if not any(k == "session-with" for k, _, _ in evs):
                    continue
                evs.sort(key=lambda e: e[1])
                for use in ast.walk(fn):
                    if not (isinstance(use, ast.Name) and use.id == name
                            and isinstance(use.ctx, ast.Load)):
                        continue
                    prior = [e for e in evs if e[1] <= use.lineno]
                    if not prior:
                        continue
                    kind, start, end = prior[-1]
                    if kind == "session-with" and use.lineno > end:
                        found.append(
                            f"{path.relative_to(BACKEND)}:{use.lineno} uses `{name}`, "
                            f"whose `async with` closed at line {end}"
                        )
    return sorted(set(found))


def test_no_session_is_used_after_its_block_closed():
    """Fails on the pre-fix tree, where agent_runner.py:1485 was the leak."""
    offenders = _offenders()
    assert not offenders, (
        "these use a DB session after its `async with` block closed. Python "
        "keeps the name bound, and a closed AsyncSession is still usable — so "
        "this silently checks out a connection nobody will ever return. Give "
        "the read its own short-lived session:\n  " + "\n  ".join(offenders)
    )


# ── the second shape: a request session captured by a streaming body ──

def _streaming_offenders() -> list[str]:
    """Generators handed to `StreamingResponse` that use the request's session.

    Different shape, same defect. Since FastAPI 0.106 the dependency
    `AsyncExitStack` is exited BEFORE the response body streams, so a
    `Depends(get_db)` session is already closed by the time the generator runs
    — and a closed session still works, so every query re-checks-out an
    ownerless connection. The codebase already knew this: `llm_proxy.py` carries
    a ten-line comment saying exactly that, and fixes it there. `chat.py` was
    the unfixed twin, plus a byte-identical dormant copy in
    `app/modules/chat/router.py`.

    The `async with ... as N` guard above cannot see this one: the session never
    came from a `with` block in the first place.
    """
    hits: list[str] = []
    for path in sorted((BACKEND / "app").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            streamed = set()
            for n in ast.walk(fn):
                if (isinstance(n, ast.Call)
                        and getattr(n.func, "id", getattr(n.func, "attr", None)) == "StreamingResponse"
                        and n.args and isinstance(n.args[0], ast.Call)):
                    g = n.args[0].func
                    nm = getattr(g, "id", getattr(g, "attr", None))
                    if nm:
                        streamed.add(nm)
            if not streamed:
                continue
            sess = {a.arg for a in fn.args.args + fn.args.kwonlyargs
                    if a.annotation and "Session" in ast.unparse(a.annotation)}
            if not sess:
                continue
            for inner in ast.walk(fn):
                if inner is fn or not isinstance(inner, ast.AsyncFunctionDef):
                    continue
                if inner.name not in streamed:
                    continue
                shadow = {a.arg for a in inner.args.args + inner.args.kwonlyargs}
                used = {n.id for n in ast.walk(inner)
                        if isinstance(n, ast.Name) and n.id in sess
                        and isinstance(n.ctx, ast.Load)} - shadow
                # follow one level of delegation to a sibling helper
                for called in {getattr(c.func, "id", None) for c in ast.walk(inner)
                               if isinstance(c, ast.Call)}:
                    for sib in ast.walk(fn):
                        if isinstance(sib, ast.AsyncFunctionDef) and sib.name == called:
                            sib_shadow = {a.arg for a in sib.args.args}
                            used |= {n.id for n in ast.walk(sib)
                                     if isinstance(n, ast.Name) and n.id in sess
                                     and isinstance(n.ctx, ast.Load)} - sib_shadow
                if used:
                    hits.append(
                        f"{path.relative_to(BACKEND)}:{inner.lineno} `{inner.name}` "
                        f"streams while using the request session {sorted(used)}"
                    )
    return sorted(set(hits))


def test_no_streaming_body_uses_the_request_session():
    """Fails on the pre-fix tree, where chat.py:316 was the offender."""
    offenders = _streaming_offenders()
    assert not offenders, (
        "these hand a generator to StreamingResponse that queries the "
        "request's `Depends(get_db)` session. FastAPI closes that session "
        "BEFORE the body streams, and a closed AsyncSession still works — so "
        "each query silently checks out a connection nobody will return. Open "
        "a fresh `async_session_maker()` inside the generator (see the note in "
        "app/api/llm_proxy.py):\n  " + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
