"""Fire-and-forget work must not be able to vanish.

`asyncio.create_task(...)` returns a task the loop holds only a WEAK reference
to. A bare call whose result nobody stores can be garbage-collected mid-await.
Where the coroutine owns a DB session that means the connection is never
checked in — the `SAWarning: ... non-checked-in connection` measured on the
canary (28 warnings / 38 turns on 2026-08-01, 24 more over the following two
days) — and it also means the work silently did not happen.

The load-bearing test here is `test_no_session_owning_bare_create_task`: it
scans the tree for the exact defect and fails on the pre-fix code, where 16
session-owning call sites used a bare `create_task`.

Direct invocation works:
    cd backend && ENVIRONMENT=development python tests/test_background_task_refs.py
"""
import ast
import asyncio
import logging
import os
import pathlib
import sys

os.environ.setdefault("ENVIRONMENT", "development")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services import background_tasks as bt  # noqa: E402

BACKEND = pathlib.Path(__file__).resolve().parents[1]


# ── the helper itself ─────────────────────────────────────────────

def test_spawn_holds_a_strong_reference_while_in_flight():
    async def _run():
        started = asyncio.Event()
        release = asyncio.Event()

        async def _work():
            started.set()
            await release.wait()
            return "done"

        task = bt.spawn(_work())
        await started.wait()
        # The caller drops its only reference; the module must still hold one.
        del task
        assert bt.pending_count() == 1, (
            "spawn() did not retain a strong reference — the task can be "
            "garbage-collected mid-await, which is the whole defect"
        )
        release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(_run())


def test_reference_is_released_when_the_task_finishes():
    """The strong-ref set must not grow without bound."""
    async def _run():
        before = bt.pending_count()
        t = bt.spawn(asyncio.sleep(0))
        await t
        await asyncio.sleep(0)
        assert bt.pending_count() == before, (
            f"leaked a reference: {bt.pending_count()} pending, expected {before}"
        )

    asyncio.run(_run())


def test_exception_is_logged_not_swallowed(caplog=None):
    """The three ad-hoc helpers this replaces never inspected the exception,
    so a background task that raised failed in total silence."""
    records = []

    class _Catch(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Catch()
    bt.logger.addHandler(handler)
    try:
        async def _run():
            async def _boom():
                raise ValueError("kaboom")

            t = bt.spawn(_boom(), name="boom-task")
            try:
                await t
            except ValueError:
                pass
            await asyncio.sleep(0)

        asyncio.run(_run())
    finally:
        bt.logger.removeHandler(handler)

    errors = [r for r in records if r.levelno >= logging.ERROR]
    assert errors, "a background task raised and nothing was logged"
    assert "kaboom" in errors[0].getMessage() or "ValueError" in errors[0].getMessage()


def test_cancellation_is_not_reported_as_an_error():
    """Shutdown cancels background work; that is normal, not an incident."""
    records = []

    class _Catch(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Catch()
    bt.logger.addHandler(handler)
    try:
        async def _run():
            t = bt.spawn(asyncio.sleep(10))
            await asyncio.sleep(0)
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            await asyncio.sleep(0)

        asyncio.run(_run())
    finally:
        bt.logger.removeHandler(handler)

    assert not [r for r in records if r.levelno >= logging.ERROR], (
        "cancellation was logged as an error; shutdown would spam the log"
    )


# ── the regression guard ──────────────────────────────────────────

def _session_owning_bare_create_tasks():
    """Every `asyncio.create_task(f(...))` used as a bare statement (result
    discarded) whose target function owns a DB session."""
    found = []
    for p in sorted((BACKEND / "app").rglob("*.py")):
        try:
            src = p.read_text(errors="ignore")
            tree = ast.parse(src)
        except SyntaxError:
            continue
        fns = {n.name: n for n in ast.walk(tree)
               if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            f = call.func
            if not (isinstance(f, ast.Attribute) and f.attr == "create_task"):
                continue
            if not call.args or not isinstance(call.args[0], ast.Call):
                continue
            g = call.args[0].func
            target = g.attr if isinstance(g, ast.Attribute) else (
                g.id if isinstance(g, ast.Name) else None)
            body = fns.get(target)
            if body is None:
                continue
            seg = ast.get_source_segment(src, body) or ""
            if "async_session_maker" in seg:
                found.append(f"{p.relative_to(BACKEND)}:{node.lineno} -> {target}()")
    return found


def test_no_session_owning_bare_create_task():
    """Fails on the pre-fix tree, which had 16 of these."""
    offenders = _session_owning_bare_create_tasks()
    assert not offenders, (
        "these fire-and-forget tasks own a DB session but are launched with a "
        "bare asyncio.create_task, so they can be garbage-collected mid-await "
        "and leak the connection (and silently not run). Use "
        "`from app.services.background_tasks import spawn`:\n  "
        + "\n  ".join(offenders)
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
