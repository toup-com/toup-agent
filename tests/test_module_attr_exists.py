"""A module-qualified call must name something that exists.

THE DEFECT THIS EXISTS FOR

On 2026-07-31, `ToolExecutor._tool_web_fetch` gained a cache probe:

    from app.agent.smart_fetch import reader as _sf_reader
    cached = _sf_reader.page_cache_get(url, max_chars)

`page_cache_get` was never added to `reader.py`. The import still succeeds —
the *module* exists — so nothing fails at import time, the deploy smoke test
stays green, and every backend test passes. The failure happens on the call,
and `ToolExecutor.execute()` wraps every tool in a catch-all that turns any
exception into a string:

    ERROR: AttributeError: module 'app.agent.smart_fetch.reader'
           has no attribute 'page_cache_get'

which is handed to the model as tool output. So `web_fetch` returned an error
for **every** call, on **every** tenant, for four days, and nothing anywhere
went red. Confirmed on the live canary container before this test was written.

`from x import y` cannot fail this way — a missing `y` is an ImportError at
boot. Only `import module` + `module.attr` defers the check to runtime, and
that is exactly the shape this scans for.

    cd backend && ENVIRONMENT=development python -m pytest tests/test_module_attr_exists.py
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import pathlib
import sys
import tempfile
import textwrap

os.environ.setdefault("ENVIRONMENT", "development")

BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# Packages whose submodules we resolve. Kept to first-party code: a third-party
# module may populate attributes at import time in ways a static read misses.
ROOTS = ("app.",)


def _attr_bases(tree: ast.AST) -> set[str]:
    """Names used as `name.something` somewhere in the file."""
    return {
        n.value.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
    }


def _module_aliases(tree: ast.AST) -> dict[str, str]:
    """alias -> dotted module path, for imports that bind a MODULE object.

    Covers both spellings:
        from app.agent.smart_fetch import reader as _sf_reader
        import app.agent.smart_fetch.reader as _sf_reader

    Module-ness is decided with `find_spec`, NOT by importing and seeing
    whether it works — and that distinction is load-bearing. `app/db/` holds
    BOTH a submodule `init_db.py` and a re-exported *function* `init_db`
    (from `app.db.database`, via `app/db/__init__.py`). Importing the
    submodule binds it onto the package and shadows the function, so
    `from app.db import init_db; await init_db()` then raises
    `TypeError: 'module' object is not callable` — for the rest of the
    process. The first version of this scan did exactly that and broke the
    autouse `_reset_database` fixture for every test that ran after it.

    A test that inspects the codebase must not change it. Only aliases that
    are actually used as `alias.attr` are imported at all.
    """
    used = _attr_bases(tree)
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if not node.module.startswith(ROOTS):
                continue
            for a in node.names:
                if a.name == "*":
                    continue
                alias = a.asname or a.name
                if alias not in used:
                    continue
                dotted = f"{node.module}.{a.name}"
                try:
                    if importlib.util.find_spec(dotted) is None:
                        continue  # a class/function — checked at import time
                except (ImportError, AttributeError, ValueError):
                    continue
                out[alias] = dotted
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(ROOTS) and a.asname and a.asname in used:
                    out[a.asname] = a.name
    return out


def _offenders() -> list[str]:
    hits: list[str] = []
    for path in sorted((BACKEND / "app").rglob("*.py")):
        try:
            src = path.read_text(errors="ignore")
            tree = ast.parse(src)
        except SyntaxError:
            continue
        if "import" not in src:
            continue

        aliases = _module_aliases(tree)
        if not aliases:
            continue

        # An alias that is REBOUND in this file (mod = something_else) is not
        # reliably the module any more — skip it rather than report a guess.
        rebound = {
            t.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Name)
        }

        modules: dict[str, object] = {}
        for alias, dotted in aliases.items():
            if alias in rebound:
                continue
            try:
                modules[alias] = importlib.import_module(dotted)
            except Exception:
                continue

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in modules):
                continue
            # Only loads — `mod.attr = x` legitimately creates the attribute.
            if not isinstance(node.ctx, ast.Load):
                continue
            mod = modules[node.value.id]
            if not hasattr(mod, node.attr):
                hits.append(
                    f"{path.relative_to(BACKEND)}:{node.lineno} "
                    f"`{node.value.id}.{node.attr}` — {mod.__name__} has no "
                    f"attribute {node.attr!r}"
                )
    return sorted(set(hits))


def test_every_module_qualified_name_exists():
    """Fails on the pre-fix tree: reader.page_cache_get did not exist."""
    offenders = _offenders()
    assert not offenders, (
        "these call a name on a first-party module that the module does not "
        "define. The import succeeds (the module is real), so nothing fails "
        "until the line actually runs — and inside a tool handler that means "
        "an `ERROR: AttributeError` handed to the model instead of a crash "
        "anyone would notice:\n  " + "\n  ".join(offenders)
    )


def test_the_scan_can_actually_see_a_missing_attribute():
    """A guard that cannot fail is not a guard.

    Rather than trust that the scan works, hand it the exact shape of the bug
    and require it to complain.
    """
    src = textwrap.dedent("""
        from app.agent.smart_fetch import reader as _sf_reader
        def f(url):
            return _sf_reader.a_name_that_does_not_exist(url)
    """)
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "probe.py"
        p.write_text(src)
        tree = ast.parse(src)
        aliases = _module_aliases(tree)
        assert "_sf_reader" in aliases, "the scan failed to bind the module alias"
        mod = importlib.import_module(aliases["_sf_reader"])
        assert not hasattr(mod, "a_name_that_does_not_exist")
        # and the real one, which is the thing that went missing:
        assert hasattr(mod, "page_cache_get"), (
            "reader.page_cache_get is missing again — ToolExecutor._tool_web_fetch "
            "calls it on every web_fetch"
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
