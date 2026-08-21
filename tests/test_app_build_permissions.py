"""Round 15 — the EACCES/exit-243 install failure, and the repair.

The observed failure, verbatim from a user's chat on 2026-08-20::

    ❌ Build failed: Nokia Snake Classic — npm install failed (stale):
    npm install failed (exit 243): npm error code EACCES

Reproduced on this machine (2026-08-21) by running the real `npm install` in a
directory the calling uid cannot write::

    npm error   code: 'EACCES',
    npm error   syscall: 'mkdir',
    npm error   path: …/node_modules
    exit 243

So the cause is not npm and not the registry: it is that the agent writes an
app's files as ROOT and then runs npm dropped to uid 1000. Two mechanisms put
it there, and both are covered here:

  1. `preexec_fn` sets HOME for the child — except every caller ALSO passes
     `env=`, and CPython hands that dict to execve directly, so the mutation
     is discarded. npx then used /root/.npm as its cache and failed before it
     started, which is what pushed scaffolding into the root-owned fallback
     branch. (`test_sandbox_environ_*`)
  2. Nothing widened the app directory afterwards, so npm's first act —
     `mkdir node_modules` — hit EACCES. (`test_npm_*`)

No test here mocks npm's exit code: the install tests run a real subprocess
against a real unwritable directory through the real `_npm_install_clean`,
with a stub `npm` on PATH that does what npm does (mkdir node_modules) and
reports what npm reports.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys

import pytest

from app.agent.app_manager import AppManager
from app.services import exec_env


# ═════════════════════════════════════════════════════════════════════
# 1. The lost HOME
# ═════════════════════════════════════════════════════════════════════

def test_preexec_env_mutation_is_lost_when_env_is_passed(tmp_path):
    """The mechanism, proven rather than asserted.

    This is the whole reason `sandbox_environ` exists. If CPython ever made
    `env=` layer over the child's mutated `os.environ`, this test would fail
    and the helper could go.
    """
    def drop() -> None:
        os.environ["HOME"] = str(tmp_path / "sandbox")

    with_env = subprocess.run(
        [sys.executable, "-c", "import os;print(os.environ.get('HOME'))"],
        env={**os.environ, "HOME": "/root"},
        preexec_fn=drop, capture_output=True, text=True,
    ).stdout.strip()
    without_env = subprocess.run(
        [sys.executable, "-c", "import os;print(os.environ.get('HOME'))"],
        preexec_fn=drop, capture_output=True, text=True,
    ).stdout.strip()

    assert with_env == "/root", (
        "env= no longer wins over a preexec os.environ write — sandbox_environ "
        "may be removable, but check every caller first"
    )
    assert without_env == str(tmp_path / "sandbox")


def test_sandbox_environ_sets_home_to_the_uid_it_drops_to(monkeypatch):
    import getpass
    import pwd

    me = getpass.getuser()
    monkeypatch.setattr(exec_env.settings, "exec_sandbox_user", me, raising=False)
    env = exec_env.sandbox_environ({"HOME": "/root", "PATH": "/usr/bin"})

    assert env["HOME"] == pwd.getpwnam(me).pw_dir
    assert env["USER"] == me and env["LOGNAME"] == me
    assert env["PATH"] == "/usr/bin", "scrubbing must not disturb unrelated vars"


def test_sandbox_environ_leaves_home_alone_when_nothing_is_dropped(monkeypatch):
    """No drop → the child runs as the agent, whose own HOME is correct.
    Rewriting it would break the un-hardened deployment instead."""
    monkeypatch.setattr(exec_env.settings, "exec_sandbox_user", "", raising=False)
    assert exec_env.sandbox_environ({"HOME": "/root"})["HOME"] == "/root"


def test_sandbox_environ_survives_an_unknown_user(monkeypatch):
    monkeypatch.setattr(exec_env.settings, "exec_sandbox_user",
                        "no_such_user_xyzzy", raising=False)
    assert exec_env.sandbox_environ({"HOME": "/root"})["HOME"] == "/root"


# ═════════════════════════════════════════════════════════════════════
# 2. Classification
# ═════════════════════════════════════════════════════════════════════

# The real tail from the reproduction, trimmed to what the classifier sees.
REAL_EACCES_TAIL = (
    "npm install failed (exit 243): npm error code EACCES\n"
    "npm error syscall mkdir\n"
    "npm error path /opt/toup-agent/apps/nokia-snake-classic/node_modules\n"
    "npm error errno -13\n"
    "npm error The operation was rejected by your operating system."
)


def test_a_permission_fault_is_not_classified_stale():
    """`stale` was the WRONG bucket and, worse, a non-retried one: the fleet's
    dominant install failure was diagnosed, never repaired, and shown raw."""
    assert AppManager._classify_install_error(REAL_EACCES_TAIL) == "permissions"


@pytest.mark.parametrize("text,expected", [
    ("npm error code EPERM, operation not permitted", "permissions"),
    ("Your cache folder contains root-owned files", "permissions"),
    ("npm error code ENOENT no such file", "stale"),
    ("npm error code ENOSPC no space left on device", "disk"),
    ("npm error code ERESOLVE unable to resolve dependency tree", "bad_dep"),
    ("npm error network ETIMEDOUT", "transient"),
])
def test_classifier_routes_each_class(text, expected):
    assert AppManager._classify_install_error(text) == expected


# ═════════════════════════════════════════════════════════════════════
# 3. The repair
# ═════════════════════════════════════════════════════════════════════

def _mode(p) -> int:
    return stat.S_IMODE(os.stat(p).st_mode)


def test_repair_permissions_widens_the_app_tree(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agent.app_manager.NPM_CACHE_DIR",
                        str(tmp_path / "cache"), raising=False)
    app_dir = tmp_path / "app"
    (app_dir / "src").mkdir(parents=True)
    (app_dir / "package.json").write_text("{}")
    (app_dir / "src" / "App.tsx").write_text("x")
    # node_modules is npm's own and is skipped — walking 27k files to chmod
    # them is the cost this pipeline exists to avoid.
    (app_dir / "node_modules" / "react").mkdir(parents=True)
    (app_dir / "node_modules" / "react" / "index.js").write_text("y")
    os.chmod(app_dir / "node_modules" / "react" / "index.js", 0o600)
    os.chmod(app_dir / "package.json", 0o600)
    os.chmod(app_dir / "src", 0o700)
    os.chmod(app_dir, 0o755)

    summary = AppManager.repair_permissions(str(app_dir))

    assert _mode(app_dir) == 0o777
    assert _mode(app_dir / "src") == 0o777
    assert _mode(app_dir / "package.json") == 0o666
    assert _mode(app_dir / "node_modules" / "react" / "index.js") == 0o600, \
        "node_modules must be skipped — it is npm's output, not ours"
    assert summary["changed"] >= 3 and not summary["truncated"]
    assert _mode(tmp_path / "cache") == 0o777, "the shared npm cache too"


def test_ensure_npm_cache_widens_an_existing_root_owned_cache(tmp_path, monkeypatch):
    """The cache is shared across UIDs. Whoever ran npm first owns it; if that
    was root, every later install as uid 1000 dies on it."""
    cache = tmp_path / "cache"
    (cache / "_cacache").mkdir(parents=True)
    os.chmod(cache / "_cacache", 0o700)
    os.chmod(cache, 0o700)
    monkeypatch.setattr("app.agent.app_manager.NPM_CACHE_DIR", str(cache), raising=False)

    from app.agent.app_manager import _ensure_npm_cache
    assert _ensure_npm_cache() == str(cache)
    assert _mode(cache) == 0o777 and _mode(cache / "_cacache") == 0o777


# ═════════════════════════════════════════════════════════════════════
# 4. End to end through the real subprocess plumbing
# ═════════════════════════════════════════════════════════════════════

# A stand-in for npm that fails the way npm fails: it tries the one thing npm
# tries first, and on refusal prints npm's words and exits with npm's code.
# Everything else in `_npm_install_clean` — env, cwd, preexec, the exit-code
# read, the tail capture — is the production path.
NPM_STUB = """#!/bin/sh
if mkdir node_modules 2>/dev/null; then
  echo "added 1 package"
  exit 0
fi
echo "npm error code EACCES" >&2
echo "npm error syscall mkdir" >&2
echo "npm error path $PWD/node_modules" >&2
exit 243
"""


@pytest.fixture()
def npm_stub(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    npm = bin_dir / "npm"
    npm.write_text(NPM_STUB)
    npm.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    monkeypatch.setattr("app.agent.app_manager.NPM_CACHE_DIR",
                        str(tmp_path / "cache"), raising=False)
    return npm


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anything")
async def test_install_recovers_from_an_unwritable_app_dir(tmp_path, npm_stub):
    """The whole failure and the whole fix, in one pass.

    The directory starts unwritable — the state `scaffold_app`'s root-owned
    fallback branch left behind. Before round 15 this raised straight out to
    `_fail_job` as "npm install failed (stale): … exit 243".

    Mutation note: TWO layers independently rescue this — the pre-install
    `_share(app_dir)` and the classify→repair→retry loop — so removing either
    one alone leaves the test green. That is deliberate defence in depth, not
    a hole. Removing BOTH fails it with the production error verbatim
    (verified 2026-08-21).
    """
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "package.json").write_text('{"name":"a","version":"1.0.0"}')
    os.chmod(app_dir, 0o555)

    mgr = AppManager.__new__(AppManager)  # no __init__: it touches APPS_DIR
    out = await mgr._npm_install_with_retry("app1234", str(app_dir))

    assert "added 1 package" in out
    assert (app_dir / "node_modules").is_dir()


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anything")
async def test_the_unrepaired_pipeline_still_fails_243(tmp_path, npm_stub, monkeypatch):
    """The control. With the repair disabled, the SAME setup reproduces the
    exact production error — so the test above is measuring the fix and not
    an environment that was writable all along."""
    monkeypatch.setattr("app.agent.app_manager._share", lambda p: None)
    monkeypatch.setattr("app.agent.app_manager._share_tree", lambda p: {})
    monkeypatch.setattr(AppManager, "repair_permissions", staticmethod(lambda d: {}))

    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "package.json").write_text('{"name":"a","version":"1.0.0"}')
    os.chmod(app_dir, 0o555)

    mgr = AppManager.__new__(AppManager)
    with pytest.raises(RuntimeError) as exc:
        await mgr._npm_install_with_retry("app1234", str(app_dir))

    assert "exit 243" in str(exc.value) and "EACCES" in str(exc.value)
    assert AppManager._classify_install_error(str(exc.value)) == "permissions"
