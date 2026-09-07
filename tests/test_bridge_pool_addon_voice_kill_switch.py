"""`BRIDGE_VOICE_TASKS_FORCE_OFF` is dead code on the deployed bridge.

`bridge/pool_addon.py::_feature_flag_env_args` carries a branch its own
comment calls "an explicit emergency rollback switch for operators who need
to force the fleet dark without reverting the image". It cannot fire. The
branch lives inside `for k in _FEATURE_FLAG_ENVS:` and `VOICE_TASKS_ENABLED`
is not in that tuple, so the loop never visits the key and the comparison is
never evaluated.

Verified against production on 2026-09-07 (read-only): the installed
`/opt/toup-bridge/pool_addon.py` is md5 f49aff272b45bd676e8a3e0088191a2b,
byte-identical to origin/main e698ea78, and `grep '"VOICE_TASKS'` on it hits
only the function body at lines 716-724 — never the tuple. `git log -S` for
the tuple entry on main is empty; only the voice branch
(codex/mobile-voice-agent-runtime) has it, as a 3-line addition to the same
tuple, and that branch is not what is installed.

Why it matters: the fleet image (7edaed3ab644) bakes
`ENV VOICE_TASKS_ENABLED=true VOICE_TASKS_FORCE_ON=true` and its Settings
validator forces the flag on. With no delivery path for a `0`, the ONLY way
to stop the voice supervisor on 61 containers is to build a new image and
walk every slot — 93-159 s per slot, 2-3 hours — which is the operation
that caused the 2026-09-06 incident in the first place.

This file EXECUTES the real function rather than modelling it, extracting it
and its tuple by AST the way `test_pool_upgrade_fairness.py` does for
`order_upgrade_candidates` — pool_addon.py is bridge-host code and importing
it whole drags in a fake `main` module and a live event loop.

Run:
    cd backend && PYTHONPATH=. pytest tests/test_bridge_pool_addon_voice_kill_switch.py
"""
from __future__ import annotations

import ast
import os
import pathlib
from typing import List

import pytest

BRIDGE = pathlib.Path(__file__).resolve().parents[2] / "bridge" / "pool_addon.py"

VOICE_FLAG = "VOICE_TASKS_ENABLED"
FORCE_OFF = "BRIDGE_VOICE_TASKS_FORCE_OFF"


@pytest.fixture(scope="module")
def flag_args():
    """The REAL `_feature_flag_env_args`, with the REAL `_FEATURE_FLAG_ENVS`."""
    tree = ast.parse(BRIDGE.read_text("utf-8"))
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "_FEATURE_FLAG_ENVS" for t in node.targets
        ):
            wanted.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_feature_flag_env_args":
            wanted.append(node)
    assert len(wanted) == 2, (
        f"expected _FEATURE_FLAG_ENVS and _feature_flag_env_args at module "
        f"level, found {len(wanted)} — this harness is no longer reading the "
        f"real code"
    )
    ns: dict = {"os": os, "List": List}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(BRIDGE), "exec"), ns)
    assert len(ns["_FEATURE_FLAG_ENVS"]) >= 5, "the tuple did not come through"
    return ns["_feature_flag_env_args"], ns["_FEATURE_FLAG_ENVS"]


def _pairs(args: List[str]) -> dict:
    out = {}
    for i, a in enumerate(args):
        if a == "-e" and i + 1 < len(args) and "=" in args[i + 1]:
            k, v = args[i + 1].split("=", 1)
            out[k] = v
    return out


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(VOICE_FLAG, raising=False)
    monkeypatch.delenv(FORCE_OFF, raising=False)


def test_the_force_off_switch_actually_reaches_a_container(flag_args, monkeypatch):
    """The falsifier. This is what an operator does in an emergency: set the
    two variables on the bridge env, restart, and let the next spawn or
    blue-green upgrade carry `-e VOICE_TASKS_ENABLED=0` into the container."""
    fn, _ = flag_args
    monkeypatch.setenv(VOICE_FLAG, "0")
    monkeypatch.setenv(FORCE_OFF, "1")
    pairs = _pairs(fn())
    assert pairs.get(VOICE_FLAG) == "0", (
        f"BRIDGE_VOICE_TASKS_FORCE_OFF=1 emitted {pairs.get(VOICE_FLAG)!r} for "
        f"{VOICE_FLAG}. The documented emergency rollback for the voice "
        f"runtime does not exist: with the image forcing the flag on, the "
        f"only remaining lever is an image rebuild plus a full fleet walk."
    )


def test_a_stale_zero_alone_does_not_darken_the_fleet(flag_args, monkeypatch):
    """The counterweight, and the reason the branch was written.

    `VOICE_TASKS_ENABLED=0` has been sitting on the bridge env since
    14:33Z on 2026-09-06. The image now defaults the supervisor ON, so
    forwarding that leftover would silently darken every container the
    bridge creates while the platform believes the feature is live. Only
    the explicit FORCE_OFF turns a stale 0 into an instruction.
    """
    fn, _ = flag_args
    monkeypatch.setenv(VOICE_FLAG, "0")
    pairs = _pairs(fn())
    assert VOICE_FLAG not in pairs, (
        f"a bare {VOICE_FLAG}=0 on the bridge env emitted "
        f"{pairs.get(VOICE_FLAG)!r} — the image default must win unless an "
        f"operator sets {FORCE_OFF}"
    )


@pytest.mark.parametrize("falsey", ["0", "false", "FALSE", "off", "no", " No "])
def test_every_falsey_spelling_is_covered(flag_args, monkeypatch, falsey):
    fn, _ = flag_args
    monkeypatch.setenv(VOICE_FLAG, falsey)
    assert VOICE_FLAG not in _pairs(fn())
    monkeypatch.setenv(FORCE_OFF, "1")
    assert _pairs(fn())[VOICE_FLAG] == falsey


def test_an_explicit_on_is_still_forwarded(flag_args, monkeypatch):
    """FORCE_OFF is a kill switch, not a mode: a `1` on the bridge env is an
    ordinary flag value and travels as one."""
    fn, _ = flag_args
    monkeypatch.setenv(VOICE_FLAG, "1")
    assert _pairs(fn())[VOICE_FLAG] == "1"


def test_unset_forwards_nothing(flag_args):
    """The state production is in today, and must stay in: nothing injected,
    so the image's own default decides."""
    fn, _ = flag_args
    assert VOICE_FLAG not in _pairs(fn())


def test_the_key_is_in_the_managed_tuple(flag_args):
    """Membership is not cosmetic, and it is the half that fixes the
    DEDICATED containers too.

    Both recreate paths strip `_FEATURE_FLAG_ENVS` members from the cloned
    env before re-injecting the current bridge values — pool_addon's
    blue-green at the `if k in _FEATURE_FLAG_ENVS: continue` line, and the
    hand-installed host main.py via
    `from pool_addon import _FEATURE_FLAG_ENVS as _managed_flags`. A key
    that is not in the tuple is never stripped, so whatever value a
    container was BORN with fossilises through every upgrade: all 7
    dedicated containers currently carry `VOICE_TASKS_ENABLED=0` from the
    voice-branch bridge that created them on 2026-09-06, a value no
    bridge-env edit can clear.
    """
    _, flags = flag_args
    assert VOICE_FLAG in flags, (
        f"{VOICE_FLAG} is not in _FEATURE_FLAG_ENVS, so the FORCE_OFF branch "
        f"inside `for k in _FEATURE_FLAG_ENVS` is unreachable and neither "
        f"recreate path strips the cloned value"
    )
