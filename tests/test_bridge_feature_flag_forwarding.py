"""Every flag the bridge forwards must be a flag the agent actually reads.

`bridge/pool_addon.py::_FEATURE_FLAG_ENVS` is the single list of env vars
copied from the bridge process into each container it creates. It is the
only thing standing between "a flag exists in Settings" and "a container
actually receives it", and it has drifted before:

  * 2026-07-26 — a rollout recreated spares WITHOUT STABLE_PREFIX_LAYOUT
    because the recreate path cloned the old container's pre-flag env.
    That is what `_FEATURE_FLAG_ENVS` was introduced to prevent.
  * 2026-08-05 — two ASSIGNED tenants (the canary and the founder's own
    account) were found running without STABLE_PREFIX_LAYOUT and, for one
    of them, without PROMPT_DIET. Different root cause — assigned tenants
    take env from a per-tenant .env written at provision time, which this
    list does not feed — but the same class of failure: a flag that exists
    in code and reaches nobody, silently, with nothing red anywhere.

The systemic fix for that second case was to make the flags default ON
(so a missing env var is no longer a silent regression). This guard covers
the remaining half: a name in the bridge list that does not correspond to
a real Settings field forwards nothing and would never be noticed, because
an env var no Settings field reads is simply ignored.

Run:
    cd backend && RUN_MODE=agent PYTHONPATH=. pytest tests/test_bridge_feature_flag_forwarding.py
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from app.config import Settings


BRIDGE = pathlib.Path(__file__).resolve().parents[2] / "bridge" / "pool_addon.py"


def _forwarded_flags() -> list[str]:
    """The literal contents of `_FEATURE_FLAG_ENVS`, read via AST.

    Parsed rather than imported: pool_addon.py is bridge-host code with
    imports (docker, psql helpers) that do not exist in the backend test
    environment.
    """
    tree = ast.parse(BRIDGE.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == "_FEATURE_FLAG_ENVS" for t in node.targets):
            continue
        return [
            el.value for el in node.value.elts          # type: ignore[attr-defined]
            if isinstance(el, ast.Constant) and isinstance(el.value, str)
        ]
    raise AssertionError("_FEATURE_FLAG_ENVS not found in bridge/pool_addon.py")


def test_the_list_is_not_empty():
    """Anti-vacuity control.

    Every assertion below iterates the parsed list. If the parse silently
    returned [], all of them would pass while proving nothing — the exact
    `all([]) is True` trap that has produced green-but-empty tests in this
    suite before.
    """
    flags = _forwarded_flags()
    assert len(flags) >= 5, f"only parsed {len(flags)} flags: {flags}"


@pytest.mark.parametrize("flag", _forwarded_flags())
def test_forwarded_flag_maps_to_a_real_settings_field(flag: str):
    """A forwarded env var with no Settings field is dead weight.

    pydantic-settings maps FOO_BAR -> settings.foo_bar. If the field does
    not exist the container receives an env var nothing reads, so the
    feature it was meant to enable stays off with no error anywhere.
    """
    field = flag.lower()
    assert field in Settings.model_fields, (
        f"bridge forwards {flag} but there is no settings.{field} — the "
        f"container gets an env var nothing reads, so the flag does nothing"
    )


@pytest.mark.parametrize("flag", [
    "STABLE_PREFIX_LAYOUT",
    "PROMPT_DIET",
    "CHANNEL_CONVERGE",
    "CACHE_AWARE_OVERFLOW",
    "EMBEDDINGS_VIA_PROXY",
    "METERING_CORRECTNESS_V2",
])
def test_flag_that_must_stay_forwardable_is_still_in_the_list(flag: str):
    """Dropping one of these from the bridge list is how a fleet silently
    loses a feature — or, for METERING_CORRECTNESS_V2, loses the ability to
    turn a gated one ON when it is finally approved."""
    assert flag in _forwarded_flags(), (
        f"{flag} left _FEATURE_FLAG_ENVS — containers can no longer receive it"
    )


def test_embeddings_via_proxy_is_forwarded_because_memory_hard_fails_without_it():
    """Called out separately because its failure mode is the worst one here.

    Without EMBEDDINGS_VIA_PROXY a bundle tenant has no embedding path at
    all (the local sentence-transformers fallback is not in the agent
    image), so memory WRITES hard-fail. That is persistent memory silently
    not persisting.
    """
    assert "EMBEDDINGS_VIA_PROXY" in _forwarded_flags()
    assert "embeddings_via_proxy" in Settings.model_fields
