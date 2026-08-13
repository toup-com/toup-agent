"""G-19a PR-B: the relay can ask the agent for voice's instructions.

Ships dark. `voice_context_from_agent` OFF means the legacy platform-side
builder is still what a session gets; `voice_context_shadow` (ON) calls the
agent too, compares the two section-by-section, logs the result, and serves
the legacy string regardless.

The property that matters most is the one a flag test usually forgets: a
Realtime session must NEVER open with no instructions. That is the
2026-07-31 shape — every voice session ran with no persona and no brain, and
the prompt looked exactly like a new user's. So every failure mode here
falls back rather than propagating.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from app.config import settings

_WS_REALTIME = Path(__file__).resolve().parents[1] / "app" / "api" / "ws_realtime.py"


# ── Flag defaults ────────────────────────────────────────────────────────


def test_ships_dark():
    """PR-B must not change what any live session gets on merge."""
    from app.config import Settings

    assert Settings.model_fields["voice_context_from_agent"].default is False, (
        "the agent path must ship OFF — flipping it changes what every voice "
        "session hears, and that needs a canary, not a merge"
    )
    assert Settings.model_fields["voice_context_shadow"].default is True, (
        "shadow is the whole point of landing this dark: compare on real "
        "traffic before anything depends on it"
    )


# ── Structure of _instructions_step ──────────────────────────────────────
#
# The step is a closure inside the WS handler, so it cannot be imported and
# driven directly without standing up a socket. These assertions are made on
# its AST rather than its source text, so they describe control flow rather
# than phrasing — a rename or a reflow does not fake them out.


def _instructions_step_ast() -> ast.AST:
    tree = ast.parse(_WS_REALTIME.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_instructions_step":
            return node
    raise AssertionError("_instructions_step not found in ws_realtime.py")


def test_legacy_builder_still_runs_and_is_the_default_return():
    """Whatever else happens, the legacy builder is called and its result is
    what a shadow run serves."""
    node = _instructions_step_ast()
    src = ast.unparse(node)
    assert "build_realtime_instructions(" in src, (
        "the legacy builder is no longer called — this PR is supposed to be "
        "additive until the canary says otherwise"
    )
    # Every `return` in the function must be able to yield the legacy value
    # or the agent value; a bare `return None` on a failure path would be the
    # no-instructions regression.
    returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
    assert returns, "no returns found"
    names = {ast.unparse(r.value) for r in returns if r.value is not None}
    assert "legacy" in names, "no path returns the legacy instructions"


def test_agent_result_is_only_served_behind_the_flag():
    node = _instructions_step_ast()
    src = ast.unparse(node)
    # The gate moved into `_agent_ctx_enabled_for` (module level, so the
    # per-user canary allowlist is unit-testable). The step must call it,
    # and the helper must read BOTH the global flag and the allowlist —
    # a helper that read neither would be an ungated fleet flip.
    assert "_agent_ctx_enabled_for(user_id)" in src, (
        "the agent path is not gated on its flag helper"
    )
    import inspect as _inspect

    import app.api.ws_realtime as _rt

    helper_src = _inspect.getsource(_rt._agent_ctx_enabled_for)
    assert "settings.voice_context_from_agent" in helper_src, (
        "the gate helper does not read the global flag"
    )
    assert "voice_context_from_agent_user_ids" in helper_src, (
        "the gate helper does not read the canary allowlist"
    )
    assert "settings.voice_context_shadow" in src, "shadow mode is not wired"
    # In shadow mode the agent string must not be returned. Walk the actual
    # `if want_shadow:` subtree rather than slicing text — an earlier version
    # of this test sliced past the branch and read the enabled path's return,
    # which made it fail for a reason that was not the property.
    shadow_if = None
    for n in ast.walk(node):
        if isinstance(n, ast.If) and ast.unparse(n.test).strip() == "want_shadow":
            shadow_if = n
            break
    assert shadow_if is not None, "no `if want_shadow:` branch found"
    for r in [x for x in ast.walk(shadow_if) if isinstance(x, ast.Return)]:
        assert ast.unparse(r.value or ast.Constant(None)) != "agent_instr", (
            "shadow mode returned the AGENT's instructions — shadow must "
            "compare and serve legacy, or it is not a shadow"
        )


def test_agent_failure_falls_back_rather_than_returning_nothing():
    node = _instructions_step_ast()
    src = ast.unparse(node)
    tail = src[src.index("if agent_instr"):]
    assert "return legacy" in tail, (
        "when the agent call fails on the enabled path there is no fallback "
        "to the legacy builder — a session would open with no instructions"
    )


# ── The comparison log must not leak the thing it compares ───────────────


def test_shadow_log_carries_fingerprints_not_content():
    """The sections being compared are the user's persona, both brains and
    the day transcript. The log line gets counts and digests.

    The comparator moved to module level on 2026-08-12 (so the offline
    parity harness imports the same judge the shadow uses instead of
    keeping a second copy), which is why this reads the real function
    object rather than the enclosing handler's source.
    """
    from app.api.ws_realtime import voice_section_fingerprints

    body = inspect.getsource(voice_section_fingerprints)
    assert "hashlib" in body and "hexdigest()[:8]" in body, (
        "fingerprints must be short digests"
    )
    # The property, asserted on the OUTPUT rather than on a source line:
    # nothing a section contains may reach what gets logged.
    out = voice_section_fingerprints(
        "# Core Identity\nsecret persona text\n\n# User Brain\nsecret fact"
    )
    assert "secret persona text" not in repr(out), "raw block text reached the output"
    assert "secret fact" not in repr(out), "raw block text reached the output"
    assert all(
        isinstance(chars, int) and len(digest) == 8
        for chars, digest in out.values()
    ), out


def test_section_fingerprints_is_stable_and_content_free():
    """The shipped helper itself — no longer recompiled out of the AST.

    It used to be nested inside the websocket handler, so the only way to
    reach it was to lift it out of the file's AST and exec it. It is
    module level now, so this imports the actual object the shadow runs.
    """
    from app.api.ws_realtime import voice_section_fingerprints as fp

    text = "# Core Identity\nsecret persona text\n\n# User Brain\nsecret fact"
    out = fp(text)
    assert set(out) == {"Core Identity", "User Brain"}, out
    for _head, (chars, digest) in out.items():
        assert isinstance(chars, int) and len(digest) == 8
    flat = repr(out)
    assert "secret persona text" not in flat and "secret fact" not in flat, (
        "the fingerprint output contains section content"
    )
    assert fp(text) == out, "fingerprints are not stable across calls"


def test_every_section_is_measured_by_the_same_ruler():
    """`split("\\n\\n# ")` eats the `# ` marker on every block but the
    first, so section 0 used to be counted and hashed over two more
    characters than an identical copy of itself. That asymmetry is the
    only reason a duplicated FIRST section was ever caught, and it made
    section 0's char count in every shadow log line two too many."""
    from app.api.ws_realtime import voice_section_fingerprints as fp

    first_then_second = fp("# Same Header\nbody\n\n# Other\nx")["Same Header"]
    second_position = fp("# Other\nx\n\n# Same Header\nbody")["Same Header"]
    assert first_then_second == second_position, (
        "the same section fingerprints differently depending on where it "
        f"sits: {first_then_second} vs {second_position}"
    )
