"""The Live prompt must describe the backend it actually has (R48, A8-1/2/5).

Three findings, one document.

A8-2: the served instructions describe a tool surface that does not exist on
the GPT-Live wire. `_tools_t` is cancelled on that path, so the model has no
function-calling surface at all — and the prompt tells it, imperatively, to
call `think`, `navigate_to`, `exec`/`read_file`/`write_file`, and to use screen
context the Live relay refuses with `live_screen_share_unavailable`. The
provider returns no error for an instruction that cannot be satisfied; the
model simply falls back to speaking, which is how "I'll do that for you"
arrives with nothing behind it.

A8-1: and nothing replaced them. Client-mode delegation is configured as
`{"type": "client"}` and carries no tool list, so the PROMPT is the only
channel that can tell GPT-Live what the backend can do and when to hand off.
Toup's described no backend at all — one recording shows 23.4 s of continuous
speech before the first delegation, another zero in 95 s.

A8-5: the strong per-turn reply-language rule lived only in
`_base_voice_instructions()`, the fallback stub a healthy session never
reaches, so neither recorded call carried it.

Pure: `render_*` take no database. The shape test that needs one is in
tests/agent/test_voice_context_parity.py.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.agent.voice_context import (
    VOICE_SECTION_ORDER,
    render_live_delegation,
    render_voice_mode,
)

FROZEN = datetime(2026, 9, 20, 18, 47, tzinfo=timezone.utc)

#: Every phrase below names a Realtime-only capability. Matched
#: case-insensitively on the rendered Live prompt.
REALTIME_ONLY_MARKERS = (
    "think tool",
    "think(task=",
    "navigate_to",
    "full access to the user's computer terminal",
    "[screen context:",
)


def test_the_live_prompt_names_no_tool_the_live_wire_does_not_have():
    live = render_voice_mode(FROZEN, live=True).lower()
    for marker in REALTIME_ONLY_MARKERS:
        assert marker.lower() not in live, marker


def test_the_realtime_prompt_still_carries_every_one_of_them():
    """The split must not quietly disarm the path that DOES have these tools.
    Realtime's `think` paragraph is the only thing routing work to the agent
    there; losing it would be a far larger outage than the one being fixed."""
    realtime = render_voice_mode(FROZEN, live=False).lower()
    for marker in REALTIME_ONLY_MARKERS:
        assert marker.lower() in realtime, marker


def test_the_default_is_the_realtime_prompt():
    """An older caller passes no flag and must get exactly today's document."""
    assert render_voice_mode(FROZEN) == render_voice_mode(FROZEN, live=False)


def test_both_paths_carry_the_speech_rules_and_the_clock():
    for live in (True, False):
        text = render_voice_mode(FROZEN, live=live)
        assert text.startswith("# Voice Conversation Mode")
        assert "Do NOT use markdown" in text
        assert "2026-09-20 18:47 UTC" in text


def test_the_reply_language_rule_is_on_BOTH_paths():
    """It was written for this bilingual account and lived in the code path a
    healthy session skips, so neither recorded call had it."""
    for live in (True, False):
        text = render_voice_mode(FROZEN, live=live)
        assert "REPLY LANGUAGE" in text
        assert "the language the user JUST SPOKE" in text
        # Language-NEUTRAL: Persian is the named example, not the only case.
        assert "the same rule holds for every other language" in text
        assert "never a reason to reply in English" in text


def test_the_tenant_language_rule_lets_an_explicit_request_outrank_mirroring():
    """V3 (§G). "Persian in, Persian out — every time" is a rule an explicit
    "speak English with me" can never win against, and the tenant prompt is
    the one the model reads first. Same precedence as the relay's copy."""

    from app.services.live_voice_protocol import LIVE_REPLY_LANGUAGE_RULE

    for live in (True, False):
        text = render_voice_mode(FROZEN, live=live)
        rule = text.split("REPLY LANGUAGE", 1)[1].split("\n- ", 1)[0]
        assert "explicitly asked" in rule
        assert "outranks" in rule
        assert "until they ask for a different" in rule
        assert "every time" not in rule
        assert rule.index("explicitly asked") < rule.index("JUST SPOKE")
        # A bare "Match the user's language." or "when the user speaks
        # Persian, reply in Farsi" elsewhere in the same document would restate
        # mirroring with no exception. The accent guidance stays.
        assert "Match the user's language." not in text
        assert "When the user speaks Persian/Farsi, reply" not in text
        assert "native Tehrani accent" in text
    for phrase in ("explicitly asked", "outranks", "until they ask for a different"):
        assert phrase in LIVE_REPLY_LANGUAGE_RULE, phrase


# ── The delegation document ─────────────────────────────────────────────

def test_the_delegation_document_lists_the_backend_capabilities():
    text = render_live_delegation()
    assert text.startswith("# Backend")
    low = text.lower()
    for capability in ("music and audio", "research", "files and documents",
                       "connected accounts", "reminders", "memory"):
        assert capability in low, capability
    assert "delegate when:" in low
    assert "do not delegate when:" in low


def test_the_delegation_document_forbids_claiming_and_forbids_denying():
    """The two rules the recordings needed. "Never say it started" is what
    stops the promise with nothing behind it; "never say you cannot" is what
    stops "I can't play music directly" from a model whose backend can."""
    low = render_live_delegation().lower()
    assert "never say that an action has started, is done, or failed" in low
    assert "never say you cannot do something listed above" in low
    # …and the progress rule that makes a status question answerable without
    # starting a second task.
    assert "progress notes" in low
    assert "without delegating again" in low
    assert "never invent progress" in low


def _agent_tool_names() -> set:
    """Every tool name the agent image can actually offer.

    Read from the DECLARATIONS, not from a list in this file: the static
    surface in `tool_definitions.py` plus the built-in skills that register
    namespaced tools (`automations__*` is where reminders, routines and
    scheduled work live — they are not in `tool_definitions.py` at all, which
    is precisely the kind of thing a hand-written list gets wrong).
    """
    import ast
    import pathlib

    names: set = set()
    root = pathlib.Path(__file__).resolve().parents[1] / "app" / "agent"
    sources = [root / "tool_definitions.py"] + sorted(
        (root / "skills" / "builtins").rglob("skill.py")
    )
    for path in sources:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # `{"name": "web_search", …}` — the one shape every tool
            # declaration in this repo is written in.
            if not isinstance(node, ast.Dict):
                continue
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and k.value == "name"
                        and isinstance(v, ast.Constant)
                        and isinstance(v.value, str)):
                    names.add(v.value)
    return names


#: Each capability the document names, and the tool that makes it true. A new
#: bullet has to be added here — and the name has to exist — or the test below
#: fails. That is the point: the six-word blacklist this replaces could not see
#: a media overclaim, because it only knew about words nobody had written yet.
_CAPABILITY_TOOLS = {
    "Music and audio": ("play_media",),
    "Research": ("web_search", "web_fetch"),
    "Files and documents": ("read_file", "write_file", "find"),
    "Connected accounts": ("automations__request_connection",),
    "Reminders, routines and scheduled work": ("automations__create",),
    "Memory": ("memory_search", "memory_store"),
}


def test_every_capability_the_document_names_maps_to_a_tool_that_exists():
    """The POSITIVE check (addendum C3).

    An overclaim here becomes a promise the backend cannot keep, and this
    document is paired with "Never say you cannot do something listed above" —
    so a capability the runner does not have becomes a delegation nothing can
    serve. The blacklist this replaces tested six words about terminals and
    screens and was blind to the line that was actually wrong.
    """
    doc = render_live_delegation()
    body = doc.split("Backend capabilities:\n", 1)[1].split("\n\n", 1)[0]
    bullets = [ln[2:] for ln in body.splitlines() if ln.startswith("- ")]
    assert bullets, doc

    available = _agent_tool_names()
    for bullet in bullets:
        capability = bullet.split(":", 1)[0].rstrip(".")
        assert capability in _CAPABILITY_TOOLS, (
            f"the document claims {capability!r} and no tool is mapped to it — "
            "name the tool that makes it true, or delete the claim"
        )
        for tool in _CAPABILITY_TOOLS[capability]:
            assert tool in available, (
                f"{capability!r} is mapped to {tool!r}, which no tool "
                "declaration in the image defines"
            )
    for capability in _CAPABILITY_TOOLS:
        assert any(b.split(":", 1)[0].rstrip(".") == capability for b in bullets), (
            f"{capability!r} is mapped but no longer claimed — drop the mapping"
        )


def test_the_media_line_claims_only_reviewed_deterministic_controls():
    """Live supports play plus the tool-less next/previous route, but still
    must not promise pause/stop/resume before those controls exist.

    The exact string is pinned because the relay's own
    `adapt_instructions_for_live` carries the same line for the window before
    the agent image rolls, and the two reach one provider session.
    """
    assert (
        "- Music and audio: start playback, or move to the next/previous track "
        "on the user's device.\n"
    ) in render_live_delegation()
    low = render_live_delegation().lower()
    for verb in ("pause", "stop playback", "resume"):
        assert verb not in low, verb


#: The one line C3 requires to be word-identical in two files. Kept as one
#: literal so the assertion below cannot drift from the assertion above.
_MEDIA_LINE = (
    "- Music and audio: start playback, or move to the next/previous track on the user's device.\n"
)


def test_the_relays_copy_of_the_media_line_is_word_identical():
    """CROSS-FILE, and the reason it exists is that the two texts reach ONE
    provider session.

    `voice_context.render_live_delegation` (this image) and
    `live_voice_protocol.LIVE_BACKEND_SECTION` (the platform relay) both
    describe the same backend to the same model — the relay's copy covers the
    window before an agent image rolls. Addendum C3 requires them to stay
    word-identical on this line; until now each file pinned only its own copy,
    so an edit to either one re-opened the pause/skip overclaim with nothing in
    the repo able to see it. The two files also roll independently, which is
    exactly the condition that makes a one-sided pin worthless.
    """
    from app.services.live_voice_protocol import LIVE_BACKEND_SECTION

    assert _MEDIA_LINE in render_live_delegation(), (
        "app/agent/voice_context.py::render_live_delegation lost the C3 media line"
    )
    assert _MEDIA_LINE in LIVE_BACKEND_SECTION, (
        "app/services/live_voice_protocol.py::LIVE_BACKEND_SECTION and "
        "app/agent/voice_context.py::render_live_delegation must carry this "
        "line WORD-IDENTICALLY (addendum C3); the relay's copy has drifted"
    )


def test_the_delegation_document_claims_nothing_the_runner_cannot_do():
    """The old blacklist, KEPT: it is cheap and it is about a different class
    of overclaim — the Realtime-only paragraphs this document replaced. It is
    no longer the only evidence."""
    low = render_live_delegation().lower()
    for overclaim in ("terminal", "shell", "screen", "phone call",
                      "your computer", "navigate"):
        assert overclaim not in low, overclaim


def test_the_delegation_section_sits_where_the_removed_paragraphs_did():
    """Removing the routing guidance without replacing it in the same position
    leaves the model with none at all — which would be worse than the defect."""
    order = list(VOICE_SECTION_ORDER)
    assert order.index("live_delegation") == order.index("voice_mode") + 1


def test_the_builder_emits_the_section_only_on_live():
    """SOURCE probe, because the section order above is only half the wiring:
    a key nothing ever sets is ordered correctly and absent. And the flag has
    to reach `render_voice_mode` too — emitting the delegation document while
    still serving the Realtime tool paragraphs would leave the model with both
    a backend and four tools it cannot call."""
    import inspect
    from app.agent import voice_context

    src = inspect.getsource(voice_context.build_voice_context)
    assert 'sections["voice_mode"] = render_voice_mode(now_utc, live=live)' in src
    i = src.find('sections["voice_mode"]')
    j = src.find('sections["live_delegation"] = render_live_delegation()')
    assert j != -1, "the Live section is ordered but never built"
    assert i < j
    # Gated, not unconditional: the Realtime path must not grow a backend
    # document describing a delegation wire it does not have.
    assert "if live:\n        sections[\"live_delegation\"]" in src


@pytest.mark.parametrize("live", [True, False])
def test_the_renderers_never_raise_on_a_naive_clock(live):
    """`now_utc` arrives from a request body and `.strftime` is the one thing
    in here that touches it."""
    assert render_voice_mode(datetime(2026, 1, 1), live=live)
