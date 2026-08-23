"""Round 25 — the build card exists while the app is being written, not after.

A build opened with a single generic "Building your app" spinner and stayed
that way for about two minutes, with no steps under it. Then, all at once, a
steps card appeared with the write already finished.

Nothing was lying. The card is minted by `steps.ensure_job`, which is called
from `_create` — and `_create` runs when the tool call is COMPLETE. For
`create_app_file` the tool call *is* the app: the entire document is one
string argument, so the model spends the whole build emitting arguments and
the pipeline does not exist until the last byte lands.

The arguments were on the wire the whole time. Both provider adapters
accumulated them privately (`input_json_delta` for Anthropic,
`response.function_call_arguments.delta` for the Responses wire, which is what
the fleet default `gpt-5.6-*` actually uses) and yielded nothing until the
call closed. `StreamEvent.type` had documented a `tool_use_input` variant since
the file was written and no code had ever emitted or consumed one.

`slug` and `title` are the first two properties of the tool's schema, and
arguments are generated in schema order — a fact this schema already depends
on, since `brief` is declared before `html` precisely so the plan is written
before the code. So the identifying fields arrive in the first tokens, and the
card can open then.

These tests drive the skill's `on_tool_input` with a REAL streamed prefix,
byte by byte, and assert on when the card appears relative to the document.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from app.agent.skills.base import SkillContext
from app.agent.skills.builtins.app_html import skill as skill_mod
from app.agent.skills.builtins.app_html import steps as steps_mod
from app.agent.skills.builtins.app_html import store
from app.agent.skills.builtins.app_html.skill import AppHtmlSkill

CTX = SkillContext(workspace="/tmp", user_id="user-1", session_id="s1")

#: A realistic call: the app is ~20 KB and it is the LAST argument, exactly as
#: the schema orders it.
BODY = "<!doctype html><html><body>" + ("<div>x</div>" * 1700) + "</body></html>"
CALL_JSON = json.dumps({
    "slug": "snake-game",
    "title": "Snake Game",
    "brief": "A plan for the game, written before the code. " * 12,
    "html": BODY,
})


@pytest.fixture()
def apps_dir(tmp_path, monkeypatch):
    root = tmp_path / "apps"
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    store.ensure_root()
    return root


@pytest.fixture()
def live(apps_dir, monkeypatch):
    """The skill, with the job layer recorded rather than hit.

    `emit_step` returns the frame it broadcast (round 25) and `retouch` reuses
    it, so the fake has to honour that shape or the byte counter cannot work.
    """
    s = AppHtmlSkill()
    opened: List[str] = []
    emitted: List[Dict[str, Any]] = []
    retouched: List[Dict[str, Any]] = []

    async def _job(user_id, slug, title):
        opened.append(slug)
        return "job-test"

    async def _emit(*, user_id, job_id, step_type, status, detail="",
                    recoverable=False):
        frame = {
            "type": "job_update", "job_id": job_id, "name": "Snake Game",
            "status": "running", "step": step_type,
            "steps": [{"type": t, "status": "pending", "label": t}
                      for t in steps_mod.PLANNED_TYPES],
        }
        for row in frame["steps"]:
            if row["type"] == step_type:
                row["status"] = status
        emitted.append({"type": step_type, "status": status, "frame": frame})
        return frame

    async def _retouch(user_id, frame, *, step_type, detail):
        retouched.append({"step_type": step_type, "detail": detail})
        return frame

    monkeypatch.setattr(steps_mod, "ensure_job", _job)
    monkeypatch.setattr(steps_mod, "emit_step", _emit)
    monkeypatch.setattr(steps_mod, "retouch", _retouch)
    return s, opened, emitted, retouched


async def _stream(skill, payload: str, *, chunk: int = 256,
                  tool="app_html__create_app_file", call_id="call-1"):
    """Feed a tool call's arguments the way a provider does, and report how
    much had arrived when the card first opened."""
    opened_at = None
    for i in range(0, len(payload), chunk):
        await skill.on_tool_input(tool, call_id, payload[: i + chunk], CTX)
        if opened_at is None and skill._live_calls.get(call_id):
            opened_at = i + chunk
    return opened_at


# ── The headline ──────────────────────────────────────────────────────

async def test_the_card_opens_before_the_app_has_been_written(live):
    skill, opened, emitted, _ = live

    opened_at = await _stream(skill, CALL_JSON)

    assert opened, "no build card was opened while the arguments streamed"
    assert opened_at is not None
    assert opened_at < 1024, (
        f"the card only appeared after {opened_at} bytes had streamed; the "
        f"identifying fields are in the first few hundred"
    )
    # The document is the overwhelming majority of the call. The card must
    # exist for essentially all of it, not at the end.
    assert opened_at < len(CALL_JSON) * 0.1, (
        f"the card appeared {100 * opened_at / len(CALL_JSON):.0f}% of the way "
        f"through the call — this is the two-minute blind spinner"
    )


async def test_the_write_phase_is_running_from_the_moment_it_opens(live):
    skill, _, emitted, _ = live
    await _stream(skill, CALL_JSON)

    first = emitted[0]
    assert first["type"] == "create"
    assert first["status"] == "running", (
        "the card opened with the write phase not yet running, so the user "
        "still had nothing to watch"
    )
    assert {r["type"] for r in first["frame"]["steps"]} == set(
        steps_mod.PLANNED_TYPES
    ), "the very first frame must carry the WHOLE plan, not just one row"


async def test_the_phase_transition_is_emitted_exactly_once_per_call(live):
    """`on_tool_input` fires on every delta. A row transition per delta would
    be a write loop on the job row for the length of the build."""
    skill, opened, emitted, _ = live
    await _stream(skill, CALL_JSON, chunk=64)

    assert len(opened) == 1, f"ensure_job ran {len(opened)} times"
    assert len(emitted) == 1, f"emit_step ran {len(emitted)} times"


async def test_the_byte_counter_advances_while_the_document_streams(live):
    skill, _, _, retouched = live
    await _stream(skill, CALL_JSON)

    assert retouched, "the card opened and then froze for the whole write"
    assert all(r["step_type"] == "create" for r in retouched)
    kb = [int(r["detail"].split()[0].replace(",", "")) for r in retouched]
    assert kb == sorted(kb), f"the byte counter went backwards: {kb}"
    assert kb[-1] >= 15, f"counter stalled at {kb[-1]} KB for a ~20 KB app"


# ── It must not open a card on a guess ────────────────────────────────

async def test_half_a_slug_opens_nothing(live):
    """Half a slug names a DIFFERENT app. The parser requires the closing
    quote, so it returns nothing until the value has actually arrived."""
    skill, opened, _, _ = live
    await skill.on_tool_input(
        "app_html__create_app_file", "call-x", '{"slug": "snake-ga', CTX,
    )
    assert not opened, "a card was opened on a truncated slug"


async def test_an_unrelated_tool_is_ignored(live):
    skill, opened, _, _ = live
    await skill.on_tool_input(
        "app_html__present_app", "call-y", '{"slug": "snake-game"}', CTX,
    )
    assert not opened, "present_app pre-opened a phase it does not own"


async def test_the_tracker_cannot_grow_without_bound(live):
    """There is no seam telling a skill that a call it watched has finished —
    `execute_tool` receives no call id — so entries are evicted by age."""
    skill, _, _, _ = live
    for i in range(skill_mod._MAX_LIVE_CALLS * 3):
        await skill.on_tool_input(
            "app_html__view_app_file", f"call-{i}",
            '{"slug": "snake-game"}', CTX,
        )
    assert len(skill._live_calls) <= skill_mod._MAX_LIVE_CALLS


# ── The parsers, on their own ─────────────────────────────────────────

def test_partial_arg_waits_for_the_closing_quote():
    assert skill_mod._partial_arg('{"slug": "snake-ga', "slug") == ""
    assert skill_mod._partial_arg('{"slug": "snake-game"', "slug") == "snake-game"
    assert skill_mod._partial_arg('{"slug": "a", "title": "B"}', "title") == "B"
    assert skill_mod._partial_arg("", "slug") == ""


def test_partial_arg_survives_an_escaped_quote_in_the_value():
    raw = json.dumps({"title": 'The "Real" Thing'})[:-1]
    assert skill_mod._partial_arg(raw, "title") == 'The "Real" Thing'


def test_the_byte_counter_measures_the_document_not_the_preamble():
    """The slug, title and brief precede the app. Counting the whole buffer
    would show a build several KB in before a line of the app existed."""
    preamble = '{"slug": "s", "title": "T", "brief": "' + "p" * 4000 + '", '
    assert skill_mod._streamed_body_len(preamble) == 0
    assert skill_mod._streamed_body_len(preamble + '"html": "') == 0
    assert skill_mod._streamed_body_len(preamble + '"html": "abcde') == 5
