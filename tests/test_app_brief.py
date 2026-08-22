"""Round 20, item 1 — every app carries a brief, and nobody but the model sees it.

Three questions, and the third is the one that would be expensive to get
wrong:

1. Is the brief WRITTEN, and written at the only moment the model still knows
   why it built what it built (`create_app_file`)?
2. Is it READ, without depending on the model remembering to ask for it?
3. Can it LEAK? It is written in the second person, to a model, about a
   person — "who it is for", "what they were trying to solve". Every test in
   §3 exists because there is no version of this feature where showing it to
   the user is acceptable, and the guarantee therefore cannot rest on one
   rule staying true.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from app.agent.skills.base import SkillContext
from app.agent.skills.builtins.app_html import appskill, store
from app.agent.skills.builtins.app_html.skill import AppHtmlSkill
from app.agent.skills.builtins.app_html.store import AppStoreError

CTX = SkillContext(user_id="u-brief", session_id="s1")

BRIEF = (
    "## What it is\n"
    "A mole-whacking game for a child of six to eight, playable in under a "
    "minute, built to be the thing a parent hands over in a queue.\n\n"
    "## Core flows\n"
    "- Tap Play, whack moles for thirty seconds, see the score.\n\n"
    "## Features, states and controls\n"
    "- States: start, playing, over. Play starts the round; each hole is a "
    "72px target; the clock counts down from 30.\n\n"
    "## Design decisions\n"
    "- Grass green field with one warm accent on the moles, so the only "
    "saturated thing on screen is the thing you are meant to hit."
)


def _html(title="Whack a Mole", body_extra="", script_extra=""):
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"<title>{title}</title>\n<style>\n"
        ":root{--field:#2F6B3A;--mole:#C4703A;--ink:#F7F4EC;}\n"
        "body{margin:0;background:var(--field);color:var(--ink);"
        "font-family:ui-sans-serif,system-ui,sans-serif;}\n"
        "#board{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}\n"
        ".hole{min-width:72px;min-height:72px;border-radius:50%;}\n"
        "</style>\n</head>\n<body>\n"
        "<header id=\"hud\"><span id=\"score\">0</span><span id=\"clock\">30</span></header>\n"
        "<main id=\"board\"><button class=\"hole\" id=\"h0\">mole</button>"
        "<button class=\"hole\" id=\"h1\">mole</button></main>\n"
        f"<button id=\"play\">Play</button>\n{body_extra}\n"
        "<script>\n"
        "let score = 0;\n"
        "let running = false;\n"
        "function startRound(){ running = true; score = 0; render(); }\n"
        "const render = () => { document.getElementById('score').textContent = score; };\n"
        "document.getElementById('play').addEventListener('click', startRound);\n"
        f"{script_extra}\n"
        "</script>\n</body>\n</html>\n"
    )


@pytest.fixture()
def apps_dir(tmp_path, monkeypatch):
    root = tmp_path / "apps"
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(root))
    monkeypatch.setenv("TOUP_APP_SMOKE_TEST", "0")
    monkeypatch.setenv("TOUP_APP_VISUAL_REVIEW", "0")
    # No model calls from a unit test. Without this the icon step on every
    # `present_app` spends `logo.LOGO_TIMEOUT_S` on a doomed network call —
    # which is also what it did in production on a container with no
    # credential, and is why `vision.can_call_model` exists.
    monkeypatch.setenv("TOUP_APP_MODEL_CALLS", "0")
    store.ensure_root()
    return root


@pytest.fixture()
def skill(apps_dir):
    return AppHtmlSkill()


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def tool(sk, name, **args):
    return _run(sk.execute_tool(f"app_html__{name}", args, CTX))


# ── 1. It is written, at the moment the model still knows ─────────────

def test_a_build_without_a_brief_is_refused(skill, apps_dir):
    """The whole feature turns on this one refusal.

    A brief that `create_app_file` merely ASKS for is a brief that exists for
    the builds where the model was not busy. Ask for it the way the file
    itself is asked for — as a required argument — and there is no such build.
    """
    out = tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
               html=_html())
    assert out.startswith("ERROR:"), out
    assert "brief is required" in out
    # And nothing was written: a refused call must not leave a half-app.
    assert not (apps_dir / "whack.html").exists()
    assert not appskill.exists("whack")


def test_a_one_line_brief_is_refused(skill, apps_dir):
    out = tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
               html=_html(), brief="A whack-a-mole game.")
    assert out.startswith("ERROR:"), out
    assert "that is a label, not a brief" in out


def test_a_build_with_a_brief_stores_it_beside_the_app(skill, apps_dir):
    out = tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
               html=_html(), brief=BRIEF)
    assert not out.startswith("ERROR:"), out

    brief = appskill.read("whack")
    assert brief is not None
    assert brief.has_narrative
    assert "child of six to eight" in brief.narrative
    assert brief.title == "Whack a Mole"
    assert brief.narrative_revision == 1
    # Stored in the dot-directory, as markdown, never as another .html.
    path = Path(appskill.brief_path("whack"))
    assert path.parent.name == ".skills"
    assert path.suffix == ".md"


def test_the_structure_half_is_derived_not_asserted(skill, apps_dir):
    """The model writes the WHY; the map is read off the bytes.

    A hand-written structure section is a structure section that is wrong two
    edits later, and a stale map is worse than no map — it sends the next edit
    to the wrong element with confidence.
    """
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    structure = appskill.read("whack").structure
    assert "`board`" in structure or "board" in structure
    assert "--field" in structure          # the palette, from the CSS itself
    assert "startRound" in structure       # a top-level function
    assert "click" in structure            # an event actually handled
    assert "self-contained" in structure   # no external libraries


def test_the_map_follows_the_file(skill, apps_dir):
    """An edit that renames a thing must not leave the old name in the map."""
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    assert "startRound" in appskill.read("whack").structure

    out = tool(skill, "edit_app_file", slug="whack",
               old_string="function startRound(){", new_string="function beginRound(){",
               reason="rename the round starter")
    assert not out.startswith("ERROR:"), out
    structure = appskill.read("whack").structure
    assert "beginRound" in structure
    assert "startRound" not in structure


def test_every_edit_leaves_a_history_line(skill, apps_dir):
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    tool(skill, "edit_app_file", slug="whack",
         old_string="<span id=\"clock\">30</span>",
         new_string="<span id=\"clock\">45</span>",
         reason="make the round 45 seconds")
    history = appskill.read("whack").history
    assert any("make the round 45 seconds" in h for h in history)
    # Round 21 versions the entry: `v<brief version> · r<app revision>`.
    assert any(h.startswith("- v2 · r2 ") for h in history), history


def test_an_edit_may_rewrite_the_narrative_but_need_not(skill, apps_dir):
    """A padding change must not be made to re-justify the whole app.

    Requiring a rewritten brief on every edit is how a brief stops being read:
    it becomes a form. Requiring it on none is how it goes stale. Optional on
    the edit, required on the build, is the split that survives both.
    """
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    tool(skill, "edit_app_file", slug="whack", old_string="gap:12px",
         new_string="gap:16px", reason="loosen the grid")
    assert "child of six to eight" in appskill.read("whack").narrative

    revised = BRIEF.replace("child of six to eight", "an adult on a commute")
    tool(skill, "edit_app_file", slug="whack", old_string="gap:16px",
         new_string="gap:20px", reason="loosen the grid again", brief=revised)
    brief = appskill.read("whack")
    assert "an adult on a commute" in brief.narrative
    assert brief.narrative_revision == 3


# ── 2. It is read, without being asked for ────────────────────────────

def test_view_hands_the_brief_back_above_the_file(skill, apps_dir):
    """Reading it is not a step the model can skip, because it is not a step.

    The prompt already requires `view_app_file` before every edit. Putting the
    brief in that same result means an edit cannot be made by a model that was
    never told what the app is for — no new tool, no new rule, and no extra
    name in the user's actions rail.
    """
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    out = tool(skill, "view_app_file", slug="whack")
    assert out.startswith("<app_brief")
    assert "child of six to eight" in out
    assert out.endswith(_html())            # the file half is still exact
    # ...and it is told, in the result itself, that this is not for the user.
    assert "Never quote it" in out


def test_the_brief_says_when_it_has_drifted(skill, apps_dir):
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    for i in range(3):
        tool(skill, "edit_app_file", slug="whack", old_string=f"gap:{12 + i * 4}px",
             new_string=f"gap:{16 + i * 4}px", reason="loosen")
    out = tool(skill, "view_app_file", slug="whack")
    assert "written against revision 1" in out
    assert "now at 4" in out


def test_an_app_with_no_brief_is_told_how_to_get_one(apps_dir, skill):
    store.write_app("legacy", "Legacy App", _html())
    out = tool(skill, "view_app_file", slug="legacy")
    assert 'status="missing"' in out
    assert "create_app_file or edit_app_file" in out


# ── 3. It cannot leak ─────────────────────────────────────────────────

def test_the_user_never_sees_the_brief_in_a_tool_result(skill, apps_dir):
    """`display` is the only string a client renders. It must never be it."""
    created = tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
                   html=_html(), brief=BRIEF)
    viewed = tool(skill, "view_app_file", slug="whack")
    for result in (created, viewed):
        display = getattr(result, "display", "") or ""
        assert "brief" not in display.lower(), display
        assert "child of six" not in display
        assert len(display) < 120, display


def test_a_refusal_about_the_brief_does_not_name_it_to_the_user(skill, apps_dir):
    """The leak this class of refusal opens, closed at every entry point.

    A refusal message is written to a model — "brief is required", "legacy has
    no brief yet". `_short` puts the first clause of one under a progress bar,
    so every one of these paths would have told the user about a file that
    describes them, that they cannot open, and that they were never meant to
    know exists.
    """
    store.write_app("legacy", "Legacy App", _html())
    refusals = [
        tool(skill, "create_app_file", slug="nobrief", title="No Brief",
             html=_html()),                                   # missing
        tool(skill, "create_app_file", slug="tiny", title="Tiny",
             html=_html(), brief="A game."),                  # too short
        tool(skill, "present_app", slug="legacy"),            # publish gate
    ]
    for out in refusals:
        assert out.startswith("ERROR:"), out                  # the model IS told
        display = (getattr(out, "display", "") or "").lower()
        assert display, out
        for leak in ("brief", "internal", "narrative", ".skills"):
            assert leak not in display, display


def test_the_brief_is_invisible_to_the_library_scanner(skill, apps_dir):
    """Three independent reasons, asserted as three.

    The library lists an app by walking the app root at depth 0, skipping
    dotted names, and requiring a `.html` suffix. The brief fails all three.
    A single guarantee that rests on one of them is one refactor from a
    Files row titled "whack" containing the user's own psychographics.
    """
    from app.services import library_service as lib

    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    tool(skill, "present_app", slug="whack")

    path = Path(appskill.brief_path("whack"))
    assert path.exists()
    assert path.parent.name.startswith(".")      # 1. a dot-directory
    assert path.parent.parent == Path(store.apps_root())   # 2. below depth 0
    assert path.suffix != ".html"                # 3. not the listed suffix

    budget = [10_000]
    keys = [c.key for c in lib._iter_app_candidates(budget)]
    assert keys == ["app:whack.html"], keys
    assert not any(".skills" in k for k in keys)


def test_no_artifact_route_can_return_a_brief(apps_dir):
    """A route is one refactor from a client. There must not be one."""
    from app.api import artifacts as agent_routes
    from app.api import artifact_proxy as platform_routes

    for module in (agent_routes, platform_routes):
        paths = [r.path for r in module.router.routes]
        assert not any("brief" in p or "skill" in p for p in paths), paths


def test_the_list_and_source_routes_carry_no_narrative(skill, apps_dir):
    from app.api import artifacts as agent_routes

    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    listing = _run(agent_routes.list_artifacts())
    source = _run(agent_routes.get_artifact_source("whack"))
    meta = _run(agent_routes.get_artifact_meta("whack"))
    for payload in (listing, source, meta):
        blob = json.dumps(payload)
        assert "child of six" not in blob
        assert "narrative" not in blob


def test_the_published_document_carries_no_brief(skill, apps_dir):
    """The one thing the browser is handed is the app. Nothing rides along."""
    from app.agent.skills.builtins.app_html import runtime

    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    served = runtime.wrap_for_runtime(store.read_app("whack"))
    assert "child of six" not in served
    assert "app_brief" not in served


# ── 4. Publishing needs one ───────────────────────────────────────────

def test_present_refuses_an_app_that_has_no_brief(apps_dir, skill):
    store.write_app("legacy", "Legacy App", _html())
    out = tool(skill, "present_app", slug="legacy")
    assert out.startswith("ERROR:"), out
    assert "has no brief yet" in out
    # And the user is told nothing about a file they must not know exists.
    assert "brief" not in (getattr(out, "display", "") or "").lower()
    # Not published: the record must not have been marked presented.
    assert store.read_manifest()["legacy"].presented_at is None


def test_present_accepts_it_once_the_brief_is_written(apps_dir, skill):
    store.write_app("legacy", "Legacy App", _html())
    tool(skill, "edit_app_file", slug="legacy", old_string="gap:12px",
         new_string="gap:14px", reason="loosen the grid", brief=BRIEF)
    out = tool(skill, "present_app", slug="legacy")
    assert not out.startswith("ERROR:"), out
    assert store.read_manifest()["legacy"].presented_at is not None


# ── 5. Backfill and cleanup ───────────────────────────────────────────

def test_backfill_gives_an_old_app_a_map_without_a_model(apps_dir):
    """The free half runs inline; it must not need anything it cannot have."""
    store.write_app("legacy", "Legacy App", _html())
    assert not appskill.exists("legacy")

    moved = appskill.backfill_missing()
    assert moved == {"legacy": "brief_created"}
    brief = appskill.read("legacy")
    assert brief is not None
    assert "startRound" in brief.structure
    # ...and it does NOT invent a purpose. A machine-written narrative would
    # be read by the next editor as the app's stated intent while being a
    # restatement of its markup.
    assert not brief.has_narrative


def test_backfill_leaves_an_existing_brief_alone(skill, apps_dir):
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    assert appskill.backfill_missing() == {}
    assert "child of six to eight" in appskill.read("whack").narrative


def test_deleting_an_app_takes_its_brief(skill, apps_dir):
    """Slugs are reusable. A brief left behind is read as the NEXT app's
    purpose — confidently wrong, which is worse than absent."""
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    assert appskill.exists("whack")
    store.delete_app("whack")
    assert not appskill.exists("whack")
    assert not os.path.exists(appskill.brief_path("whack"))


def test_a_purged_record_takes_its_brief_too(skill, apps_dir):
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    tool(skill, "create_app_file", slug="other", title="Other",
         html=_html("Other"), brief=BRIEF)
    # The file goes, the history goes, the record remains: the exact state
    # `reconcile` purges.
    os.unlink(store.app_path("whack"))
    import shutil
    shutil.rmtree(os.path.join(store.apps_root(), ".versions", "whack"),
                  ignore_errors=True)
    assert store.reconcile("whack") == "purged"
    assert not appskill.exists("whack")
    assert appskill.exists("other")


# ── 6. The document round-trips ───────────────────────────────────────

def test_a_brief_parses_back_into_its_three_parts():
    brief = appskill.Brief(
        slug="whack", title="Whack a Mole", narrative=BRIEF,
        structure="- **Size:** 1,000 bytes", history=["- r1 · t · built"],
        narrative_revision=3,
    )
    again = appskill.parse(brief.render(), "whack")
    assert again.title == "Whack a Mole"
    assert again.narrative_revision == 3
    assert again.narrative.strip() == BRIEF.strip()
    assert "1,000 bytes" in again.structure
    assert again.history == ["- r1 · t · built"]


def test_a_corrupt_brief_degrades_to_its_narrative():
    """A file two code paths and a model can write must never raise."""
    again = appskill.parse("just some prose with no frontmatter at all\n", "x")
    assert again.narrative.startswith("just some prose")
    assert again.structure == ""
    assert again.history == []


def test_validate_narrative_is_the_only_gate():
    with pytest.raises(AppStoreError):
        appskill.validate_narrative("", required=True)
    with pytest.raises(AppStoreError):
        appskill.validate_narrative("tiny", required=False)
    assert appskill.validate_narrative("", required=False) == ""
    assert appskill.validate_narrative(BRIEF, required=True) == BRIEF.strip()
