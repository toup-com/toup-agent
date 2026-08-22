"""Round 21 — the logo is part of the build, the app is in Files at once,
Apps is a shelf and not a folder that follows you around, and the app's
memory is versioned.

Six items, and each one is asserted the way it was found:

1.  The mark is drawn as its own phase of the publish, before anything is
    handed over, and the payload the card is drawn from carries it. Control:
    on ``origin/main`` there is no ``logo`` step and no ``icon_svg``.
2.  A published app is in Files on the NEXT listing, not on the next listing
    that happens to fall outside the library sync's two-second throttle.
    Control: `test_r21_probe` on main returned ``[]`` from that listing and
    only found the app with ``?refresh=true``.
3.  A system folder is returned at the root and nowhere else, and one that
    somehow acquired a parent is put back.
4.  Three chained edits produce three versions, in order, each recording why
    it happened and what the bytes did.
5.  ``GET /api/apps/`` costs ONE query for N single-file apps. Control: 26
    queries for 25 apps — one per app, each opening its own session.
6.  An image row carries a thumbnail URL, the route serves a small JPEG, and
    a generated image is named in words.

RUN_MODE=agent: user_files/user_folders/build_jobs/apps are AGENT_ONLY.
Listed in COVERAGE_DEBT.txt as `# agent-mode`.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.agent.skills.base import SkillContext
from app.agent.skills.builtins.app_html import appskill, logo, steps as steps_mod, store
from app.agent.skills.builtins.app_html.skill import AppHtmlSkill
from app.services import library_service as lib

USER = "871bac24-c366-42b5-b224-8802c73aef3a"
CTX = SkillContext(user_id=USER, session_id="s-r21")

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


def _html(field="#2F6B3A", extra_control=""):
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        "<title>Whack a Mole</title>\n<style>\n"
        f":root{{--field:{field};--mole:#C4703A;--ink:#F7F4EC;}}\n"
        "body{margin:0;background:var(--field);color:var(--ink);}\n"
        "#board{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}\n"
        ".hole{min-width:72px;min-height:72px;border-radius:50%;}\n"
        "</style>\n</head>\n<body>\n"
        "<header id=\"hud\"><span id=\"score\">0</span><span id=\"clock\">30</span></header>\n"
        "<main id=\"board\"><button class=\"hole\" id=\"h0\">mole</button></main>\n"
        f"<button id=\"play\">Play</button>\n{extra_control}\n"
        "<script>\n"
        "let score = 0;\n"
        "function startRound(){ score = 0; render(); }\n"
        "const render = () => { document.getElementById('score').textContent = score; };\n"
        "document.getElementById('play').addEventListener('click', startRound);\n"
        "</script>\n</body>\n</html>\n"
    )


@pytest.fixture()
def apps_dir(tmp_path, monkeypatch):
    from app.config import settings
    root = tmp_path / "apps"
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(root))
    monkeypatch.setenv("TOUP_APP_SMOKE_TEST", "0")
    monkeypatch.setenv("TOUP_APP_VISUAL_REVIEW", "0")
    monkeypatch.setenv("TOUP_APP_MODEL_CALLS", "0")
    monkeypatch.setattr(settings, "agent_workspace_dir", str(tmp_path))
    monkeypatch.setattr(settings, "user_id", USER, raising=False)
    monkeypatch.setattr(lib, "_last_sync_at", {})
    monkeypatch.setattr(lib, "_sync_locks", {})
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


async def atool(sk, name, **args):
    return await sk.execute_tool(f"app_html__{name}", args, CTX)


# ═════════════════════════════════════════════════════════════════════
# 1. The logo is drawn DURING the build, and travels with the card
# ═════════════════════════════════════════════════════════════════════

def test_publishing_leaves_the_app_with_a_mark(skill, apps_dir):
    """No model on this box, so the mark is the monogram — the point is that
    the app is never handed over without one."""
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    assert logo.read_icon("whack") is None, "nothing drawn before the publish"
    tool(skill, "present_app", slug="whack")
    assert logo.read_icon("whack"), "published with no mark at all"


def test_the_mark_is_its_own_phase_of_the_publish():
    """A phase, not a side effect inside `present`. Round 20 drew the icon
    with no row of its own, so a slow or failed drawing was invisible."""
    assert "logo" in steps_mod.STEP_TYPES
    assert steps_mod.STEP_TYPES.index("logo") < steps_mod.STEP_TYPES.index("present")
    assert steps_mod.phase_label("logo", "running") == "Drawing the app's icon"
    # …and every phase has words of its own. A phase with no entry falls back
    # to "Working on your app", which is what a card should never say.
    for t in steps_mod.STEP_TYPES:
        assert steps_mod.phase_label(t, "done") != "Working on your app"


def test_the_card_payload_carries_the_drawing_not_just_a_flag(skill, apps_dir):
    """Item 1: "the present_app response must include the logo data".

    The live frame carries the SVG so the card paints without a second
    request; the copy persisted into a message's metadata carries only the
    validator, so a thread with thirty app cards does not carry thirty SVGs
    in its history for ever.
    """
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    tool(skill, "present_app", slug="whack")

    live = steps_mod.artifact_payload("whack", include_icon=True)
    persisted = steps_mod.artifact_payload("whack")

    assert live["icon_svg"].lstrip().startswith("<svg")
    assert live["icon_svg"] == logo.read_icon("whack")
    assert "icon_svg" not in persisted
    # Both halves know WHICH drawing — that is what lets a client cache it.
    assert live["icon_etag"] == persisted["icon_etag"] == logo.icon_etag("whack")
    assert len(persisted["icon_etag"]) == 32
    assert persisted["has_icon"] is True


def test_an_oversized_mark_is_not_inlined(skill, apps_dir):
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    tool(skill, "present_app", slug="whack")
    path = logo.icon_path("whack")
    with open(path, "w") as fh:
        fh.write("<svg viewBox='0 0 64 64'>" + "<rect width='1' height='1'/>"
                 * 1000 + "</svg>")
    assert os.path.getsize(path) > steps_mod.MAX_INLINE_ICON_BYTES
    payload = steps_mod.artifact_payload("whack", include_icon=True)
    assert "icon_svg" not in payload, "a 24 KB SVG must not ride a WS frame"
    assert payload["icon_etag"], "…but the validator still travels"


def test_repainting_the_app_makes_the_mark_stale(skill, apps_dir):
    """Round 20 keyed the icon on title + purpose, so an edit that changed
    every colour on screen left the tile in the old palette. The palette is
    part of what the mark IS."""
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(field="#2F6B3A"), brief=BRIEF)
    purpose = "A mole-whacking game for a child of six to eight."
    palette = ["#2f6b3a", "#c4703a", "#f7f4ec"]
    # A DESIGNED mark, as a reachable model would have left it. (A holding
    # monogram is provisional by definition and always reads as stale — that
    # is a different rule, and it would hide this one.)
    logo._store_icon(
        "whack", logo.fallback_icon("whack", "Whack a Mole", palette),
        source="model", title="Whack a Mole", purpose=purpose,
        subject="mole with mallet", palette=palette,
    )

    assert not logo.is_stale("whack", title="Whack a Mole", purpose=purpose,
                             palette=palette)
    # The same app, repainted.
    assert logo.is_stale("whack", title="Whack a Mole", purpose=purpose,
                         palette=["#101820", "#c4703a", "#f7f4ec"])
    # …and renaming it, or changing what it is for, still counts too.
    assert logo.is_stale("whack", title="Mole Panic", purpose=purpose,
                         palette=palette)


def test_a_padding_change_does_not_make_the_mark_stale(skill, apps_dir):
    """The other half of the rule: redrawing on every edit would spend two
    model calls on a gap change and make the tile flicker."""
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    tool(skill, "present_app", slug="whack")
    before = logo.read_icon("whack")
    tool(skill, "edit_app_file", slug="whack", old_string="gap:12px",
         new_string="gap:16px", reason="loosen the board")
    tool(skill, "present_app", slug="whack")
    assert logo.read_icon("whack") == before


def test_the_list_route_stats_the_icon_rather_than_reading_it(skill, apps_dir):
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    assert logo.has_icon("whack") is False
    tool(skill, "present_app", slug="whack")
    assert logo.has_icon("whack") is True
    assert logo.has_icon("no-such-app") is False


# ═════════════════════════════════════════════════════════════════════
# 4. The app's memory is versioned
# ═════════════════════════════════════════════════════════════════════

def test_three_chained_edits_produce_three_accurate_versions(skill, apps_dir):
    """The acceptance test the round asked for, in full.

    Every entry has to answer both questions a later editor arrives with:
    WHY was this done (the reason the editor gave) and WHAT did it actually
    do (derived from the file, so it cannot claim a change that is not
    there).
    """
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    assert appskill.read("whack").version == 1

    tool(skill, "edit_app_file", slug="whack",
         old_string="<span id=\"clock\">30</span>",
         new_string="<span id=\"clock\">45</span>",
         reason="make the round 45 seconds")
    tool(skill, "edit_app_file", slug="whack",
         old_string="<button id=\"play\">Play</button>",
         new_string="<button id=\"play\">Play</button>\n"
                    "<button id=\"pause\">Pause</button>",
         reason="add a pause button")
    tool(skill, "edit_app_file", slug="whack", old_string="gap:12px",
         new_string="gap:20px", reason="loosen the board")

    brief = appskill.read("whack")
    assert brief.version == 4, brief.history

    # In order, one per version, newest last.
    assert [h.split(" · ")[0] for h in brief.history] == [
        "- v1", "- v2", "- v3", "- v4",
    ]
    # …and each carries the app revision it was written against.
    assert [h.split(" · ")[1] for h in brief.history] == [
        "r1", "r2", "r3", "r4",
    ]

    whys = [h.split("why: ")[1].split(" · what:")[0] for h in brief.history]
    assert whys == ["built", "make the round 45 seconds", "add a pause button",
                    "loosen the board"]

    whats = [h.split(" · what: ")[1] for h in brief.history]
    # v2 changed a number inside an element: bytes move, structure does not.
    assert "+0 bytes" not in whats[1]
    # v3 added a control, and the derived half says which one.
    assert "pause" in whats[2], whats[2]
    # v4 is a CSS value of the same length: no bytes moved and no structure
    # changed — and it says BOTH, because "same size" alone reads as
    # "nothing happened".
    assert whats[3] == "same size; no structural change", whats[3]


def test_the_memory_is_written_on_a_publish_too(skill, apps_dir):
    """"Always updated after every edit (not just builds)" — and a publish is
    the moment the user's copy changes, which is exactly what the next
    editor needs to be able to tell from "what I changed"."""
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    tool(skill, "edit_app_file", slug="whack", old_string="gap:12px",
         new_string="gap:16px", reason="loosen the board")
    before = appskill.read("whack").version
    tool(skill, "present_app", slug="whack")
    after = appskill.read("whack")
    assert after.version == before + 1
    assert "published revision" in after.history[-1]
    assert "loosen the board" in after.history[-1]


def test_an_edit_hands_the_memory_back(skill, apps_dir):
    """Read before editing, write after. The model is given the version its
    own edit just produced, so the NEXT change in the chain is informed
    whether or not it remembered to call view_app_file."""
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    out = tool(skill, "edit_app_file", slug="whack", old_string="gap:12px",
               new_string="gap:16px", reason="loosen the board")
    body = str(out)
    assert "<app_brief" in body and 'version="2"' in body
    # …and it is MODEL-facing only. The user's line never mentions it.
    assert "brief" not in (getattr(out, "display", "") or "").lower()

    # A second edit in the same chain does not repeat a document the model is
    # already holding at that version… until the write moves it on again.
    out2 = tool(skill, "edit_app_file", slug="whack", old_string="gap:16px",
                new_string="gap:20px", reason="loosen it more")
    assert 'version="3"' in str(out2)


def test_the_version_is_the_memory_s_own_counter(skill, apps_dir):
    """Not the app's revision. `create_app_file` on an existing slug replaces
    the file and can leave the revision where it was; the memory has still
    moved on, and a reader has to be able to tell."""
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    tool(skill, "present_app", slug="whack")
    brief = appskill.read("whack")
    assert brief.version > store.read_manifest()["whack"].revision


def test_a_brief_written_by_this_round_still_round_trips(skill, apps_dir):
    """The history is parsed back out of markdown, so an entry that spans
    lines would come back as three entries. One line per version."""
    tool(skill, "create_app_file", slug="whack", title="Whack a Mole",
         html=_html(), brief=BRIEF)
    tool(skill, "edit_app_file", slug="whack", old_string="gap:12px",
         new_string="gap:16px", reason="loosen the board")
    first = appskill.read("whack")
    again = appskill.parse(first.render(), "whack")
    assert again.version == first.version
    assert again.history == first.history
    assert again.narrative == first.narrative


def test_a_word_ending_in_hex_is_not_an_id():
    """"facade" is six hex characters AND a word. A shorter id rule ate it."""
    assert lib.humanise_generated_name("sunset_over_a_facade.png") == (
        "Sunset over a facade.png")
    assert lib.humanise_generated_name("a_long_slow_decade.png") == (
        "A long slow decade.png")


# ═════════════════════════════════════════════════════════════════════
# 2 + 3 + 6. Files
# ═════════════════════════════════════════════════════════════════════

AGENT_KEY = "test-agent-key-0123456789"


@pytest.fixture
def agent_headers(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "agent_api_key", AGENT_KEY)
    monkeypatch.setattr(settings, "user_id", USER)
    return {"X-Agent-Key": AGENT_KEY}


@pytest_asyncio.fixture
async def api(apps_dir):
    from fastapi import FastAPI
    from app.api.library import router
    from app.api.files import router as files_router
    from app.config import settings
    app = FastAPI()
    app.include_router(router, prefix=settings.api_prefix)
    app.include_router(files_router, prefix=settings.api_prefix)
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://agent") as ac:
        yield ac


async def _names(api, headers, path=""):
    r = await api.get("/api/workspace/files", params={"path": path},
                      headers=headers)
    assert r.status_code == 200, r.text
    return [e["name"] for e in r.json()["files"]]


async def test_a_published_app_is_in_files_on_the_very_next_listing(
    api, agent_headers, skill, apps_dir,
):
    """Item 2. Before this, the listing straight after a publish was answered
    from the sync throttle and came back empty; only `?refresh=true` found
    the app."""
    # Warm the throttle exactly as a client would: list Files, then publish.
    assert await _names(api, agent_headers, "Apps") == []

    await atool(skill, "create_app_file", slug="whack", title="Whack a Mole",
                html=_html(), brief=BRIEF)
    await atool(skill, "present_app", slug="whack")

    assert await _names(api, agent_headers, "Apps") == ["Whack a Mole"]


async def test_the_row_carries_the_name_size_and_time_of_the_app(
    api, agent_headers, skill, apps_dir,
):
    await atool(skill, "create_app_file", slug="whack", title="Whack a Mole",
                html=_html(), brief=BRIEF)
    await atool(skill, "present_app", slug="whack")
    r = await api.get("/api/workspace/files", params={"path": "Apps"},
                      headers=agent_headers)
    row = r.json()["files"][0]
    assert row["name"] == "Whack a Mole"           # the title, never the slug
    assert row["kind"] == "app"
    assert row["app_slug"] == "whack"              # the artifact handle
    assert row["size"] == os.path.getsize(store.app_path("whack"))
    assert row["modified"], "no timestamp on the row"
    assert ".html" not in json.dumps(row)


async def test_republishing_does_not_produce_a_second_row(
    api, agent_headers, skill, apps_dir,
):
    await atool(skill, "create_app_file", slug="whack", title="Whack a Mole",
                html=_html(), brief=BRIEF)
    await atool(skill, "present_app", slug="whack")
    await atool(skill, "edit_app_file", slug="whack", old_string="gap:12px",
                new_string="gap:16px", reason="loosen the board")
    await atool(skill, "present_app", slug="whack")
    assert await _names(api, agent_headers, "Apps") == ["Whack a Mole"]


async def test_apps_is_returned_at_the_root_and_nowhere_else(
    api, agent_headers, apps_dir,
):
    """Item 3. Each folder's listing is its own contents."""
    r = await api.get("/api/workspace/files", params={"path": ""},
                      headers=agent_headers)
    root = r.json()["files"]
    assert [e["name"] for e in root if e["type"] == "dir"] == [
        "Documents", "Images", "Apps", "Uploads",
    ]
    for folder in ("Documents", "Images", "Uploads", "Apps"):
        r = await api.get("/api/workspace/files", params={"path": folder},
                          headers=agent_headers)
        inside = r.json()["files"]
        assert not [e for e in inside if e.get("system")], (
            f"a system folder was listed inside {folder}: {inside}"
        )


async def test_a_nested_system_folder_is_returned_to_the_root(
    api, agent_headers, apps_dir,
):
    """The self-heal. Whatever put Apps inside Uploads — a lost race, an old
    migration — it is put back rather than listed there."""
    from sqlalchemy import select
    from app.db.database import async_session_maker
    from app.db.models.user_file import UserFolder

    await api.get("/api/workspace/files", params={"path": ""},
                  headers=agent_headers)
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(UserFolder).where(UserFolder.user_id == USER)
        )).scalars().all()
        by_key = {f.system_key: f for f in rows}
        by_key["apps"].parent_id = by_key["uploads"].id
        await db.commit()

    # The listing must not show it under Uploads…
    r = await api.get("/api/workspace/files", params={"path": "Uploads"},
                      headers=agent_headers)
    assert [e["name"] for e in r.json()["files"]] == []
    # …and the next sync puts it back where it belongs.
    async with async_session_maker() as db:
        await lib.sync_user_library(db, USER, force=True)
        moved = await db.get(UserFolder, by_key["apps"].id)
        assert moved.parent_id is None
    assert "Apps" in await _names(api, agent_headers, "")


# ── 6. Thumbnails and readable names ─────────────────────────────────

def _png(path: str, size=(900, 600), colour=(40, 120, 90)) -> str:
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, colour).save(path, "PNG")
    return path


async def test_an_image_row_offers_a_thumbnail_and_the_route_serves_one(
    api, agent_headers, apps_dir, tmp_path,
):
    _png(str(tmp_path / "generated" / USER /
             f"{'a' * 32}_a-fox-in-deep-snow.png"))
    r = await api.get("/api/library/files", params={"folder": "all",
                                                    "refresh": "true"},
                      headers=agent_headers)
    items = r.json()["items"]
    img = next(f for f in items if f["kind"] == "image")

    assert img["thumbnail_url"].endswith(f"/library/files/{img['id']}/thumbnail")
    t = await api.get(img["thumbnail_url"].replace("/api", "/api", 1),
                      headers=agent_headers)
    assert t.status_code == 200, t.text
    assert t.headers["content-type"].startswith("image/jpeg")
    assert 0 < len(t.content) < 120_000, "a thumbnail, not the original"
    assert int(t.headers["content-length"]) < os.path.getsize(
        str(tmp_path / "generated" / USER / f"{'a' * 32}_a-fox-in-deep-snow.png")
    )
    # …and it is small in PIXELS, not just in bytes.
    from io import BytesIO
    from PIL import Image
    with Image.open(BytesIO(t.content)) as im:
        assert max(im.size) <= 320

    # Cached hard, and revalidated by a validator that moves with the source.
    etag = t.headers["etag"]
    again = await api.get(img["thumbnail_url"], headers={**agent_headers,
                                                         "If-None-Match": etag})
    assert again.status_code == 304


async def test_a_document_has_no_thumbnail_url_and_the_route_refuses(
    api, agent_headers, apps_dir, tmp_path,
):
    p = tmp_path / "generated" / USER / f"{'b' * 32}_quarterly-report.pdf"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    p.write_bytes(b"%PDF-1.4\n" + b"0" * 40000)
    r = await api.get("/api/library/files", params={"folder": "all",
                                                    "refresh": "true"},
                      headers=agent_headers)
    doc = next(f for f in r.json()["items"] if f["kind"] == "document")
    assert doc["thumbnail_url"] is None
    t = await api.get(f"/api/library/files/{doc['id']}/thumbnail",
                      headers=agent_headers)
    assert t.status_code == 415


async def test_a_generated_image_is_named_in_words(
    api, agent_headers, apps_dir, tmp_path,
):
    _png(str(tmp_path / "generated" / USER /
             f"{'c' * 32}_A_fox_in_deep_snow.png"))
    _png(str(tmp_path / "generated" / USER / f"{'d' * 32}_image_a1b2c3d4.png"),
         colour=(10, 20, 30))
    # A photo the agent persisted for the user keeps the name the camera gave
    # it — "Img 3145" is not an improvement on IMG_3145.
    _png(str(tmp_path / "generated" / USER / f"{'e' * 32}_IMG_3145.jpg"),
         colour=(90, 20, 30))

    names = await _names(api, agent_headers, "Images")
    assert sorted(names) == ["A fox in deep snow.png", "IMG_3145.jpg",
                             "Image.png"], names


def test_humanising_is_only_for_names_that_are_machine_minted():
    h = lib.humanise_generated_name
    assert h("A_fox_in_snow.png") == "A fox in snow.png"
    assert h("muscular-veiny-hand-steering-wheel.png") == (
        "Muscular veiny hand steering wheel.png")
    assert h("image_a1b2c3d4.png") == "Image.png"
    assert h("edited_9f0e1d2c.png") == "Edited.png"
    # Left alone
    assert h("IMG_3145.jpg") == "IMG_3145.jpg"
    assert h("DSC_0042.jpg") == "DSC_0042.jpg"
    assert h("Q3 report.pdf") == "Q3 report.pdf"


# ═════════════════════════════════════════════════════════════════════
# 5. The apps list is one query, whatever the library holds
# ═════════════════════════════════════════════════════════════════════

async def test_listing_apps_does_not_cost_a_query_per_app(monkeypatch, tmp_path):
    """Item 5, the measured half. Control on origin/main: 26 queries for 25
    apps — `_resolve_app_dir` opened its own session for every row the
    caller was already holding."""
    from sqlalchemy import event
    from app.config import settings
    from app.db.database import async_session_maker, engine
    from app.db.models import App
    import app.api.apps as apps_api

    uid = str(uuid.uuid4())
    monkeypatch.setattr(settings, "user_id", uid, raising=False)
    monkeypatch.setattr(settings, "agent_workspace_dir", str(tmp_path))
    root = tmp_path / "apps"
    root.mkdir(parents=True, exist_ok=True)
    async with async_session_maker() as db:
        for i in range(25):
            p = root / f"app-{i}.html"
            p.write_text("<!doctype html><html><body>x</body></html>")
            db.add(App(id=str(uuid.uuid4()), user_id=uid, name=f"App {i}",
                       slug=f"app-{i}", app_dir=str(p), source="html_artifact",
                       status="ready", platforms="web,ios", db_type="none",
                       deps_json="{}", files_json="{}"))
        await db.commit()

    class _Manager:
        """The real resolver's shape: a row it is not given costs a query."""
        async def _resolve_app_dir(self, app_id, app_dir=None):
            if app_dir and os.path.exists(app_dir):
                return app_dir
            async with async_session_maker() as db:
                row = await db.get(App, app_id)
                return row.app_dir if row else ""

        async def get_qr_url(self, app_id):
            return None

        async def get_web_url(self, app_id):
            return None

    async def _no_reconcile(user_id, db):
        return None

    monkeypatch.setattr(apps_api, "_app_manager", _Manager())
    monkeypatch.setattr(apps_api, "_reconcile_local", _no_reconcile)

    seen = {"n": 0}

    def _count(conn, cur, statement, params, ctx, many):
        seen["n"] += 1

    event.listen(engine.sync_engine, "before_cursor_execute", _count)
    try:
        rows = await apps_api.list_apps()
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _count)

    assert len(rows) == 25
    assert seen["n"] == 1, (
        f"{seen['n']} queries for 25 apps — one per app is the N+1 this "
        f"round removed"
    )
