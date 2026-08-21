"""Round 15 — the artifact's life OUTSIDE the turn that built it.

Round 12 shipped the pipeline that makes an app and the sandbox that runs it,
and stopped there: once the reply scrolled away, an app could only be reached
by finding the card again. It also shipped a client written against a contract
this backend implements differently — the phone listened for a frame nobody
sends and called routes that do not exist — and nothing on either side could
see it, because each half is internally consistent.

So these tests pin the SEAM, not the pipeline:

  1. `announce_ready` carries the SLUG. It is the handle every client acts on
     (`/api/artifacts/{slug}`, the runner, the chat card). Without it a client
     has only `app_id`, and must fetch the `apps` row to translate — a
     round-trip between the reply landing and the card appearing.
  2. The manifest is listable, renameable and deletable. The `apps` table
     cannot answer for these apps: `AppResponse` exposes neither size (it is
     inside an opaque `files_json` blob) nor revision, and a Files page that
     renders name · modified · size would be missing two of its three columns.
  3. A rename moves the TITLE and nothing else. The slug is the identity every
     chat card is keyed on and the revision is what makes an open runner
     reload; a rename that moved either would orphan cards or restart games.
  4. Delete is total — file, history, state, manifest row — because a record
     for something that is gone reads as breakage, not as deletion.
"""

from __future__ import annotations

import json
import os

import pytest

from app.agent.skills.builtins.app_html import steps as steps_mod, store
from app.agent.skills.builtins.app_html.store import AppStoreError


@pytest.fixture()
def apps_dir(tmp_path, monkeypatch):
    root = tmp_path / "apps"
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    store.ensure_root()
    return root


# Past MIN_HTML_BYTES: the store refuses a stub outright (the 952-byte `x.pdf`
# lesson), so a fixture app has to be a plausible document.
_APP_HTML = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>Fixture</title><style>"
    + ("body{margin:0;font-family:system-ui;background:#111;color:#eee}" * 6)
    + "</style></head><body><canvas id='c'></canvas><script>"
    + ("var s=0;function tick(){s++;}" * 6)
    + "</script></body></html>"
)


def _make(slug="snake", title="Nokia Snake Classic"):
    rec, _ = store.write_app(slug, title, _APP_HTML)
    return rec


# ── 1. The announcement carries the handle ───────────────────────────

@pytest.mark.asyncio
async def test_announce_ready_carries_the_slug(monkeypatch):
    sent = {}

    async def _capture(user_id, payload):
        sent.update(payload)

    monkeypatch.setattr(steps_mod, "_broadcast", _capture)
    await steps_mod.announce_ready(
        user_id="u1", job_id="j1", app_id="a1", title="Nokia Snake", slug="nokia-snake",
    )
    assert sent["type"] == "app_ready"
    assert sent["kind"] == "html_artifact"
    # Both, and for different readers: `kind` says WHICH pipeline (so the Expo
    # preview path is not taken), `slug` says WHICH APP (so the runner can open
    # it without a round-trip).
    assert sent["slug"] == "nokia-snake"
    assert sent["name"] == "Nokia Snake"


@pytest.mark.asyncio
async def test_present_passes_the_slug_through(apps_dir, monkeypatch):
    """The skill is what actually calls it — a default-None parameter that no
    caller fills is the same as not having one."""
    from app.agent.skills.base import SkillContext
    from app.agent.skills.builtins.app_html.skill import AppHtmlSkill

    _make("budget", "Budget Tracker")
    seen = {}

    async def _noop(*_a, **_k):
        return None

    async def _job(*_a, **_k):
        return "job-1"

    async def _app_row(*_a, **_k):
        return "app-1"

    async def _announce(**kw):
        seen.update(kw)

    monkeypatch.setattr(steps_mod, "ensure_job", _job)
    monkeypatch.setattr(steps_mod, "emit_step", _noop)
    monkeypatch.setattr(steps_mod, "finish_job", _noop)
    monkeypatch.setattr(steps_mod, "upsert_app_row", _app_row)
    monkeypatch.setattr(steps_mod, "announce_ready", _announce)

    skill = AppHtmlSkill()
    ctx = SkillContext(user_id="u1", session_id="s1")
    out = await skill.execute_tool("app_html__present_app", {"slug": "budget"}, ctx)

    assert seen.get("slug") == "budget"

    # ── Where the slug lives now ─────────────────────────────────────
    # This used to assert that the RESULT STRING contained
    # `/api/artifacts/budget` and `[[open_app:budget]]`, because those two
    # anchors were the only durable record of which app a turn handed over —
    # the clients regexed the slug back out of the tool's prose.
    #
    # The cost of that was paid by the user: both anchors are rendered in
    # chat, so every published app put an internal route and a directive
    # token in the transcript, and the chip drew a second Open button under a
    # card that already had one (round 18, items 2 and 6).
    #
    # The slug now rides a field — `app_artifact` on the assistant message,
    # `app_slug` on the tool record — so the prose does not have to carry it,
    # and this test asserts the prose does NOT.
    assert "/api/artifacts" not in out
    assert "[[" not in out
    assert "Budget Tracker" in out


# ── 2. The manifest answers the questions the apps table cannot ──────

def test_the_listing_carries_size_and_revision(apps_dir):
    _make("snake", "Nokia Snake Classic")
    rows = store.read_manifest()
    rec = rows["snake"].to_dict()
    assert rec["title"] == "Nokia Snake Classic"
    assert rec["size_bytes"] > 0
    assert rec["revision"] == 1
    assert rec["updated_at"]


# ── 3. A rename moves exactly one field ──────────────────────────────

def test_rename_moves_the_title_and_nothing_else(apps_dir):
    before = _make("snake", "Nokia Snake Classic").to_dict()
    after = store.retitle_record("snake", "Snake").to_dict()

    assert after["title"] == "Snake"
    # The identity every card, every runner and the apps row key on.
    assert after["slug"] == before["slug"]
    # A bump would make every OPEN runner reload — mid-game.
    assert after["revision"] == before["revision"]
    assert after["size_bytes"] == before["size_bytes"]
    assert after["created_at"] == before["created_at"]
    # …and the file is untouched.
    assert os.path.isfile(store.app_path("snake"))


def test_rename_persists_across_a_reread(apps_dir):
    _make("snake", "Nokia Snake Classic")
    store.retitle_record("snake", "Snake")
    assert store.read_manifest()["snake"].title == "Snake"


def test_rename_normalises_whitespace_and_refuses_an_empty_name(apps_dir):
    _make("snake")
    assert store.retitle_record("snake", "  Snake   II  ").title == "Snake II"
    with pytest.raises(AppStoreError):
        store.retitle_record("snake", "   ")


def test_renaming_an_unknown_app_is_a_miss_not_a_crash(apps_dir):
    assert store.retitle_record("ghost", "Ghost") is None


# ── 4. Delete is total ───────────────────────────────────────────────

def test_delete_removes_the_file_the_state_and_the_record(apps_dir):
    _make("snake")
    store.write_state("snake", {"high": 42})
    assert store.read_state("snake") == {"high": 42}

    assert store.delete_app("snake") is True
    assert not os.path.isfile(store.app_path("snake"))
    assert "snake" not in store.read_manifest()
    # Nothing an app saved may outlive the app: the next app to take the slug
    # would inherit a stranger's saved game.
    assert store.read_state("snake") == {}


def test_delete_of_an_unknown_app_is_not_an_error(apps_dir):
    assert store.delete_app("ghost") is False
