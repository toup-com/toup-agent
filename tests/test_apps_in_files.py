"""Round 15 — an app the agent built appears in Files.

Before this, `apps/` was in the library scanner's `_DENIED_DIRS` — correct for
the Expo era, when an app was a 452 MiB build tree. A round-12 app is ONE
self-contained `.html` file, and the user who asked for a snake game has no
reason to learn that it lives somewhere other than their documents.

What the entry has to be, and what it must not:

  * named "Nokia Snake Classic", never `nokia-snake-classic.html`, never a
    UUID, never a path;
  * `kind: "app"`, because it is opened to be RUN, not read;
  * openable ONLY through the sandboxed artifact frame — the ordinary file
    routes would serve model-authored script on an origin holding the
    account's session;
  * renameable and deletable, with both landing everywhere the app appears.
"""

from __future__ import annotations

import json
import os

import pytest

from app.db.models.user_file import SYSTEM_FOLDER_APPS
from app.services import library_service as lib

USER = "871bac24-c366-42b5-b224-8802c73aef3a"

APP_HTML = (
    "<!doctype html><html><head><meta charset='utf-8'><title>Snake</title>"
    "<style>body{background:#9ead86}</style></head><body>"
    "<canvas id='g'></canvas><script>const g=document.getElementById('g');"
    "</script></body></html>"
)


def _write_app(root: str, slug: str, title: str, *, presented: bool = True,
               html: str = APP_HTML) -> None:
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, f"{slug}.html"), "w") as fh:
        fh.write(html)
    manifest_path = os.path.join(root, "manifest.json")
    try:
        with open(manifest_path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {"version": 1, "apps": {}}
    data["apps"][slug] = {
        "slug": slug, "title": title,
        "created_at": "2026-08-21T00:00:00Z", "updated_at": "2026-08-21T00:00:00Z",
        "revision": 2, "size_bytes": len(html),
        "presented_at": "2026-08-21T00:00:01Z" if presented else None,
    }
    with open(manifest_path, "w") as fh:
        json.dump(data, fh)


@pytest.fixture
def apps_root(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "agent_workspace_dir", str(tmp_path))
    monkeypatch.setattr(settings, "user_id", USER, raising=False)
    root = str(tmp_path / "apps")
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", root)
    os.makedirs(root, exist_ok=True)
    return root


# ═════════════════════════════════════════════════════════════════════
# The scanner
# ═════════════════════════════════════════════════════════════════════

def test_a_presented_app_is_a_library_candidate(apps_root):
    _write_app(apps_root, "nokia-snake-classic", "Nokia Snake Classic")
    keys = [c.key for c in lib.iter_physical_candidates(USER)]
    assert "app:nokia-snake-classic.html" in keys


def test_bookkeeping_and_history_are_not_files(apps_root):
    _write_app(apps_root, "snake", "Snake")
    os.makedirs(os.path.join(apps_root, ".versions", "snake"), exist_ok=True)
    with open(os.path.join(apps_root, ".versions", "snake", "1-x.html"), "w") as fh:
        fh.write(APP_HTML)

    keys = [c.key for c in lib.iter_physical_candidates(USER)]
    assert keys == ["app:snake.html"], keys
    assert not any("manifest" in k or "versions" in k for k in keys)


def test_an_unpresented_draft_is_not_listed(apps_root):
    """A file the model wrote and abandoned mid-turn is not something the
    user asked to keep — and it would appear in Files before it appeared
    anywhere else."""
    _write_app(apps_root, "half-written", "Half Written", presented=False)
    assert [c.key for c in lib.iter_physical_candidates(USER)] == []


def test_the_expo_build_tree_stays_denied(apps_root, tmp_path):
    """`_DENIED_DIRS` still refuses the generic walk into apps/: a legacy
    project's App.tsx and its node_modules are not deliverables."""
    legacy = tmp_path / "apps" / "Nokia-Snake-Arcade"
    (legacy / "node_modules" / "left-pad").mkdir(parents=True)
    (legacy / "App.tsx").write_text("export default 1")
    (legacy / "node_modules" / "left-pad" / "index.js").write_text("x")
    _write_app(apps_root, "snake", "Snake")

    keys = [c.key for c in lib.iter_physical_candidates(USER)]
    assert keys == ["app:snake.html"], keys


# ═════════════════════════════════════════════════════════════════════
# Naming and classification
# ═════════════════════════════════════════════════════════════════════

def test_the_name_is_the_title_not_the_slug(apps_root):
    _write_app(apps_root, "nokia-snake-classic", "Nokia Snake Classic")
    assert lib.app_title("nokia-snake-classic") == "Nokia Snake Classic"


def test_a_missing_title_degrades_to_words_not_a_slug(apps_root):
    """Never show `nokia-snake-classic` to a person, even when the manifest
    has nothing better."""
    _write_app(apps_root, "nokia-snake-classic", "")
    assert lib.app_title("nokia-snake-classic") == "Nokia Snake Classic"


def test_an_app_is_kind_app_not_kind_document():
    """`kind_of` sees a name with no extension and a text/html mime and would
    call it a document — which is a client offering to READ it."""
    class _Row:
        storage_key = "app:snake.html"
        name = "Nokia Snake Classic"
        mime_type = "text/html"
    assert lib.kind_of_row(_Row()) == lib.KIND_APP
    assert lib.kind_of("Nokia Snake Classic", "text/html") == lib.KIND_DOCUMENT


def test_an_app_is_placed_in_the_apps_folder():
    from app.db.models.user_file import ORIGIN_AGENT, ORIGIN_UPLOAD
    assert lib.default_system_key(ORIGIN_AGENT, lib.KIND_APP) == SYSTEM_FOLDER_APPS
    # Origin never overrides it: an app is an app.
    assert lib.default_system_key(ORIGIN_UPLOAD, lib.KIND_APP) == SYSTEM_FOLDER_APPS


def test_the_entry_carries_the_artifact_handle_and_no_preview(apps_root):
    class _Row:
        id = "11111111-1111-1111-1111-111111111111"
        storage_key = "app:nokia-snake-classic.html"
        name = "Nokia Snake Classic"
        mime_type = "text/html"
        size_bytes = 4698
        origin = "agent"
        folder_id = None
        modified_at = None
        created_at = None

    e = lib.file_entry(_Row(), "Apps/Nokia Snake Classic")
    assert e["kind"] == "app"
    assert e["app_slug"] == "nokia-snake-classic"
    assert e["preview_url"] is None, (
        "a preview would render model-authored script on the API origin"
    )
    # Nothing internal in what a client displays.
    shown = f"{e['name']} {e['path']} {e['size_label']}"
    assert ".html" not in shown and "/app/" not in shown and e["id"] not in shown


def test_app_slug_of_only_answers_for_app_keys():
    assert lib.app_slug_of("app:snake.html") == "snake"
    assert lib.app_slug_of("gen:871bac24/report.pdf") is None
    assert lib.app_slug_of("ws:notes.md") is None
    assert lib.app_slug_of("nonsense") is None


# ═════════════════════════════════════════════════════════════════════
# Sync, rename, delete
# ═════════════════════════════════════════════════════════════════════

async def _sync(user_id: str):
    from app.db.database import async_session_maker
    async with async_session_maker() as db:
        await lib.sync_user_library(db, user_id, force=True)


async def _rows(user_id: str):
    from sqlalchemy import select
    from app.db.database import async_session_maker
    from app.db.models.user_file import UserFile
    async with async_session_maker() as db:
        return list((await db.execute(
            select(UserFile).where(UserFile.user_id == user_id,
                                   UserFile.deleted_at.is_(None))
        )).scalars().all())


async def test_sync_imports_the_app_under_its_title(apps_root, test_user_id, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "user_id", test_user_id, raising=False)
    _write_app(apps_root, "nokia-snake-classic", "Nokia Snake Classic")

    await _sync(test_user_id)
    rows = [r for r in await _rows(test_user_id) if r.storage_key.startswith("app:")]

    assert len(rows) == 1
    row = rows[0]
    assert row.name == "Nokia Snake Classic"
    assert row.mime_type == "text/html"
    assert row.size_bytes == len(APP_HTML)
    assert lib.kind_of_row(row) == lib.KIND_APP

    from app.db.database import async_session_maker
    from app.db.models.user_file import UserFolder
    async with async_session_maker() as db:
        folder = await db.get(UserFolder, row.folder_id)
    assert folder.system_key == SYSTEM_FOLDER_APPS and folder.name == "Apps"


async def test_rename_in_files_renames_the_app_everywhere(apps_root, test_user_id,
                                                          monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "user_id", test_user_id, raising=False)
    _write_app(apps_root, "snake", "Snake")
    await _sync(test_user_id)
    row = [r for r in await _rows(test_user_id) if r.storage_key.startswith("app:")][0]

    from app.db.database import async_session_maker
    async with async_session_maker() as db:
        f = await lib.get_file(db, test_user_id, row.id)
        await lib.update_file(db, test_user_id, f, name="Serpent Deluxe")

    assert lib.app_title("snake") == "Serpent Deluxe", "the manifest still says Snake"
    # The bytes never move — the slug, the URL and the chat card's link all
    # keep working.
    assert os.path.isfile(os.path.join(apps_root, "snake.html"))


async def test_delete_in_files_retires_the_app(apps_root, test_user_id, monkeypatch):
    """A half-delete is worse than none: the bytes gone while the Apps list
    still offers it and the artifact route still answers from the manifest."""
    from app.config import settings
    monkeypatch.setattr(settings, "user_id", test_user_id, raising=False)
    _write_app(apps_root, "snake", "Snake")
    await _sync(test_user_id)
    row = [r for r in await _rows(test_user_id) if r.storage_key.startswith("app:")][0]

    from app.db.database import async_session_maker
    async with async_session_maker() as db:
        f = await lib.get_file(db, test_user_id, row.id)
        await lib.delete_file(db, test_user_id, f)

    assert not os.path.isfile(os.path.join(apps_root, "snake.html"))
    assert "snake" not in lib.app_manifest()
    assert not [r for r in await _rows(test_user_id) if r.storage_key.startswith("app:")]


async def test_a_deleted_app_does_not_come_back_on_the_next_sync(
    apps_root, test_user_id, monkeypatch,
):
    from app.config import settings
    monkeypatch.setattr(settings, "user_id", test_user_id, raising=False)
    _write_app(apps_root, "snake", "Snake")
    await _sync(test_user_id)
    row = [r for r in await _rows(test_user_id) if r.storage_key.startswith("app:")][0]

    from app.db.database import async_session_maker
    async with async_session_maker() as db:
        f = await lib.get_file(db, test_user_id, row.id)
        await lib.delete_file(db, test_user_id, f)

    await _sync(test_user_id)
    assert not [r for r in await _rows(test_user_id) if r.storage_key.startswith("app:")]


# ═════════════════════════════════════════════════════════════════════
# The sandbox boundary
# ═════════════════════════════════════════════════════════════════════

def test_the_file_routes_refuse_an_app():
    """Model-authored HTML with model-authored script may be served ONLY by
    the artifact route: a cookieless origin, a strict CSP, and a frame
    sandboxed without allow-same-origin. Every route in library.py answers on
    an origin that carries the account's session."""
    from fastapi import HTTPException
    from app.api.library import _refuse_if_app

    class _App:
        storage_key = "app:snake.html"
        name = "Snake"
        mime_type = "text/html"

    class _Doc:
        storage_key = "gen:871bac24/report.pdf"
        name = "Q3 report.pdf"
        mime_type = "application/pdf"

    with pytest.raises(HTTPException) as exc:
        _refuse_if_app(_App())
    assert exc.value.status_code == 415
    assert "sandboxed" in exc.value.detail

    _refuse_if_app(_Doc())  # must not raise
