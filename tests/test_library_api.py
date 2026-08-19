"""The file library API on the agent (app/api/library.py) — agent lane.

RUN_MODE=agent: user_files/user_folders/messages/conversations are
AGENT_ONLY tables. Listed in COVERAGE_DEBT.txt as `# agent-mode`.

Every test drives the real router with a real JWT against a tmp workspace
laid out like a real tenant, so what is asserted is what a client sees:
no internals in any response body, correct metadata, working operations,
traversal refused, tenants isolated, physical files never moved by a
rename, chat attachment pointers still valid afterwards.
"""

from __future__ import annotations

import io
import json
import os
import re
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.services import library_service as lib

HEX = "0123456789abcdef"
_INTERNAL_MARKERS = ("/app/workspace", "generated/", "storage_path", "storage_key", "/data/agents",
                     "vibecoding", "apps/", "docker", "toup-code", ".whatsapp_auth", ".dashboard")
_HEX32_UNDERSCORE = re.compile(r"\b[0-9a-f]{32}_")


def _w(path, data=b"x" * 2048, mtime=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _assert_clean(payload, user_id: str):
    """No response may carry a physical path, storage key, tenant UUID
    directory or pipeline folder name."""
    text = json.dumps(payload)
    for m in _INTERNAL_MARKERS:
        assert m not in text, f"internal marker {m!r} leaked: {text[:400]}"
    assert user_id not in text, f"tenant uuid leaked: {text[:400]}"
    assert not _HEX32_UNDERSCORE.search(text), f"storage prefix leaked: {text[:400]}"


@pytest.fixture
def ws(tmp_path, monkeypatch):
    from app.config import settings
    from app.services import file_storage
    monkeypatch.setattr(settings, "agent_workspace_dir", str(tmp_path))
    monkeypatch.setattr(file_storage, "_backend", None)
    monkeypatch.setattr(lib, "_last_sync_at", {})
    monkeypatch.setattr(lib, "_sync_locks", {})
    return str(tmp_path)


AGENT_KEY = "test-agent-key-0123456789"


@pytest.fixture
def agent_headers(monkeypatch, test_user_id):
    """The agent's real auth: X-Agent-Key resolves to the tenant owner
    (settings.user_id). The platform proxy sends exactly this."""
    from app.config import settings
    monkeypatch.setattr(settings, "agent_api_key", AGENT_KEY)
    monkeypatch.setattr(settings, "user_id", test_user_id)
    return {"X-Agent-Key": AGENT_KEY}


@pytest_asyncio.fixture
async def api(ws):
    from fastapi import FastAPI
    from app.api.library import router
    from app.api.files import router as files_router
    from app.config import settings
    from app.db.database import engine
    from app.db.models import Base, UserSession
    # Bearer auth's per-session revocation check reads user_sessions, a
    # PLATFORM_ONLY table absent under RUN_MODE=agent. The isolation test
    # authenticates a SECOND user with a bearer JWT (the agent gate only
    # knows one user), so give this harness the table.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, tables=[UserSession.__table__])
    app = FastAPI()
    app.include_router(router, prefix=settings.api_prefix)
    app.include_router(files_router, prefix=settings.api_prefix)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as ac:
        yield ac


async def _seed_chat(user_id: str, attachments_by_role: list[tuple[str, list[dict]]]):
    """One conversation, one message per (role, attachments) tuple."""
    from app.db import async_session_maker
    from app.db.models import Conversation, Message
    ids = []
    async with async_session_maker() as db:
        conv = Conversation(id=str(uuid.uuid4()), user_id=user_id, title="t", channel="web")
        db.add(conv)
        await db.flush()
        for i, (role, atts) in enumerate(attachments_by_role):
            m = Message(id=str(uuid.uuid4()), conversation_id=conv.id, role=role, content="…",
                        created_at=datetime(2026, 8, 1 + i, 12, 0, 0), attachments=atts)
            db.add(m)
            ids.append(m.id)
        await db.commit()
    return ids


def _att(storage_path: str, filename: str, mime: str, size: int, created="2026-08-10T10:00:00+00:00"):
    return {"id": uuid.uuid4().hex, "filename": filename, "mime_type": mime, "size_bytes": size,
            "storage_path": storage_path, "created_at": created}


@pytest_asyncio.fixture
async def tenant(ws, test_user_id):
    """A tenant workspace with the fleet's real junk AND a real library."""
    uid = test_user_id
    other = str(uuid.uuid4())
    root = ws
    # internals
    _w(f"{root}/.dashboard/state.json"); _w(f"{root}/.whatsapp_auth/creds.json")
    _w(f"{root}/apps/Nokia-Snake-Arcade/App.tsx"); _w(f"{root}/vibecoding/todo/index.html")
    _w(f"{root}/{uid}/README.md", b"# Toup Agent Workspace\n")
    _w(f"{root}/{uid}/scratch.py")
    _w(f"{root}/generated/e2e-final/c2abccc5c7254d6eb480fcac37572070_A_fox_safe.png", b"F" * 5000)
    _w(f"{root}/generated/{other}/0000aaaa0000aaaa0000aaaa0000aaaa_theirs.pdf", b"P" * 4000)
    _w(f"{root}/generated/shared/5535d3bc3a4a4d80ae71745c99b5f85c_x.pdf", b"%PDF" + b"0" * 948)
    _w(f"{root}/generated/{uid}/b271948939724bb7a37d2855bde3e2c7_e2e_test.png")
    _w(f"{root}/generated/{uid}/c9a145bfa1934b609c4b9250708b91c8_hq_test.png")
    _w(f"{root}/A_fox_safe.png", b"F" * 5000)
    _w(f"{root}/edited_A.png"); _w(f"{root}/hq_test.png")
    # the library
    ts = datetime(2026, 7, 9, 23, 43, 46, tzinfo=timezone.utc).timestamp()
    img1 = f"{uid}/03291e2d652b4e20a5af0b3075ca3d71_muscular-veiny-hand-steering-wheel.png"
    img2 = f"{uid}/25c57c8afca14356902e10507ff195ba_muscular-veiny-hand-steering-wheel.png"
    up1 = f"{uid}/045a5029509d462293e94689f200986a_IMG_3145.jpg"
    up2 = f"{uid}/7d4d525217c1464aa3803883982c9ae9_Resume_2.pdf"
    doc1 = "shared/f112c3d24b3246798cfffed224f64bc8_uoft-events.docx"
    stub = "shared/5535d3bc3a4a4d80ae71745c99b5f85c_x.pdf"
    _w(f"{root}/generated/{img1}", b"M" * 3000, mtime=ts)
    _w(f"{root}/generated/{img2}", b"N" * 3100, mtime=ts + 100)
    _w(f"{root}/muscular-veiny-hand-steering-wheel.png", b"M" * 3000)     # workspace copy of img1
    _w(f"{root}/generated/{up1}", b"J" * 7000)
    _w(f"{root}/generated/{up2}", b"%PDF" + b"R" * 35000)
    _w(f"{root}/generated/{doc1}", b"D" * 40000)
    _w(f"{root}/generated/{doc1}.preview.pdf", b"%PDF" * 5000)
    _w(f"{root}/generated/summary.md", b"# summary\n" * 20)
    _w(f"{root}/{uid}/generated/plan.md", b"# plan\n" * 20)
    _w(f"{root}/{uid}/thoughts.md", b"# thoughts\n" * 10)
    _w(f"{root}/project-management-tools-comparison-2026.md", b"# pm\n" * 400)
    atts = [
        ("assistant", _att(img1, "muscular-veiny-hand-steering-wheel.png", "image/png", 3000, "2026-07-09T23:43:46+00:00")),
        ("assistant", _att(img2, "muscular-veiny-hand-steering-wheel.png", "image/png", 3100, "2026-07-10T03:50:01+00:00")),
        ("user", _att(up1, "IMG_3145.jpg", "image/jpeg", 7000, "2026-08-09T17:32:43+00:00")),
        ("user", _att(up2, "Resume_2.pdf", "application/pdf", 35004, "2026-08-18T21:55:08+00:00")),
        ("assistant", _att(doc1, "uoft-events.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 40000)),
        ("assistant", _att(stub, "x.pdf", "application/pdf", 952)),
    ]
    msg_ids = await _seed_chat(uid, [(role, [a]) for role, a in atts])
    return {"uid": uid, "other": other, "root": root, "msg_ids": msg_ids,
            "att_ids": [a["id"] for _, a in atts],
            "img1": img1, "img2": img2, "up1": up1, "up2": up2, "doc1": doc1}


async def _list(api, headers, path=""):
    r = await api.get("/api/workspace/files", params={"path": path}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


async def _folders(api, headers):
    r = await api.get("/api/library/folders", headers=headers)
    assert r.status_code == 200, r.text
    return {f["name"]: f for f in r.json()["items"]}


# ═════════════════════════════════════════════════════════════════════
# The clean library
# ═════════════════════════════════════════════════════════════════════

async def test_root_is_three_system_folders_and_no_internals(api, agent_headers, tenant):
    body = await _list(api, agent_headers)
    assert body["base"] == "/" and body["curated"] is True and body["path"] == ""
    names = [f["name"] for f in body["files"]]
    assert names == ["Documents", "Images", "Uploads"]
    for f in body["files"]:
        assert f["type"] == "dir" and f["size"] == 0 and f["modified"].endswith("Z")
        assert f["system"] in ("documents", "images", "uploads")
    _assert_clean(body, tenant["uid"])


async def test_junk_and_internals_never_appear_anywhere(api, agent_headers, tenant):
    seen = []
    for folder in ("", "Documents", "Images", "Uploads"):
        body = await _list(api, agent_headers, folder)
        _assert_clean(body, tenant["uid"])
        seen += [f["name"] for f in body["files"]]
    r = await api.get("/api/library/files", params={"folder": "all", "limit": 200}, headers=agent_headers)
    all_names = [f["name"] for f in r.json()["items"]]
    _assert_clean(r.json(), tenant["uid"])
    for junk in ("e2e_test.png", "hq_test.png", "edited_A.png", "A_fox_safe.png", "x.pdf", "README.md",
                 "scratch.py", "theirs.pdf", "uoft-events.docx.preview.pdf", "App.tsx", "index.html",
                 tenant["uid"], tenant["other"], "apps", "vibecoding", "generated", ".dashboard"):
        assert junk not in seen and junk not in all_names, junk


async def test_files_land_in_the_right_system_folder_with_clean_names(api, agent_headers, tenant):
    docs = await _list(api, agent_headers, "Documents")
    imgs = await _list(api, agent_headers, "Images")
    ups = await _list(api, agent_headers, "Uploads")
    assert sorted(f["name"] for f in docs["files"]) == sorted([
        "uoft-events.docx", "summary.md", "plan.md", "thoughts.md",
        "project-management-tools-comparison-2026.md",
    ])
    # two attachments with the same display name → second is uniquified;
    # the workspace copy of the first is NOT a third entry
    assert sorted(f["name"] for f in imgs["files"]) == [
        "muscular-veiny-hand-steering-wheel (2).png", "muscular-veiny-hand-steering-wheel.png",
    ]
    assert sorted(f["name"] for f in ups["files"]) == ["IMG_3145.jpg", "Resume_2.pdf"]
    for f in ups["files"]:
        assert f["origin"] == "upload"
    for f in imgs["files"] + docs["files"]:
        assert f["origin"] == "agent"


async def test_metadata_is_real(api, agent_headers, tenant):
    imgs = await _list(api, agent_headers, "Images")
    by = {f["name"]: f for f in imgs["files"]}
    f = by["muscular-veiny-hand-steering-wheel.png"]
    assert f["type"] == "file" and f["kind"] == "image" and f["mime"] == "image/png"
    assert f["size"] == 3000 and f["size_label"] == "2.9 KB"
    assert f["modified"] == "2026-07-09T23:43:46Z"       # the file's real mtime, UTC, Z
    assert f["path"] == "Images/muscular-veiny-hand-steering-wheel.png"
    assert f["id"] and "/" not in f["id"]
    ups = await _list(api, agent_headers, "Uploads")
    up = {x["name"]: x for x in ups["files"]}["Resume_2.pdf"]
    assert up["kind"] == "document" and up["mime"] == "application/pdf" and up["size"] == 35004
    r = await api.get(f"/api/library/files/{up['id']}", headers=agent_headers)
    e = r.json()
    assert e["created"] == "2026-08-18T21:55:08Z"      # the chat turn's timestamp
    assert e["download_url"].endswith(f"/library/files/{up['id']}/download")
    assert e["preview_url"].endswith(f"/library/files/{up['id']}/preview")
    _assert_clean(e, tenant["uid"])


async def test_listing_needs_auth(api, tenant):
    assert (await api.get("/api/workspace/files")).status_code == 401
    assert (await api.get("/api/library/files")).status_code == 401
    assert (await api.get("/api/library/folders")).status_code == 401


# ═════════════════════════════════════════════════════════════════════
# Pagination / search / sort
# ═════════════════════════════════════════════════════════════════════

async def test_pagination_and_search(api, agent_headers, ws, test_user_id):
    uid = test_user_id
    for i in range(60):
        _w(f"{ws}/generated/{uid}/{HEX * 2}_report-{i:02d}.md", b"# r\n" * 50)
    _w(f"{ws}/generated/{uid}/{HEX * 2}_invoice-final.pdf", b"%PDF" + b"I" * 5000)
    r = await api.get("/api/library/files", params={"folder": "all", "limit": 25}, headers=agent_headers)
    b = r.json()
    assert b["total"] == 61 and len(b["items"]) == 25 and b["next_offset"] == 25
    r = await api.get("/api/library/files", params={"folder": "all", "limit": 25, "offset": 50}, headers=agent_headers)
    b = r.json()
    assert len(b["items"]) == 11 and b["next_offset"] is None
    r = await api.get("/api/library/files", params={"q": "invoice"}, headers=agent_headers)
    assert [f["name"] for f in r.json()["items"]] == ["invoice-final.pdf"]
    r = await api.get("/api/library/files", params={"q": "report-0", "limit": 5}, headers=agent_headers)
    assert r.json()["total"] == 10 and len(r.json()["items"]) == 5
    r = await api.get("/api/library/files", params={"q": "%"}, headers=agent_headers)
    assert r.json()["total"] == 0                       # LIKE wildcards are escaped
    r = await api.get("/api/library/files", params={"folder": "all", "kind": "document", "sort": "size",
                                                    "order": "desc", "limit": 1}, headers=agent_headers)
    assert r.json()["items"][0]["name"] == "invoice-final.pdf"
    r = await api.get("/api/library/files", params={"folder": "all", "kind": "image"}, headers=agent_headers)
    assert r.json()["total"] == 0


# ═════════════════════════════════════════════════════════════════════
# Folders
# ═════════════════════════════════════════════════════════════════════

async def test_folder_create_rename_move_delete(api, agent_headers, tenant):
    r = await api.post("/api/library/folders", json={"name": "Reports"}, headers=agent_headers)
    assert r.status_code == 201, r.text
    fid = r.json()["id"]
    assert r.json()["path"] == "Reports" and r.json()["system"] is None
    # duplicate (case-insensitive) → 409; system-folder name → 409; bad names → 400
    assert (await api.post("/api/library/folders", json={"name": "reports"}, headers=agent_headers)).status_code == 409
    assert (await api.post("/api/library/folders", json={"name": "documents"}, headers=agent_headers)).status_code == 409
    for bad in ("../x", ".hidden", "a/b", "x" * 201):
        assert (await api.post("/api/library/folders", json={"name": bad}, headers=agent_headers)).status_code == 400, bad
    # nested + rename + move to root
    r = await api.post("/api/library/folders", json={"name": "2026", "parent_id": fid}, headers=agent_headers)
    sub = r.json()["id"]
    assert r.json()["path"] == "Reports/2026"
    r = await api.patch(f"/api/library/folders/{sub}", json={"name": "Q3"}, headers=agent_headers)
    assert r.json()["path"] == "Reports/Q3"
    r = await api.patch(f"/api/library/folders/{sub}", json={"parent_id": None}, headers=agent_headers)
    assert r.json()["path"] == "Q3" and r.json()["parent_id"] is None
    # cannot move a folder into its own subtree
    r = await api.patch(f"/api/library/folders/{sub}", json={"parent_id": fid}, headers=agent_headers)
    assert r.status_code == 200
    r = await api.patch(f"/api/library/folders/{fid}", json={"parent_id": sub}, headers=agent_headers)
    assert r.status_code == 400
    # non-empty delete → 409 with a code; recursive works
    r = await api.delete(f"/api/library/folders/{fid}", headers=agent_headers)
    assert r.status_code == 409 and r.json() == {"detail": "Folder is not empty", "code": "folder_not_empty"}
    r = await api.delete(f"/api/library/folders/{fid}", params={"recursive": "true"}, headers=agent_headers)
    assert r.status_code == 200
    assert (await api.get(f"/api/library/folders", params={"parent": fid}, headers=agent_headers)).status_code == 404
    _assert_clean(r.json(), tenant["uid"])


async def test_system_folders_can_be_renamed_but_not_deleted_or_moved(api, agent_headers, tenant):
    folders = await _folders(api, agent_headers)
    docs = folders["Documents"]
    assert (await api.delete(f"/api/library/folders/{docs['id']}", params={"recursive": "true"},
                             headers=agent_headers)).status_code == 400
    r = await api.patch(f"/api/library/folders/{docs['id']}", json={"name": "Reports"}, headers=agent_headers)
    assert r.status_code == 200 and r.json()["name"] == "Reports" and r.json()["system"] == "documents"
    r = await api.patch(f"/api/library/folders/{docs['id']}", json={"parent_id": folders["Images"]["id"]},
                        headers=agent_headers)
    assert r.status_code == 400
    # new agent documents still land in it under its new name
    _w(f"{tenant['root']}/generated/{tenant['uid']}/{HEX * 2}_late.md", b"# late\n" * 10)
    r = await api.get("/api/workspace/files", params={"path": "Reports", "refresh": "true"}, headers=agent_headers)
    assert "late.md" in [f["name"] for f in r.json()["files"]]


# ═════════════════════════════════════════════════════════════════════
# File operations
# ═════════════════════════════════════════════════════════════════════

async def test_rename_and_move_never_touch_the_bytes(api, agent_headers, tenant):
    ups = await _list(api, agent_headers, "Uploads")
    f = {x["name"]: x for x in ups["files"]}["Resume_2.pdf"]
    physical = f"{tenant['root']}/generated/{tenant['up2']}"
    assert os.path.isfile(physical)
    r = await api.patch(f"/api/library/files/{f['id']}", json={"name": "CV 2026.pdf"}, headers=agent_headers)
    assert r.status_code == 200 and r.json()["name"] == "CV 2026.pdf" and r.json()["path"] == "Uploads/CV 2026.pdf"
    folders = await _folders(api, agent_headers)
    r = await api.patch(f"/api/library/files/{f['id']}", json={"folder_id": folders["Documents"]["id"]},
                        headers=agent_headers)
    assert r.json()["path"] == "Documents/CV 2026.pdf"
    r = await api.patch(f"/api/library/files/{f['id']}", json={"folder_id": None}, headers=agent_headers)
    assert r.json()["path"] == "CV 2026.pdf" and r.json()["folder_id"] is None
    assert os.path.isfile(physical), "a rename/move must not move the bytes"
    # the chat card that points at the same bytes still resolves
    r = await api.get(f"/api/files/{tenant['msg_ids'][3]}/{tenant['att_ids'][3]}", headers=agent_headers)
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    # name clash → 409, bad name → 400, unknown id → 404
    ups = await _list(api, agent_headers, "Uploads")
    img = {x["name"]: x for x in ups["files"]}["IMG_3145.jpg"]
    r = await api.patch(f"/api/library/files/{img['id']}", json={"name": "CV 2026.pdf", "folder_id": None},
                        headers=agent_headers)
    assert r.status_code == 409
    assert (await api.patch(f"/api/library/files/{img['id']}", json={"name": "../x"}, headers=agent_headers)).status_code == 400
    assert (await api.patch(f"/api/library/files/{uuid.uuid4()}", json={"name": "x"}, headers=agent_headers)).status_code == 404


async def test_delete_removes_bytes_and_stays_deleted_across_syncs(api, agent_headers, tenant):
    docs = await _list(api, agent_headers, "Documents")
    f = {x["name"]: x for x in docs["files"]}["uoft-events.docx"]
    physical = f"{tenant['root']}/generated/{tenant['doc1']}"
    preview = physical + ".preview.pdf"
    assert os.path.isfile(physical) and os.path.isfile(preview)
    r = await api.delete(f"/api/library/files/{f['id']}", headers=agent_headers)
    assert r.status_code == 200
    assert not os.path.isfile(physical) and not os.path.isfile(preview)
    assert (await api.get(f"/api/library/files/{f['id']}", headers=agent_headers)).status_code == 404
    # a forced re-sync must not resurrect it (the message attachment still
    # names it, but the bytes are gone)
    r = await api.post("/api/library/sync", headers=agent_headers)
    assert r.status_code == 200 and r.json()["restored"] == 0
    docs = await _list(api, agent_headers, "Documents")
    assert "uoft-events.docx" not in [x["name"] for x in docs["files"]]
    # the chat card now reports the file as gone, not a 500
    r = await api.get(f"/api/files/{tenant['msg_ids'][4]}/{tenant['att_ids'][4]}", headers=agent_headers)
    assert r.status_code == 410


async def test_deleted_then_recreated_newer_file_comes_back(api, agent_headers, tenant):
    docs = await _list(api, agent_headers, "Documents")
    f = {x["name"]: x for x in docs["files"]}["thoughts.md"]
    physical = f"{tenant['root']}/{tenant['uid']}/thoughts.md"
    r = await api.delete(f"/api/library/files/{f['id']}", headers=agent_headers)
    assert r.status_code == 200 and not os.path.exists(physical)
    # the agent writes a NEW thoughts.md later (newer mtime than the tombstone)
    import time
    _w(physical, b"# fresh\n", mtime=time.time() + 5)
    r = await api.get("/api/workspace/files", params={"path": "Documents", "refresh": "true"}, headers=agent_headers)
    names = [x["name"] for x in r.json()["files"]]
    assert "thoughts.md" in names


async def test_files_that_vanish_from_disk_are_hidden_on_the_next_sync(api, agent_headers, tenant):
    docs = await _list(api, agent_headers, "Documents")
    assert "plan.md" in [x["name"] for x in docs["files"]]
    os.unlink(f"{tenant['root']}/{tenant['uid']}/generated/plan.md")
    r = await api.get("/api/workspace/files", params={"path": "Documents", "refresh": "true"}, headers=agent_headers)
    assert "plan.md" not in [x["name"] for x in r.json()["files"]]


async def test_upload_download_preview_content(api, agent_headers, tenant, _test_user):
    files = {"file": ("Q3 notes.md", io.BytesIO(b"# Q3\n\nhello **world**\n"), "text/markdown")}
    r = await api.post("/api/library/upload", files=files, headers=agent_headers)
    assert r.status_code == 201, r.text
    e = r.json()
    assert e["name"] == "Q3 notes.md" and e["path"] == "Uploads/Q3 notes.md" and e["origin"] == "upload"
    assert e["kind"] == "document" and e["mime"] == "text/markdown" and e["size"] == 22
    _assert_clean(e, tenant["uid"])
    # bytes live under the storage backend for this tenant, never at the root
    stored = [p for p in os.listdir(f"{tenant['root']}/generated/{tenant['uid']}") if p.endswith("_Q3 notes.md")]
    assert len(stored) == 1 and re.match(r"^[0-9a-f]{32}_", stored[0])
    assert not os.path.exists(f"{tenant['root']}/Q3 notes.md")
    # download (attachment), inline, ?token auth for embeds, no auth → 401
    r = await api.get(f"/api/library/files/{e['id']}/download", headers=agent_headers)
    assert r.status_code == 200 and r.content == b"# Q3\n\nhello **world**\n"
    assert r.headers["content-disposition"].startswith('attachment; filename="Q3 notes.md"')
    assert r.headers["x-content-type-options"] == "nosniff"
    r = await api.get(f"/api/library/files/{e['id']}/download", params={"inline": "1", "token": _test_user["token"]})
    assert r.status_code == 200 and r.headers["content-disposition"].startswith("inline;")
    assert (await api.get(f"/api/library/files/{e['id']}/download")).status_code == 401
    # text content + overwrite
    r = await api.get(f"/api/library/files/{e['id']}/content", headers=agent_headers)
    assert r.json()["content"].startswith("# Q3") and r.json()["path"] == "Uploads/Q3 notes.md"
    r = await api.put(f"/api/library/files/{e['id']}/content", json={"content": "# Q3 v2\n"}, headers=agent_headers)
    assert r.status_code == 200 and r.json()["size"] == 8
    r = await api.get(f"/api/library/files/{e['id']}/download", headers=agent_headers)
    assert r.content == b"# Q3 v2\n"
    # a binary upload previews inline and refuses the text route
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
    r = await api.post("/api/library/upload", files={"file": ("shot.png", io.BytesIO(png), "image/png")},
                       headers=agent_headers)
    pid = r.json()["id"]
    r = await api.get(f"/api/library/files/{pid}/preview", headers=agent_headers)
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content == png
    assert (await api.get(f"/api/library/files/{pid}/content", headers=agent_headers)).status_code == 415
    # unknown-type preview → 415 with a download_url in JSON, an HTML page for a browser
    r = await api.post("/api/library/upload", files={"file": ("data.bin", io.BytesIO(b"\x00\x01" * 100),
                                                              "application/octet-stream")}, headers=agent_headers)
    bid = r.json()["id"]
    r = await api.get(f"/api/library/files/{bid}/preview", headers=agent_headers)
    assert r.status_code == 415 and r.json()["download_url"].endswith(f"/library/files/{bid}/download")
    r = await api.get(f"/api/library/files/{bid}/preview", headers={**agent_headers, "accept": "text/html"})
    assert r.status_code == 415 and "Download" in r.text and "generated" not in r.text
    # empty upload → 400; upload into an explicit folder; oversized → 413
    assert (await api.post("/api/library/upload", files={"file": ("e.txt", io.BytesIO(b""), "text/plain")},
                           headers=agent_headers)).status_code == 400
    folders = await _folders(api, agent_headers)
    r = await api.post("/api/library/upload", files={"file": ("in-docs.txt", io.BytesIO(b"hi"), "text/plain")},
                       data={"folder_id": folders["Documents"]["id"]}, headers=agent_headers)
    assert r.json()["path"] == "Documents/in-docs.txt"
    from app.api import library as api_mod
    old = api_mod.MAX_UPLOAD_BYTES
    api_mod.MAX_UPLOAD_BYTES = 10
    try:
        r = await api.post("/api/library/upload", files={"file": ("big.bin", io.BytesIO(b"x" * 11), "application/octet-stream")},
                           headers=agent_headers)
        assert r.status_code == 413
    finally:
        api_mod.MAX_UPLOAD_BYTES = old


async def test_upload_filename_traversal_and_unicode(api, agent_headers, tenant):
    r = await api.post("/api/library/upload",
                       files={"file": ("../../etc/passwd", io.BytesIO(b"root:x"), "text/plain")}, headers=agent_headers)
    assert r.status_code == 201 and r.json()["name"] == "passwd"
    assert not os.path.exists(f"{tenant['root']}/etc/passwd")
    r = await api.post("/api/library/upload",
                       files={"file": ("Rapport été — Q3.pdf", io.BytesIO(b"%PDF" + b"x" * 2000), "application/pdf")},
                       headers=agent_headers)
    assert r.status_code == 201 and r.json()["name"] == "Rapport été — Q3.pdf"
    r = await api.get(f"/api/library/files/{r.json()['id']}/download", headers=agent_headers)
    assert "filename*=UTF-8''Rapport%20%C3%A9t%C3%A9" in r.headers["content-disposition"]


# ═════════════════════════════════════════════════════════════════════
# Traversal — nothing a client sends is a filesystem path
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("path", [
    "../", "..", "../../etc/passwd", "Documents/../..", "%2e%2e/%2e%2e", "/etc/passwd", "\\..\\..",
    "apps/Nokia-Snake-Arcade", "generated/shared", ".whatsapp_auth", "vibecoding",
])
async def test_compat_paths_that_are_not_in_the_tree_404(api, agent_headers, tenant, path):
    r = await api.get("/api/workspace/files", params={"path": path}, headers=agent_headers)
    assert r.status_code == 404, (path, r.text)
    r = await api.get("/api/workspace/file-content", params={"path": path}, headers=agent_headers)
    assert r.status_code == 404, (path, r.text)
    r = await api.get("/api/workspace/file-download", params={"path": path}, headers=agent_headers)
    assert r.status_code == 404, (path, r.text)
    r = await api.delete("/api/workspace/file", params={"path": path}, headers=agent_headers)
    assert r.status_code in (400, 404), (path, r.text)


async def test_uuid_dir_and_generated_dir_are_not_addressable_even_by_exact_name(api, agent_headers, tenant):
    for path in (tenant["uid"], f"generated/{tenant['uid']}", "generated", f"{tenant['uid']}/generated"):
        assert (await api.get("/api/workspace/files", params={"path": path}, headers=agent_headers)).status_code == 404
    # and the tree never lists them
    r = await api.get("/api/workspace/tree", params={"depth": 5}, headers=agent_headers)
    _assert_clean(r.json(), tenant["uid"])
    assert r.json()["base"] == "/"


async def test_write_cannot_escape(api, agent_headers, tenant):
    r = await api.post("/api/workspace/file-write", json={"path": "../escape.md", "content": "x"}, headers=agent_headers)
    assert r.status_code == 400
    r = await api.post("/api/workspace/create-dir", json={"path": "../../etc"}, headers=agent_headers)
    assert r.status_code == 400
    r = await api.post("/api/workspace/file-rename", json={"old_path": "Documents/summary.md", "new_path": "../out.md"},
                       headers=agent_headers)
    assert r.status_code == 400
    assert not os.path.exists(os.path.join(os.path.dirname(tenant["root"]), "escape.md"))


# ═════════════════════════════════════════════════════════════════════
# Tenant isolation
# ═════════════════════════════════════════════════════════════════════

async def test_another_user_cannot_see_or_touch_my_files(api, agent_headers, tenant):
    from app.db import async_session_maker
    from app.db.models import User
    from app.services.auth_service import create_access_token, get_password_hash
    other_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=other_id, email=f"o-{other_id[:8]}@example.com",
                    hashed_password=get_password_hash("pw-12345678"), name="Other"))
        await db.commit()
    other_headers = {"Authorization": f"Bearer {create_access_token(other_id)}"}
    _w(f"{tenant['root']}/generated/{other_id}/{HEX * 2}_theirs.md", b"# theirs\n")

    mine = await _list(api, agent_headers, "Uploads")
    my_file = mine["files"][0]
    theirs = await api.get("/api/library/files", params={"folder": "all"}, headers=other_headers)
    their_names = [f["name"] for f in theirs.json()["items"]]
    assert their_names == ["theirs.md"]                # not one of mine, not the shared/root docs of my tenant
    r = await api.get("/api/library/files", params={"folder": "all"}, headers=agent_headers)
    assert "theirs.md" not in [f["name"] for f in r.json()["items"]]
    for method, url in (("get", f"/api/library/files/{my_file['id']}"),
                        ("get", f"/api/library/files/{my_file['id']}/download"),
                        ("get", f"/api/library/files/{my_file['id']}/content"),
                        ("delete", f"/api/library/files/{my_file['id']}")):
        r = await getattr(api, method)(url, headers=other_headers)
        assert r.status_code == 404, (method, url, r.status_code)
    r = await api.patch(f"/api/library/files/{my_file['id']}", json={"name": "pwned.pdf"}, headers=other_headers)
    assert r.status_code == 404
    my_folders = await _folders(api, agent_headers)
    r = await api.delete(f"/api/library/folders/{my_folders['Uploads']['id']}", headers=other_headers)
    assert r.status_code == 404
    # my file is untouched
    r = await api.get(f"/api/library/files/{my_file['id']}", headers=agent_headers)
    assert r.status_code == 200 and r.json()["name"] == my_file["name"]


# ═════════════════════════════════════════════════════════════════════
# The path-based compat surface (what the shipped phone calls)
# ═════════════════════════════════════════════════════════════════════

async def test_compat_write_rename_mkdir_delete_upload_download(api, agent_headers, tenant, _test_user):
    # mkdir -p
    r = await api.post("/api/workspace/create-dir", json={"path": "Projects/Alpha"}, headers=agent_headers)
    assert r.status_code == 200 and r.json()["path"] == "Projects/Alpha"
    assert [f["name"] for f in (await _list(api, agent_headers, "Projects"))["files"]] == ["Alpha"]
    # write creates (auto-creating parents), then overwrites
    r = await api.post("/api/workspace/file-write", json={"path": "Projects/Alpha/notes.md", "content": "# a\n"},
                       headers=agent_headers)
    assert r.status_code == 200 and r.json()["path"] == "Projects/Alpha/notes.md" and r.json()["size"] == 4
    fid = r.json()["id"]
    r = await api.post("/api/workspace/file-write", json={"path": "Projects/Alpha/notes.md", "content": "# aa\n"},
                       headers=agent_headers)
    assert r.json()["id"] == fid and r.json()["size"] == 5
    r = await api.get("/api/workspace/file-content", params={"path": "Projects/Alpha/notes.md"}, headers=agent_headers)
    assert r.json()["content"] == "# aa\n" and r.json()["path"] == "Projects/Alpha/notes.md" and r.json()["mime"] == "text/markdown"
    # rename + move in one call; parents auto-created
    r = await api.post("/api/workspace/file-rename",
                       json={"old_path": "Projects/Alpha/notes.md", "new_path": "Projects/Beta/NOTES.md"}, headers=agent_headers)
    assert r.status_code == 200 and r.json()["new_path"] == "Projects/Beta/NOTES.md" and r.json()["id"] == fid
    assert (await api.get("/api/workspace/file-content", params={"path": "Projects/Alpha/notes.md"},
                          headers=agent_headers)).status_code == 404
    # folder rename
    r = await api.post("/api/workspace/file-rename", json={"old_path": "Projects/Beta", "new_path": "Projects/Gamma"},
                       headers=agent_headers)
    assert r.status_code == 200 and r.json()["new_path"] == "Projects/Gamma"
    # upload via multipart with `path` as a query param (the phone's shape)
    r = await api.post("/api/workspace/file-upload", params={"path": "Projects/Gamma"},
                       files={"file": ("photo.png", io.BytesIO(b"\x89PNG" + b"\x00" * 50), "image/png")}, headers=agent_headers)
    assert r.status_code == 201 and r.json()["path"] == "Projects/Gamma/photo.png" and r.json()["name"] == "photo.png"
    # empty path → Uploads
    r = await api.post("/api/workspace/file-upload", files={"file": ("root.txt", io.BytesIO(b"r"), "text/plain")},
                       headers=agent_headers)
    assert r.json()["path"] == "Uploads/root.txt"
    # download by path (bearer) and inline with ?token
    r = await api.get("/api/workspace/file-download", params={"path": "Projects/Gamma/photo.png"}, headers=agent_headers)
    assert r.status_code == 200 and r.content.startswith(b"\x89PNG") and r.headers["content-type"] == "image/png"
    r = await api.get("/api/workspace/file-download", params={"path": "Projects/Gamma/photo.png", "inline": "1",
                                                              "token": _test_user["token"]})
    assert r.status_code == 200 and r.headers["content-disposition"].startswith("inline;")
    # non-empty folder delete → 409 + code; recursive → gone with bytes
    r = await api.delete("/api/workspace/file", params={"path": "Projects"}, headers=agent_headers)
    assert r.status_code == 409 and r.json()["code"] == "folder_not_empty"
    r = await api.delete("/api/workspace/file", params={"path": "Projects", "recursive": "true"}, headers=agent_headers)
    assert r.status_code == 200
    assert (await api.get("/api/workspace/files", params={"path": "Projects"}, headers=agent_headers)).status_code == 404
    stored = [p for p in os.listdir(f"{tenant['root']}/generated/{tenant['uid']}") if p.endswith("_photo.png")]
    assert stored == []
    # system folder delete → 400
    r = await api.delete("/api/workspace/file", params={"path": "Uploads", "recursive": "true"}, headers=agent_headers)
    assert r.status_code == 400
    # the removed vibe preview route
    assert (await api.get("/api/workspace/preview/vibecoding/todo/index.html", headers=agent_headers)).status_code == 410


async def test_compat_file_content_resolves_legacy_physical_links(api, agent_headers, tenant):
    """Old toup://report?path=<physical> links in chat history keep opening,
    resolved through the manifest — never by touching the filesystem."""
    await _list(api, agent_headers)  # sync
    for legacy, expect in (
        ("generated/summary.md", "Documents/summary.md"),
        (f"{tenant['uid']}/thoughts.md", "Documents/thoughts.md"),
        ("thoughts.md", "Documents/thoughts.md"),
        ("generated/plan.md", "Documents/plan.md"),
        ("project-management-tools-comparison-2026.md", "Documents/project-management-tools-comparison-2026.md"),
    ):
        r = await api.get("/api/workspace/file-content", params={"path": legacy}, headers=agent_headers)
        assert r.status_code == 200, (legacy, r.text)
        assert r.json()["path"] == expect
        _assert_clean(r.json(), tenant["uid"])
    # a legacy-looking path that is NOT in the manifest is a 404 even if the
    # bytes exist on disk (apps/, README, another scope)
    for legacy in ("apps/Nokia-Snake-Arcade/App.tsx", f"{tenant['uid']}/README.md",
                   f"generated/{tenant['other']}/0000aaaa0000aaaa0000aaaa0000aaaa_theirs.pdf", "../etc/passwd"):
        assert (await api.get("/api/workspace/file-content", params={"path": legacy},
                              headers=agent_headers)).status_code == 404, legacy


async def test_compat_tree_is_the_virtual_tree(api, agent_headers, tenant):
    r = await api.get("/api/workspace/tree", params={"depth": 3}, headers=agent_headers)
    assert r.status_code == 200
    body = r.json()
    names = [e["name"] for e in body["tree"]]
    assert names[:3] == ["Documents", "Images", "Uploads"]
    assert "IMG_3145.jpg" in names and "uoft-events.docx" in names
    for e in body["tree"]:
        assert e["modified"] and e["modified"].endswith("Z")
        if e["type"] == "dir":
            assert e["size"] == 0
    _assert_clean(body, tenant["uid"])


# ═════════════════════════════════════════════════════════════════════
# Duplicates and lazy names (from the 2026-08-19 fleet dry run)
# ═════════════════════════════════════════════════════════════════════

async def test_identical_bytes_persisted_repeatedly_are_one_entry(api, agent_headers, ws, test_user_id):
    uid = test_user_id
    img = b"\x89PNG" + bytes(range(256)) * 300
    for i in range(4):
        _w(f"{ws}/generated/{uid}/{i:032x}_IMG_1100.jpg", img)      # four storage keys, one content
    # same name + same size, DIFFERENT bytes → a genuine second file
    _w(f"{ws}/generated/{uid}/{'f' * 32}_IMG_1100.jpg", b"\x89PNG" + bytes(reversed(range(256))) * 300)
    imgs = await _list(api, agent_headers, "Images")
    names = sorted(f["name"] for f in imgs["files"])
    assert names == ["IMG_1100 (2).jpg", "IMG_1100.jpg"], names


async def test_lazily_named_real_documents_are_shown_but_empty_office_stubs_are_not(api, agent_headers, ws, test_user_id):
    import zipfile
    uid = test_user_id

    def docx(path, text):
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("word/styles.xml", "<w:styles>" + "x" * 30000 + "</w:styles>")
            z.writestr("word/document.xml", f"<w:document><w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>")
    os.makedirs(f"{ws}/generated/shared", exist_ok=True)
    docx(f"{ws}/generated/shared/{HEX * 2}_x.docx", "Q3 revenue grew 14% on the back of the new pricing tiers.")
    docx(f"{ws}/generated/shared/{'a' * 32}_test.docx", "A second real document, named lazily by the model.")
    docx(f"{ws}/generated/shared/{'b' * 32}_Quarterly plan.docx", "")            # a stub with a nice name
    docs = await _list(api, agent_headers, "Documents")
    names = sorted(f["name"] for f in docs["files"])
    assert names == ["test.docx", "x.docx"], names
