"""Round 14 — the picture in a thread is downloaded once, not once per visit.

Two defects, one symptom (a spinner on a card whose image the browser already
had on disk):

1. **No validator anywhere.** `/api/files/{m}/{a}` answered `Cache-Control:
   private, max-age=3600` and nothing else, so the browser had exactly two
   states for a 2.4 MB PNG — fresh for an hour, then a full unconditional
   re-fetch. There was no ETag to revalidate WITH, so an image looked at
   yesterday came down again in full today. The bytes behind an attachment id
   cannot change (`_persist` mints a uuid4 per file and writes its key once),
   so the correct answer is a strong ETag plus `immutable`, and a bodyless 304
   for a conditional GET.

2. **The proxy erased what did exist.** On `run_mode=platform` — where every
   real user is — `_proxy_file` forwarded neither the caller's
   `If-None-Match` nor the agent's `ETag`. The agent could answer 304 all day
   and no browser would ever ask it to.

Also pins the sibling half already shipped in `c1faac26` / `3584d54a`, which
had no test: `_persist` stamps intrinsic `width`/`height` on an image
attachment so a card can be laid out at the right shape on FIRST paint (the
placeholder was 1050x843 landscape for a 1050x1461 portrait image, and the
container nearly doubled in height when the picture decoded — the scroll
jump). Those fields ride the `Message.attachments` JSON column, so every read
path that spreads the dict carries them; the WS frame is built field by field
and names them explicitly.
"""
from __future__ import annotations

import io

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import app.api.files as files_mod
from app.config import settings


class _User:
    id = "u1"
    is_active = True


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
THUMB_BYTES = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 8
PDF_BYTES = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\ntrailer<<>>\n%%EOF\n"

ATTACHMENTS = {
    "png": {"id": "png", "filename": "chart.png", "mime_type": "image/png",
            "size_bytes": len(PNG_BYTES), "storage_path": "u1/k_png",
            "width": 1050, "height": 1461, "has_thumb": True},
    # No derivative on disk — too small to be worth one.
    "small": {"id": "small", "filename": "icon.png", "mime_type": "image/png",
              "size_bytes": len(PNG_BYTES), "storage_path": "u1/k_small"},
    "pdf": {"id": "pdf", "filename": "report.pdf", "mime_type": "application/pdf",
            "size_bytes": len(PDF_BYTES), "storage_path": "u1/k_pdf"},
}
BLOBS = {
    "u1/k_png": PNG_BYTES,
    "u1/k_png.thumb.webp": THUMB_BYTES,
    "u1/k_small": PNG_BYTES,       # deliberately no .thumb.webp beside it
    "u1/k_pdf": PDF_BYTES,
}


class _Backend:
    def __init__(self, blobs):
        self.blobs = blobs

    def exists(self, key):
        return key in self.blobs

    def size(self, key):
        return len(self.blobs[key])

    def path(self, key):
        raise AssertionError("path() must not be needed here")

    def open(self, key):
        return io.BytesIO(self.blobs[key])


@pytest.fixture
def agent_app(monkeypatch):
    """The files router in AGENT run_mode, auth + lookup + storage stubbed."""
    monkeypatch.setattr(settings, "run_mode", "agent")

    async def _user(request, token, db):
        return _User()

    async def _att(message_id, attachment_id, user_id, db):
        return ATTACHMENTS[attachment_id]

    async def _stream(key):
        yield BLOBS[key]

    monkeypatch.setattr(files_mod, "_get_user_for_file", _user)
    monkeypatch.setattr(files_mod, "_load_attachment", _att)
    monkeypatch.setattr(files_mod, "get_storage_backend", lambda: _Backend(BLOBS))
    monkeypatch.setattr(files_mod, "stream_file", _stream)

    async def _no_db():
        yield None

    app = FastAPI()
    app.include_router(files_mod.router, prefix="/api")
    app.dependency_overrides[files_mod.get_db] = _no_db
    return app


@pytest.fixture
async def client(agent_app):
    async with AsyncClient(transport=ASGITransport(app=agent_app), base_url="http://agent") as ac:
        yield ac


# ── 1. The download route validates and pins ───────────────────────


async def test_download_sends_a_strong_etag_and_immutable_freshness(client):
    r = await client.get("/api/files/m1/png")
    assert r.status_code == 200
    assert r.content == PNG_BYTES
    etag = r.headers["etag"]
    # Strong (no W/ prefix) and quoted — a weak validator would be refused
    # for the Range requests a video/PDF viewer makes.
    assert etag.startswith('"') and etag.endswith('"')
    assert not etag.startswith("W/")
    cc = r.headers["cache-control"]
    assert "immutable" in cc, cc
    assert "max-age=31536000" in cc, cc


async def test_conditional_get_is_a_bodyless_304(client):
    first = await client.get("/api/files/m1/png")
    etag = first.headers["etag"]

    again = await client.get("/api/files/m1/png", headers={"If-None-Match": etag})
    assert again.status_code == 304
    assert again.content == b""
    assert again.headers["etag"] == etag
    assert "immutable" in again.headers["cache-control"]
    # A 304 must not describe a body it is not sending.
    assert "content-length" not in again.headers
    assert "content-type" not in again.headers


async def test_a_stale_validator_still_gets_the_bytes(client):
    r = await client.get("/api/files/m1/png", headers={"If-None-Match": '"not-this-one"'})
    assert r.status_code == 200
    assert r.content == PNG_BYTES


async def test_if_none_match_accepts_the_list_and_weak_forms(client):
    etag = (await client.get("/api/files/m1/png")).headers["etag"]
    inner = etag.strip('"')

    for header in (f'"other", {etag}', f'W/"{inner}"', "*"):
        r = await client.get("/api/files/m1/png", headers={"If-None-Match": header})
        assert r.status_code == 304, header


# ── 2. The derivative is not the original ──────────────────────────


async def test_thumb_and_original_carry_different_validators(client):
    original = await client.get("/api/files/m1/png")
    thumb = await client.get("/api/files/m1/png?variant=thumb")

    assert thumb.status_code == 200
    assert thumb.content == THUMB_BYTES
    assert thumb.headers["content-type"].startswith("image/webp")
    assert thumb.headers["etag"] != original.headers["etag"], (
        "one ETag across both would let a client holding the thumbnail be told "
        "its copy of the full-size image is current"
    )
    assert "immutable" in thumb.headers["cache-control"]


async def test_a_thumb_etag_does_not_satisfy_the_original(client):
    thumb_etag = (await client.get("/api/files/m1/png?variant=thumb")).headers["etag"]
    r = await client.get("/api/files/m1/png", headers={"If-None-Match": thumb_etag})
    assert r.status_code == 200
    assert r.content == PNG_BYTES


async def test_a_thumb_request_that_falls_through_is_tagged_as_the_original(client):
    """`small` has no derivative on disk: the route serves the original, so it
    must say so. Tagging the fallback as a thumbnail would poison the client's
    cache entry for a derivative it never received."""
    fell_through = await client.get("/api/files/m1/small?variant=thumb")
    plain = await client.get("/api/files/m1/small")

    assert fell_through.status_code == 200
    assert fell_through.content == PNG_BYTES
    assert fell_through.headers["etag"] == plain.headers["etag"]
    assert "thumb" not in fell_through.headers["etag"]


# ── 3. The preview route validates too ─────────────────────────────


@pytest.mark.parametrize("aid,body,ctype", [
    ("png", PNG_BYTES, "image/png"),
    ("pdf", PDF_BYTES, "application/pdf"),
])
async def test_preview_validates_and_revalidates_free(client, aid, body, ctype):
    first = await client.get(f"/api/files/m1/{aid}/preview")
    assert first.status_code == 200
    assert first.content == body
    assert first.headers["content-type"].startswith(ctype)
    # Round 12's inline serving is untouched.
    assert first.headers["content-disposition"].startswith("inline;")
    assert first.headers.get("x-content-type-options") == "nosniff"

    etag = first.headers["etag"]
    assert "immutable" in first.headers["cache-control"]

    again = await client.get(f"/api/files/m1/{aid}/preview", headers={"If-None-Match": etag})
    assert again.status_code == 304
    assert again.content == b""


async def test_preview_and_download_of_one_file_share_a_validator(client):
    """Same bytes, two URLs — the same ETag, so a client that pulled the file
    through one route revalidates the other for free."""
    dl = await client.get("/api/files/m1/pdf")
    pv = await client.get("/api/files/m1/pdf/preview")
    assert dl.headers["etag"] == pv.headers["etag"]


# ── 4. The unpreviewable page still refuses to be cached ───────────


async def test_the_415_fallback_page_is_never_stored(client, monkeypatch):
    async def _att(*a, **k):
        return {"id": "zip", "filename": "bundle.zip", "mime_type": "application/zip",
                "size_bytes": 12, "storage_path": "u1/k_zip"}
    monkeypatch.setattr(files_mod, "_load_attachment", _att)
    monkeypatch.setattr(files_mod, "get_storage_backend",
                        lambda: _Backend({**BLOBS, "u1/k_zip": b"PK\x03\x04"}))

    r = await client.get("/api/files/m1/zip/preview", headers={"Accept": "text/html"})
    assert r.status_code == 415
    assert r.headers["cache-control"] == "no-store"
    assert "etag" not in r.headers


# ── 5. The platform proxy relays validators in both directions ─────


class _FakeResp:
    def __init__(self, status, headers, body=b""):
        self.status_code = status
        self.headers = headers
        self._body = body
        self.closed = False

    async def aread(self):
        return self._body

    async def aclose(self):
        self.closed = True

    async def aiter_bytes(self, chunk_size=65536):
        yield self._body


class _FakeClient:
    def __init__(self, resp):
        self.resp = resp
        self.last_headers = None

    def build_request(self, method, url, headers=None, params=None, timeout=None, **kw):
        self.last_headers = headers or {}
        return ("req", url, params)

    async def send(self, req, stream=True):
        return self.resp


async def _proxy(monkeypatch, resp, **kwargs):
    import app.services.agent_http as ah
    fake = _FakeClient(resp)
    monkeypatch.setattr(ah, "get_agent_http_client", lambda: fake)
    it, headers, status = await files_mod._proxy_file(
        "https://agent.example", "key", "m1/png", {}, **kwargs
    )
    body = b"".join([c async for c in it])
    return body, headers, status, fake


async def test_proxy_forwards_the_callers_conditional_headers(monkeypatch):
    _, _, _, fake = await _proxy(
        monkeypatch,
        _FakeResp(200, {"content-type": "image/png"}, PNG_BYTES),
        conditional={"if-none-match": '"png-72"'},
    )
    assert fake.last_headers["if-none-match"] == '"png-72"'
    assert fake.last_headers["X-Agent-Key"] == "key"


async def test_proxy_relays_the_agents_validator(monkeypatch):
    _, headers, status, _ = await _proxy(
        monkeypatch,
        _FakeResp(200, {"content-type": "image/png",
                        "etag": '"png-72"',
                        "cache-control": "private, max-age=31536000, immutable"},
                  PNG_BYTES),
    )
    assert status == 200
    assert headers["etag"] == '"png-72"'
    assert "immutable" in headers["cache-control"]


async def test_proxy_turns_an_upstream_304_into_a_bodyless_304(monkeypatch):
    resp = _FakeResp(304, {"etag": '"png-72"', "cache-control": "private, max-age=31536000, immutable"})
    body, headers, status, _ = await _proxy(monkeypatch, resp, conditional={"if-none-match": '"png-72"'})
    assert status == 304
    assert body == b""
    assert headers["etag"] == '"png-72"'
    # The upstream response is released even though nobody iterated it.
    assert resp.closed is True


async def test_conditional_headers_are_taken_from_the_request_verbatim():
    class _Req:
        headers = {"if-none-match": '"a"', "if-modified-since": "Wed, 20 Aug 2026 00:00:00 GMT",
                   "authorization": "Bearer secret"}

    out = files_mod._conditional_headers(_Req())
    assert out == {"if-none-match": '"a"',
                   "if-modified-since": "Wed, 20 Aug 2026 00:00:00 GMT"}
    assert files_mod._conditional_headers(None) == {}


async def test_platform_mode_answers_304_without_a_body(monkeypatch, agent_app):
    """End to end through the route, not just the helper: a 304 relayed as a
    StreamingResponse would frame a body the status forbids."""
    monkeypatch.setattr(settings, "run_mode", "platform")

    async def _proxy_info(user_id, db):
        return ("https://agent.example", "key")
    monkeypatch.setattr(files_mod, "_get_agent_proxy_info", _proxy_info)

    import app.services.agent_http as ah
    monkeypatch.setattr(ah, "get_agent_http_client",
                        lambda: _FakeClient(_FakeResp(304, {"etag": '"png-72"'})))

    async with AsyncClient(transport=ASGITransport(app=agent_app), base_url="http://plat") as ac:
        r = await ac.get("/api/files/m1/png", headers={"If-None-Match": '"png-72"'})
    assert r.status_code == 304
    assert r.content == b""
    assert r.headers["etag"] == '"png-72"'


# ── 6. The validator itself ────────────────────────────────────────


def test_attachment_etag_is_derived_not_read():
    a = files_mod.attachment_etag("abc", 1024)
    b = files_mod.attachment_etag("abc", 1024)
    assert a == b == '"abc-1024"'
    # Different bytes under the same id cannot happen, but a different id or
    # a different length must never collide.
    assert files_mod.attachment_etag("abd", 1024) != a
    assert files_mod.attachment_etag("abc", 1025) != a
    assert files_mod.attachment_etag("abc", 1024, "thumb") != a


def test_suffix_etag_names_a_derivation():
    base = files_mod.attachment_etag("abc", 10)
    assert files_mod._suffix_etag(base, "pdf") == '"abc-10-pdf"'
    assert files_mod._suffix_etag('W/"abc-10"', "pdf") == '"abc-10-pdf"'
    assert files_mod._suffix_etag(None, "pdf") is None


# ── 6b. The library validates without claiming immutability ────────


def test_library_validator_moves_when_the_bytes_do(tmp_path):
    """A library file id is a STABLE handle whose bytes a re-upload can
    replace — unlike an attachment id, which is minted per file. So its
    validator folds in size and mtime, and its freshness stays revalidating."""
    import app.api.library as library_mod

    p = tmp_path / "notes.md"
    p.write_text("one")

    class _F:
        id = "f1"

    first = library_mod._library_etag(_F(), str(p))
    assert first.startswith('"f1-')

    # Same bytes, same validator.
    assert library_mod._library_etag(_F(), str(p)) == first

    # Replaced bytes, different validator. (os.utime so the test does not
    # depend on the filesystem's mtime resolution.)
    import os as _os
    p.write_text("a longer body")
    st = _os.stat(str(p))
    _os.utime(str(p), (st.st_atime, st.st_mtime + 10))
    assert library_mod._library_etag(_F(), str(p)) != first

    # A missing file still yields a usable tag rather than raising.
    assert library_mod._library_etag(_F(), str(tmp_path / "gone")) == '"f1"'
    # And the library never claims immutability.
    assert "immutable" not in library_mod._LIBRARY_CACHE


async def test_the_library_proxy_relays_a_304_without_a_body(monkeypatch):
    """`if-none-match` has always been forwarded here, but until the library
    routes carried an ETag the agent had nothing to match, so this branch was
    unreachable. Making it reachable is what makes it a bug."""
    import app.api.workspace_proxy as wp
    import app.services.agent_http as ah

    resp = _FakeResp(304, {"etag": '"f1-3-100"', "content-type": "text/markdown",
                           "cache-control": "private, max-age=3600"})

    async def _proxy_info(user_id, db):
        return ("https://agent.example", "key")

    monkeypatch.setattr(wp, "_get_agent_proxy_info", _proxy_info)
    monkeypatch.setattr(ah, "get_agent_http_client", lambda: _FakeClient(resp))

    class _Req:
        method = "GET"
        headers = {"if-none-match": '"f1-3-100"'}

        class _QP:
            @staticmethod
            def multi_items():
                return []
        query_params = _QP()

    out = await wp._forward(_Req(), "u1", "library/files/f1/download", None)
    assert out.status_code == 304
    assert out.body == b""
    assert out.headers["etag"] == '"f1-3-100"'
    # A 304 describes no content, so it must not claim a type for one.
    assert "content-type" not in out.headers
    assert resp.closed is True


# ── 7. Bug 4 — the shape is known before the bytes arrive ──────────


async def test_persist_stamps_intrinsic_dimensions_on_an_image(monkeypatch, tmp_path):
    """A 3:4 portrait must persist as 3:4. The scroll jump was a client sizing
    a 1050x1461 image into a 1050x843 landscape box and re-laying out the
    thread when it decoded."""
    from PIL import Image as PILImage
    import app.agent.doc_generators as dg

    buf = io.BytesIO()
    PILImage.new("RGB", (1050, 1461), "white").save(buf, format="PNG")
    png = buf.getvalue()

    written = {}

    class _B:
        async def put(self, key, data):
            written[key] = data
            return key

    monkeypatch.setattr(dg, "get_storage_backend", lambda: _B())
    monkeypatch.setattr(dg, "_prewarm_preview", lambda *a, **k: None)

    att = await dg._persist(png, "portrait.png", "image/png", "u1")
    assert att.width == 1050
    assert att.height == 1461
    assert att.to_dict()["width"] == 1050
    assert att.to_dict()["height"] == 1461


async def test_persist_leaves_dimensions_absent_for_a_non_image(monkeypatch):
    import app.agent.doc_generators as dg

    class _B:
        async def put(self, key, data):
            return key

    monkeypatch.setattr(dg, "get_storage_backend", lambda: _B())
    monkeypatch.setattr(dg, "_prewarm_preview", lambda *a, **k: None)

    att = await dg._persist(b"%PDF-1.4", "a.pdf", "application/pdf", "u1")
    assert att.width is None and att.height is None
    assert att.has_thumb is False


def test_a_header_that_will_not_parse_is_not_fatal():
    """Best-effort by design — a picture is worth delivering even when Pillow
    cannot read its header."""
    import app.agent.doc_generators as dg
    assert dg._image_dimensions(b"not an image at all", "image/png") == (None, None)
    assert dg._image_dimensions(b"", "image/jpeg") == (None, None)


def test_the_live_ws_frame_names_width_and_height():
    """`ws_chat.on_attachment` builds its payload field by field, so a field
    nobody names is dropped on the LIVE path while working on reload — the
    exact asymmetry that makes a layout shift look intermittent."""
    import inspect
    import app.api.ws_chat as ws

    src = inspect.getsource(ws)
    assert 'payload["width"]' in src
    assert 'payload["height"]' in src


def test_the_session_fallback_advertises_the_same_urls_as_day_chats():
    """The mobile client falls back to /api/sessions whenever /api/day-chats
    fails. An attachment that arrives there without `thumb_url` pulls the
    multi-megabyte original into a card sized for a tenth of it."""
    import inspect
    import app.api.sessions as sessions_mod

    src = inspect.getsource(sessions_mod)
    assert "_attachment_urls" in src
