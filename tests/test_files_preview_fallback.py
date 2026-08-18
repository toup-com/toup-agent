"""The unrequested x.pdf (founder, iOS, 2026-08-18) — layer (c): the viewer.

Opening the attachment on iOS showed a black screen with

    {"detail":"{\\"detail\\":\\"No preview for application/pdf\\"}"}

Three things stacked up, all in app/api/files.py and pinned here:

1. `preview_url` is advertised for application/pdf and image/* (ws_chat.py,
   day_chats.py) but `preview_file` only rendered DOCX/PPTX/XLSX, so a PDF
   fell through to `HTTPException(415, "No preview for application/pdf")`.
   → PDFs and images are now served INLINE from /preview (a PDF is its own
     preview; WKWebView / <iframe> render it natively).
2. The platform proxy (`_proxy_file`) re-raised any agent ≥400 as
   `HTTPException(detail=<agent body as text>)` — wrapping the agent's
   JSON error in a second JSON envelope. → agent errors pass through
   verbatim (status, content-type, body), and the client's Accept header
   is forwarded so the agent can negotiate.
3. The 415 was JSON regardless of who asked. → a browser/WebView (Accept:
   text/html) gets a self-contained "Preview unavailable · Download" page;
   a fetch()/API caller gets structured JSON carrying download_url.
"""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import app.api.files as files_mod
from app.config import settings


class _User:
    id = "u1"
    is_active = True


class _Backend:
    """Minimal in-memory storage backend for the preview route."""
    def __init__(self, blobs):
        self.blobs = blobs
        self.root = "/tmp/never-used"

    def exists(self, key):
        return key in self.blobs

    def size(self, key):
        return len(self.blobs[key])

    def path(self, key):
        raise AssertionError("path() must not be needed for pdf/image/fallback")


PDF_BYTES = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

ATTACHMENTS = {
    "pdf": {"id": "pdf", "filename": "august-ai-news.pdf", "mime_type": "application/pdf",
            "size_bytes": len(PDF_BYTES), "storage_path": "u1/k_pdf"},
    "png": {"id": "png", "filename": "chart.png", "mime_type": "image/png",
            "size_bytes": len(PNG_BYTES), "storage_path": "u1/k_png"},
    "zip": {"id": "zip", "filename": "bundle.zip", "mime_type": "application/zip",
            "size_bytes": 12, "storage_path": "u1/k_zip"},
    "pptx": {"id": "pptx", "filename": "deck.pptx",
             "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
             "size_bytes": 12, "storage_path": "u1/k_pptx"},
}
BLOBS = {"u1/k_pdf": PDF_BYTES, "u1/k_png": PNG_BYTES, "u1/k_zip": b"PK\x03\x04" + b"0" * 8,
         "u1/k_pptx": b"PK\x03\x04" + b"0" * 8}


@pytest.fixture
def agent_app(monkeypatch):
    """The files router in AGENT run_mode with auth + attachment lookup +
    storage stubbed. Only the preview rendering logic is real."""
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


# ── 1. PDF / image are their own preview ───────────────────────────


async def test_pdf_preview_serves_the_pdf_inline(client):
    r = await client.get("/api/files/m1/pdf/preview?format=html")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/pdf")
    assert r.headers["content-disposition"].startswith("inline;")
    assert 'filename="august-ai-news.pdf"' in r.headers["content-disposition"]
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.content == PDF_BYTES


async def test_image_preview_serves_the_image_inline(client):
    r = await client.get("/api/files/m1/png/preview")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/png")
    assert r.headers["content-disposition"].startswith("inline;")
    assert r.content == PNG_BYTES


# ── 3. Unpreviewable → negotiated 415 ─────────────────────────────


async def test_unpreviewable_json_for_api_callers(client):
    r = await client.get("/api/files/m1/zip/preview", headers={"Accept": "*/*"})
    assert r.status_code == 415
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    # A single, plain-string detail — never a JSON string inside a string.
    assert body["detail"] == "Preview unavailable for application/zip"
    assert body["code"] == "preview_unavailable"
    assert body["mime_type"] == "application/zip"
    assert body["filename"] == "bundle.zip"
    assert body["download_url"] == f"{settings.api_prefix}/files/m1/zip"
    with pytest.raises(json.JSONDecodeError):
        json.loads(body["detail"])


async def test_unpreviewable_html_page_for_browsers_and_webviews(client):
    r = await client.get(
        "/api/files/m1/zip/preview?token=abc.def",
        headers={"Accept": "text/html,application/xhtml+xml,*/*;q=0.8"},
    )
    assert r.status_code == 415
    assert r.headers["content-type"].startswith("text/html")
    page = r.text
    assert "Preview unavailable" in page
    assert "bundle.zip" in page
    assert ">Download</a>" in page
    # The download link carries the same token the iframe/WebView used to
    # get here — it cannot send headers, so this is its only credential.
    assert f'href="{settings.api_prefix}/files/m1/zip?token=abc.def"' in page
    assert '{"detail"' not in page


async def test_html_fallback_escapes_filename(client, monkeypatch):
    evil = dict(ATTACHMENTS["zip"], filename='<img src=x onerror=alert(1)>.zip')

    async def _att(*a, **k):
        return evil
    monkeypatch.setattr(files_mod, "_load_attachment", _att)
    r = await client.get("/api/files/m1/zip/preview", headers={"Accept": "text/html"})
    assert r.status_code == 415
    assert "<img src=x" not in r.text
    assert "&lt;img src=x onerror=alert(1)&gt;.zip" in r.text


# ── 2. Platform proxy passes agent errors through verbatim ─────────


class _FakeResp:
    def __init__(self, status, headers, body):
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

    def build_request(self, method, url, headers=None, params=None, timeout=None):
        self.last_headers = headers or {}
        return ("req", url, params)

    async def send(self, req, stream=True):
        return self.resp


async def _proxy_with(monkeypatch, resp, accept=None):
    import app.services.agent_http as ah
    fake = _FakeClient(resp)
    monkeypatch.setattr(ah, "get_agent_http_client", lambda: fake)
    it, headers, status = await files_mod._proxy_file(
        "https://agent.example", "key", "m1/pdf/preview", {"format": "html"}, accept=accept,
    )
    chunks = [c async for c in it]
    return b"".join(chunks), headers, status, fake


async def test_proxy_passes_agent_json_error_through_without_double_encoding(monkeypatch):
    agent_body = json.dumps({"detail": "Preview unavailable for application/zip",
                             "code": "preview_unavailable"}).encode()
    body, headers, status, _ = await _proxy_with(
        monkeypatch, _FakeResp(415, {"content-type": "application/json"}, agent_body),
    )
    assert status == 415
    assert headers["content-type"] == "application/json"
    assert body == agent_body                      # byte-identical relay
    assert json.loads(body)["detail"] == "Preview unavailable for application/zip"
    assert b'\\"detail\\"' not in body              # the double-encoding is gone


async def test_proxy_passes_agent_html_fallback_through(monkeypatch):
    html = b"<!doctype html><title>x</title>Preview unavailable"
    body, headers, status, _ = await _proxy_with(
        monkeypatch, _FakeResp(415, {"content-type": "text/html; charset=utf-8"}, html),
        accept="text/html",
    )
    assert status == 415
    assert headers["content-type"].startswith("text/html")
    assert body == html


async def test_proxy_forwards_accept_header(monkeypatch):
    _, _, _, fake = await _proxy_with(
        monkeypatch, _FakeResp(200, {"content-type": "application/pdf"}, b"%PDF"),
        accept="text/html,*/*;q=0.8",
    )
    assert fake.last_headers["Accept"] == "text/html,*/*;q=0.8"
    assert fake.last_headers["X-Agent-Key"] == "key"


async def test_proxy_omits_accept_when_caller_had_none(monkeypatch):
    _, _, _, fake = await _proxy_with(
        monkeypatch, _FakeResp(200, {"content-type": "application/pdf"}, b"%PDF"),
    )
    assert "Accept" not in fake.last_headers


async def test_proxy_success_path_unchanged(monkeypatch):
    body, headers, status, _ = await _proxy_with(
        monkeypatch,
        _FakeResp(200, {"content-type": "application/pdf",
                        "content-disposition": 'inline; filename="a.pdf"',
                        "content-length": "4"}, b"%PDF"),
    )
    assert status == 200 and body == b"%PDF"
    assert headers["content-type"] == "application/pdf"
    assert headers["content-disposition"] == 'inline; filename="a.pdf"'
    assert "content-length" not in headers      # framing stays chunked (pre-existing rule)
