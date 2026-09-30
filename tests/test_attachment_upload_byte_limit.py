"""`POST /api/chat/attachments` — the body ceiling is enforced on the BYTES
that arrive, not on the number a client declares.

The gap this pins (found 2026-09-28): `AttachmentBodyLimitMiddleware` read only
`Content-Length`. A body sent with no length (chunked), a malformed length or an
UNDERSTATED length went straight past the 26 MiB ceiling into fastapi's
`await request.form()`, which runs before any dependency or handler statement:
file parts spooled to the container's disk, non-file fields accumulated in
memory, and starlette's 1 MiB `max_file_size` is only a SPOOL threshold, not a
maximum. On the platform that disk and that RSS are shared by every tenant.

Every test here drives the REAL route through a real FastAPI app and its real
multipart parser — a raw ASGI scope/receive/send (so the test, not httpx,
decides which `Content-Length` goes on the wire) or httpx's ASGITransport with
a streaming body. The refusal must be the SAME typed 413 the declared-length
fast path sends (status, `X-Toup-Reason`, `detail`), never fastapi's generic
400 "There was an error parsing the body" or a 500, and the handler — the
per-part read, the ingest, the store — must never run.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import httpx
import pytest
from fastapi import FastAPI, Request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures"))
import attachments as fx  # noqa: E402

from app.agent import attachment_limits as L  # noqa: E402
from app.api import chat_attachments as CA  # noqa: E402

MiB = 1024 * 1024
CAP = CA.BODY_HARD_MAX
PATH = "/api/chat/attachments"
BOUNDARY = "toupRxLimitBoundary7c1e"
CONTENT_TYPE = f"multipart/form-data; boundary={BOUNDARY}".encode()
USER = "user-rxlimit0929"
#: starlette >= 0.40 can bound one non-file part itself; the pinned 0.35.1
#: (fastapi 0.109, production and CI) cannot. Read from starlette, not from the
#: module under test, so this file runs — and fails for the real reason —
#: against code that predates the fix.
PARSER_BOUNDS_A_PART = "max_part_size" in inspect.signature(Request.form).parameters


# ── a real app around the real route ─────────────────────────────────────

class _Who:
    id = USER


@dataclass
class Spies:
    read_capped: int = 0
    ingest: List[int] = None  # type: ignore[assignment]
    other_bytes: List[int] = None  # type: ignore[assignment]


@pytest.fixture
def spies(monkeypatch, tmp_path):
    """Local ingest path (the agent side), with the handler's first statement
    and the ingest instrumented so a test can say they NEVER ran."""
    fx.local_storage(tmp_path)
    CA._inflight.clear()
    s = Spies(ingest=[], other_bytes=[])

    real_read_capped = CA._read_capped

    async def _read_capped_spy(file):
        s.read_capped += 1
        return await real_read_capped(file)

    real_ingest = CA.ingest_and_store

    async def _ingest_spy(data, filename, declared_mime, user_id):
        s.ingest.append(len(data))
        if len(data) > 4 * MiB:
            # The boundary test's 25 MiB file: that it ARRIVED intact is the
            # point; running the real extractor over it only costs time.
            return {
                "attachment_id": "att-boundary", "sha256": "0" * 64,
                "name": filename, "mime": "text/plain", "kind": "document",
                "size_bytes": len(data), "attachment": {"size_bytes": len(data)},
                "ingest": {"status": "ok"}, "_deduped": False,
            }
        return await real_ingest(data, filename, declared_mime, user_id)

    async def _no_proxy(*_a, **_k):
        return None

    monkeypatch.setattr(CA, "_read_capped", _read_capped_spy)
    monkeypatch.setattr(CA, "ingest_and_store", _ingest_spy)
    monkeypatch.setattr(CA, "agent_proxy_info", _no_proxy)
    monkeypatch.setattr(CA, "serving_locally", lambda: True)
    yield s
    from app.services import file_storage

    file_storage._backend = None


def _build_app(spies: Spies) -> FastAPI:
    app = FastAPI()
    app.include_router(CA.router, prefix="/api")

    @app.post("/api/workspace/sink")
    async def _other_route(request: Request):
        n = 0
        async for chunk in request.stream():
            n += len(chunk)
        spies.other_bytes.append(n)
        return {"n": n}

    async def _no_db():
        yield None

    app.dependency_overrides[CA.get_current_user] = lambda: _Who()
    app.dependency_overrides[CA.get_db] = _no_db
    app.add_middleware(CA.AttachmentBodyLimitMiddleware)
    return app


@pytest.fixture
def app(spies):
    return _build_app(spies)


def _blobs(tmp_path) -> int:
    n = 0
    for _dir, _sub, files in os.walk(str(tmp_path)):
        n += len(files)
    return n


# ── multipart bodies ─────────────────────────────────────────────────────

def _file_part(name: str, filename: str, ctype: str, data: bytes) -> bytes:
    return (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + data + b"\r\n"


def _field_part(name: str, value: bytes) -> bytes:
    return (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
    ).encode() + value + b"\r\n"


_CLOSE = f"--{BOUNDARY}--\r\n".encode()


def _upload_body(data: bytes, filename: str = "notes.txt",
                 ctype: str = "text/plain", *, extra: bytes = b"") -> bytes:
    return extra + _file_part("file", filename, ctype, data) + _CLOSE


def _exact_body(total: int, file_bytes: int) -> bytes:
    """A valid upload of `file_bytes`, padded to exactly `total` bytes with a
    multipart EPILOGUE (bytes after the closing boundary, which the parser
    skips) — so the byte counter sees every one of them and the handler still
    gets a well-formed form."""
    body = _upload_body(b"a" * file_bytes)
    assert len(body) <= total, "the file alone overshoots the target size"
    return body + b"e" * (total - len(body))


def _chunks(body: bytes, size: int = MiB) -> List[bytes]:
    return [body[i:i + size] for i in range(0, len(body), size)] or [b""]


# ── a raw ASGI client: the TEST decides the headers ─────────────────────

@dataclass
class Result:
    status: int
    headers: Dict[bytes, bytes]
    body: bytes
    starts: int
    receives: int
    chunks: int

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


async def _asgi(app, chunks: Iterable[bytes], *, content_length: Optional[bytes] = None,
                path: str = PATH, method: str = "POST",
                content_type: Optional[bytes] = CONTENT_TYPE,
                chunked: bool = False) -> Result:
    chunks = list(chunks)
    state = {"receives": 0}
    sent: List[Dict[str, Any]] = []

    async def receive():
        i = state["receives"]
        state["receives"] += 1
        if i < len(chunks):
            return {"type": "http.request", "body": chunks[i],
                    "more_body": i < len(chunks) - 1}
        await asyncio.sleep(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    headers = [(b"host", b"testserver")]
    if content_type is not None:
        headers.append((b"content-type", content_type))
    if content_length is not None:
        headers.append((b"content-length", content_length))
    if chunked:
        headers.append((b"transfer-encoding", b"chunked"))
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": path,
        "raw_path": path.encode(), "root_path": "", "query_string": b"",
        "headers": headers, "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    await app(scope, receive, send)
    starts = [m for m in sent if m["type"] == "http.response.start"]
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return Result(
        status=starts[0]["status"] if starts else 0,
        headers={k.lower(): v for k, v in (starts[0].get("headers") or [])} if starts else {},
        body=body, starts=len(starts), receives=state["receives"], chunks=len(chunks),
    )


def _assert_typed_413(r: Result) -> None:
    assert r.status == 413, (r.status, r.body[:300])
    assert r.starts == 1, "a second response was sent"
    assert r.headers.get(b"x-toup-reason") == L.REASON_TOO_LARGE.encode()
    assert r.json() == {"detail": L.too_large_detail(None)}


def _assert_handler_never_ran(spies: Spies, tmp_path) -> None:
    assert spies.read_capped == 0, "the upload handler ran on an over-cap body"
    assert spies.ingest == [], "an over-cap body reached the ingest"
    assert _blobs(tmp_path) == 0, "an over-cap body left something in storage"


def _over_cap_upload() -> bytes:
    # One file part whose FILE is under nothing in particular — it is the body
    # that is over the ceiling, by two MiB.
    return _upload_body(b"z" * (CAP + 2 * MiB))


# ── no declared length ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_body_with_no_length_over_the_cap_is_the_typed_413(app, spies, tmp_path):
    chunks = _chunks(_over_cap_upload())
    r = await _asgi(app, chunks, content_length=None)
    _assert_typed_413(r)
    _assert_handler_never_ran(spies, tmp_path)
    # Feeding stopped at the chunk that crossed the ceiling: the rest of the
    # body was never handed to the multipart parser.
    crossing = CAP // MiB + 1
    assert r.receives <= crossing, (r.receives, r.chunks)
    assert r.receives < r.chunks


@pytest.mark.asyncio
async def test_a_chunked_stream_over_the_cap_is_the_typed_413_through_httpx(app, spies, tmp_path):
    """httpx sends a generator body with `Transfer-Encoding: chunked` and NO
    `Content-Length` — the shape a hostile or broken client actually uses."""
    body = _over_cap_upload()
    sent_chunks = {"n": 0}

    async def _stream():
        for c in _chunks(body):
            sent_chunks["n"] += 1
            yield c

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            PATH, content=_stream(),
            headers={"content-type": CONTENT_TYPE.decode()},
        )
        assert "content-length" not in {k.lower() for k in resp.request.headers.keys()}
    assert resp.status_code == 413, resp.text[:300]
    assert resp.headers.get("x-toup-reason") == L.REASON_TOO_LARGE
    assert resp.json() == {"detail": L.too_large_detail(None)}
    _assert_handler_never_ran(spies, tmp_path)


@pytest.mark.asyncio
async def test_an_explicitly_chunked_raw_request_over_the_cap_is_refused(app, spies, tmp_path):
    r = await _asgi(app, _chunks(_over_cap_upload(), 256 * 1024), chunked=True)
    _assert_typed_413(r)
    _assert_handler_never_ran(spies, tmp_path)


# ── a declared length that lies ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_an_understated_length_is_enforced_by_the_count(app, spies, tmp_path):
    r = await _asgi(app, _chunks(_over_cap_upload()), content_length=b"1000")
    _assert_typed_413(r)
    _assert_handler_never_ran(spies, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [b"abc", b"-5", b"", b"12abc", b"1e9"])
async def test_a_malformed_length_over_the_cap_is_the_typed_413_never_a_500(app, spies, tmp_path, bad):
    r = await _asgi(app, _chunks(_over_cap_upload()), content_length=bad)
    _assert_typed_413(r)
    _assert_handler_never_ran(spies, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [b"abc", b"-5"])
async def test_a_malformed_length_under_the_cap_is_served_normally(app, spies, tmp_path, bad):
    body = _upload_body(fx.png_image(), "a.png", "image/png")
    r = await _asgi(app, _chunks(body), content_length=bad)
    assert r.status == 200, r.body[:300]
    assert r.json()["attachment_id"]
    assert spies.ingest == [len(fx.png_image())]
    assert _blobs(tmp_path) > 0


# ── what the multipart parser would otherwise accumulate ───────────────

@pytest.mark.asyncio
async def test_one_huge_non_file_field_over_the_cap_is_refused(app, spies, tmp_path):
    """A non-file field is held IN MEMORY by starlette's parser, whole."""
    body = (_field_part("client_attachment_id", b"q" * (CAP + MiB))
            + _file_part("file", "a.png", "image/png", fx.png_image()) + _CLOSE)
    r = await _asgi(app, _chunks(body), content_length=None)
    if PARSER_BOUNDS_A_PART:
        # starlette >= 0.40 bounds a non-file part itself (the route passes
        # 64 KiB) and refuses it long before the ceiling. Production pins
        # fastapi 0.109 / starlette 0.35.1, which has no such bound — there
        # the byte count is the only thing between this field and the RSS.
        assert r.status == 400, (r.status, r.body[:300])
        assert r.starts == 1
        assert "maximum size" in r.json()["detail"]
    else:
        _assert_typed_413(r)
    assert r.receives < r.chunks
    _assert_handler_never_ran(spies, tmp_path)


@pytest.mark.asyncio
async def test_a_flood_of_small_fields_is_a_bounded_refusal(app, spies, tmp_path):
    fields = b"".join(_field_part(f"f{i}", b"v") for i in range(64))
    body = fields + _file_part("file", "a.png", "image/png", fx.png_image()) + _CLOSE
    r = await _asgi(app, _chunks(body), content_length=None)
    assert r.status == 400, (r.status, r.body[:300])
    assert r.starts == 1
    assert "field" in r.json()["detail"].lower()
    _assert_handler_never_ran(spies, tmp_path)


@pytest.mark.asyncio
async def test_a_flood_of_file_parts_is_a_bounded_refusal(app, spies, tmp_path):
    """The route takes ONE file. Each extra file part is a spooled temp file
    the parser opens before any handler could say no."""
    parts = b"".join(_file_part("file", f"p{i}.png", "image/png", b"x" * 64) for i in range(16))
    r = await _asgi(app, _chunks(parts + _CLOSE), content_length=None)
    assert r.status == 400, (r.status, r.body[:300])
    assert r.starts == 1
    assert "file" in r.json()["detail"].lower()
    _assert_handler_never_ran(spies, tmp_path)


# ── the exact boundary ──────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [False, True])
async def test_a_body_of_exactly_the_cap_reaches_the_handler(app, spies, tmp_path, declared):
    body = _exact_body(CAP, L.MAX_BYTES_PER_DOCUMENT)
    assert len(body) == CAP
    r = await _asgi(app, _chunks(body),
                    content_length=str(len(body)).encode() if declared else None)
    assert r.status == 200, (r.status, r.body[:300])
    assert spies.read_capped == 1
    assert spies.ingest == [L.MAX_BYTES_PER_DOCUMENT]
    assert r.receives == r.chunks, "the whole body was not delivered"


@pytest.mark.asyncio
async def test_one_byte_over_the_cap_is_refused(app, spies, tmp_path):
    body = _exact_body(CAP + 1, L.MAX_BYTES_PER_DOCUMENT)
    assert len(body) == CAP + 1
    chunks = _chunks(body)
    # The last chunk is the one that crosses — the refusal is decided by the
    # very last byte, after the whole file part has already been parsed.
    assert sum(len(c) for c in chunks[:-1]) <= CAP
    r = await _asgi(app, chunks, content_length=None)
    _assert_typed_413(r)
    _assert_handler_never_ran(spies, tmp_path)


# ── the paths that must not change ─────────────────────────────────────

@pytest.mark.asyncio
async def test_a_declared_over_cap_length_is_refused_with_zero_receives(app, spies, tmp_path):
    r = await _asgi(app, _chunks(_over_cap_upload()),
                    content_length=str(CAP + 1).encode())
    _assert_typed_413(r)
    assert r.receives == 0, "the fast reject read the body"
    _assert_handler_never_ran(spies, tmp_path)


@pytest.mark.asyncio
async def test_a_normal_small_upload_is_unchanged(app, spies, tmp_path):
    data = fx.png_image()
    body = _upload_body(data, "a.png", "image/png",
                        extra=_field_part("client_attachment_id", b"row-7"))
    r = await _asgi(app, _chunks(body), content_length=str(len(body)).encode())
    assert r.status == 200, r.body[:300]
    out = r.json()
    assert out["attachment_id"] and out["client_attachment_id"] == "row-7"
    assert out["deduped"] is False
    assert spies.read_capped == 1 and spies.ingest == [len(data)]
    assert _blobs(tmp_path) > 0


@pytest.mark.asyncio
async def test_a_normal_small_upload_through_httpx_is_unchanged(app, spies, tmp_path):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            PATH, files={"file": ("a.png", fx.png_image(), "image/png")},
            data={"client_attachment_id": "row-1"},
        )
    assert resp.status_code == 200, resp.text[:300]
    assert resp.json()["client_attachment_id"] == "row-1"
    assert _blobs(tmp_path) > 0


@pytest.mark.asyncio
async def test_other_routes_are_not_counted(app, spies):
    body = b"k" * (CAP + 4 * MiB)
    r = await _asgi(app, _chunks(body), content_length=None,
                    path="/api/workspace/sink", content_type=b"application/octet-stream")
    assert r.status == 200, r.body[:300]
    assert spies.other_bytes == [len(body)]


@pytest.mark.asyncio
async def test_a_get_on_the_upload_path_is_not_counted(app, spies):
    r = await _asgi(app, _chunks(b"k" * (CAP + MiB)), content_length=None, method="GET")
    assert r.status == 405


# ── the contract of the wrapped send ───────────────────────────────────

@pytest.mark.asyncio
async def test_no_second_response_when_the_app_already_answered():
    """An app that has already started its response and THEN reads past the
    ceiling must not get a 413 stacked on top — one response, ever."""
    sent: List[Dict[str, Any]] = []
    chunks = _chunks(b"y" * (CAP + 4 * MiB))
    state = {"i": 0}

    async def receive():
        i = state["i"]
        state["i"] += 1
        return {"type": "http.request", "body": chunks[i], "more_body": i < len(chunks) - 1}

    async def send(m):
        sent.append(m)

    async def inner(scope, recv, snd):
        await snd({"type": "http.response.start", "status": 202, "headers": []})
        while True:
            m = await recv()
            if not m.get("more_body"):
                break

    scope = {"type": "http", "method": "POST", "path": PATH, "headers": []}
    await CA.AttachmentBodyLimitMiddleware(inner)(scope, receive, send)
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert [s["status"] for s in starts] == [202]
    # ...and the app still was not fed past the ceiling.
    assert state["i"] == CAP // MiB + 1 < len(chunks)


@pytest.mark.asyncio
async def test_a_refusal_that_escapes_the_app_is_still_the_typed_413():
    """An app with no fastapi exception handler between it and the body (a
    raw ASGI reader) lets the refusal propagate: the middleware answers it."""
    sent: List[Dict[str, Any]] = []
    chunks = _chunks(b"y" * (CAP + MiB))
    state = {"i": 0}

    async def receive():
        i = state["i"]
        state["i"] += 1
        return {"type": "http.request", "body": chunks[i], "more_body": i < len(chunks) - 1}

    async def send(m):
        sent.append(m)

    async def inner(scope, recv, snd):
        while True:
            m = await recv()
            if not m.get("more_body"):
                break
        raise AssertionError("the app read past the ceiling")  # pragma: no cover

    scope = {"type": "http", "method": "POST", "path": PATH, "headers": []}
    await CA.AttachmentBodyLimitMiddleware(inner)(scope, receive, send)
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1 and starts[0]["status"] == 413
    assert dict(starts[0]["headers"])[b"x-toup-reason"] == L.REASON_TOO_LARGE.encode()
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert json.loads(body) == {"detail": L.too_large_detail(None)}
    assert state["i"] == CAP // MiB + 1


# ── the shipped platform app, with its whole middleware stack ──────────

@pytest.mark.asyncio
async def test_the_platform_app_refuses_a_no_length_over_cap_body_before_auth():
    """fastapi parses the form BEFORE dependencies — including auth — so the
    refusal holds for an anonymous caller on the process every tenant shares."""
    from platform_main import app as platform_app

    r = await _asgi(platform_app, _chunks(_over_cap_upload()), content_length=None)
    _assert_typed_413(r)
    assert r.receives < r.chunks
