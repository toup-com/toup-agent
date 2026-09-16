"""`POST /api/chat/attachments` — one identity per (user, bytes).

The idempotency is the point of the whole route. Without it:
  * the 15-attempt warm-up loop re-sent every byte, every time;
  * a user who re-picked the same photo got a second blob;
  * a partial failure had nothing to retry ALONE, so it cost the whole set.

Mostly exercised through `ingest_and_store` rather than over HTTP, because
THAT is the behaviour that has to hold under a retry and under two simultaneous
callers. The route's own gates are driven here too: an earlier version of this
docstring routed the reader to "the limits suite" for them, and the limits
suite only ever exercised the pure `check_one`/`kind_for` functions — so the
server's only backstop against an old or hostile client (the 415, the per-file
413 and the read ceiling) had zero behavioural coverage, and deleting any of
them left every test green.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures"))
import attachments as fx  # noqa: E402

from app.api import chat_attachments as CA  # noqa: E402

USER = "user-feed0082"


@pytest.fixture(autouse=True)
def storage(tmp_path):
    fx.local_storage(tmp_path)
    CA._inflight.clear()
    yield
    from app.services import file_storage

    file_storage._backend = None


def count_blobs(tmp_path) -> int:
    root = os.path.join(str(tmp_path), "generated")
    n = 0
    for _dir, _sub, files in os.walk(root):
        n += len([f for f in files if not f.endswith(".json")])
    return n


@pytest.mark.asyncio
async def test_the_same_bytes_twice_is_one_attachment(tmp_path):
    data = fx.png_image()
    first = await CA.ingest_and_store(data, "a.png", "image/png", USER)
    blobs_after_first = count_blobs(tmp_path)
    second = await CA.ingest_and_store(data, "a.png", "image/png", USER)

    assert first["attachment_id"] == second["attachment_id"]
    assert second["_deduped"] is True
    assert first["_deduped"] is False
    assert count_blobs(tmp_path) == blobs_after_first, "a dedupe wrote a second blob"


@pytest.mark.asyncio
async def test_the_same_name_with_different_bytes_is_two_attachments():
    a = await CA.ingest_and_store(fx.png_image(64, 48, "red"), "photo.png", "image/png", USER)
    b = await CA.ingest_and_store(fx.png_image(64, 48, "blue"), "photo.png", "image/png", USER)
    assert a["attachment_id"] != b["attachment_id"]


@pytest.mark.asyncio
async def test_identity_is_scoped_to_the_user():
    data = fx.png_image()
    a = await CA.ingest_and_store(data, "a.png", "image/png", USER)
    b = await CA.ingest_and_store(data, "a.png", "image/png", "someone-else")
    assert a["attachment_id"] != b["attachment_id"]


@pytest.mark.asyncio
async def test_two_simultaneous_uploads_of_one_file_produce_one_identity(tmp_path):
    """The container is the single writer for this tenant, which is the whole
    concurrency story: the lock is in-process on purpose."""
    data = fx.jpeg_image(400, 300)
    a, b = await asyncio.gather(
        CA.ingest_and_store(data, "x.jpg", "image/jpeg", USER),
        CA.ingest_and_store(data, "x.jpg", "image/jpeg", USER),
    )
    assert a["attachment_id"] == b["attachment_id"]
    assert [a["_deduped"], b["_deduped"]].count(True) == 1


@pytest.mark.asyncio
async def test_the_stored_record_can_be_read_back_for_the_turn():
    data = fx.png_image()
    rec = await CA.ingest_and_store(data, "a.png", "image/png", USER)
    turn = CA.build_turn_record(USER, rec["attachment_id"])
    assert turn is not None
    assert turn["kind"] == "image"
    assert turn["name"] == "a.png"
    assert turn["image_b64"], "the model's copy was not readable"
    assert turn["attachment"]["id"]


@pytest.mark.asyncio
async def test_an_unknown_id_reads_back_as_none_rather_than_raising():
    assert CA.build_turn_record(USER, "deadbeef" * 4) is None


@pytest.mark.asyncio
async def test_a_document_keeps_its_extracted_text_out_of_the_index(tmp_path):
    """The record that is persisted onto Message.attachments and logged around
    carries counts, never contents."""
    if not fx.have("pypdf"):
        pytest.skip("pypdf not installed here")
    rec = await CA.ingest_and_store(
        fx.long_text_pdf("SECRETWORD ", 3000), "cv.pdf", "application/pdf", USER
    )
    assert "SECRETWORD" not in str(rec["ingest"])
    turn = CA.build_turn_record(USER, rec["attachment_id"])
    assert "SECRETWORD" in (turn["text"] or ""), "the turn lost the text"


@pytest.mark.asyncio
async def test_an_unreachable_tenant_is_a_503_never_a_platform_local_success(monkeypatch):
    """`agent_proxy_info` answers None for TWO reasons — "I am the agent" and
    "this user has no active AgentConfig" — and the route fell through to the
    local ingest for both. On the PLATFORM that stored the file in the shared
    container's storage and answered 200 with an attachment_id the turn can
    never resolve (the agent reads its own store, so it rejects it as
    `attachment_not_found`), while running every untrusted parser in this module
    multi-tenant. The module docstring promised the opposite from day one."""
    from fastapi import HTTPException

    monkeypatch.setattr(CA, "serving_locally", lambda: False)

    async def _no_agent(user_id, db):
        return None

    monkeypatch.setattr(CA, "agent_proxy_info", _no_agent)
    called = {"n": 0}

    async def _must_not_run(*a, **kw):
        called["n"] += 1
        raise AssertionError("the platform ingested the file locally")

    monkeypatch.setattr(CA, "ingest_and_store", _must_not_run)

    class _File:
        filename = "a.png"
        content_type = "image/png"

        def __init__(self):
            self._sent = False

        async def read(self, n):
            if self._sent:
                return b""
            self._sent = True
            return fx.png_image()

    class _User:
        id = USER

    with pytest.raises(HTTPException) as exc:
        await CA.upload_chat_attachment(
            file=_File(), sha256=None, client_attachment_id=None,
            current_user=_User(), db=None,
        )
    assert exc.value.status_code == 503
    assert exc.value.headers.get("X-Toup-Reason") == "agent_unreachable"
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_the_agent_itself_still_ingests_locally(monkeypatch, tmp_path):
    """The other half of the same branch: on the agent `serving_locally()` is
    true, `agent_proxy_info` returns None for the right reason, and the work
    happens here."""
    monkeypatch.setattr(CA, "serving_locally", lambda: True)

    async def _no_proxy(user_id, db):
        return None

    monkeypatch.setattr(CA, "agent_proxy_info", _no_proxy)

    class _File:
        filename = "a.png"
        content_type = "image/png"

        def __init__(self):
            self._sent = False

        async def read(self, n):
            if self._sent:
                return b""
            self._sent = True
            return fx.png_image()

    class _User:
        id = USER

    body = await CA.upload_chat_attachment(
        file=_File(), sha256=None, client_attachment_id=None,
        current_user=_User(), db=None,
    )
    assert body["attachment_id"]
    assert body["kind"] == "image"


@pytest.mark.asyncio
async def test_the_public_payload_never_leaks_a_storage_path():
    rec = await CA.ingest_and_store(fx.png_image(), "a.png", "image/png", USER)
    body = CA._public(rec, deduped=False)
    assert "storage_path" not in body
    assert set(body) >= {
        "attachment_id", "sha256", "filename", "mime_type",
        "size_bytes", "kind", "deduped", "ingest",
    }


# ── the route's own gates, driven ────────────────────────────────────────

class _Sized:
    """An `UploadFile` stand-in that yields `total` bytes in 1 MiB chunks."""

    def __init__(self, total: int, filename="big.pdf", content_type="application/pdf"):
        self.filename = filename
        self.content_type = content_type
        self._left = total

    async def read(self, n):
        if self._left <= 0:
            return b""
        take = min(n, self._left)
        self._left -= take
        return b"\0" * take


class _Who:
    id = USER


@pytest.mark.asyncio
async def test_an_oversized_part_is_refused_before_anything_is_ingested(monkeypatch):
    """`_read_capped`'s ceiling: deleting it or flipping the comparison used to
    leave every test green while an app build that does not enforce the table
    spooled an unbounded body into one contiguous `bytes`."""
    from fastapi import HTTPException

    called = {"n": 0}

    async def _must_not_run(*a, **kw):
        called["n"] += 1
        raise AssertionError("ingest ran on a refused upload")

    monkeypatch.setattr(CA, "ingest_and_store", _must_not_run)

    with pytest.raises(HTTPException) as exc:
        await CA.upload_chat_attachment(
            file=_Sized(CA._ABSOLUTE_MAX + 1), sha256=None,
            client_attachment_id=None, current_user=_Who(), db=None,
        )
    assert exc.value.status_code == 413
    assert exc.value.headers.get("X-Toup-Reason") == "attachment_too_large"
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_an_unsupported_type_is_named_not_silently_accepted(monkeypatch):
    from fastapi import HTTPException

    async def _must_not_run(*a, **kw):
        raise AssertionError("ingest ran on an unsupported type")

    monkeypatch.setattr(CA, "ingest_and_store", _must_not_run)

    with pytest.raises(HTTPException) as exc:
        await CA.upload_chat_attachment(
            file=_Sized(64, filename="tool.exe",
                        content_type="application/x-msdownload"),
            sha256=None, client_attachment_id=None, current_user=_Who(), db=None,
        )
    assert exc.value.status_code == 415
    assert exc.value.headers.get("X-Toup-Reason") == "attachment_unsupported"


@pytest.mark.asyncio
async def test_a_document_over_its_own_ceiling_is_refused_by_type(monkeypatch):
    """Between the per-type limit and the absolute one: a 20 MB IMAGE is fine,
    a 20 MB image is not — `check_one` decides, and the route has to act on it."""
    from fastapi import HTTPException
    from app.agent.attachment_limits import MAX_BYTES_PER_IMAGE

    async def _must_not_run(*a, **kw):
        raise AssertionError("ingest ran on an over-limit file")

    monkeypatch.setattr(CA, "ingest_and_store", _must_not_run)

    with pytest.raises(HTTPException) as exc:
        await CA.upload_chat_attachment(
            file=_Sized(MAX_BYTES_PER_IMAGE + 1, filename="p.png",
                        content_type="image/png"),
            sha256=None, client_attachment_id=None, current_user=_Who(), db=None,
        )
    assert exc.value.status_code == 413
    assert exc.value.headers.get("X-Toup-Reason") == "attachment_too_large"


# ── the pre-parse body ceiling ───────────────────────────────────────────

async def _drive(scope):
    """Run the middleware over one ASGI scope; return (status, headers, body)."""
    sent = []

    async def _send(msg):
        sent.append(msg)

    async def _inner(_scope, _receive, _send):
        sent.append({"type": "http.response.start", "status": 200, "headers": []})

    await CA.AttachmentBodyLimitMiddleware(_inner)(scope, None, _send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], dict(start.get("headers") or [])


def _scope(path, length, method="POST"):
    headers = [(b"host", b"x")]
    if length is not None:
        headers.append((b"content-length", str(length).encode()))
    return {"type": "http", "method": method, "path": path, "headers": headers}


@pytest.mark.asyncio
async def test_an_oversized_body_is_refused_before_the_form_is_parsed():
    """fastapi calls `await request.form()` before `solve_dependencies`, so the
    whole multipart body — file parts spooled to the container's disk, non-file
    fields accumulated in memory with no ceiling in this starlette — is already
    there when the handler's first statement runs. On the PLATFORM that disk
    and that memory are shared by every tenant."""
    status, headers = await _drive(
        _scope("/api/chat/attachments", CA.BODY_HARD_MAX + 1))
    assert status == 413
    assert headers[b"x-toup-reason"] == b"attachment_too_large"


@pytest.mark.asyncio
async def test_a_body_inside_the_ceiling_reaches_the_app():
    status, _ = await _drive(_scope("/api/chat/attachments", CA.BODY_HARD_MAX - 1))
    assert status == 200


@pytest.mark.asyncio
async def test_the_ceiling_only_guards_this_route():
    status, _ = await _drive(_scope("/api/workspace/files", CA.BODY_HARD_MAX * 4))
    assert status == 200
    status, _ = await _drive(
        _scope("/api/chat/attachments", CA.BODY_HARD_MAX * 4, method="GET"))
    assert status == 200


@pytest.mark.asyncio
async def test_a_body_with_no_declared_length_still_reaches_the_read_cap():
    """Content-Length is the only thing available before a body byte is read.
    A chunked body is not refused here — `_read_capped` bounds the part."""
    status, _ = await _drive(_scope("/api/chat/attachments", None))
    assert status == 200


# ── the typed refusal survives the platform proxy ────────────────────────

@pytest.mark.asyncio
async def test_an_agent_refusal_keeps_its_typed_reason_across_the_proxy(monkeypatch):
    """`X-Toup-Reason` IS the refusal contract: both clients read it to name
    WHICH file failed and why, and `attachment_unsupported` can never succeed
    while `upload_failed` invites a retry. The platform's 4xx re-raise dropped
    the header, so an agent-authored refusal reached the user as the generic
    "could not be uploaded. Try again." — latent only while the platform's own
    pre-check and the agent's agree on every input."""
    import httpx
    from fastapi import HTTPException
    from app.api import tenant_proxy as TP

    class _Client:
        async def request(self, method, url, **kw):
            return httpx.Response(
                415,
                json={"detail": "That kind of file cannot be read yet."},
                headers={"X-Toup-Reason": "attachment_unsupported",
                         "Retry-After": "0",
                         "Set-Cookie": "leak=1"},
                request=httpx.Request(method, url),
            )

    from app.services import agent_http
    monkeypatch.setattr(agent_http, "get_agent_http_client", lambda: _Client())

    with pytest.raises(HTTPException) as exc:
        await TP.proxy_to_agent("https://agent.invalid", "k", "chat/attachments", "POST")
    assert exc.value.status_code == 415
    assert exc.value.headers.get("X-Toup-Reason") == "attachment_unsupported"
    assert exc.value.headers.get("Retry-After") == "0"
    # Only the typed pair crosses: everything else the agent set stays there.
    assert not any(k.lower() == "set-cookie" for k in (exc.value.headers or {}))
