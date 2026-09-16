"""Chat attachment upload — one identity per file, before the turn.

TOPOLOGY (see app/api/tenant_proxy.py)
    The store is the tenant's storage backend and, at turn time, the tenant's
    `messages.attachments` column — both AGENT_ONLY. Mounted on BOTH apps: on
    the platform every handler proxies to the user's agent with X-Agent-Key
    exactly as `documents.py` does; on the agent `serving_locally()` is true
    and the handler does the work. Nothing here falls back to a platform-local
    answer: an unreachable tenant is a 503, never a fabricated success.

WHY IT EXISTS (round 46, incident 4)
    Attachments rode as base64 INSIDE the chat WS frame. That gave every file
    the same fate as the turn:

      * no identity — a partial failure cost the whole set, there was nothing
        to retry alone, and nothing to show a status against;
      * no bound — the only ceiling was uvicorn's 16 MiB frame, whose breach
        closed the socket and reached the user as "check your internet";
      * no reuse — the warm-up loop re-sent every byte up to 15 times;
      * all the CPU on the turn's critical path, on a 1.00-CPU container.

    An upload fixes all four at once, so the ingest (sniff, extract, downscale,
    rasterize) also happens HERE, on a thread, before the user has pressed
    send — the turn then only reads what this route already produced.

IDEMPOTENCY
    On `(user_id, sha256)`. A repeat returns the SAME `attachment_id` with
    `deduped: true` and writes nothing. The lock is in-process because a
    tenant's store is served by exactly one container: that makes two
    simultaneous POSTs of the same bytes produce one identity and one storage
    write, which no amount of retry logic upstream can undo.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.attachment_limits import (
    MAX_BYTES_PER_DOCUMENT,
    MAX_BYTES_PER_IMAGE,
    REASON_TOO_LARGE,
    REASON_UNSUPPORTED,
    check_one,
    normalize_mime,
)
from app.api.auth import get_current_user
from app.api.tenant_proxy import (
    UPLOAD_PROXY_TIMEOUT_S,
    agent_proxy_info,
    proxy_to_agent,
    serving_locally,
)
from app.db import get_db
from app.services.file_storage import get_storage_backend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat attachments"])

#: The largest single file any kind may be. The per-kind cap is applied after
#: the type is known; this is the read ceiling so a 200 MB body is refused
#: without being buffered.
_ABSOLUTE_MAX = max(MAX_BYTES_PER_IMAGE, MAX_BYTES_PER_DOCUMENT)
_READ_CHUNK = 1024 * 1024

_INDEX_PREFIX = "chat-attachments"
_inflight: Dict[str, asyncio.Lock] = {}


def _scope(user_id: str) -> str:
    """A storage-key-safe scope. Ids are uuids; this is belt and braces against
    a key that could escape the backend root."""
    return "".join(c for c in (user_id or "anon") if c.isalnum() or c in "-_")[:64] or "anon"


def _record_key(user_id: str, attachment_id: str) -> str:
    return f"{_INDEX_PREFIX}/{_scope(user_id)}/{_scope(attachment_id)}.json"


def _sha_key(user_id: str, sha: str) -> str:
    return f"{_INDEX_PREFIX}/{_scope(user_id)}/by-sha/{_scope(sha)}.json"


def _text_key(user_id: str, attachment_id: str) -> str:
    return f"{_INDEX_PREFIX}/{_scope(user_id)}/{_scope(attachment_id)}.text"


def _page_key(user_id: str, attachment_id: str, n: int) -> str:
    return f"{_INDEX_PREFIX}/{_scope(user_id)}/{_scope(attachment_id)}.page{n}.webp"


def _attachment_id_for(user_id: str, sha: str) -> str:
    """Deterministic identity for (user, bytes).

    Content-addressed on purpose: two uploads of the same file can never become
    two attachments, whatever happens to the index write in between.
    """
    return hashlib.sha256(f"{user_id}|{sha}".encode("utf-8")).hexdigest()[:32]


def _read_json(key: str) -> Optional[Dict[str, Any]]:
    backend = get_storage_backend()
    try:
        if not backend.exists(key):
            return None
        with backend.open(key) as fh:
            return json.loads(fh.read().decode("utf-8"))
    except Exception:
        logger.debug("chat attachment: index read failed", exc_info=True)
        return None


async def _write_json(key: str, payload: Dict[str, Any]) -> None:
    await get_storage_backend().put(key, json.dumps(payload).encode("utf-8"))


def _public(record: Dict[str, Any], *, deduped: bool) -> Dict[str, Any]:
    """The C6 response shape. Never carries text, bytes, or a storage path."""
    att = record.get("attachment") or {}
    return {
        "attachment_id": record.get("attachment_id"),
        "sha256": record.get("sha256"),
        "filename": att.get("filename") or record.get("name"),
        "mime_type": att.get("mime_type") or record.get("mime"),
        "size_bytes": att.get("size_bytes") or record.get("size_bytes") or 0,
        "width": att.get("width"),
        "height": att.get("height"),
        "has_thumb": bool(att.get("has_thumb")),
        "kind": record.get("kind"),
        "deduped": deduped,
        "ingest": record.get("ingest") or {"status": "ok"},
    }


def _refuse(reason: str, detail: str, code: int) -> HTTPException:
    """A typed refusal. The CODE crosses the wire; the client composes the
    sentence next to the file's own name — a refusal the user cannot attribute
    to a file is the silent drop this round removed."""
    return HTTPException(status_code=code, detail=detail, headers={"X-Toup-Reason": reason})


async def _read_capped(file: UploadFile) -> bytes:
    """Refuse a part past the absolute ceiling instead of holding it in RAM.

    NOT a body-size defence, and the docstring here used to claim it was: by
    the time this runs FastAPI has already called `await request.form()`, so
    the whole multipart body has been parsed — file parts spooled to disk past
    starlette's 1 MiB threshold, non-file fields accumulated in memory with no
    ceiling at all. The pre-parse ceiling is
    :class:`AttachmentBodyLimitMiddleware`, installed on both apps; this keeps
    a single oversized part out of one contiguous `bytes` after that.
    """
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > _ABSOLUTE_MAX:
            raise _refuse(
                REASON_TOO_LARGE,
                "That file is larger than this chat accepts.",
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def ingest_and_store(
    data: bytes,
    filename: str,
    declared_mime: Optional[str],
    user_id: str,
) -> Dict[str, Any]:
    """The agent-side half: classify, persist, index. Returns the stored record.

    Serialized per (user, sha256): the container is the single writer for this
    tenant, so this is the whole concurrency story.
    """
    from app.agent.attachment_ingest import ingest_one
    from app.agent.doc_generators import _persist

    sha = hashlib.sha256(data).hexdigest()
    att_id = _attachment_id_for(user_id, sha)
    # Bounded: one entry per distinct (user, sha256) for the life of the
    # process is a slow leak in a 768 MiB container. An unheld lock is safe to
    # drop — the next upload of those bytes mints a new one and finds the
    # record already written.
    if len(_inflight) >= 512:
        for stale in [k for k, l in _inflight.items() if not l.locked()][:256]:
            _inflight.pop(stale, None)
    lock = _inflight.setdefault(f"{user_id}:{sha}", asyncio.Lock())
    async with lock:
        existing = _read_json(_record_key(user_id, att_id))
        if existing:
            existing["_deduped"] = True
            return existing

        ing = await asyncio.to_thread(ingest_one, data, filename, declared_mime)
        # role='source': this file is the model's INPUT, not the agent's output.
        # Clients that draw a card per attachment must not draw one for it twice.
        att = await _persist(data, ing.name, ing.mime, user_id, role="source")

        text_key: Optional[str] = None
        if ing.text:
            text_key = _text_key(user_id, att_id)
            await get_storage_backend().put(text_key, ing.text.encode("utf-8"))
        page_keys: List[str] = []
        for i, page in enumerate(ing.page_images):
            key = _page_key(user_id, att_id, i)
            await get_storage_backend().put(key, page)
            page_keys.append(key)

        record = {
            "attachment_id": att_id,
            "sha256": sha,
            "name": ing.name,
            "mime": ing.mime,
            "kind": ing.kind,
            "size_bytes": ing.size_bytes,
            "attachment": att.to_dict(),
            "ingest": ing.to_record(),
            "text_key": text_key,
            "page_keys": page_keys,
            # The derivative _persist already wrote at MODEL_IMAGE_LONG_EDGE IS
            # the model's copy — an uploaded image is never resized twice.
            "model_image_key": (
                _thumb_key_for(att) if (ing.kind == "image" and att.has_thumb) else att.storage_path
            ) if ing.kind == "image" else None,
            "model_image_mime": ("image/webp" if (ing.kind == "image" and att.has_thumb) else ing.mime)
            if ing.kind == "image"
            else None,
        }
        await _write_json(_record_key(user_id, att_id), record)
        await _write_json(_sha_key(user_id, sha), {"attachment_id": att_id})
        record["_deduped"] = False
        return record


def _thumb_key_for(att) -> str:
    from app.agent.doc_generators import thumb_key

    return thumb_key(att.storage_path)


def load_attachment_record(user_id: str, attachment_id: str) -> Optional[Dict[str, Any]]:
    """The stored record for one id, or None. Read-only, no ingest."""
    return _read_json(_record_key(user_id, attachment_id))


def build_turn_record(user_id: str, attachment_id: str) -> Optional[Dict[str, Any]]:
    """One uploaded attachment in the shape `_build_attachment_blocks` reads.

    Reads the already-ingested text / already-downscaled image out of storage:
    the turn pays no sniff, no extraction, no resize. Synchronous (storage is
    local disk); callers run it on a thread with the rest of the block.
    """
    import base64

    rec = load_attachment_record(user_id, attachment_id)
    if not rec:
        return None
    backend = get_storage_backend()
    out: Dict[str, Any] = {
        "name": rec.get("name"),
        "mime": rec.get("mime"),
        "kind": rec.get("kind"),
        "status": (rec.get("ingest") or {}).get("status", "ok"),
        "reason": (rec.get("ingest") or {}).get("reason"),
        "page_count": (rec.get("ingest") or {}).get("page_count"),
        "truncated": bool((rec.get("ingest") or {}).get("truncated")),
        "attachment_id": rec.get("attachment_id"),
        "attachment": rec.get("attachment"),
        "text": None,
    }
    try:
        if rec.get("text_key") and backend.exists(rec["text_key"]):
            with backend.open(rec["text_key"]) as fh:
                out["text"] = fh.read().decode("utf-8", errors="replace")
        if rec.get("model_image_key") and backend.exists(rec["model_image_key"]):
            with backend.open(rec["model_image_key"]) as fh:
                out["image_b64"] = base64.standard_b64encode(fh.read()).decode("ascii")
            out["image_mime"] = rec.get("model_image_mime") or rec.get("mime")
        pages = []
        for key in rec.get("page_keys") or []:
            if backend.exists(key):
                with backend.open(key) as fh:
                    pages.append(base64.standard_b64encode(fh.read()).decode("ascii"))
        if pages:
            out["page_images_b64"] = pages
    except Exception:
        # A readable record whose blob vanished is still an attachment the user
        # can be told about — it must not take the turn down with it.
        logger.warning("chat attachment: blob missing for a stored record")
        out["status"] = "unreadable"
        out["reason"] = "unreadable"
    return out


@router.post("/attachments")
async def upload_chat_attachment(
    file: UploadFile = File(...),
    sha256: Optional[str] = Form(None),
    client_attachment_id: Optional[str] = Form(None),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload ONE file for an upcoming chat turn.

    `sha256` is an optional client hint (the app has no cheap digest and sends
    none); the server always computes its own, so the hint can never decide
    identity. `client_attachment_id` is echoed back so a client can correlate a
    slow response with the tray row that asked for it.
    """
    data = await _read_capped(file)
    declared = file.content_type or None
    name = (file.filename or "file").split("/")[-1].split("\\")[-1]
    mime = normalize_mime(declared, name)
    kind, reason = check_one(len(data), mime, name)
    if reason == REASON_UNSUPPORTED:
        raise _refuse(
            REASON_UNSUPPORTED,
            "That kind of file cannot be read yet.",
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )
    if reason == REASON_TOO_LARGE:
        raise _refuse(
            REASON_TOO_LARGE,
            "That file is larger than this chat accepts.",
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    # Route to the tenant BEFORE any local work — the storage backend and the
    # turn that will read it both live there. The cheap gates above run first so
    # a refusal does not cost a 25 MB hop.
    proxy = await agent_proxy_info(current_user.id, db)
    if proxy:
        form: Dict[str, str] = {}
        if sha256:
            form["sha256"] = sha256
        if client_attachment_id:
            form["client_attachment_id"] = client_attachment_id
        result = await proxy_to_agent(
            proxy[0], proxy[1], "chat/attachments", "POST",
            data=form,
            files={"file": (name, data, mime)},
            timeout=UPLOAD_PROXY_TIMEOUT_S,
        )
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Your agent stored the file but returned no details.",
            )
        return JSONResponse(content=result)

    if not serving_locally():
        # `agent_proxy_info` answers None for TWO reasons — "I am the agent" and
        # "this user has no active AgentConfig" — and only the first one may
        # reach the local ingest. Falling through on the second stored the file
        # in the SHARED PLATFORM container's storage and answered 200 with an
        # attachment_id no turn can ever resolve (the agent reads its own store,
        # so the turn rejects it as `attachment_not_found`), while running every
        # untrusted parser in this module's ingest path multi-tenant. The
        # docstring at the top of this file always said so; nothing implemented it.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Your agent is not reachable right now.",
            headers={"Retry-After": "3", "X-Toup-Reason": "agent_unreachable"},
        )

    record = await ingest_and_store(data, name, declared, current_user.id)
    body = _public(record, deduped=bool(record.get("_deduped")))
    if client_attachment_id:
        body["client_attachment_id"] = client_attachment_id
    logger.info(
        "[ATTACH] stored kind=%s status=%s size_class=%s deduped=%s user=%s",
        record.get("kind"),
        (record.get("ingest") or {}).get("status"),
        _size_class(record.get("size_bytes") or 0),
        int(bool(record.get("_deduped"))),
        current_user.id[:8],
    )
    return body


def plan_uploaded_attachments(
    attachment_ids: Any,
    resolve: Any,
) -> tuple:
    """Which uploaded ids enter this turn — and, for each, the index the CLIENT
    used. Returns ``(records, wire_indices, rejected)``.

    Pure apart from ``resolve(attachment_id) -> record | None`` (the caller
    binds :func:`build_turn_record` to the user and runs the whole pass on one
    thread instead of hopping per item).

    Round 46 review: the ingest report numbered accepted items by their
    position in the ACCEPTED list and stamped every rejection ``index: -1``.
    One rejected file therefore painted every later file's status onto the
    wrong row, and a rejection reached the client as ``sent[-1]`` — undefined,
    i.e. no status at all, which is the silent drop this round set out to
    remove, at the seam.
    """
    from app.agent.attachment_limits import (
        MAX_ATTACHMENTS_PER_TURN,
        MAX_TOTAL_BYTES_PER_TURN,
        REASON_TOTAL_TOO_LARGE,
    )

    records: List[Dict[str, Any]] = []
    indices: List[int] = []
    rejected: List[Dict[str, Any]] = []
    turn_bytes = 0
    for idx, aid in enumerate(attachment_ids or ()):
        if not isinstance(aid, str) or not aid:
            continue
        if len(records) >= MAX_ATTACHMENTS_PER_TURN:
            rejected.append({
                "index": idx, "name": "", "reason": "attachment_count_exceeded",
            })
            continue
        rec = resolve(aid)
        if rec is None:
            # An id we cannot resolve is NOT a file we silently forget: the
            # user attached something and is owed a line saying it did not
            # make it — against the row they attached it to.
            rejected.append({
                "index": idx, "name": "", "reason": "attachment_not_found",
            })
            continue
        try:
            size = int(((rec.get("attachment") or {}).get("size_bytes")) or 0)
        except (TypeError, ValueError):
            size = 0
        if turn_bytes + size > MAX_TOTAL_BYTES_PER_TURN:
            rejected.append({
                "index": idx,
                "name": rec.get("name") or "",
                "reason": REASON_TOTAL_TOO_LARGE,
            })
            continue
        turn_bytes += size
        records.append(rec)
        indices.append(idx)
    return records, indices, rejected


#: A body this route can never legitimately need: the largest single file the
#: table allows, plus a MiB of multipart framing and form fields.
BODY_HARD_MAX = _ABSOLUTE_MAX + 1024 * 1024


class AttachmentBodyLimitMiddleware:
    """Refuse an oversized upload body BEFORE the framework buffers it.

    A `Depends` cannot do this: fastapi calls `await request.form()` in
    `routing.py` before `solve_dependencies`, so the whole multipart body is
    already parsed — file parts spooled to the container's disk, non-file
    fields accumulated in memory — before the handler's first statement runs.
    On the PLATFORM that disk and that memory are shared by every tenant.

    Content-Length only, deliberately: it is the one thing available before a
    single body byte is read, and an honest client always sends it. A chunked
    body still reaches the parser, where `_read_capped` bounds the part.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and str(scope.get("path") or "").endswith("/chat/attachments")
        ):
            declared = 0
            for key, value in scope.get("headers") or ():
                if key == b"content-length":
                    try:
                        declared = int(value)
                    except (TypeError, ValueError):
                        declared = 0
                    break
            if declared > BODY_HARD_MAX:
                await send({
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"x-toup-reason", REASON_TOO_LARGE.encode()),
                    ],
                })
                await send({
                    "type": "http.response.body",
                    "body": json.dumps(
                        {"detail": "That file is larger than this chat accepts."}
                    ).encode(),
                })
                return
        await self.app(scope, receive, send)


def _size_class(n: int) -> str:
    """Telemetry carries a SIZE CLASS, never a filename and never the bytes."""
    if n < 256 * 1024:
        return "xs"
    if n < 2 * 1024 * 1024:
        return "s"
    if n < 10 * 1024 * 1024:
        return "m"
    return "l"
