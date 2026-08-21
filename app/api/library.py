"""The file library API — runs on the AGENT (the data owner).

Two surfaces over ONE virtual tree (``app/services/library_service.py``):

  ``/api/library/*``    id-based, paginated, searchable. The canonical API.
  ``/api/workspace/*``  the path-based shapes the shipped mobile app and web
                        client already call (``files``, ``file-content``,
                        ``file-write``, ``file`` DELETE, ``file-rename``,
                        ``create-dir``, ``file-download``, ``file-upload``,
                        ``tree``). Same tree, virtual name paths.

Nothing here reads a client-supplied filesystem path. Clients send ids or
virtual paths ("Documents/Q3 report.pdf"); the service maps them to a
manifest row; only the service turns a row's storage key into bytes, with a
realpath containment check. Responses never carry storage keys, absolute
paths, container names or tenant UUIDs — ``base`` is ``"/"``.

Auth: the agent's ``AgentAPIKeyMiddleware`` gates every /api route behind
X-Agent-Key; ``get_current_user`` resolves that to the tenant owner. The
binary routes additionally accept ``?token=<jwt>`` (embeds cannot set
headers) through the same explicit-before-ambient ladder as /api/files.
The platform proxies all of this 1:1 (``app/api/workspace_proxy.py``).
"""

from __future__ import annotations

import logging
import os
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.api.files import (
    etag_matches, get_user_for_file, not_modified, render_preview,
)
from app.config import settings
from app.db import get_db
from app.db.models.user_file import ORIGIN_UPLOAD, UserFile, UserFolder
from app.services import library_service as lib

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Library"])

# Byte caps. Uploads through the platform ride a Railway request; 50 MB is
# what /workspace/file-download already allowed and comfortably above any
# document the agent produces.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
# Text returned inside a JSON body (reports, notes). Larger → download.
MAX_TEXT_BYTES = 1_000_000
MAX_LIST_LIMIT = 200
_COMPAT_LIST_CAP = 500
# Extensions the text routes will read even when the MIME guess is generic.
_TEXT_EXTS = ("md", "markdown", "txt", "csv", "tsv", "json", "html", "htm", "xml", "yaml", "yml", "log")


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _api() -> str:
    return settings.api_prefix


async def _synced(db: AsyncSession, user_id: str, refresh: bool = False) -> None:
    try:
        await lib.sync_user_library(db, user_id, force=refresh)
    except Exception:
        # A sync failure must never take the listing down with it — the
        # manifest is still readable; the next call retries.
        logger.exception("[library] sync failed for user=%s", user_id)
        try:
            await db.rollback()
        except Exception:
            pass


def _err(exc: Exception) -> HTTPException:
    if isinstance(exc, lib.InvalidName):
        return HTTPException(400, str(exc))
    if isinstance(exc, lib.NameConflict):
        return HTTPException(409, str(exc))
    if isinstance(exc, lib.NotEmpty):
        return HTTPException(409, "Folder is not empty")
    if isinstance(exc, lib.SystemFolderError):
        return HTTPException(400, str(exc))
    if isinstance(exc, LookupError):
        return HTTPException(404, str(exc) or "Not found")
    if isinstance(exc, ValueError):
        return HTTPException(400, str(exc))
    return HTTPException(500, "Library operation failed")


def _folder_not_empty() -> JSONResponse:
    # Shape the phone reads: a flat detail + a machine code (FastAPI would
    # nest a dict detail one level down).
    return JSONResponse(status_code=409, content={"detail": "Folder is not empty", "code": "folder_not_empty"})


def _content_disposition(disposition: str, filename: str) -> str:
    """RFC 6266: an ASCII fallback plus the UTF-8 name in filename*."""
    ascii_name = filename.encode("ascii", "ignore").decode("ascii").replace('"', "'").strip() or "file"
    return f"{disposition}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename, safe='')}"


def _stream_path(path: str, chunk: int = 64 * 1024):
    async def _gen():
        with open(path, "rb") as f:
            while True:
                data = f.read(chunk)
                if not data:
                    break
                yield data
    return _gen()


async def _folder_paths(db: AsyncSession, user_id: str) -> tuple[dict[str, UserFolder], dict[str, str]]:
    folders = await lib.all_folders(db, user_id)
    return folders, {fid: lib.folder_path(f, folders) for fid, f in folders.items()}


def _file_path(f: UserFile, paths: dict[str, str]) -> str:
    parent = paths.get(f.folder_id or "", "") if f.folder_id else ""
    return f"{parent}/{f.name}" if parent else f.name


async def _require_file(db: AsyncSession, user_id: str, file_id: str) -> UserFile:
    f = await lib.get_file(db, user_id, file_id)
    if f is None:
        raise HTTPException(404, "File not found")
    return f


async def _require_folder(db: AsyncSession, user_id: str, folder_id: str) -> UserFolder:
    fo = await lib.get_folder(db, user_id, folder_id)
    if fo is None:
        raise HTTPException(404, "Folder not found")
    return fo


async def _read_upload(upload: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File is too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
        chunks.append(chunk)
    return b"".join(chunks)


_LIBRARY_CACHE = "private, max-age=3600"


def _library_etag(f: UserFile, path: str) -> str:
    """Validator for a library file's bytes.

    Unlike a chat attachment, a library file id is a STABLE handle whose
    bytes a re-upload can replace, so the validator has to move when they do
    — hence size and mtime, not the id alone. That is also why the library
    keeps a revalidating `max-age` rather than `immutable`: the ETag makes
    revalidation free, but it must still happen.
    """
    try:
        st = os.stat(path)
        return f'"{f.id}-{st.st_size}-{int(st.st_mtime)}"'
    except OSError:
        return f'"{f.id}"'


async def _download_response(
    user_id: str, f: UserFile, *, inline: bool, request: Optional[Request] = None
) -> Response:
    path = lib.file_abspath(user_id, f)
    if not path:
        raise HTTPException(410, "This file is no longer available")
    length = os.path.getsize(path)
    etag = _library_etag(f, path)
    if etag_matches(request, etag):
        return not_modified(etag, _LIBRARY_CACHE)
    return StreamingResponse(
        _stream_path(path),
        media_type=f.mime_type or "application/octet-stream",
        headers={
            "Content-Length": str(length),
            "Content-Disposition": _content_disposition("inline" if inline else "attachment", f.name),
            "Cache-Control": _LIBRARY_CACHE,
            "ETag": etag,
            "X-Content-Type-Options": "nosniff",
        },
    )


async def _text_of(user_id: str, f: UserFile) -> str:
    path = lib.file_abspath(user_id, f)
    if not path:
        raise HTTPException(410, "This file is no longer available")
    size = os.path.getsize(path)
    if size > MAX_TEXT_BYTES:
        raise HTTPException(413, "File too large (>1MB) — use download")
    mime = lib.guess_mime(f.name, f.mime_type)
    if not lib.is_text_mime(mime) and lib.ext_of(f.name) not in _TEXT_EXTS:
        raise HTTPException(415, f"Binary file ({mime}) — use the preview/download route")
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8", errors="replace")


# ═════════════════════════════════════════════════════════════════════
# /api/library — id-based
# ═════════════════════════════════════════════════════════════════════

class FolderCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    parent_id: Optional[str] = None


class FolderPatch(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    parent_id: Optional[str] = None


class FilePatch(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    folder_id: Optional[str] = None


class ContentBody(BaseModel):
    content: str


@router.get("/library/folders")
async def library_list_folders(
    parent: str = Query("root", description="'root' or a folder id"),
    refresh: bool = Query(False),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    await _synced(db, uid, refresh)
    parent_id = None if parent in ("", "root") else parent
    if parent_id:
        await _require_folder(db, uid, parent_id)
    folders, paths = await _folder_paths(db, uid)
    file_counts, sub_counts = await lib.folder_counts(db, uid)
    rows = await lib.list_folders(db, uid, parent_id)
    return {"items": [
        lib.folder_entry(fo, paths.get(fo.id, fo.name),
                         file_count=file_counts.get(fo.id, 0), folder_count=sub_counts.get(fo.id, 0))
        for fo in rows
    ], "parent_id": parent_id}


@router.post("/library/folders", status_code=201)
async def library_create_folder(
    body: FolderCreate,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    await _synced(db, uid)
    try:
        fo = await lib.create_folder(db, uid, body.name, body.parent_id)
    except Exception as e:  # noqa: BLE001
        raise _err(e)
    folders, paths = await _folder_paths(db, uid)
    return lib.folder_entry(fo, paths.get(fo.id, fo.name))


@router.patch("/library/folders/{folder_id}")
async def library_update_folder(
    folder_id: str,
    body: FolderPatch,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    fo = await _require_folder(db, uid, folder_id)
    move = "parent_id" in body.model_fields_set
    try:
        fo = await lib.update_folder(db, uid, fo, name=body.name, parent_id=body.parent_id, move=move)
    except Exception as e:  # noqa: BLE001
        raise _err(e)
    folders, paths = await _folder_paths(db, uid)
    file_counts, sub_counts = await lib.folder_counts(db, uid)
    return lib.folder_entry(fo, paths.get(fo.id, fo.name),
                            file_count=file_counts.get(fo.id, 0), folder_count=sub_counts.get(fo.id, 0))


@router.delete("/library/folders/{folder_id}")
async def library_delete_folder(
    folder_id: str,
    recursive: bool = Query(False),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    fo = await _require_folder(db, uid, folder_id)
    try:
        n = await lib.delete_folder(db, uid, fo, recursive=recursive)
    except lib.NotEmpty:
        return _folder_not_empty()
    except Exception as e:  # noqa: BLE001
        raise _err(e)
    lib.invalidate_sync(uid)
    return {"success": True, "id": folder_id, "files_deleted": n}


@router.get("/library/files")
async def library_list_files(
    folder: str = Query("root", description="'root', 'all' or a folder id"),
    q: str = Query("", max_length=200),
    kind: str = Query("", description="document|image|audio|video|archive|other"),
    origin: str = Query("", description="upload|agent"),
    sort: str = Query("name"),
    order: str = Query("asc"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=MAX_LIST_LIMIT),
    refresh: bool = Query(False),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    await _synced(db, uid, refresh)
    if folder not in (lib.FOLDER_ROOT, lib.FOLDER_ALL, ""):
        await _require_folder(db, uid, folder)
    if sort not in lib.SORT_FIELDS:
        sort = "name"
    scope = lib.FOLDER_ALL if q.strip() and folder in ("", lib.FOLDER_ROOT) else (folder or lib.FOLDER_ROOT)
    rows, total = await lib.list_files(
        db, uid, folder=scope, q=q.strip(), kind=kind, origin=origin, sort=sort,
        order="desc" if order == "desc" else "asc", offset=offset, limit=limit,
    )
    folders, paths = await _folder_paths(db, uid)
    items = [lib.file_entry(f, _file_path(f, paths), api_prefix=_api()) for f in rows]
    next_offset = offset + len(rows) if offset + len(rows) < total else None
    return {"items": items, "total": total, "offset": offset, "limit": limit,
            "next_offset": next_offset, "folder": scope, "q": q.strip()}


@router.post("/library/upload", status_code=201)
async def library_upload(
    request: Request,
    file: UploadFile = File(...),
    folder_id: Optional[str] = Form(None),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    await _synced(db, uid)
    dest = folder_id or request.query_params.get("folder_id") or None
    if not dest:
        sys_folders = await lib.ensure_system_folders(db, uid)
        dest = sys_folders[lib.SYSTEM_FOLDER_UPLOADS].id
    data = await _read_upload(file)
    if not data:
        raise HTTPException(400, "Empty file")
    name = os.path.basename((file.filename or "").replace("\\", "/")).strip() or "upload"
    try:
        row = await lib.store_new_file(db, uid, name=name, data=data,
                                       mime=file.content_type or "", origin=ORIGIN_UPLOAD, folder_id=dest)
    except Exception as e:  # noqa: BLE001
        raise _err(e)
    lib.invalidate_sync(uid)
    folders, paths = await _folder_paths(db, uid)
    return lib.file_entry(row, _file_path(row, paths), api_prefix=_api())


@router.get("/library/files/{file_id}")
async def library_get_file(
    file_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    f = await _require_file(db, uid, file_id)
    folders, paths = await _folder_paths(db, uid)
    return lib.file_entry(f, _file_path(f, paths), api_prefix=_api())


@router.patch("/library/files/{file_id}")
async def library_update_file(
    file_id: str,
    body: FilePatch,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    f = await _require_file(db, uid, file_id)
    move = "folder_id" in body.model_fields_set
    try:
        f = await lib.update_file(db, uid, f, name=body.name, folder_id=body.folder_id, move=move)
    except Exception as e:  # noqa: BLE001
        raise _err(e)
    folders, paths = await _folder_paths(db, uid)
    return lib.file_entry(f, _file_path(f, paths), api_prefix=_api())


@router.delete("/library/files/{file_id}")
async def library_delete_file(
    file_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    f = await _require_file(db, uid, file_id)
    try:
        await lib.delete_file(db, uid, f)
    except OSError as e:
        logger.warning("[library] delete failed for %s: %s", file_id, e)
        raise HTTPException(500, "Could not delete the file")
    lib.invalidate_sync(uid)
    return {"success": True, "id": file_id}


@router.get("/library/files/{file_id}/download")
async def library_download(
    file_id: str,
    request: Request,
    inline: bool = Query(False),
    token: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    user = await get_user_for_file(request, token, db)
    f = await _require_file(db, user.id, file_id)
    return await _download_response(user.id, f, inline=inline, request=request)


@router.get("/library/files/{file_id}/preview")
async def library_preview(
    file_id: str,
    request: Request,
    format: str = Query("html"),
    token: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    user = await get_user_for_file(request, token, db)
    f = await _require_file(db, user.id, file_id)
    path = lib.file_abspath(user.id, f)
    if not path:
        raise HTTPException(410, "This file is no longer available")
    return await render_preview(
        request,
        stream=lambda: _stream_path(path),
        open_file=lambda: open(path, "rb"),
        path_of=lambda: path,
        size=os.path.getsize(path),
        backend_key=lib.storage_backend_key(f.storage_key),
        mime=f.mime_type,
        filename=f.name,
        download_url=f"{_api()}/library/files/{f.id}/download",
        token=token,
        format=format,
        etag=_library_etag(f, path),
    )


@router.get("/library/files/{file_id}/content")
async def library_get_content(
    file_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    f = await _require_file(db, uid, file_id)
    text = await _text_of(uid, f)
    folders, paths = await _folder_paths(db, uid)
    e = lib.file_entry(f, _file_path(f, paths), api_prefix=_api())
    return {"id": f.id, "name": f.name, "path": e["path"], "content": text,
            "size": e["size"], "mime": e["mime"], "modified": e["modified"]}


@router.put("/library/files/{file_id}/content")
async def library_put_content(
    file_id: str,
    body: ContentBody,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    f = await _require_file(db, uid, file_id)
    data = body.content.encode("utf-8")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Content too large")
    try:
        f = await lib.overwrite_file_bytes(db, uid, f, data)
    except Exception as e:  # noqa: BLE001
        raise _err(e)
    folders, paths = await _folder_paths(db, uid)
    return lib.file_entry(f, _file_path(f, paths), api_prefix=_api())


@router.post("/library/sync")
async def library_force_sync(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        counts = await lib.sync_user_library(db, current_user.id, force=True)
    except Exception:  # noqa: BLE001
        logger.exception("[library] forced sync failed")
        raise HTTPException(500, "Sync failed")
    return {"success": True, **counts}


# ═════════════════════════════════════════════════════════════════════
# /api/workspace — path-based compat (the shipped mobile + web shapes)
# ═════════════════════════════════════════════════════════════════════

_COMPAT_BASE = "/"


def _compat_envelope(path: str, files: list[dict]) -> dict:
    # `base: "/"` is the discriminator the phone reads to trust the listing
    # verbatim (anything else → its legacy client-side curation). Keep it
    # on EVERY response, empty folders and the root included.
    return {"path": path, "files": files, "base": _COMPAT_BASE, "curated": True}


async def _resolve_or_404(db: AsyncSession, uid: str, path: str) -> lib.PathResolution:
    res = await lib.resolve_virtual_path(db, uid, path)
    if res is None:
        raise HTTPException(404, "Not found")
    return res


async def _folder_listing(db: AsyncSession, uid: str, folder: Optional[UserFolder], path: str) -> dict:
    folders, paths = await _folder_paths(db, uid)
    subs = await lib.list_folders(db, uid, folder.id if folder else None)
    files, _ = await lib.list_files(db, uid, folder=(folder.id if folder else lib.FOLDER_ROOT),
                                    sort="name", order="asc", offset=0, limit=MAX_LIST_LIMIT)
    # The compat surface is unpaginated; page the rest in behind the scenes.
    all_files = list(files)
    off = len(files)
    while len(files) == MAX_LIST_LIMIT and off < _COMPAT_LIST_CAP:
        files, _ = await lib.list_files(db, uid, folder=(folder.id if folder else lib.FOLDER_ROOT),
                                        sort="name", order="asc", offset=off, limit=MAX_LIST_LIMIT)
        all_files.extend(files)
        off += len(files)
    entries = [lib.compat_dir_entry(fo, paths.get(fo.id, fo.name)) for fo in subs]
    entries += [lib.compat_file_entry(f, _file_path(f, paths)) for f in all_files[:_COMPAT_LIST_CAP]]
    return _compat_envelope(path, entries)


@router.get("/workspace/files")
async def compat_list_files(
    path: str = Query("", description="Virtual folder path ('' = root)"),
    refresh: bool = Query(False),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    await _synced(db, uid, refresh)
    res = await _resolve_or_404(db, uid, path)
    if res.kind == "file":
        raise HTTPException(404, "Directory not found")
    return await _folder_listing(db, uid, res.folder, res.path)


@router.get("/workspace/tree")
async def compat_tree(
    path: str = Query(""),
    depth: int = Query(3, ge=1, le=5),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Recursive virtual tree — replaces the old container find + apps merge."""
    uid = current_user.id
    await _synced(db, uid)
    res = await _resolve_or_404(db, uid, path)
    if res.kind == "file":
        raise HTTPException(404, "Directory not found")
    folders, paths = await _folder_paths(db, uid)
    root_id = res.folder.id if res.folder else None
    out: list[dict] = []

    async def _walk(fid: Optional[str], d: int):
        if d <= 0 or len(out) >= 1000:
            return
        for fo in await lib.list_folders(db, uid, fid):
            p = paths.get(fo.id, fo.name)
            out.append({"name": fo.name, "path": p, "type": "dir", "size": 0,
                        "modified": lib._iso(fo.updated_at), "id": fo.id, "kind": "folder",
                        "mime": None, "system": fo.system_key})
            await _walk(fo.id, d - 1)
        rows, _ = await lib.list_files(db, uid, folder=(fid or lib.FOLDER_ROOT), limit=MAX_LIST_LIMIT)
        for f in rows:
            e = lib.compat_file_entry(f, _file_path(f, paths))
            out.append({"name": f.name, "path": e["path"], "type": "file", "size": e["size"],
                        "modified": e["modified"], "id": f.id, "kind": e["kind"], "mime": f.mime_type,
                        "system": None})

    await _walk(root_id, depth)
    # Same order the old route promised: directories first, then files, by path.
    out.sort(key=lambda e: (0 if e["type"] == "dir" else 1, e["path"].lower()))
    return {"path": res.path, "tree": out, "base": _COMPAT_BASE, "curated": True}


def _content_payload(f: UserFile, path: str, text: str) -> dict:
    e = lib.file_entry(f, path)
    return {"path": path, "name": f.name, "id": f.id, "content": text,
            "size": e["size"], "mime": e["mime"], "modified": e["modified"]}


@router.get("/workspace/file-content")
async def compat_file_content(
    path: str = Query(..., description="Virtual file path"),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    await _synced(db, uid)
    res = await lib.resolve_virtual_path(db, uid, path)
    f: Optional[UserFile] = res.file if res and res.kind == "file" else None
    if f is None:
        # Old toup://report?path=<physical rel> links in chat history.
        f = await lib.resolve_legacy_physical_path(db, uid, path)
    if f is None:
        raise HTTPException(404, "File not found")
    text = await _text_of(uid, f)
    folders, paths = await _folder_paths(db, uid)
    return _content_payload(f, _file_path(f, paths), text)


# Alias kept for the agent image's old route name (workspace_files.py had
# GET /workspace/file); the platform proxy called it before this rewrite.
@router.get("/workspace/file")
async def compat_file_content_alias(
    path: str = Query(...),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await compat_file_content(path=path, current_user=current_user, db=db)


class CompatWrite(BaseModel):
    path: str
    content: str


@router.post("/workspace/file-write")
async def compat_file_write(
    body: CompatWrite,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    await _synced(db, uid)
    try:
        parts = lib.split_virtual_path(body.path)
    except ValueError:
        raise HTTPException(400, "Invalid path")
    if not parts:
        raise HTTPException(400, "Path is required")
    data = body.content.encode("utf-8")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Content too large")
    res = await lib.resolve_virtual_path(db, uid, body.path)
    try:
        if res is not None and res.kind == "file":
            f = await lib.overwrite_file_bytes(db, uid, res.file, data)
        elif res is not None:
            raise HTTPException(409, "A folder with that name exists")
        else:
            parent = await lib.ensure_folder_path(db, uid, parts[:-1])
            f = await lib.store_new_file(db, uid, name=parts[-1], data=data,
                                         mime=lib.guess_mime(parts[-1], "text/plain"),
                                         origin=ORIGIN_UPLOAD, folder_id=parent.id if parent else None)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise _err(e)
    lib.invalidate_sync(uid)
    folders, paths = await _folder_paths(db, uid)
    return {"success": True, "path": _file_path(f, paths), "size": int(f.size_bytes or 0), "id": f.id}


@router.delete("/workspace/file")
async def compat_delete(
    path: str = Query(...),
    recursive: bool = Query(False),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    await _synced(db, uid)
    res = await _resolve_or_404(db, uid, path)
    if res.kind == "root":
        raise HTTPException(400, "Cannot delete the library root")
    try:
        if res.kind == "file":
            await lib.delete_file(db, uid, res.file)
        else:
            await lib.delete_folder(db, uid, res.folder, recursive=recursive)
    except lib.NotEmpty:
        return _folder_not_empty()
    except OSError:
        raise HTTPException(500, "Could not delete the file")
    except Exception as e:  # noqa: BLE001
        raise _err(e)
    lib.invalidate_sync(uid)
    return {"success": True, "path": res.path}


class CompatRename(BaseModel):
    old_path: str
    new_path: str


@router.post("/workspace/file-rename")
async def compat_rename(
    body: CompatRename,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    await _synced(db, uid)
    res = await _resolve_or_404(db, uid, body.old_path)
    if res.kind == "root":
        raise HTTPException(400, "Cannot rename the library root")
    try:
        new_parts = lib.split_virtual_path(body.new_path)
    except ValueError:
        raise HTTPException(400, "Invalid path")
    if not new_parts:
        raise HTTPException(400, "new_path is required")
    try:
        parent = await lib.ensure_folder_path(db, uid, new_parts[:-1])
        parent_id = parent.id if parent else None
        if res.kind == "file":
            f = res.file
            f = await lib.update_file(db, uid, f, name=new_parts[-1], folder_id=parent_id, move=True)
            entity_id = f.id
        else:
            fo = res.folder
            if fo.system_key and parent_id != fo.parent_id:
                raise lib.SystemFolderError("System folders stay at the top level")
            fo = await lib.update_folder(db, uid, fo, name=new_parts[-1], parent_id=parent_id,
                                         move=(parent_id != fo.parent_id))
            entity_id = fo.id
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        await db.rollback()
        raise _err(e)
    lib.invalidate_sync(uid)
    folders, paths = await _folder_paths(db, uid)
    if res.kind == "file":
        new_path = _file_path(await lib.get_file(db, uid, entity_id), paths)
    else:
        new_path = paths.get(entity_id, "/".join(new_parts))
    return {"success": True, "old_path": res.path, "new_path": new_path, "id": entity_id}


class CompatMkdir(BaseModel):
    path: str


@router.post("/workspace/create-dir")
async def compat_mkdir(
    body: CompatMkdir,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = current_user.id
    await _synced(db, uid)
    try:
        parts = lib.split_virtual_path(body.path)
    except ValueError:
        raise HTTPException(400, "Invalid path")
    if not parts:
        raise HTTPException(400, "Path is required")
    try:
        leaf = await lib.ensure_folder_path(db, uid, parts)
        await db.commit()
    except Exception as e:  # noqa: BLE001
        await db.rollback()
        raise _err(e)
    folders, paths = await _folder_paths(db, uid)
    return {"success": True, "path": paths.get(leaf.id, "/".join(parts)), "id": leaf.id}


@router.get("/workspace/file-download")
async def compat_download(
    request: Request,
    path: str = Query(...),
    inline: bool = Query(False),
    token: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    user = await get_user_for_file(request, token, db)
    await _synced(db, user.id)
    res = await lib.resolve_virtual_path(db, user.id, path)
    if res is None or res.kind != "file":
        raise HTTPException(404, "File not found")
    return await _download_response(user.id, res.file, inline=inline, request=request)


@router.post("/workspace/file-upload", status_code=201)
async def compat_upload(
    request: Request,
    file: UploadFile = File(...),
    path: Optional[str] = Form(None),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """multipart `file` + `path` (form or query) = destination FOLDER path.
    Empty path → Uploads."""
    uid = current_user.id
    await _synced(db, uid)
    dest_path = path if path is not None else request.query_params.get("path", "")
    try:
        parts = lib.split_virtual_path(dest_path or "")
    except ValueError:
        raise HTTPException(400, "Invalid path")
    data = await _read_upload(file)
    if not data:
        raise HTTPException(400, "Empty file")
    name = os.path.basename((file.filename or "").replace("\\", "/")).strip() or "upload"
    try:
        if parts:
            res = await lib.resolve_virtual_path(db, uid, dest_path)
            if res is not None and res.kind == "file":
                raise HTTPException(409, "Destination is a file")
            folder = res.folder if res is not None else await lib.ensure_folder_path(db, uid, parts)
            folder_id = folder.id if folder else None
        else:
            sys_folders = await lib.ensure_system_folders(db, uid)
            folder_id = sys_folders[lib.SYSTEM_FOLDER_UPLOADS].id
        row = await lib.store_new_file(db, uid, name=name, data=data, mime=file.content_type or "",
                                       origin=ORIGIN_UPLOAD, folder_id=folder_id)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        await db.rollback()
        raise _err(e)
    lib.invalidate_sync(uid)
    folders, paths = await _folder_paths(db, uid)
    return {"success": True, "path": _file_path(row, paths), "name": row.name,
            "size": int(row.size_bytes or 0), "id": row.id}


@router.get("/workspace/preview/{path:path}")
async def compat_preview_removed(path: str):
    """The vibe-app raw file server is gone: it served any workspace file
    by physical path (source included). App previews live at
    /api/apps/{id}/preview."""
    raise HTTPException(410, "This preview route was removed. Use /api/library/files/{id}/preview.")
