"""
File serving endpoints for generated attachments.

GET  /api/files/{message_id}/{attachment_id}
GET  /api/files/{message_id}/{attachment_id}/preview?format=html

Auth: JWT via existing get_current_user dependency. The endpoint verifies
that the requested attachment belongs to a conversation owned by the
caller — cross-user access returns 404 (not 403, to avoid leaking
attachment existence).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user, SSO_COOKIE_NAME
from app.config import settings
from app.db import get_db
from app.db.models import Conversation, Message, User
from app.services import decode_access_token, get_user_by_id
from app.services.file_storage import get_storage_backend, stream_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["Files"])


async def _get_user_for_file(
    request: Request,
    token: Optional[str],
    db: AsyncSession,
):
    """Auth for file endpoints. Accepts JWT via:
      1) Authorization: Bearer <token> header (standard)
      2) hex_sso_token cookie (SSO, same-origin)
      3) ?token=<jwt> query param — required for <iframe src> / <img src>
         embeds that can't set headers.
      4) X-Agent-Key header (platform proxy mode, reused from auth.py).
    """
    user_id: Optional[str] = None

    # Bearer header
    auth_header = request.headers.get("authorization", "") if request else ""
    if auth_header.lower().startswith("bearer "):
        user_id = decode_access_token(auth_header[7:])

    # SSO cookie
    if not user_id and request is not None:
        cookie_tok = request.cookies.get(SSO_COOKIE_NAME)
        if cookie_tok:
            user_id = decode_access_token(cookie_tok)

    # Query param (for embeds)
    if not user_id and token:
        user_id = decode_access_token(token)

    # Agent-mode fallback
    if not user_id and request is not None:
        agent_key = request.headers.get("x-agent-key", "")
        if settings.agent_api_key and agent_key == settings.agent_api_key and settings.user_id:
            user_id = settings.user_id

    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    user = await get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


async def _load_attachment(
    message_id: str, attachment_id: str, user_id: str, db: AsyncSession
) -> dict:
    """Fetch the attachment dict, verifying the caller owns the parent conversation.
    Returns 404 on miss OR on cross-user access (don't leak existence)."""
    msg = (await db.execute(
        select(Message).where(Message.id == message_id)
    )).scalar_one_or_none()
    if not msg or not msg.attachments:
        raise HTTPException(status_code=404, detail="Attachment not found")

    conv = (await db.execute(
        select(Conversation).where(Conversation.id == msg.conversation_id)
    )).scalar_one_or_none()
    if not conv or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="Attachment not found")

    # Column is JSON(B) — already a Python list when fetched via SQLAlchemy.
    # Belt-and-braces: some drivers may still return a string (legacy rows).
    attachments = msg.attachments
    if isinstance(attachments, str):
        try:
            attachments = json.loads(attachments)
        except (TypeError, ValueError):
            raise HTTPException(status_code=404, detail="Attachment not found")
    if not isinstance(attachments, list):
        raise HTTPException(status_code=404, detail="Attachment not found")

    for att in attachments:
        if isinstance(att, dict) and att.get("id") == attachment_id:
            return att
    raise HTTPException(status_code=404, detail="Attachment not found")


@router.get("/{message_id}/{attachment_id}")
async def download_file(
    message_id: str,
    attachment_id: str,
    request: Request,
    token: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Stream the raw attachment bytes with correct Content-Type + Content-Disposition."""
    current_user = await _get_user_for_file(request, token, db)
    att = await _load_attachment(message_id, attachment_id, current_user.id, db)
    key = att.get("storage_path")
    if not key:
        raise HTTPException(status_code=500, detail="Attachment has no storage_path")

    backend = get_storage_backend()
    if not backend.exists(key):
        logger.warning("Attachment file missing on disk: key=%s message=%s", key, message_id)
        raise HTTPException(status_code=410, detail="File has been deleted from storage")

    filename = att.get("filename", "download.bin")
    mime = att.get("mime_type", "application/octet-stream")

    return StreamingResponse(
        stream_file(key),
        media_type=mime,
        headers={
            "Content-Length": str(backend.size(key)),
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "private, max-age=3600",
        },
    )


@router.get("/{message_id}/{attachment_id}/preview")
async def preview_file(
    message_id: str,
    attachment_id: str,
    request: Request,
    format: str = Query("html"),
    token: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Server-side render for the DocumentViewer pane.

    DOCX  → HTML via mammoth
    XLSX  → HTML tables (one per sheet) via openpyxl
    PPTX  → 204 (frontend falls back to download)
    Other → 415 Unsupported Media Type (client should use /download)
    """
    current_user = await _get_user_for_file(request, token, db)
    att = await _load_attachment(message_id, attachment_id, current_user.id, db)
    key = att.get("storage_path")
    mime = att.get("mime_type", "")
    backend = get_storage_backend()
    if not key or not backend.exists(key):
        raise HTTPException(status_code=410, detail="File unavailable")

    if format != "html" and format != "png":
        raise HTTPException(status_code=400, detail="format must be 'html' or 'png'")

    # ── DOCX → HTML ─────────────────────────────────────────────
    if mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            import mammoth  # type: ignore
        except ImportError:
            raise HTTPException(status_code=501, detail="mammoth not installed")
        with backend.open(key) as f:
            result = mammoth.convert_to_html(f)
        # Minimal shell so pane renders with basic typography.
        html = (
            "<!doctype html><meta charset='utf-8'>"
            "<style>body{font-family:system-ui,-apple-system,sans-serif;"
            "line-height:1.6;max-width:760px;margin:2rem auto;padding:0 1rem;color:#222}"
            "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.3em .6em}"
            "img{max-width:100%}</style>"
            f"<body>{result.value}</body>"
        )
        return HTMLResponse(content=html)

    # ── XLSX → HTML (one <table> per sheet with tabs) ─────────────
    if mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        try:
            from openpyxl import load_workbook  # type: ignore
        except ImportError:
            raise HTTPException(status_code=501, detail="openpyxl not installed")
        wb = load_workbook(backend.path(key), read_only=True, data_only=True)
        parts: list[str] = [
            "<!doctype html><meta charset='utf-8'>",
            "<style>body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:1rem;color:#222}"
            ".tabs{display:flex;gap:.25rem;margin-bottom:.5rem;border-bottom:1px solid #ddd}"
            ".tab{padding:.4rem .8rem;cursor:pointer;border:1px solid #ddd;border-bottom:none;"
            "background:#f6f6f6;border-radius:.3rem .3rem 0 0;user-select:none}"
            ".tab.active{background:#fff;font-weight:600}"
            ".sheet{display:none;overflow:auto;max-height:80vh}"
            ".sheet.active{display:block}"
            "table{border-collapse:collapse;font-size:.9em}"
            "td,th{border:1px solid #ddd;padding:.25em .5em;white-space:nowrap}"
            "th{background:#f6f6f6;font-weight:600}</style>",
            "<div class='tabs'>",
        ]
        for i, name in enumerate(wb.sheetnames):
            parts.append(
                f"<div class='tab{' active' if i == 0 else ''}' onclick=\"for(var e of document.querySelectorAll('.tab,.sheet'))e.classList.remove('active');this.classList.add('active');document.getElementById('s{i}').classList.add('active')\">{name}</div>"
            )
        parts.append("</div>")
        for i, name in enumerate(wb.sheetnames):
            ws = wb[name]
            parts.append(f"<div class='sheet{' active' if i == 0 else ''}' id='s{i}'><table>")
            for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
                tag = "th" if row_idx == 0 else "td"
                parts.append("<tr>" + "".join(
                    f"<{tag}>{'' if v is None else str(v)}</{tag}>" for v in row
                ) + "</tr>")
            parts.append("</table></div>")
        return HTMLResponse(content="".join(parts))

    # ── PPTX → first-slide PNG (best-effort; libreoffice or placeholder) ─────
    if mime == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        # Full rendering is out of scope. Return 204 so frontend falls back
        # to "download + see filename" UX. Follow-up PR can add libreoffice
        # or python-pptx-to-image rendering.
        return Response(status_code=204)

    raise HTTPException(status_code=415, detail=f"No preview for {mime}")
