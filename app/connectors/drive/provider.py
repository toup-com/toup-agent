"""T3d — Google Drive connector provider.

Drive v3 REST + the simple Docs upload path. Same shared
`_google_base` helpers as Gmail/Calendar.

Doc creation is a two-step ritual:
  1. Create a Drive file with mimeType `application/vnd.google-apps.document`
  2. Get its `documentId` and POST to docs.googleapis.com/v1/documents/<id>:batchUpdate
     to insert the body text.

Drive v3 does not let you specify content at create-time for a Google
Doc; that's a Docs API responsibility. We lazily inline the Docs API
call here without adding a dedicated Docs scope — `drive.file` covers
edits to files we created.
"""

from __future__ import annotations

import json
from typing import ClassVar, Optional

from app.connectors._google_base import (
    _GoogleConnectorError,
    google_refresh,
    google_request,
    google_revoke,
)
from app.connectors.base import (
    BaseConnectorProvider,
    ConnectorContext,
    ConnectorOk,
    ConnectorReauthRequired,
    ConnectorResult,
    ConnectorToolError,
    HealthResult,
    RefreshResult,
)
from app.db.database import async_session_maker
from app.services import connector_vault as _vault

DRIVE_API = "https://www.googleapis.com/drive/v3"
DOCS_API = "https://docs.googleapis.com/v1"


async def _resolve_token(user_id: str) -> str:
    async with async_session_maker() as db:
        ident = await _vault.get(db, user_id, "drive")
    if ident is None or not ident.access_token:
        raise _GoogleConnectorError(
            ConnectorToolError(message="No active Drive identity", retryable=False),
        )
    return ident.access_token


def _retarget_reauth(result: ConnectorResult) -> ConnectorResult:
    if isinstance(result, ConnectorReauthRequired):
        return ConnectorReauthRequired(reauth_url="/agent/integrations/drive")
    return result


class DriveProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "drive"

    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ConnectorContext,
    ) -> ConnectorResult:
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
        except _GoogleConnectorError as e:
            return _retarget_reauth(e.result)

        try:
            if tool_name == "drive__list_files":
                params: dict = {
                    "pageSize": int(tool_input.get("max_results", 20)),
                    "fields": "files(id,name,mimeType,modifiedTime,webViewLink),nextPageToken",
                }
                if tool_input.get("query"):
                    params["q"] = tool_input["query"]
                result = await google_request(
                    "GET",
                    f"{DRIVE_API}/files",
                    access_token=access_token,
                    params=params,
                    scope_hint="drive.file",
                )
                return ConnectorOk(content=json.dumps(result))

            if tool_name == "drive__get_file_text":
                fid = tool_input.get("file_id")
                if not fid:
                    return ConnectorToolError(message="file_id required", retryable=False)
                # Step 1: metadata fetch to learn mimeType.
                meta = await google_request(
                    "GET",
                    f"{DRIVE_API}/files/{fid}",
                    access_token=access_token,
                    params={"fields": "id,name,mimeType,size"},
                    scope_hint="drive.file",
                )
                mime = meta.get("mimeType", "")
                # Google native types need export to text/plain.
                if mime.startswith("application/vnd.google-apps."):
                    text = await google_request(
                        "GET",
                        f"{DRIVE_API}/files/{fid}/export",
                        access_token=access_token,
                        params={"mimeType": "text/plain"},
                        scope_hint="drive.file",
                    )
                    body = text.get("raw", "")
                else:
                    # Binary download: alt=media returns the bytes; we
                    # try utf-8 decode and fall back to a [binary]
                    # marker for non-text.
                    raw = await google_request(
                        "GET",
                        f"{DRIVE_API}/files/{fid}",
                        access_token=access_token,
                        params={"alt": "media"},
                        scope_hint="drive.file",
                    )
                    body = raw.get("raw", "")
                return ConnectorOk(content=json.dumps({
                    "id": meta.get("id"),
                    "name": meta.get("name"),
                    "mimeType": mime,
                    "text": body[:50_000],  # safety cap pre-truncation
                    "truncated": len(body) > 50_000,
                }))

            if tool_name == "drive__create_doc":
                title = tool_input.get("title")
                body_text = tool_input.get("body")
                if not (title and body_text):
                    return ConnectorToolError(message="title/body required", retryable=False)
                # Step 1: create the empty Google Doc via Drive.
                created = await google_request(
                    "POST",
                    f"{DRIVE_API}/files",
                    access_token=access_token,
                    json_body={
                        "name": title,
                        "mimeType": "application/vnd.google-apps.document",
                    },
                    scope_hint="drive.file",
                )
                doc_id = created.get("id")
                if not doc_id:
                    return ConnectorToolError(
                        message="Drive returned no documentId",
                        retryable=True,
                    )
                # Step 2: insert the body via Docs batchUpdate.
                await google_request(
                    "POST",
                    f"{DOCS_API}/documents/{doc_id}:batchUpdate",
                    access_token=access_token,
                    json_body={
                        "requests": [
                            {
                                "insertText": {
                                    "location": {"index": 1},
                                    "text": body_text,
                                }
                            }
                        ]
                    },
                    scope_hint="drive.file",
                )
                return ConnectorOk(content=json.dumps({
                    "id": doc_id,
                    "title": title,
                    "url": f"https://docs.google.com/document/d/{doc_id}/edit",
                }))

            return ConnectorToolError(
                message=f"unknown drive tool {tool_name!r}",
                retryable=False,
            )
        except _GoogleConnectorError as e:
            return _retarget_reauth(e.result)

    async def revoke(self, user_id, access_token, refresh_token=None):
        await google_revoke(access_token)

    async def refresh(
        self,
        refresh_token: str,
        *,
        scopes: Optional[list[str]] = None,
    ) -> RefreshResult:
        return await google_refresh(refresh_token)

    async def health_probe(self, ctx: ConnectorContext) -> HealthResult:
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
            await google_request(
                "GET",
                f"{DRIVE_API}/about",
                access_token=access_token,
                params={"fields": "user(emailAddress)"},
                scope_hint="drive.file",
            )
            return HealthResult(ok=True)
        except _GoogleConnectorError as e:
            return HealthResult(ok=False, detail=repr(e.result))
        except Exception as e:
            return HealthResult(ok=False, detail=f"{type(e).__name__}: {e}")
