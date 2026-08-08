"""Google Docs connector provider.

Reads + writes Google Docs via the v1 REST API. Uses the same shared
Google OAuth client + helpers as Gmail/Calendar/Drive — only the API
endpoint differs.

Doc fetch path: GET /v1/documents/{docId}. Response is a structured
body tree; we flatten it to plain text on the way out so the agent
isn't paying tokens for the StructuralElement schema.

Doc write path: batchUpdate request with `insertText` requests at a
specific index. Append-text computes the "end of body" index by
inspecting the latest body content offsets returned by GET.
"""

from __future__ import annotations

import json
import re
from typing import ClassVar, Optional

from app.connectors._google_base import (
    _GoogleConnectorError,
    google_liveness,
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

DOCS_API = "https://docs.googleapis.com/v1"

# A syntactically valid document id that will never resolve. Used only
# by `health_probe`; GETting it costs one small request and answers
# 404 on a healthy path. Keep it well-formed — a malformed id could be
# rejected at request validation, before Google checks the token, which
# would make the probe blind to an expired credential.
_PROBE_DOC_ID = "toup-health-probe-does-not-exist"

# Match a Google Docs URL and capture the document id. Tolerates the
# `/edit`, `/preview`, and `?usp=...` suffixes Google emits.
_DOC_URL_RE = re.compile(
    r"docs\.google\.com/document/d/([a-zA-Z0-9_-]+)",
)


async def _resolve_token(user_id: str) -> str:
    async with async_session_maker() as db:
        ident = await _vault.get(db, user_id, "docs")
    if ident is None or not ident.access_token:
        raise _GoogleConnectorError(
            ConnectorToolError(message="No active Google Docs identity", retryable=False),
        )
    return ident.access_token


def _retarget_reauth(result: ConnectorResult) -> ConnectorResult:
    """Per-connector reauth URL so the user lands on the right card."""
    if isinstance(result, ConnectorReauthRequired):
        return ConnectorReauthRequired(reauth_url="/agent/integrations/docs")
    return result


def _extract_doc_id(tool_input: dict) -> Optional[str]:
    """Accept either `document_id` (preferred) or `document_url` and
    return the bare id. Returns None when neither is usable so the
    caller can produce a clean ConnectorToolError."""
    raw_id = (tool_input.get("document_id") or "").strip()
    if raw_id:
        return raw_id
    url = (tool_input.get("document_url") or "").strip()
    if url:
        m = _DOC_URL_RE.search(url)
        if m:
            return m.group(1)
    return None


def _flatten_body(body: dict) -> tuple[str, int]:
    """Walk the doc body and join every textRun. Returns the plain
    text plus the maximum `endIndex` seen — which is the right index
    to anchor `insertText` requests at the end of the doc."""
    text_parts: list[str] = []
    max_end = 1
    for element in body.get("content", []):
        end = element.get("endIndex")
        if isinstance(end, int) and end > max_end:
            max_end = end
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        for run in paragraph.get("elements", []):
            text_run = run.get("textRun")
            if text_run and isinstance(text_run.get("content"), str):
                text_parts.append(text_run["content"])
    return "".join(text_parts), max_end


class DocsProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "docs"

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
            if tool_name == "docs__get":
                doc_id = _extract_doc_id(tool_input)
                if not doc_id:
                    return ConnectorToolError(
                        message="document_id or document_url required",
                        retryable=False,
                    )
                doc = await google_request(
                    "GET",
                    f"{DOCS_API}/documents/{doc_id}",
                    access_token=access_token,
                    scope_hint="documents",
                )
                text, _ = _flatten_body(doc.get("body") or {})
                return ConnectorOk(content=json.dumps({
                    "id": doc.get("documentId", doc_id),
                    "title": doc.get("title", ""),
                    "url": f"https://docs.google.com/document/d/{doc_id}/edit",
                    "text": text[:50_000],
                    "truncated": len(text) > 50_000,
                }))

            if tool_name == "docs__create":
                title = (tool_input.get("title") or "").strip()
                body_text = tool_input.get("body") or ""
                if not title:
                    return ConnectorToolError(
                        message="title required", retryable=False,
                    )
                # Step 1: create the doc with the title. The Docs API
                # `documents.create` only accepts a title; body content
                # is added via a follow-up batchUpdate.
                created = await google_request(
                    "POST",
                    f"{DOCS_API}/documents",
                    access_token=access_token,
                    json_body={"title": title},
                    scope_hint="documents",
                )
                doc_id = created.get("documentId")
                if not doc_id:
                    return ConnectorToolError(
                        message="Docs API returned no documentId",
                        retryable=True,
                    )
                # Step 2: insert body text at index 1 (the very start
                # of the body — index 0 is the implicit pre-body anchor).
                if body_text:
                    await google_request(
                        "POST",
                        f"{DOCS_API}/documents/{doc_id}:batchUpdate",
                        access_token=access_token,
                        json_body={
                            "requests": [{
                                "insertText": {
                                    "location": {"index": 1},
                                    "text": body_text,
                                },
                            }],
                        },
                        scope_hint="documents",
                    )
                return ConnectorOk(content=json.dumps({
                    "id": doc_id,
                    "title": title,
                    "url": f"https://docs.google.com/document/d/{doc_id}/edit",
                }))

            if tool_name == "docs__append_text":
                doc_id = _extract_doc_id(tool_input)
                if not doc_id:
                    return ConnectorToolError(
                        message="document_id or document_url required",
                        retryable=False,
                    )
                text = tool_input.get("text") or ""
                if not text:
                    return ConnectorToolError(
                        message="text to append cannot be empty",
                        retryable=False,
                    )
                # First fetch the current end-of-body index. Docs API
                # rejects insertText at the absolute last index (which
                # is the implicit trailing newline anchor) — we anchor
                # one position before it.
                doc = await google_request(
                    "GET",
                    f"{DOCS_API}/documents/{doc_id}",
                    access_token=access_token,
                    params={"fields": "body(content(endIndex))"},
                    scope_hint="documents",
                )
                _, end_index = _flatten_body(doc.get("body") or {})
                # Always prefix with newline so the appended text
                # starts on its own line.
                payload = "\n" + text
                await google_request(
                    "POST",
                    f"{DOCS_API}/documents/{doc_id}:batchUpdate",
                    access_token=access_token,
                    json_body={
                        "requests": [{
                            "insertText": {
                                "location": {"index": max(end_index - 1, 1)},
                                "text": payload,
                            },
                        }],
                    },
                    scope_hint="documents",
                )
                return ConnectorOk(content=json.dumps({
                    "id": doc_id,
                    "url": f"https://docs.google.com/document/d/{doc_id}/edit",
                    "appended_chars": len(payload),
                }))

            return ConnectorToolError(
                message=f"unknown docs tool {tool_name!r}",
                retryable=False,
            )
        except _GoogleConnectorError as e:
            return _retarget_reauth(e.result)

    async def revoke(self, user_id, access_token, refresh_token=None):
        await google_revoke(access_token)

    async def refresh(self, refresh_token: str) -> RefreshResult:
        return await google_refresh(refresh_token)

    async def health_probe(self, ctx: ConnectorContext) -> HealthResult:
        """Docs has no /me and no /about, so we probe by GETting a
        document id that cannot exist. Google authenticates before it
        resolves the entity, so a 404 is proof of a healthy path: API
        enabled, token accepted, scope sufficient. See `google_liveness`.

        This used to only check that a token could be decrypted, on the
        reasoning that "the next real tool call surfaces any deeper
        rot". It does not — nothing feeds a failed tool call back into
        the identity's status. On 2026-08-07 this connector reported
        `active` for two days while every call returned 403
        SERVICE_DISABLED, because the Docs API had never been enabled
        on the Cloud project. The card read "Connected" throughout.
        A probe that cannot fail is not a probe.
        """
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
        except _GoogleConnectorError as e:
            return HealthResult(ok=False, detail=repr(e.result))
        except Exception as e:
            return HealthResult(ok=False, detail=f"{type(e).__name__}: {e}")

        ok, detail = await google_liveness(
            f"{DOCS_API}/documents/{_PROBE_DOC_ID}",
            access_token=access_token,
        )
        return HealthResult(ok=ok, detail=detail)
