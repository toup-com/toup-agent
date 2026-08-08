"""Google Sheets connector provider.

Sheets v4 REST, on the same shared Google OAuth client and
`_google_base` helpers as Gmail/Calendar/Drive/Docs. Only the API host
and the error surface differ.

The manifest for this connector shipped on 2026-05-10 and this file did
not, so `connector_registry` hit `_SkipConnector("provider.py is
missing")` at every boot and Sheets never appeared in the catalogue for
anyone. The manifest is the contract; this is the implementation of it.

Two things worth knowing before editing:

  - **A1 notation must be percent-encoded into the path.** Ranges carry
    `!` and `:` by design and sheet TITLES routinely carry spaces and
    non-ASCII ("Q3 Pipeline", "Ventas 2026"). Interpolating one raw
    produces a URL that either 400s or, worse, silently addresses a
    different range. Every range goes through `_quote_range`.
  - **`sheets__list_spreadsheets` is the odd one out.** Sheets has no
    "list my spreadsheets" endpoint — enumeration is a Drive concern,
    and it needs `drive.readonly`, which is RESTRICTED in Google's
    verification policy and would drag the whole project into a CASA
    assessment. So it stays in `scopes_optional` (never requested by
    `oauth.py`, which only sends `oauth.scopes`), and this tool checks
    the grant and returns `ConnectorScopeMissing` instead of burning a
    round-trip on a 403 the agent can't interpret. Everything else here
    works on a spreadsheet the user names, with `spreadsheets` alone.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import Any, ClassVar, Optional

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
    ConnectorScopeMissing,
    ConnectorToolError,
    HealthResult,
    RefreshResult,
)
from app.db.database import async_session_maker
from app.services import connector_vault as _vault

SHEETS_API = "https://sheets.googleapis.com/v4"
DRIVE_API = "https://www.googleapis.com/drive/v3"

SPREADSHEET_MIME = "application/vnd.google-apps.spreadsheet"

# Scope needed to ENUMERATE spreadsheets (a Drive capability). Optional
# by design — see the module docstring.
DRIVE_LIST_SCOPE = "https://www.googleapis.com/auth/drive.readonly"

# Capture the id out of a full Sheets URL. Tolerates the `/edit`,
# `#gid=0` and `?usp=sharing` suffixes Google emits.
_SHEET_URL_RE = re.compile(
    r"docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)",
)

# A well-formed spreadsheet id that will never resolve. Used only by
# `health_probe` — see `google_liveness` for why a 404 is the healthy
# answer.
_PROBE_SHEET_ID = "toup-health-probe-does-not-exist"

_VALUE_INPUT_OPTIONS = {"RAW", "USER_ENTERED"}


async def _resolve_identity(user_id: str):
    """Return the decrypted identity. We need the granted SCOPES as
    well as the token, so this returns the whole struct rather than
    just the access token like its siblings do."""
    async with async_session_maker() as db:
        ident = await _vault.get(db, user_id, "sheets")
    if ident is None or not ident.access_token:
        raise _GoogleConnectorError(
            ConnectorToolError(
                message="No active Google Sheets identity", retryable=False,
            ),
        )
    return ident


def _retarget_reauth(result: ConnectorResult) -> ConnectorResult:
    """Per-connector reauth URL so the user lands on the right card."""
    if isinstance(result, ConnectorReauthRequired):
        return ConnectorReauthRequired(reauth_url="/agent/integrations/sheets")
    return result


def _extract_sheet_id(raw: str) -> Optional[str]:
    """Accept a bare id or any Sheets URL and return the bare id.

    The manifest declares one `spreadsheet_id` field, so a user who
    pastes a link puts the whole URL in it. Extracting here means that
    just works instead of 404ing on an id like
    `https://docs.google.com/spreadsheets/d/abc/edit`.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    m = _SHEET_URL_RE.search(raw)
    if m:
        return m.group(1)
    # Not a URL we recognise — if it still looks like a URL, refuse
    # rather than send a doomed request with a slash-laden path.
    if "/" in raw or raw.startswith("http"):
        return None
    return raw


def _quote_range(a1: str) -> str:
    """Percent-encode A1 notation for use as a single path segment.

    `safe=""` is deliberate: `/` inside a sheet title would otherwise
    split the path and address a different resource entirely.
    """
    return urllib.parse.quote(a1, safe="")


def _drive_escape(s: str) -> str:
    """Escape a literal for a Drive `q` string. Drive uses single-quoted
    literals, so an apostrophe in a filename ("Nariman's budget") ends
    the literal early and makes the rest of the query syntax."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _coerce_rows(raw: Any) -> Optional[list[list[Any]]]:
    """Validate `values` is a 2D array. Returns None when it isn't.

    The agent sometimes hands a flat list for a single row. Accept that
    and lift it, rather than writing a column of one-character cells —
    which is what Sheets does with a flat list and is very hard for a
    user to spot after the fact.
    """
    if not isinstance(raw, list) or not raw:
        return None
    if all(isinstance(r, list) for r in raw):
        return raw
    if any(isinstance(r, list) for r in raw):
        return None  # mixed — ambiguous, refuse rather than guess
    return [raw]


def _value_input_option(tool_input: dict) -> str:
    opt = str(tool_input.get("value_input_option") or "USER_ENTERED").upper()
    return opt if opt in _VALUE_INPUT_OPTIONS else "USER_ENTERED"


def _sheet_url(spreadsheet_id: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"


class SheetsProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "sheets"

    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ConnectorContext,
    ) -> ConnectorResult:
        try:
            ident = await _resolve_identity(ctx.user_id)
        except _GoogleConnectorError as e:
            return _retarget_reauth(e.result)
        access_token = ctx.access_token or ident.access_token
        granted = set(ident.scopes or [])

        try:
            if tool_name == "sheets__list_spreadsheets":
                return await self._list_spreadsheets(
                    tool_input, access_token, granted,
                )
            if tool_name == "sheets__read_range":
                return await self._read_range(tool_input, access_token)
            if tool_name == "sheets__append_rows":
                return await self._write_rows(
                    tool_input, access_token, append=True,
                )
            if tool_name == "sheets__update_range":
                return await self._write_rows(
                    tool_input, access_token, append=False,
                )
            if tool_name == "sheets__create_spreadsheet":
                return await self._create(tool_input, access_token)

            return ConnectorToolError(
                message=f"unknown sheets tool {tool_name!r}", retryable=False,
            )
        except _GoogleConnectorError as e:
            return _retarget_reauth(e.result)

    # ── tools ──────────────────────────────────────────────────────

    async def _list_spreadsheets(
        self, tool_input: dict, access_token: str, granted: set[str],
    ) -> ConnectorResult:
        if DRIVE_LIST_SCOPE not in granted:
            # Answered locally, not via a 403. The agent's contract for
            # ConnectorScopeMissing is "ask the user to reconnect with
            # the broader permission, or work from an id/URL they give
            # you" — which is actionable. A raw 403 is not.
            return ConnectorScopeMissing(required_scope=DRIVE_LIST_SCOPE)

        clauses = [f"mimeType='{SPREADSHEET_MIME}'", "trashed=false"]
        name_query = (tool_input.get("query") or "").strip()
        if name_query:
            clauses.append(f"name contains '{_drive_escape(name_query)}'")

        max_results = max(1, min(int(tool_input.get("max_results", 20) or 20), 50))
        body = await google_request(
            "GET",
            f"{DRIVE_API}/files",
            access_token=access_token,
            params={
                "q": " and ".join(clauses),
                "pageSize": max_results,
                "orderBy": "modifiedTime desc",
                "fields": "files(id,name,modifiedTime,webViewLink)",
            },
            scope_hint=DRIVE_LIST_SCOPE,
        )
        return ConnectorOk(content=json.dumps({
            "spreadsheets": [
                {
                    "id": f.get("id"),
                    "title": f.get("name"),
                    "modified_at": f.get("modifiedTime"),
                    "url": f.get("webViewLink") or _sheet_url(f.get("id", "")),
                }
                for f in body.get("files", [])
            ],
        }))

    async def _read_range(
        self, tool_input: dict, access_token: str,
    ) -> ConnectorResult:
        sheet_id = _extract_sheet_id(tool_input.get("spreadsheet_id", ""))
        a1 = (tool_input.get("range") or "").strip()
        if not sheet_id:
            return ConnectorToolError(
                message="spreadsheet_id required (id or Sheets URL)",
                retryable=False,
            )
        if not a1:
            return ConnectorToolError(
                message='range required, in A1 notation (e.g. "Sheet1!A1:D20")',
                retryable=False,
            )
        body = await google_request(
            "GET",
            f"{SHEETS_API}/spreadsheets/{sheet_id}/values/{_quote_range(a1)}",
            access_token=access_token,
            scope_hint="spreadsheets",
        )
        values = body.get("values", [])
        return ConnectorOk(content=json.dumps({
            "id": sheet_id,
            "range": body.get("range", a1),
            "rows": len(values),
            "values": values,
            "url": _sheet_url(sheet_id),
        }))

    async def _write_rows(
        self, tool_input: dict, access_token: str, *, append: bool,
    ) -> ConnectorResult:
        sheet_id = _extract_sheet_id(tool_input.get("spreadsheet_id", ""))
        a1 = (tool_input.get("range") or "").strip()
        rows = _coerce_rows(tool_input.get("values"))
        if not sheet_id:
            return ConnectorToolError(
                message="spreadsheet_id required (id or Sheets URL)",
                retryable=False,
            )
        if not a1:
            return ConnectorToolError(
                message="range required, in A1 notation", retryable=False,
            )
        if rows is None:
            return ConnectorToolError(
                message=(
                    "values must be a non-empty array of rows, each row an "
                    "array of cell values"
                ),
                retryable=False,
            )

        quoted = _quote_range(a1)
        option = _value_input_option(tool_input)
        if append:
            body = await google_request(
                "POST",
                f"{SHEETS_API}/spreadsheets/{sheet_id}/values/{quoted}:append",
                access_token=access_token,
                params={
                    "valueInputOption": option,
                    # Without this Sheets OVERWRITES whatever sits below
                    # the detected table instead of shifting rows down.
                    "insertDataOption": "INSERT_ROWS",
                },
                json_body={"values": rows},
                scope_hint="spreadsheets",
            )
            updates = body.get("updates", {}) or {}
            written_range = updates.get("updatedRange", a1)
            cells = updates.get("updatedCells", 0)
        else:
            body = await google_request(
                "PUT",
                f"{SHEETS_API}/spreadsheets/{sheet_id}/values/{quoted}",
                access_token=access_token,
                params={"valueInputOption": option},
                json_body={"values": rows},
                scope_hint="spreadsheets",
            )
            written_range = body.get("updatedRange", a1)
            cells = body.get("updatedCells", 0)

        return ConnectorOk(content=json.dumps({
            "id": sheet_id,
            "range": written_range,
            "rows_written": len(rows),
            "cells_written": cells,
            "url": _sheet_url(sheet_id),
        }))

    async def _create(
        self, tool_input: dict, access_token: str,
    ) -> ConnectorResult:
        title = (tool_input.get("title") or "").strip()
        if not title:
            return ConnectorToolError(message="title required", retryable=False)
        sheet_title = (tool_input.get("sheet_title") or "Sheet1").strip() or "Sheet1"
        body = await google_request(
            "POST",
            f"{SHEETS_API}/spreadsheets",
            access_token=access_token,
            json_body={
                "properties": {"title": title},
                "sheets": [{"properties": {"title": sheet_title}}],
            },
            scope_hint="spreadsheets",
        )
        sheet_id = body.get("spreadsheetId")
        if not sheet_id:
            return ConnectorToolError(
                message="Sheets API returned no spreadsheetId", retryable=True,
            )
        return ConnectorOk(content=json.dumps({
            "id": sheet_id,
            "title": title,
            "sheet_title": sheet_title,
            "url": body.get("spreadsheetUrl") or _sheet_url(sheet_id),
        }))

    # ── lifecycle ──────────────────────────────────────────────────

    async def revoke(self, user_id, access_token, refresh_token=None):
        await google_revoke(access_token)

    async def refresh(self, refresh_token: str) -> RefreshResult:
        return await google_refresh(refresh_token)

    async def health_probe(self, ctx: ConnectorContext) -> HealthResult:
        """Probe `spreadsheets.get` on an id that cannot exist.

        Deliberately NOT the manifest's declared `health.probe` tool
        (`sheets__list_spreadsheets`) — that one needs the optional
        `drive.readonly` scope, which no user holds, so probing through
        it would report every healthy identity as down. That exact
        mistake shipped in the Calendar connector and made it read
        "Provider down" for every user on the platform. The manifest
        field only has to name a real tool for the registry lint; the
        probe that actually runs is this method.
        """
        try:
            ident = await _resolve_identity(ctx.user_id)
        except _GoogleConnectorError as e:
            return HealthResult(ok=False, detail=repr(e.result))
        except Exception as e:
            return HealthResult(ok=False, detail=f"{type(e).__name__}: {e}")

        ok, detail = await google_liveness(
            f"{SHEETS_API}/spreadsheets/{_PROBE_SHEET_ID}",
            access_token=ctx.access_token or ident.access_token,
        )
        return HealthResult(ok=ok, detail=detail)
