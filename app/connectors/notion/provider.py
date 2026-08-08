"""Notion connector provider — REST API v1 (api.notion.com).

Bearer auth like every other connector here, but three things about
Notion are genuinely different and each one is load-bearing:

  1. **There are no OAuth scopes.** Access comes from the connection's
     *capabilities* (set once by whoever owns the Notion integration)
     and from the user sharing individual pages with it. Nothing about
     a permission failure can be repaired by re-authorizing with a
     bigger scope set, so this provider NEVER returns
     ConnectorScopeMissing. See `_handle_notion_error`.

  2. **`Notion-Version` is mandatory on every request** and the API is
     versioned by date, not by URL. Omit the header and Notion answers
     400 `missing_version`; send an old date and you silently get an
     older object model back. Pinned at `_NOTION_VERSION` below.

  3. **The OAuth token endpoint does not look like the others.** It
     wants HTTP Basic auth (client_id:client_secret) and a JSON body —
     not the form-encoded `client_id=…&client_secret=…` that
     `app/api/oauth.py::_exchange_code_for_tokens` posts. `refresh()`
     and `revoke()` below therefore build their own requests rather
     than reusing anything shared, and the *authorization-code* leg
     needs the same treatment in the OAuth API before this connector
     can complete a connect at all.

Databases vs data sources: since API version 2025-09-03 a Notion
database is a container that holds one or more *data sources*, and the
queryable endpoint is `/v1/data_sources/{id}/query` — the old
`/v1/databases/{id}/query` is deprecated. Users, and therefore the
agent, still say "database", so `notion__query_database` accepts a
database id and resolves it to its data source. That resolution is a
real extra round-trip; it is cached nowhere because the mapping can
change under us and the read is cheap next to the query itself.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from typing import Any, ClassVar, Optional

import httpx

from app.connectors.base import (
    BaseConnectorProvider,
    ConnectorContext,
    ConnectorOk,
    ConnectorProviderDown,
    ConnectorRateLimited,
    ConnectorReauthRequired,
    ConnectorResult,
    ConnectorToolError,
    HealthResult,
    RefreshFailed,
    RefreshResult,
)
from app.db.database import async_session_maker
from app.services import connector_vault as _vault
from app.services.provider_apps import get_provider_app_async

logger = logging.getLogger(__name__)


NOTION_API = "https://api.notion.com/v1"

# Pinned deliberately. Latest documented version as of 2026-08; it is
# the one whose request/response shapes this file was written against:
# `in_trash` (not `archived`), data-source query paths, and the
# `page_or_data_source` search response. Bumping this is a code change,
# not a config change — read Notion's upgrade guide first.
_NOTION_VERSION = "2026-03-11"

_HTTP_TIMEOUT_S = 15.0

# Notion documents `Retry-After` on every 429, so this is a fallback for
# a malformed header rather than a tuned value. Its rate limit is an
# average of ~3 requests/second per connection.
_DEFAULT_RETRY_AFTER_S = 30

# Notion's own per-request ceilings (developers.notion.com/reference/request-limits).
_MAX_PAGE_SIZE = 100          # pagination ceiling on every list endpoint
_MAX_RICH_TEXT_CHARS = 2000   # per rich_text element
_MAX_BLOCKS_PER_REQUEST = 100 # per `children` array

# Guards for `notion__get_page_content`. A page can nest arbitrarily
# deep and each level costs a request against a 3 rps budget, so the
# request count — not the depth — is the real limiter.
_MAX_BLOCK_REQUESTS = 25
_MAX_BLOCK_DEPTH = 4

_REAUTH_URL = "/agent/integrations/notion"


# ─── Pooled client ───────────────────────────────────────────────────
#
# Same rationale as `_google_base._get_google_client`: without pooling
# every call pays a fresh TLS handshake (~200-400 ms), which is
# especially wasteful here because several tools issue 2-3 requests.

_NOTION_CLIENT: Optional[httpx.AsyncClient] = None
_NOTION_CLIENT_LOCK = asyncio.Lock()


async def _get_notion_client() -> httpx.AsyncClient:
    global _NOTION_CLIENT
    if _NOTION_CLIENT is not None:
        return _NOTION_CLIENT
    async with _NOTION_CLIENT_LOCK:
        if _NOTION_CLIENT is None:
            try:
                _NOTION_CLIENT = httpx.AsyncClient(
                    timeout=_HTTP_TIMEOUT_S,
                    limits=httpx.Limits(
                        max_connections=50,
                        max_keepalive_connections=50,
                        keepalive_expiry=300.0,
                    ),
                )
            except Exception as e:
                logger.error(
                    "[notion] pooled AsyncClient construction failed "
                    "(%s: %s) — falling back to per-call clients. "
                    "Investigate immediately.",
                    type(e).__name__, e,
                )
                _NOTION_CLIENT = httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S)
    return _NOTION_CLIENT


async def shutdown_notion_client() -> None:
    global _NOTION_CLIENT
    if _NOTION_CLIENT is not None:
        await _NOTION_CLIENT.aclose()
        _NOTION_CLIENT = None


class _NotionError(Exception):
    """Wraps a `ConnectorResult` so any depth of the call chain can exit
    via `raise` and translate once, at `execute`'s outer except. Same
    pattern as `_GoogleConnectorError` / `_LinkedInError`."""

    def __init__(self, result: ConnectorResult):
        super().__init__(repr(result))
        self.result = result


async def _resolve_token(user_id: str) -> str:
    async with async_session_maker() as db:
        ident = await _vault.get(db, user_id, "notion")
    if ident is None or not ident.access_token:
        raise _NotionError(
            ConnectorToolError(message="No active Notion identity", retryable=False),
        )
    return ident.access_token


# ─── Id normalisation ────────────────────────────────────────────────

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_HEX32_RE = re.compile(r"\A[0-9a-fA-F]{32}\Z")


def _normalize_id(raw: Any) -> str:
    """Accept a bare id, a dashed UUID, or a pasted Notion URL.

    The agent is handed Notion URLs by users constantly, and Notion's
    API only accepts an id, so normalising here removes an entire class
    of `validation_error`.

    Two traps, both silent if you get them wrong:

      - **Strip the query string first.** A database URL carries
        `?v=<32 hex>` — the *view* id, which is id-shaped. Scanning the
        whole URL for the last hex run picks the view and queries the
        wrong object.
      - **Do not remove dashes globally before matching.** Notion's URL
        slug is the page title, and titles contain hex-looking words
        ("Cafe", "Facade", "Added"). Joining the slug to the id slides
        the 32-character window off by however many hex letters the
        title ended with. Split on `-` and require a whole segment.
    """
    if raw is None:
        return ""
    val = str(raw).strip()
    if not val:
        return ""
    val = val.split("?", 1)[0].split("#", 1)[0]

    m = _UUID_RE.search(val)
    if m:
        return m.group(0).lower()

    segment = val.rstrip("/").rsplit("/", 1)[-1]
    for part in reversed(segment.split("-")):
        if _HEX32_RE.match(part):
            h = part.lower()
            return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

    # Not id-shaped. Hand it to Notion unchanged rather than guessing —
    # its `validation_error` message names the problem better than we can.
    return val


def _clamp_page_size(raw: Any, default: int = 25) -> int:
    """Notion 400s on page_size > 100 rather than clamping it itself."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, min(value, _MAX_PAGE_SIZE))


# ─── Error mapping ───────────────────────────────────────────────────


def _error_body(resp: httpx.Response) -> dict:
    try:
        parsed = resp.json()
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _handle_notion_error(resp: httpx.Response) -> None:
    """Map a Notion HTTP response onto the ConnectorResult sum type.

    The interesting decision here is what NOT to return.

    **403 is never ConnectorScopeMissing, and neither is 404.** Notion
    has no OAuth scopes at all — this connector requests none and is
    granted none — so naming a "required scope" would be inventing a
    cause, and the remedy it implies (re-authorize with more scopes)
    does not exist in Notion's product. The two real causes are:

      - 403 `restricted_resource`: the *connection's capabilities* are
        too narrow (e.g. read-only when the tool wants to insert). Only
        the owner of the Notion integration can widen those, in the
        connection's settings — not the user, and not by reconnecting.
      - 404 `object_not_found`: far more often "the user never shared
        this page with the connection" than "this page does not exist".
        Notion returns 404 rather than 403 for unshared content on
        purpose, so we must not translate it into a permission verdict.

    Both therefore come back as ConnectorToolError carrying **Notion's
    own message** plus the one action that actually fixes it. An honest
    "here is what the provider said, here is where to look" beats a
    confident wrong diagnosis, which is what the agent would otherwise
    repeat to the user as fact.
    """
    if 200 <= resp.status_code < 300:
        return

    body = _error_body(resp)
    code = str(body.get("code") or "")
    message = str(body.get("message") or "").strip()
    detail = message or (resp.text or "")[:200]

    # `invalid_grant` arrives as a 400 but means the token is dead —
    # the one 4xx besides 401 that is genuinely a re-auth, not a bug in
    # the request we sent.
    if resp.status_code == 401 or code in ("unauthorized", "invalid_grant"):
        raise _NotionError(ConnectorReauthRequired(reauth_url=_REAUTH_URL))

    if resp.status_code == 429:
        retry_after = _DEFAULT_RETRY_AFTER_S
        try:
            retry_after = int(resp.headers.get("Retry-After", _DEFAULT_RETRY_AFTER_S))
        except (TypeError, ValueError):
            pass
        reason = (body.get("additional_data") or {}).get("rate_limit_reason")
        logger.info(
            "[notion] rate limited retry_after=%ss reason=%s", retry_after, reason,
        )
        raise _NotionError(ConnectorRateLimited(retry_after_s=max(retry_after, 1)))

    if resp.status_code == 403:
        raise _NotionError(ConnectorToolError(
            message=(
                f"Notion refused this request ({code or 'restricted_resource'}): "
                f"{detail}. This is a capability limit on the Notion connection "
                f"itself, not a missing permission the user can re-grant — the "
                f"integration's read/update/insert capabilities are set in its "
                f"Notion connection settings."
            ),
            retryable=False,
        ))

    if resp.status_code == 404:
        raise _NotionError(ConnectorToolError(
            message=(
                f"Notion could not find this object ({code or 'object_not_found'}): "
                f"{detail}. Notion answers 404 for content that exists but has not "
                f"been shared with this connection — ask the user to open the page "
                f"in Notion and add it from ⋯ → Connections."
            ),
            retryable=False,
        ))

    if resp.status_code == 409:
        # `conflict_error` is a save collision: another writer touched
        # the same object mid-transaction. Retrying is the documented fix.
        raise _NotionError(ConnectorToolError(
            message=f"Notion transaction conflict: {detail}",
            retryable=True,
        ))

    if resp.status_code >= 500:
        # Covers 500/502/503/504 and Notion's 529 `service_overload`.
        raise _NotionError(ConnectorProviderDown(
            provider_status_url="https://status.notion.so",
        ))

    raise _NotionError(ConnectorToolError(
        message=f"{resp.status_code} from Notion ({code or 'error'}): {detail}",
        retryable=False,
    ))


async def _notion_request(
    method: str,
    path: str,
    *,
    access_token: str,
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
) -> dict:
    """Authenticated request → parsed JSON. Raises `_NotionError` on any
    non-2xx with the right ConnectorResult variant attached."""
    client = await _get_notion_client()
    try:
        resp = await client.request(
            method,
            f"{NOTION_API}{path}",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Notion-Version": _NOTION_VERSION,
                "Accept": "application/json",
            },
            json=json_body,
            params=params,
        )
    except httpx.HTTPError as e:
        # Transport-level: unreachable, TLS, timeout. The health probe
        # flips the card to `down` on its next sweep.
        raise _NotionError(ConnectorProviderDown(
            provider_status_url="https://status.notion.so",
        )) from e

    _handle_notion_error(resp)
    if not resp.content:
        return {}
    try:
        parsed = resp.json()
    except ValueError:
        raise _NotionError(ConnectorToolError(
            message=f"Notion returned a non-JSON 2xx body: {resp.text[:200]}",
            retryable=False,
        )) from None
    return parsed if isinstance(parsed, dict) else {"results": parsed}


# ─── Object shaping ──────────────────────────────────────────────────


def _rich_text_to_plain(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    return "".join(
        str(it.get("plain_text") or "")
        for it in items
        if isinstance(it, dict)
    )


def _title_of(obj: dict) -> str:
    """Title of a page or a data source.

    They differ: a data source carries a top-level `title` rich-text
    array, while a page's title lives inside `properties` under whatever
    the user named the title column ("Name", "Task", "Aufgabe"…), found
    by its `type`, never by its key.
    """
    top = obj.get("title")
    if isinstance(top, list):
        return _rich_text_to_plain(top)
    for prop in (obj.get("properties") or {}).values():
        if isinstance(prop, dict) and prop.get("type") == "title":
            return _rich_text_to_plain(prop.get("title") or [])
    return ""


def _flatten_property(prop: dict) -> Any:
    """One Notion property value → a plain JSON scalar/list.

    Raw property values are deeply nested and mostly styling; handing
    them to the model verbatim burns its budget for no information.
    Unknown types fall through with their raw value rather than being
    dropped — a new Notion property type should degrade to noisy, not
    to invisible.
    """
    if not isinstance(prop, dict):
        return None
    ptype = prop.get("type")
    val = prop.get(ptype) if ptype else None

    if ptype in ("title", "rich_text"):
        return _rich_text_to_plain(val or [])
    if ptype in (
        "number", "checkbox", "url", "email", "phone_number",
        "created_time", "last_edited_time",
    ):
        return val
    if ptype in ("select", "status"):
        return (val or {}).get("name")
    if ptype == "multi_select":
        return [o.get("name") for o in (val or []) if isinstance(o, dict)]
    if ptype == "date":
        if not val:
            return None
        return {"start": val.get("start"), "end": val.get("end")}
    if ptype == "people":
        return [
            p.get("name") or p.get("id")
            for p in (val or []) if isinstance(p, dict)
        ]
    if ptype == "files":
        return [f.get("name") for f in (val or []) if isinstance(f, dict)]
    if ptype == "relation":
        return [r.get("id") for r in (val or []) if isinstance(r, dict)]
    if ptype == "formula":
        f = val or {}
        return f.get(f.get("type"))
    if ptype == "rollup":
        r = val or {}
        rtype = r.get("type")
        if rtype == "array":
            # Each array element is itself a property-value object.
            return [_flatten_property(x) for x in (r.get("array") or [])]
        return r.get(rtype)
    if ptype in ("created_by", "last_edited_by"):
        return (val or {}).get("name") or (val or {}).get("id")
    if ptype == "unique_id":
        u = val or {}
        prefix = u.get("prefix")
        return f"{prefix}-{u.get('number')}" if prefix else u.get("number")
    if ptype == "verification":
        return (val or {}).get("state")
    return val


def _flatten_properties(props: Any) -> tuple[dict, list[str]]:
    """Flatten every property, and report which ones Notion truncated.

    Notion caps relation / people / mention lists at 25 references per
    page read and sets `has_more` on the property. Silently returning 25
    of 60 related rows is how an agent confidently reports a wrong
    total, so the names of truncated properties ride along with the
    payload.
    """
    flat: dict[str, Any] = {}
    truncated: list[str] = []
    if not isinstance(props, dict):
        return flat, truncated
    for name, prop in props.items():
        flat[name] = _flatten_property(prop)
        if isinstance(prop, dict) and prop.get("has_more"):
            truncated.append(name)
    return flat, truncated


def _page_summary(obj: dict) -> dict:
    """Compact shape shared by search results and database rows."""
    flat, truncated = _flatten_properties(obj.get("properties"))
    row: dict[str, Any] = {
        "id": obj.get("id"),
        "object": obj.get("object"),
        "title": _title_of(obj),
        "url": obj.get("url"),
        "last_edited_time": obj.get("last_edited_time"),
        "properties": flat,
    }
    if truncated:
        row["properties_truncated"] = truncated
    return row


# ─── Block rendering ─────────────────────────────────────────────────

_BLOCK_PREFIX = {
    "heading_1": "# ",
    "heading_2": "## ",
    "heading_3": "### ",
    "heading_4": "#### ",
    "bulleted_list_item": "- ",
    "numbered_list_item": "1. ",
    "quote": "> ",
    "toggle": "- ",
    "callout": "> ",
}


def _render_block(block: dict, depth: int) -> str:
    """One block → one line of plain text, or "" to skip it."""
    btype = str(block.get("type") or "")
    payload = block.get(btype)
    payload = payload if isinstance(payload, dict) else {}
    indent = "  " * depth

    if btype == "divider":
        return f"{indent}---"
    if btype in ("child_page", "child_database"):
        label = "page" if btype == "child_page" else "database"
        return f"{indent}[{label}] {payload.get('title') or 'Untitled'}"
    if btype in ("bookmark", "embed", "link_preview"):
        return f"{indent}{payload.get('url') or ''}".rstrip()
    if btype == "equation":
        return f"{indent}{payload.get('expression') or ''}".rstrip()
    if btype in ("image", "video", "file", "pdf", "audio"):
        src = (payload.get("external") or {}).get("url") or (
            payload.get("file") or {}
        ).get("url") or ""
        caption = _rich_text_to_plain(payload.get("caption") or [])
        label = f"[{btype}]"
        # File URLs are short-lived signed links; keep the caption, which
        # is the only durable description, and note the source host only.
        return f"{indent}{label} {caption}".rstrip() if caption else f"{indent}{label} {src}".rstrip()
    if btype == "table_row":
        cells = payload.get("cells") or []
        return f"{indent}| " + " | ".join(_rich_text_to_plain(c) for c in cells) + " |"

    text = _rich_text_to_plain(payload.get("rich_text") or [])
    if btype == "code":
        lang = payload.get("language") or ""
        return f"{indent}```{lang}\n{text}\n{indent}```"
    if btype == "to_do":
        mark = "x" if payload.get("checked") else " "
        return f"{indent}- [{mark}] {text}"
    if not text:
        return ""
    return f"{indent}{_BLOCK_PREFIX.get(btype, '')}{text}"


async def _collect_blocks(
    block_id: str,
    *,
    access_token: str,
    state: dict,
    depth: int = 0,
) -> list[str]:
    """Depth-first walk of a block subtree, budgeted by request count."""
    lines: list[str] = []
    cursor: Optional[str] = None

    while True:
        if state["requests"] >= _MAX_BLOCK_REQUESTS:
            state["truncated"] = True
            return lines
        params: dict[str, Any] = {"page_size": _MAX_PAGE_SIZE}
        if cursor:
            params["start_cursor"] = cursor
        state["requests"] += 1
        data = await _notion_request(
            "GET", f"/blocks/{block_id}/children",
            access_token=access_token, params=params,
        )

        for block in (data.get("results") or []):
            if not isinstance(block, dict):
                continue
            rendered = _render_block(block, depth)
            if rendered:
                lines.append(rendered)
                state["chars"] += len(rendered) + 1
                if state["chars"] >= state["max_chars"]:
                    state["truncated"] = True
                    return lines
            # A child_page is a DIFFERENT page with its own id and its
            # own sharing. Inlining its body here would silently return
            # content the caller did not ask for and would blow the
            # request budget on the first page that has a few sub-pages.
            if (
                block.get("has_children")
                and block.get("type") != "child_page"
                and depth + 1 <= _MAX_BLOCK_DEPTH
            ):
                lines.extend(await _collect_blocks(
                    str(block.get("id") or ""),
                    access_token=access_token, state=state, depth=depth + 1,
                ))
                if state["truncated"]:
                    return lines

        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
        if not cursor:
            break

    return lines


# ─── Database → data source resolution ───────────────────────────────


async def _resolve_data_source_id(database_id: str, *, access_token: str) -> str:
    """A database id → the id of its single data source.

    Since API version 2025-09-03 a database holds one or more data
    sources and only a data source is queryable. Nearly every database
    has exactly one, so this is one extra GET. When there are several we
    refuse rather than picking `[0]` — an arbitrary pick silently
    queries the wrong table and returns a plausible, wrong answer. The
    error names the alternatives so the agent can retry with an explicit
    `data_source_id`.
    """
    db = await _notion_request(
        "GET", f"/databases/{database_id}", access_token=access_token,
    )
    sources = [s for s in (db.get("data_sources") or []) if isinstance(s, dict)]
    if not sources:
        raise _NotionError(ConnectorToolError(
            message=(
                f"Notion database {database_id} exposes no data sources, so it "
                f"cannot be queried."
            ),
            retryable=False,
        ))
    if len(sources) > 1:
        listed = ", ".join(
            f"{s.get('name') or 'Untitled'} ({s.get('id')})" for s in sources
        )
        raise _NotionError(ConnectorToolError(
            message=(
                f"This Notion database has {len(sources)} data sources; say which "
                f"one by passing data_source_id. Available: {listed}."
            ),
            retryable=False,
        ))
    return str(sources[0].get("id") or "")


async def _title_property_name(data_source_id: str, *, access_token: str) -> str:
    """The name of a data source's title column.

    Required to create a row: Notion keys `properties` by the column's
    display name, and the title column is only identifiable by its
    `type`. Defaults to "Name" if the schema somehow has no title
    column, which is Notion's own default for a new database.
    """
    schema = await _notion_request(
        "GET", f"/data_sources/{data_source_id}", access_token=access_token,
    )
    for name, spec in (schema.get("properties") or {}).items():
        if isinstance(spec, dict) and spec.get("type") == "title":
            return name
    return "Name"


def _paragraph_blocks(content: str) -> tuple[list[dict], bool]:
    """Plain text → Notion paragraph blocks, within Notion's ceilings.

    Two hard limits apply and both are silent 400s when crossed: a
    rich_text element may hold 2000 characters, and a `children` array
    may hold 100 elements. Returns `(blocks, truncated)`.
    """
    blocks: list[dict] = []
    for para in content.split("\n"):
        chunks = (
            [para[i:i + _MAX_RICH_TEXT_CHARS]
             for i in range(0, len(para), _MAX_RICH_TEXT_CHARS)]
            if para else [""]
        )
        for chunk in chunks:
            if len(blocks) >= _MAX_BLOCKS_PER_REQUEST:
                return blocks, True
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": (
                        [{"type": "text", "text": {"content": chunk}}]
                        if chunk else []
                    ),
                },
            })
    return blocks, False


# ─── OAuth: refresh + revoke ─────────────────────────────────────────
#
# Both endpoints authenticate the CLIENT, not the user: HTTP Basic with
# `base64(client_id:client_secret)` and a JSON body. Notion's own docs
# render the header as `Authorization: Basic "$CLIENT_ID:$CLIENT_SECRET"`
# — that is a documentation typo (the prose one paragraph up says the
# pair is base64-encoded); sending it literally yields 401.


def _basic_auth(client_id: str, client_secret: str) -> str:
    raw = f"{client_id}:{client_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _oauth_headers(client_id: str, client_secret: str) -> dict:
    return {
        "Authorization": _basic_auth(client_id, client_secret),
        "Notion-Version": _NOTION_VERSION,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def notion_refresh(refresh_token: str) -> RefreshResult:
    """Rotate a refresh_token for a fresh access_token.

    Notion returns NO `expires_in`, so `expires_at` stays None and the
    dispatcher's `_needs_refresh` never schedules a proactive refresh —
    in practice a dead token surfaces as a 401 on the next tool call and
    becomes ConnectorReauthRequired. This path is implemented correctly
    regardless so it works the day Notion starts expiring our tokens.

    The refresh_token IS rotated on every successful call; returning the
    new one is what keeps the chain alive.
    """
    app_cfg = await get_provider_app_async("notion")
    if app_cfg is None:
        raise RefreshFailed(
            "notion provider not registered — set credentials via "
            "/admin/oauth-apps or NOTION_OAUTH_CLIENT_ID env var"
        )
    client = await _get_notion_client()
    resp = await client.post(
        app_cfg.token_url,
        headers=_oauth_headers(app_cfg.client_id, app_cfg.client_secret),
        json={"grant_type": "refresh_token", "refresh_token": refresh_token},
    )
    if resp.status_code in (400, 401, 403):
        # `invalid_grant` here means the user removed the connection or
        # the token was already rotated — permanently unrecoverable.
        raise RefreshFailed(
            f"notion refresh returned {resp.status_code}: {resp.text[:200]}"
        )
    if resp.status_code >= 500:
        resp.raise_for_status()
    if not resp.is_success:
        raise RefreshFailed(f"notion refresh unexpected status {resp.status_code}")

    try:
        body = resp.json()
    except ValueError:
        raise RefreshFailed(
            f"notion refresh returned non-JSON body: {resp.text[:200]}"
        ) from None
    access_token = body.get("access_token")
    if not access_token:
        raise RefreshFailed(
            f"notion refresh returned no access_token (keys={sorted(body.keys())})"
        )
    return RefreshResult(
        access_token=access_token,
        refresh_token=body.get("refresh_token"),
        expires_at=None,
    )


async def notion_revoke(access_token: str) -> None:
    """Revoke the grant at Notion via `POST /v1/oauth/revoke`.

    Unlike Microsoft and LinkedIn, Notion does expose a real revocation
    endpoint, so this is not a no-op — disconnecting in Toup actually
    ends the connection's access rather than leaving a live token to age
    out. A 401/404 means the token is already dead, which is the outcome
    we wanted; anything else raises so `/oauth/disconnect` audits a
    `revocation_provider_failed` event.
    """
    app_cfg = await get_provider_app_async("notion")
    if app_cfg is None or not access_token:
        return
    client = await _get_notion_client()
    resp = await client.post(
        f"{NOTION_API}/oauth/revoke",
        headers=_oauth_headers(app_cfg.client_id, app_cfg.client_secret),
        json={"token": access_token},
    )
    if resp.is_success or resp.status_code in (401, 404):
        return
    raise RuntimeError(
        f"notion revoke returned {resp.status_code}: {resp.text[:200]}"
    )


# ─── Provider ────────────────────────────────────────────────────────


class NotionProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "notion"

    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ConnectorContext,
    ) -> ConnectorResult:
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
        except _NotionError as e:
            return e.result

        try:
            if tool_name == "notion__search":
                return await self._search(tool_input, access_token)
            if tool_name == "notion__get_page":
                return await self._get_page(tool_input, access_token)
            if tool_name == "notion__get_page_content":
                return await self._get_page_content(tool_input, access_token)
            if tool_name == "notion__query_database":
                return await self._query_database(tool_input, access_token)
            if tool_name == "notion__create_page":
                return await self._create_page(tool_input, access_token)
            return ConnectorToolError(
                message=f"unknown notion tool {tool_name!r}", retryable=False,
            )
        except _NotionError as e:
            return e.result

    # ── Tools ──

    async def _search(self, tool_input: dict, access_token: str) -> ConnectorResult:
        body: dict[str, Any] = {
            "page_size": _clamp_page_size(tool_input.get("page_size", 25)),
        }
        query = (tool_input.get("query") or "").strip()
        if query:
            body["query"] = query
        if tool_input.get("start_cursor"):
            body["start_cursor"] = tool_input["start_cursor"]

        filt = (tool_input.get("filter") or "any").strip().lower()
        if filt in ("page", "data_source"):
            body["filter"] = {"property": "object", "value": filt}
        elif filt == "database":
            # Users and older prompts say "database"; the searchable
            # object has been called a data source since 2025-09-03.
            body["filter"] = {"property": "object", "value": "data_source"}

        sort = (tool_input.get("sort") or "").strip().lower()
        if sort == "relevance":
            body["sort"] = {"property": "relevance"}
        elif sort in ("last_edited_desc", "last_edited_asc"):
            body["sort"] = {
                "timestamp": "last_edited_time",
                "direction": "descending" if sort.endswith("desc") else "ascending",
            }

        data = await _notion_request(
            "POST", "/search", access_token=access_token, json_body=body,
        )
        results = []
        for obj in (data.get("results") or []):
            if not isinstance(obj, dict):
                continue
            row = {
                "id": obj.get("id"),
                "object": obj.get("object"),
                "title": _title_of(obj),
                "url": obj.get("url"),
                "last_edited_time": obj.get("last_edited_time"),
            }
            if obj.get("object") == "data_source":
                # Hand back the containing database id too: users refer
                # to that one, and it is what notion__query_database and
                # notion__create_page take as a parent.
                row["database_id"] = (
                    (obj.get("database_parent") or {}).get("database_id")
                )
            results.append(row)

        return ConnectorOk(content=json.dumps({
            "results": results,
            "has_more": bool(data.get("has_more")),
            "next_cursor": data.get("next_cursor"),
        }))

    async def _get_page(self, tool_input: dict, access_token: str) -> ConnectorResult:
        page_id = _normalize_id(tool_input.get("page_id"))
        if not page_id:
            return ConnectorToolError(message="page_id required", retryable=False)
        page = await _notion_request(
            "GET", f"/pages/{page_id}", access_token=access_token,
        )
        flat, truncated = _flatten_properties(page.get("properties"))
        payload: dict[str, Any] = {
            "id": page.get("id"),
            "title": _title_of(page),
            "url": page.get("url"),
            "created_time": page.get("created_time"),
            "last_edited_time": page.get("last_edited_time"),
            "in_trash": page.get("in_trash"),
            "parent": page.get("parent"),
            "properties": flat,
        }
        if truncated:
            payload["properties_truncated"] = truncated
            payload["note"] = (
                "Notion returns at most 25 references per relation/people "
                "property; the listed properties have more."
            )
        return ConnectorOk(content=json.dumps(payload))

    async def _get_page_content(
        self, tool_input: dict, access_token: str,
    ) -> ConnectorResult:
        page_id = _normalize_id(tool_input.get("page_id"))
        if not page_id:
            return ConnectorToolError(message="page_id required", retryable=False)
        try:
            max_chars = int(tool_input.get("max_chars", 40_000))
        except (TypeError, ValueError):
            max_chars = 40_000
        max_chars = max(500, min(max_chars, 100_000))

        state = {"requests": 0, "chars": 0, "max_chars": max_chars, "truncated": False}
        lines = await _collect_blocks(
            page_id, access_token=access_token, state=state,
        )
        return ConnectorOk(content=json.dumps({
            "page_id": page_id,
            "content": "\n".join(lines),
            "truncated": bool(state["truncated"]),
            "blocks_requests": state["requests"],
        }))

    async def _query_database(
        self, tool_input: dict, access_token: str,
    ) -> ConnectorResult:
        data_source_id = _normalize_id(tool_input.get("data_source_id"))
        if not data_source_id:
            database_id = _normalize_id(tool_input.get("database_id"))
            if not database_id:
                return ConnectorToolError(
                    message="database_id or data_source_id required",
                    retryable=False,
                )
            data_source_id = await _resolve_data_source_id(
                database_id, access_token=access_token,
            )

        body: dict[str, Any] = {
            "page_size": _clamp_page_size(tool_input.get("page_size", 25)),
        }
        if isinstance(tool_input.get("filter"), dict) and tool_input["filter"]:
            body["filter"] = tool_input["filter"]
        if isinstance(tool_input.get("sorts"), list) and tool_input["sorts"]:
            body["sorts"] = tool_input["sorts"]
        if tool_input.get("start_cursor"):
            body["start_cursor"] = tool_input["start_cursor"]

        data = await _notion_request(
            "POST", f"/data_sources/{data_source_id}/query",
            access_token=access_token, json_body=body,
        )
        rows = [
            _page_summary(obj)
            for obj in (data.get("results") or [])
            if isinstance(obj, dict)
        ]
        payload: dict[str, Any] = {
            "data_source_id": data_source_id,
            "results": rows,
            "has_more": bool(data.get("has_more")),
            "next_cursor": data.get("next_cursor"),
        }
        # Notion caps a single query at 10k rows and says so here rather
        # than by omitting `has_more`.
        status = data.get("request_status") or {}
        if isinstance(status, dict) and status.get("type") == "incomplete":
            payload["incomplete_reason"] = status.get("incomplete_reason")
        return ConnectorOk(content=json.dumps(payload))

    async def _create_page(
        self, tool_input: dict, access_token: str,
    ) -> ConnectorResult:
        title = (tool_input.get("title") or "").strip()
        if not title:
            return ConnectorToolError(message="title required", retryable=False)

        parent_page = _normalize_id(tool_input.get("parent_page_id"))
        parent_db = _normalize_id(tool_input.get("parent_database_id"))
        parent_ds = _normalize_id(tool_input.get("parent_data_source_id"))
        given = [p for p in (parent_page, parent_db, parent_ds) if p]
        if len(given) != 1:
            return ConnectorToolError(
                message=(
                    "exactly one of parent_page_id, parent_database_id or "
                    "parent_data_source_id is required"
                ),
                retryable=False,
            )

        title_value = {"title": [{"type": "text", "text": {"content": title[:_MAX_RICH_TEXT_CHARS]}}]}

        if parent_page:
            # Under a page parent `title` is the only legal property and
            # its key is literally "title" — there is no user-named
            # column to look up.
            parent: dict[str, Any] = {"page_id": parent_page}
            properties: dict[str, Any] = {"title": title_value}
        else:
            data_source_id = parent_ds or await _resolve_data_source_id(
                parent_db, access_token=access_token,
            )
            title_key = await _title_property_name(
                data_source_id, access_token=access_token,
            )
            parent = {"data_source_id": data_source_id}
            properties = {title_key: title_value}
            extra = tool_input.get("properties")
            if isinstance(extra, dict):
                # Caller-supplied columns win over nothing but the title
                # key, which they may legitimately want to override with
                # a richer rich_text value.
                properties.update(extra)

        body: dict[str, Any] = {"parent": parent, "properties": properties}
        content = tool_input.get("content") or ""
        content_truncated = False
        if content:
            children, content_truncated = _paragraph_blocks(str(content))
            if children:
                body["children"] = children

        page = await _notion_request(
            "POST", "/pages", access_token=access_token, json_body=body,
        )
        payload: dict[str, Any] = {
            "id": page.get("id"),
            "url": page.get("url"),
            "title": title,
            "created": True,
        }
        if content_truncated:
            payload["content_truncated"] = True
            payload["note"] = (
                f"Only the first {_MAX_BLOCKS_PER_REQUEST} paragraphs were written; "
                f"Notion accepts at most that many blocks per request."
            )
        return ConnectorOk(content=json.dumps(payload))

    # ── Lifecycle ──

    async def revoke(self, user_id, access_token, refresh_token=None):
        await notion_revoke(access_token)

    async def refresh(self, refresh_token: str) -> RefreshResult:
        return await notion_refresh(refresh_token)

    async def health_probe(self, ctx: ConnectorContext) -> HealthResult:
        """Cheapest authenticated round-trip Notion offers: a one-result
        search. Matches `health.probe: notion__search` in the manifest.

        An empty result set is still healthy — it means the user has not
        shared anything yet, which is a setup state, not an outage.
        """
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
            await _notion_request(
                "POST", "/search",
                access_token=access_token, json_body={"page_size": 1},
            )
            return HealthResult(ok=True)
        except _NotionError as e:
            return HealthResult(ok=False, detail=repr(e.result))
        except Exception as e:
            return HealthResult(ok=False, detail=f"{type(e).__name__}: {e}")
