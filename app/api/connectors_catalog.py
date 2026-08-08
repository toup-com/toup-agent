"""
T2b — Connector catalog endpoint.

Returns the full manifest catalogue overlaid with per-user identity
status. Replaces the frontend's hardcoded mock list. Drives:

  - `/agent/integrations` page tile rendering (T2a card per entry)
  - Detail modal scope review (T2b)
  - Channel-access toggles + event log filters (T2c)

Scope: read-only. Connect/disconnect remain on the `/api/oauth/*`
surface — this endpoint just describes what's available + what's
already connected.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.config import settings
from app.db import get_db
from app.db.models import ConnectorIdentity
from app.services import connector_vault as vault
from app.services.connector_registry import get_registry
from app.services.provider_apps import list_configured_async

router = APIRouter(prefix="/connectors", tags=["Connectors"])


class CatalogToolEntry(BaseModel):
    name: str
    description: str
    mutates: bool
    elevation: bool
    pii: Optional[str] = None
    rate_limit: Optional[str] = None


class CatalogScopeEntry(BaseModel):
    """Per-scope plain-language description for the OAuth scope review
    UI (T2b §5.3). T3+ providers populate `description` via the
    manifest's `scope_descriptions` map; defaults to the raw scope."""

    scope: str
    description: str
    required: bool


class CatalogConnectorEntry(BaseModel):
    id: str
    name: str
    short_description: str
    icon: Optional[str] = None
    category: str
    docs_url: Optional[str] = None
    status: Optional[str] = None  # 'active' | 'reauth_required' | 'revoked' | None
    last_used_at: Optional[str] = None
    scopes: list[CatalogScopeEntry]
    tools: list[CatalogToolEntry]
    # True when this connector cannot currently be connected. Both the
    # web page and the mobile Connectors screen already render this as
    # a dimmed, non-actionable tile, so it is the right channel for
    # "advertised but unreachable" — see `unavailable_reason` for why.
    coming_soon: bool = False
    # Machine-readable cause when `coming_soon` is set by the platform
    # rather than by the connector's own roadmap status. Currently only
    # 'credentials_missing'. Nothing renders this yet; it exists so the
    # catalogue stays honest to an operator debugging a dimmed tile,
    # instead of claiming a finished connector is "coming soon".
    unavailable_reason: Optional[str] = None
    # Per-identity read-only flag. Defaults to False; only meaningful
    # when status is 'active' or 'reauth_required'.
    read_only: bool = False


class CatalogResponse(BaseModel):
    connectors: list[CatalogConnectorEntry]


# Plain-language scope descriptions. Manifest YAML can override per
# provider via `scope_descriptions: { "<scope>": "..." }`. Falls back
# to the raw scope value here if absent.
DEFAULT_SCOPE_DESCRIPTIONS: dict[str, str] = {
    # Gmail
    "https://www.googleapis.com/auth/gmail.readonly": "Read your Gmail messages and threads.",
    "https://www.googleapis.com/auth/gmail.send": "Send mail on your behalf.",
    "https://www.googleapis.com/auth/gmail.modify": "Read, label, and modify your mail (but not delete).",
    # Calendar
    "https://www.googleapis.com/auth/calendar.readonly": "List events and check availability.",
    "https://www.googleapis.com/auth/calendar.events": "Create and edit events on your calendars.",
    # Drive
    "https://www.googleapis.com/auth/drive.readonly": "View files in your Drive.",
    "https://www.googleapis.com/auth/drive.file": "Manage files the agent creates in Drive.",
    # GitHub
    "repo": "Read and write to your repositories.",
    "read:org": "Read your organization memberships.",
    "user:email": "Read your primary email address.",
}


@router.get("/catalog", response_model=CatalogResponse)
async def get_connector_catalog(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CatalogResponse:
    """Per-user connector catalogue.

    Each entry's `status` is the user's `ConnectorIdentity.status`
    (None when never connected). Tool listings come from the manifest;
    scope descriptions resolve via manifest override → built-in
    fallback → raw scope string.
    """
    user_id = str(current_user.id)
    registry = get_registry()

    # Which provider apps actually have OAuth credentials. A connector
    # whose provider app is missing them cannot complete a connect —
    # `/api/oauth/connect` answers 503 `credentials_missing` — so
    # advertising a live Connect button for it is a button that is
    # guaranteed to fail. LinkedIn and Outlook did exactly that on
    # every client for months.
    configured = await list_configured_async()

    # One indexed query — same shape as `vault.list_active` but
    # includes non-active rows (we want `reauth_required` shown too).
    from sqlalchemy import select
    rows = (
        await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == user_id,
            )
        )
    ).scalars().all()
    status_by_id: dict[str, ConnectorIdentity] = {
        r.connector_id: r for r in rows
    }

    out: list[CatalogConnectorEntry] = []
    for entry in registry.list_all():
        manifest = entry.manifest
        # Skip experimental connectors in production unless the user
        # is explicitly on a non-prod env — matches the registry's
        # boot-time gate.
        if (
            manifest.status == "experimental"
            and settings.environment == "production"
        ):
            continue

        ident = status_by_id.get(manifest.id)
        # An ALREADY-CONNECTED identity keeps its live tile even if the
        # provider app's credentials were later removed: the user has a
        # working grant, the tools still dispatch, and dimming their
        # connected account would read as data loss. The gate is about
        # not offering a connect that cannot succeed.
        # `configured is None` means the credential lookup failed — gate
        # nothing rather than greying out the whole page on a DB blip.
        unconfigured = (
            configured is not None
            and manifest.oauth.provider_app not in configured
            and ident is None
        )
        scope_descs = getattr(manifest, "scope_descriptions", None) or {}
        all_scopes = [(s, True) for s in manifest.oauth.scopes] + [
            (s, False) for s in (manifest.oauth.scopes_optional or [])
        ]
        catalog_scopes = [
            CatalogScopeEntry(
                scope=s,
                description=scope_descs.get(s)
                or DEFAULT_SCOPE_DESCRIPTIONS.get(s, s),
                required=required,
            )
            for s, required in all_scopes
        ]

        out.append(
            CatalogConnectorEntry(
                id=manifest.id,
                name=manifest.name,
                short_description=manifest.short_description,
                icon=manifest.icon,
                category=manifest.category,
                docs_url=manifest.docs_url,
                status=ident.status if ident else None,
                last_used_at=(
                    ident.last_used_at.isoformat() + "Z"
                    if ident and ident.last_used_at
                    else None
                ),
                read_only=bool(getattr(ident, "read_only", False)) if ident else False,
                scopes=catalog_scopes,
                tools=[
                    CatalogToolEntry(
                        name=t.name,
                        description=t.description,
                        mutates=t.mutates,
                        elevation=t.elevation,
                        pii=t.pii,
                        rate_limit=t.rate_limit,
                    )
                    for t in manifest.tools
                ],
                coming_soon=unconfigured,
                unavailable_reason=(
                    "credentials_missing" if unconfigured else None
                ),
            )
        )

    return CatalogResponse(connectors=out)
