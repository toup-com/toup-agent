"""
Admin OAuth provider-app credentials.

Lets the platform owner paste Google / GitHub / Notion / etc. OAuth
client_id + client_secret through an admin UI instead of editing
Railway env vars and redeploying — the same UX Base44, Zapier, and
similar platforms expose to their owners.

Endpoints (all admin-gated):

  GET    /api/admin/provider-apps
         List every known provider with status (configured | env_only |
         missing) and a redacted client_id. Never returns the secret.

  PUT    /api/admin/provider-apps/{name}
         Upsert client_id + client_secret. Secret is Fernet-encrypted
         via `credential_crypto.encrypt_str` before persistence.

  DELETE /api/admin/provider-apps/{name}
         Drop the DB row. Env-var registration (if any) silently
         resumes precedence on the next /connect.

Audit: each PUT/DELETE writes a ConnectorEvent row keyed on the
operator's user_id so we can trace who rotated the secret. The event
metadata records `client_id` (not the secret) for forensic context.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import delete as sa_delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_admin
from app.config import settings
from app.db import get_db
from app.db.models import ConnectorEvent, ProviderAppCredential
from app.services.credential_crypto import encrypt_str
from app.services.provider_apps import _TEMPLATES, _apps as _env_apps

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/provider-apps", tags=["Admin — Provider apps"])


# Friendly per-provider metadata that drives the admin UI walkthrough.
# Keep this aligned with `_TEMPLATES` in provider_apps.py — both indexed
# by the same provider name.
_PROVIDER_META: dict[str, dict] = {
    "google": {
        "label": "Google",
        "description": (
            "Google OAuth client. One client backs Gmail, Calendar, and "
            "Drive — distinct manifests, distinct scopes, one credential "
            "pair."
        ),
        "console_url": "https://console.cloud.google.com/apis/credentials",
        "redirect_uri_hint": "Add this exact redirect URI to your Google OAuth client.",
    },
    "github": {
        "label": "GitHub",
        "description": (
            "GitHub OAuth App. Used by the GitHub connector for repo / "
            "issues / PR access."
        ),
        "console_url": "https://github.com/settings/developers",
        "redirect_uri_hint": "Add this as the Authorization callback URL on the OAuth App.",
    },
}


class ProviderAppOut(BaseModel):
    """One row for the admin list. NEVER carries the secret."""
    name: str
    label: str
    description: str
    console_url: str
    redirect_uri: str
    redirect_uri_hint: str
    status: str  # "configured_db" | "configured_env" | "missing"
    client_id: Optional[str] = None  # full id (not a secret)
    client_secret_set: bool = False
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class ProviderAppListResponse(BaseModel):
    providers: list[ProviderAppOut]


class ProviderAppUpsertRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=512)
    client_secret: str = Field(min_length=1, max_length=512)


class ProviderAppUpsertResponse(BaseModel):
    name: str
    status: str
    updated_at: datetime


class ProviderAppDeleteResponse(BaseModel):
    name: str
    status: str  # "missing" | "configured_env" after delete


# ─── Helpers ─────────────────────────────────────────────────────────


async def _resolve_status(
    db: AsyncSession, name: str,
) -> tuple[str, Optional[str], bool, Optional[datetime], Optional[str]]:
    """Return (status, client_id, secret_set, updated_at, updated_by)
    for one provider name. Status is one of:

      - "configured_db"  — DB row exists, takes precedence
      - "configured_env" — env-var-only (admin can override via DB)
      - "missing"        — no creds anywhere
    """
    row = (await db.execute(
        select(ProviderAppCredential).where(
            ProviderAppCredential.name == name,
        ),
    )).scalar_one_or_none()
    if row is not None:
        return ("configured_db", row.client_id, True, row.updated_at, row.updated_by)
    env_cfg = _env_apps.get(name)
    if env_cfg is not None:
        return ("configured_env", env_cfg.client_id, True, None, None)
    return ("missing", None, False, None, None)


def _build_out(name: str, *, status_: str, client_id: Optional[str],
               secret_set: bool, updated_at: Optional[datetime],
               updated_by: Optional[str]) -> ProviderAppOut:
    meta = _PROVIDER_META.get(name, {
        "label": name.title(),
        "description": f"OAuth provider — {name}",
        "console_url": "",
        "redirect_uri_hint": "",
    })
    return ProviderAppOut(
        name=name,
        label=meta["label"],
        description=meta["description"],
        console_url=meta["console_url"],
        redirect_uri=settings.oauth_callback_url,
        redirect_uri_hint=meta["redirect_uri_hint"],
        status=status_,
        client_id=client_id,
        client_secret_set=secret_set,
        updated_at=updated_at,
        updated_by=updated_by,
    )


def _audit_metadata(*, client_id: str, action: str) -> str:
    return json.dumps({
        "action": action,
        "client_id_prefix": client_id[:12],  # never log the full id either
    })


# ─── List ────────────────────────────────────────────────────────────


@router.get("", response_model=ProviderAppListResponse)
async def list_provider_apps(
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Return every provider known to the platform (anything in
    `_TEMPLATES`) with its current credential status. The frontend
    uses this to decide which tiles are connectable and to surface
    a "Configure" CTA on the rest."""
    out: list[ProviderAppOut] = []
    for name in sorted(_TEMPLATES.keys()):
        if name == "stub_provider_app":
            continue  # internal test fixture, never surface to UI
        status_, client_id, secret_set, updated_at, updated_by = await _resolve_status(db, name)
        out.append(_build_out(
            name,
            status_=status_,
            client_id=client_id,
            secret_set=secret_set,
            updated_at=updated_at,
            updated_by=updated_by,
        ))
    return ProviderAppListResponse(providers=out)


# ─── Upsert ──────────────────────────────────────────────────────────


@router.put("/{name}", response_model=ProviderAppUpsertResponse)
async def upsert_provider_app(
    name: str,
    req: ProviderAppUpsertRequest,
    admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create or update DB-backed credentials for one provider. The
    secret is Fernet-encrypted before insertion. Idempotent — repeated
    PUTs with new values overwrite cleanly."""
    if name not in _TEMPLATES or name == "stub_provider_app":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown provider {name!r}",
        )

    encrypted = encrypt_str(req.client_secret)
    now = datetime.utcnow()

    existing = (await db.execute(
        select(ProviderAppCredential).where(
            ProviderAppCredential.name == name,
        ),
    )).scalar_one_or_none()

    if existing is not None:
        existing.client_id = req.client_id
        existing.client_secret_enc = encrypted
        existing.updated_at = now
        existing.updated_by = str(admin.id)
        action = "updated"
    else:
        db.add(ProviderAppCredential(
            name=name,
            client_id=req.client_id,
            client_secret_enc=encrypted,
            updated_at=now,
            updated_by=str(admin.id),
        ))
        action = "created"

    # Audit row — keyed on the admin user_id so the connector_events
    # log shows who rotated the secret. connector_id reuses the
    # provider name; event_type is namespaced so it's distinguishable
    # from per-user lifecycle events.
    db.add(ConnectorEvent(
        user_id=str(admin.id),
        connector_id=f"provider_app:{name}",
        event_type=f"provider_app_{action}",
        metadata_json=_audit_metadata(client_id=req.client_id, action=action),
        occurred_at=now,
    ))
    await db.commit()

    logger.info(
        "[admin/provider-apps] %s %r by user=%s",
        action, name, admin.id,
    )
    return ProviderAppUpsertResponse(
        name=name,
        status="configured_db",
        updated_at=now,
    )


# ─── Delete ──────────────────────────────────────────────────────────


@router.delete("/{name}", response_model=ProviderAppDeleteResponse)
async def delete_provider_app(
    name: str,
    admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Drop the DB-backed credential row. If env-var registration was
    populated at boot, that takes over silently on the next /connect."""
    if name not in _TEMPLATES or name == "stub_provider_app":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown provider {name!r}",
        )
    existing = (await db.execute(
        select(ProviderAppCredential).where(
            ProviderAppCredential.name == name,
        ),
    )).scalar_one_or_none()
    if existing is None:
        # Idempotent — nothing to do, return current status.
        env_status = "configured_env" if _env_apps.get(name) else "missing"
        return ProviderAppDeleteResponse(name=name, status=env_status)

    captured_client_id = existing.client_id
    await db.execute(
        sa_delete(ProviderAppCredential).where(
            ProviderAppCredential.name == name,
        ),
    )
    db.add(ConnectorEvent(
        user_id=str(admin.id),
        connector_id=f"provider_app:{name}",
        event_type="provider_app_deleted",
        metadata_json=_audit_metadata(client_id=captured_client_id, action="deleted"),
        occurred_at=datetime.utcnow(),
    ))
    await db.commit()

    env_status = "configured_env" if _env_apps.get(name) else "missing"
    logger.info(
        "[admin/provider-apps] deleted %r by user=%s (post=%s)",
        name, admin.id, env_status,
    )
    return ProviderAppDeleteResponse(name=name, status=env_status)
