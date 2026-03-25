"""Streaming credentials API — store/retrieve/delete per-user channel credentials."""

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Query
from pydantic import BaseModel
from sqlalchemy import select, and_, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.api.auth import get_current_user

router = APIRouter(prefix="/streaming", tags=["Streaming"])


class CredentialIn(BaseModel):
    channel: str
    email: str
    password: str


class CredentialOut(BaseModel):
    channel: str
    email: str
    has_password: bool
    connected_at: Optional[str] = None


@router.get("/credentials")
async def list_credentials(
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all connected streaming channels (passwords redacted)."""
    from app.db.models import StreamingCredential

    result = await db.execute(
        select(StreamingCredential).where(StreamingCredential.user_id == user.id)
    )
    creds = result.scalars().all()
    return [
        CredentialOut(
            channel=c.channel,
            email=c.email,
            has_password=bool(c.password),
            connected_at=c.created_at.isoformat() if c.created_at else None,
        )
        for c in creds
    ]


@router.put("/credentials")
async def upsert_credential(
    body: CredentialIn,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Connect or update a streaming channel."""
    from app.db.models import StreamingCredential

    result = await db.execute(
        select(StreamingCredential).where(
            and_(
                StreamingCredential.user_id == user.id,
                StreamingCredential.channel == body.channel,
            )
        )
    )
    cred = result.scalar_one_or_none()

    if cred:
        cred.email = body.email.strip()
        cred.password = body.password
        cred.updated_at = datetime.utcnow()
    else:
        cred = StreamingCredential(
            id=str(uuid.uuid4()),
            user_id=user.id,
            channel=body.channel,
            email=body.email.strip(),
            password=body.password,
        )
        db.add(cred)

    await db.commit()
    return {"status": "connected", "channel": body.channel}


@router.delete("/credentials/{channel}")
async def disconnect_credential(
    channel: str,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Disconnect a streaming channel."""
    from app.db.models import StreamingCredential

    await db.execute(
        delete(StreamingCredential).where(
            and_(
                StreamingCredential.user_id == user.id,
                StreamingCredential.channel == channel,
            )
        )
    )
    await db.commit()
    return {"status": "disconnected", "channel": channel}


@router.get("/credentials/internal/{channel}")
async def get_credential_internal(
    channel: str,
    user_id: str = Query(...),
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
    db: AsyncSession = Depends(get_db),
):
    """Internal endpoint for the agent to fetch credentials (with password).
    Auth: X-Agent-Key must match the user's agent config."""
    if not x_agent_key:
        raise HTTPException(status_code=401, detail="X-Agent-Key required")

    from app.db.models import AgentConfig, StreamingCredential

    # Verify agent key belongs to this user
    cfg = await db.execute(
        select(AgentConfig).where(
            and_(AgentConfig.user_id == user_id, AgentConfig.agent_api_key == x_agent_key)
        )
    )
    if not cfg.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Invalid agent key")

    result = await db.execute(
        select(StreamingCredential).where(
            and_(
                StreamingCredential.user_id == user_id,
                StreamingCredential.channel == channel,
            )
        )
    )
    cred = result.scalar_one_or_none()
    if not cred:
        raise HTTPException(status_code=404, detail=f"No {channel} credentials found")

    return {"channel": cred.channel, "email": cred.email, "password": cred.password}


@router.get("/netflix-search")
async def netflix_search(
    q: str = Query(...),
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
):
    """Search for a Netflix title ID. Tries Google → DuckDuckGo → fallback."""
    import re
    import httpx as _httpx

    if not x_agent_key:
        raise HTTPException(status_code=401, detail="X-Agent-Key required")

    ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/137.0.0.0 Safari/537.36"
    netflix_id = None
    title = q

    import logging
    _log = logging.getLogger(__name__)

    async with _httpx.AsyncClient(timeout=12, follow_redirects=True, headers={"User-Agent": ua}) as client:
        # Try 1: DuckDuckGo (most reliable, no captcha)
        try:
            resp = await client.get("https://html.duckduckgo.com/html/", params={"q": f"{q} site:netflix.com"})
            _log.info("[NF-SEARCH] DDG status=%d len=%d", resp.status_code, len(resp.text))
            ids = re.findall(r'netflix\.com/(?:title|watch)/(\d+)', resp.text)
            if ids:
                netflix_id = ids[0]
                tm = re.search(r'class="result__a"[^>]*>(?:Watch\s+)?([^<|]+?)(?:\s*[|\-–])\s*Netflix', resp.text)
                if tm:
                    import html as _html
                    title = _html.unescape(tm.group(1).strip())
                _log.info("[NF-SEARCH] DDG found: id=%s title=%s", netflix_id, title)
        except Exception as e:
            _log.warning("[NF-SEARCH] DDG failed: %s", e)

        # Try 2: Google
        if not netflix_id:
            try:
                resp = await client.get("https://www.google.com/search", params={"q": f"{q} site:netflix.com", "num": 5})
                _log.info("[NF-SEARCH] Google status=%d len=%d", resp.status_code, len(resp.text))
                ids = re.findall(r'netflix\.com/(?:title|watch)/(\d+)', resp.text)
                if ids:
                    netflix_id = ids[0]
                    tm = re.search(r'(?:Watch |)([^<]+?)(?:\s*[|–-])\s*Netflix', resp.text)
                    if tm:
                        import html as _html
                        title = _html.unescape(tm.group(1).strip())
                    _log.info("[NF-SEARCH] Google found: id=%s title=%s", netflix_id, title)
            except Exception as e:
                _log.warning("[NF-SEARCH] Google failed: %s", e)

        # Try 3: Bing
        if not netflix_id:
            try:
                resp = await client.get("https://www.bing.com/search", params={"q": f"{q} site:netflix.com"})
                _log.info("[NF-SEARCH] Bing status=%d len=%d", resp.status_code, len(resp.text))
                ids = re.findall(r'netflix\.com/(?:title|watch)/(\d+)', resp.text)
                if ids:
                    netflix_id = ids[0]
                    _log.info("[NF-SEARCH] Bing found: id=%s", netflix_id)
            except Exception as e:
                _log.warning("[NF-SEARCH] Bing failed: %s", e)

    _log.info("[NF-SEARCH] Final: q=%s id=%s title=%s", q, netflix_id, title)
    return {"netflix_id": netflix_id, "title": title}
