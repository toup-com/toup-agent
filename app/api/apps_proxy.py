"""
Apps proxy routes — platform forwards apps/jobs requests to the user's VPS agent.

App data (built apps, build jobs) lives on the user's VPS.
The platform is a passthrough proxy only.
"""

import logging
import re
from typing import Optional, Tuple
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db, AgentConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/apps", tags=["Apps Proxy"])


# ── Agent proxy helpers ─────────────────────────────────────

async def _get_agent(user_id: str, db: AsyncSession) -> Optional[Tuple[str, str]]:
    result = await db.execute(
        select(AgentConfig.agent_url, AgentConfig.agent_api_key)
        .where(
            AgentConfig.user_id == user_id,
            AgentConfig.deploy_status == "active",
        )
    )
    row = result.first()
    if row and row.agent_url and row.agent_api_key:
        return (row.agent_url, row.agent_api_key)
    return None


async def _proxy(
    agent_url: str, agent_api_key: str, path: str,
    method: str = "GET", body: Optional[dict] = None,
    timeout: float = 30.0,
):
    url = f"{agent_url}/api/apps/{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            headers = {"X-Agent-Key": agent_api_key}
            if method == "GET":
                resp = await client.get(url, headers=headers)
            elif method == "POST":
                resp = await client.post(url, headers=headers, json=body or {})
            elif method == "DELETE":
                resp = await client.delete(url, headers=headers)
            else:
                return None
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        logger.warning("Apps proxy %s %s failed: %s", method, url, e)
        raise HTTPException(502, "Agent unreachable")


def _require(info):
    if not info:
        raise HTTPException(503, "Agent not deployed or not reachable.")
    return info


def _rewrite_app_urls(data):
    """Replace raw VPS web_url with platform proxy path."""
    def _fix(app: dict):
        if isinstance(app, dict) and app.get("id"):
            app["web_url"] = f"/api/apps/{app['id']}/preview/"
        return app

    if isinstance(data, list):
        return [_fix(a) for a in data]
    elif isinstance(data, dict):
        return _fix(data)
    return data


# ── Server info ─────────────────────────────────────────────

@router.get("/server")
async def get_server_info(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get VPS system info (CPU, RAM, disk, uptime, running apps)."""
    agent_info = await _get_agent(current_user.id, db)
    if not agent_info:
        return JSONResponse(content={"status": "offline"}, status_code=200)
    agent_url, key = agent_info
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{agent_url}/agent/system",
                headers={"X-Agent-Key": key},
            )
            if resp.status_code == 200:
                data = resp.json()
                data["status"] = "online"
                data["ip"] = agent_url.replace("http://", "").replace("https://", "").split(":")[0]
                return JSONResponse(content=data)
    except Exception as e:
        logger.warning("Server info proxy failed: %s", e)
    return JSONResponse(content={"status": "offline"})


# ── Agent capabilities ──────────────────────────────────────

@router.get("/capabilities")
async def get_capabilities(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Fetch all loaded tools and skills from the VPS agent."""
    agent_info = await _get_agent(current_user.id, db)
    if not agent_info:
        return JSONResponse(content={"core_tools": [], "skills": [], "total_tools": 0})
    agent_url, key = agent_info
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{agent_url}/agent/capabilities",
                headers={"X-Agent-Key": key},
            )
            if resp.status_code == 200:
                return JSONResponse(content=resp.json())
    except Exception as e:
        logger.warning("Capabilities proxy failed: %s", e)
    return JSONResponse(content={"core_tools": [], "skills": [], "total_tools": 0})


# ── App endpoints ───────────────────────────────────────────

@router.get("/")
async def list_apps(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    url = f"{agent_url}/api/apps/"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers={"X-Agent-Key": key})
            data = _rewrite_app_urls(resp.json())
            return JSONResponse(content=data, status_code=resp.status_code)
    except Exception as e:
        logger.warning("Apps proxy list failed: %s", e)
        raise HTTPException(502, "Agent unreachable")


@router.get("/jobs/")
async def list_jobs(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, "jobs/")


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, f"jobs/{job_id}")


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, f"jobs/{job_id}", method="DELETE")


# ── Web Preview Proxy ──────────────────────────────────────
# Reverse-proxies the Expo web dev server through toup.ai so the
# mobile app can load it over HTTPS without direct VPS port access.

async def _get_app_web_port(app_id: str, agent_url: str, agent_api_key: str) -> int:
    """Fetch the app's web_port from the VPS agent API."""
    url = f"{agent_url}/api/apps/{app_id}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers={"X-Agent-Key": agent_api_key})
        if resp.status_code != 200:
            raise HTTPException(404, "App not found")
        data = resp.json()
        web_port = data.get("web_port")
        if not web_port:
            raise HTTPException(503, "App web server not running")
        return web_port


async def _get_user_from_token(token: str, db: AsyncSession):
    """Validate JWT from query param for preview auth."""
    from app.services.auth_service import decode_access_token
    try:
        user_id = decode_access_token(token)
        if not user_id:
            return None
        return type("User", (), {"id": user_id})()
    except Exception:
        return None


@router.get("/{app_id}/preview/{path:path}")
@router.get("/{app_id}/preview")
async def preview_proxy(
    app_id: str, request: Request,
    path: str = "",
    token: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Reverse-proxy the Expo web dev server for in-app preview.

    Auth via ?token=JWT (SFSafariViewController can't send Bearer headers).
    Injects <base href> into HTML so sub-resources (JS bundles, etc.)
    route back through this proxy instead of hitting toup.ai root.
    """
    # Try Bearer header first, then query param token, then cookie
    user = None
    try:
        user = await get_current_user(request, db)
    except Exception:
        pass
    if not user and token:
        user = await _get_user_from_token(token, db)
    if not user:
        cookie_token = request.cookies.get("preview_token")
        if cookie_token:
            user = await _get_user_from_token(cookie_token, db)
    if not user:
        raise HTTPException(401, "Not authenticated")

    agent_info = await _get_agent(user.id, db)
    agent_url, key = _require(agent_info)

    from urllib.parse import urlparse
    vps_host = urlparse(agent_url).hostname

    web_port = await _get_app_web_port(app_id, agent_url, key)
    target = f"http://{vps_host}:{web_port}/{path}"

    # Forward query string (except our token param)
    params = {k: v for k, v in request.query_params.items() if k != "token"}
    if params:
        target += f"?{urlencode(params)}"

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(target)
            content_type = resp.headers.get("content-type", "text/html")

            body = resp.content

            # For HTML responses (the initial page), inject <base href> so
            # relative URLs like /index.ts.bundle resolve through the proxy
            # path instead of toup.ai root.
            # Also rewrite script src to include ?token= so sub-resource
            # requests authenticate without relying on cookies (WebView
            # may not send cookies for cross-origin sub-requests).
            if "text/html" in content_type:
                base_href = f"/api/apps/{app_id}/preview/"
                base_tag = f'<base href="{base_href}">'
                # Inject agent bridge globals so the app can connect
                # to the user's real agent via WebSocket
                agent_globals = (
                    f'<script>'
                    f'window.__TOUP_AUTH_TOKEN="{token or ""}";'
                    f'window.__TOUP_APP_ID="{app_id}";'
                    f'window.__TOUP_WS_URL="wss://toup.ai/api/ws/chat";'
                    f'</script>'
                )
                # Emoji font CSS — iOS WKWebView doesn't auto-fallback to emoji fonts.
                # react-native-web sets font-family to "-apple-system,...,sans-serif"
                # which lacks emoji fallback, causing ? boxes on mobile.
                emoji_css = (
                    '<style id="emoji-fix">'
                    '* { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
                    'Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", '
                    '"Noto Color Emoji" !important; }'
                    '</style>'
                )
                html = body.decode("utf-8", errors="replace")
                html = html.replace("<head>", f"<head>\n{base_tag}\n{agent_globals}\n{emoji_css}", 1)
                # Rewrite absolute src="/..." to relative so <base href>
                # routes them through the preview proxy path.
                # Also inject ?token= so bundle requests are authenticated
                # (WebView may not send cookies for sub-resource requests).
                def _rewrite_src(m):
                    src = m.group(1)
                    # Strip leading / to make relative (so <base href> applies)
                    if src.startswith("/"):
                        src = src[1:]
                    # Inject auth token
                    if token:
                        sep = "&" if "?" in src else "?"
                        src = f"{src}{sep}token={token}"
                    return f'src="{src}"'
                html = re.sub(r'src="(/[^"]*)"', _rewrite_src, html)
                body = html.encode("utf-8")

            # Ensure charset=utf-8 is in Content-Type for text responses.
            # iOS WKWebView uses the HTTP header (not <meta charset>) to decode,
            # and defaults to ASCII when charset is missing — corrupting emoji bytes.
            resp_content_type = content_type
            if "text/html" in content_type and "charset" not in content_type:
                resp_content_type = "text/html; charset=utf-8"
            elif "javascript" in content_type and "charset" not in content_type:
                resp_content_type = content_type + "; charset=utf-8"

            response = StreamingResponse(
                iter([body]),
                status_code=resp.status_code,
                media_type=resp_content_type,
            )

            # Set auth cookie so sub-resource requests (JS bundles, etc.)
            # are authenticated without needing ?token= on every URL.
            if token and "text/html" in content_type:
                response.set_cookie(
                    key="preview_token",
                    value=token,
                    max_age=3600,
                    httponly=True,
                    samesite="none",
                    secure=True,
                )

            return response
    except Exception as e:
        logger.warning("Preview proxy failed: %s → %s", target, e)
        raise HTTPException(502, "App preview unreachable")


@router.get("/{app_id}")
async def get_app(app_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    url = f"{agent_url}/api/apps/{app_id}"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers={"X-Agent-Key": key})
            data = _rewrite_app_urls(resp.json())
            return JSONResponse(content=data, status_code=resp.status_code)
    except Exception as e:
        logger.warning("Apps proxy get failed: %s", e)
        raise HTTPException(502, "Agent unreachable")


@router.post("/{app_id}/start")
async def start_app(app_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, f"{app_id}/start", method="POST", timeout=60.0)


@router.post("/{app_id}/stop")
async def stop_app(app_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, f"{app_id}/stop", method="POST")


@router.post("/{app_id}/publish-web")
async def publish_web(app_id: str, request: Request, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    body = None
    try:
        body = await request.json()
    except Exception:
        pass
    return await _proxy(agent_url, key, f"{app_id}/publish-web", method="POST", body=body, timeout=120.0)


@router.post("/{app_id}/push-github")
async def push_github(app_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, f"{app_id}/push-github", method="POST", timeout=60.0)


@router.delete("/{app_id}")
async def delete_app(app_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, app_id, method="DELETE")
