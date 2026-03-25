"""
Netflix Stream API — start/stop streams + serve HLS segments.

Endpoints:
  POST /api/netflix-stream/start  → start a stream, returns stream_id + HLS URL
  POST /api/netflix-stream/stop   → stop a stream
  GET  /api/netflix-stream/{stream_id}/{filename} → serve HLS segments
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Header
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/netflix-stream", tags=["Netflix Stream"])


class StartStreamRequest(BaseModel):
    netflix_url: str
    email: str
    password: str
    profile: str = ""


class StartStreamResponse(BaseModel):
    stream_id: str
    hls_url: str


@router.post("/start", response_model=StartStreamResponse)
async def start_stream(
    body: StartStreamRequest,
    agent_key: Optional[str] = Query(None, alias="agent_key"),
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
):
    """Start a Netflix streaming session."""
    from app.config import settings

    # Auth: agent key
    key = agent_key or x_agent_key
    if not key or key != settings.agent_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    from app.services.netflix_stream import start_netflix_stream

    stream_id = str(uuid.uuid4())[:8]

    try:
        hls_dir = await start_netflix_stream(
            stream_id=stream_id,
            netflix_url=body.netflix_url,
            email=body.email,
            password=body.password,
            profile=body.profile,
        )
    except Exception as e:
        logger.exception("[NF-STREAM] Failed to start")
        raise HTTPException(status_code=500, detail=str(e))

    hls_url = f"/api/netflix-stream/{stream_id}/stream.m3u8"
    return StartStreamResponse(stream_id=stream_id, hls_url=hls_url)


@router.post("/stop")
async def stop_stream(
    stream_id: str,
    agent_key: Optional[str] = Query(None, alias="agent_key"),
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
):
    """Stop a Netflix streaming session."""
    from app.config import settings

    key = agent_key or x_agent_key
    if not key or key != settings.agent_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    from app.services.netflix_stream import stop_netflix_stream
    await stop_netflix_stream(stream_id)
    return {"status": "stopped"}


@router.get("/{stream_id}/{filename}")
async def serve_hls(stream_id: str, filename: str):
    """Serve HLS playlist and segments."""
    from app.services.netflix_stream import get_stream_hls_dir

    hls_dir = get_stream_hls_dir(stream_id)
    if not hls_dir:
        raise HTTPException(status_code=404, detail="Stream not found")

    file_path = hls_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Segment not found")

    # Set correct content types
    if filename.endswith(".m3u8"):
        return Response(
            content=file_path.read_bytes(),
            media_type="application/vnd.apple.mpegurl",
            headers={
                "Cache-Control": "no-cache, no-store",
                "Access-Control-Allow-Origin": "*",
            },
        )
    elif filename.endswith(".ts"):
        return Response(
            content=file_path.read_bytes(),
            media_type="video/MP2T",
            headers={
                "Cache-Control": "max-age=3600",
                "Access-Control-Allow-Origin": "*",
            },
        )
    else:
        raise HTTPException(status_code=400, detail="Unknown file type")


class EpisodeLookupRequest(BaseModel):
    show_id: str
    season: int
    episode: int
    email: str
    password: str


@router.post("/episode-lookup")
async def episode_lookup(
    body: EpisodeLookupRequest,
    agent_key: Optional[str] = Query(None, alias="agent_key"),
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
):
    """Find the Netflix watch ID for a specific episode using Playwright."""
    from app.config import settings

    key = agent_key or x_agent_key
    if not key or key != settings.agent_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        episode_id = await _get_episode_id(
            body.show_id, body.season, body.episode,
            body.email, body.password,
        )
        return {"episode_id": episode_id}
    except Exception as e:
        logger.exception("[NF] Episode lookup failed")
        raise HTTPException(status_code=500, detail=str(e))


async def _get_episode_id(show_id: str, season: int, episode: int, email: str, password: str) -> str:
    """Use Playwright to navigate Netflix show page and extract episode watch ID."""
    import asyncio
    import os
    import shutil
    import re

    try:
        from patchright.async_api import async_playwright
    except ImportError:
        from playwright.async_api import async_playwright

    chrome = shutil.which("google-chrome-stable") or "/opt/google/chrome/google-chrome"
    data_dir = "/tmp/nf-chrome-shared"
    os.makedirs(data_dir, exist_ok=True)

    # Find a free Xvfb display
    display = None
    for num in range(130, 145):
        if not os.path.exists(f"/tmp/.X{num}-lock"):
            display = f":{num}"
            break
    if not display:
        raise RuntimeError("No free display")

    xvfb = await asyncio.create_subprocess_exec(
        "Xvfb", display, "-screen", "0", "1280x720x24", "-ac",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    await asyncio.sleep(0.5)

    try:
        os.environ["DISPLAY"] = display
        pw = await async_playwright().start()
        browser = await pw.chromium.launch_persistent_context(
            user_data_dir=data_dir,
            executable_path=chrome,
            headless=False,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True,
        )

        page = browser.pages[0] if browser.pages else await browser.new_page()

        # Login if needed
        await page.goto(f"https://www.netflix.com/title/{show_id}", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3000)

        if "/login" in page.url:
            logger.info("[NF-EPISODE] Login required")
            email_input = page.locator('input[name="userLoginId"], input[type="email"]').first
            await email_input.fill(email)
            await page.locator('button:has-text("Continue"), button[type="submit"]').first.click()
            await page.wait_for_timeout(3000)
            try:
                pw_input = page.locator('input[type="password"]').first
                await pw_input.fill(password, timeout=5000)
                await page.locator('button:has-text("Sign In"), button[type="submit"]').first.click()
                await page.wait_for_timeout(5000)
            except Exception:
                pass

            # Profile picker
            try:
                await page.locator('.profile-link, .profile-icon, [data-profile-guid]').first.click(timeout=5000)
                await page.wait_for_timeout(3000)
            except Exception:
                pass

            # Navigate to show page
            await page.goto(f"https://www.netflix.com/title/{show_id}", wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(3000)

        # Select the correct season
        logger.info("[NF-EPISODE] On show page, selecting season %d", season)
        try:
            # Click season dropdown
            season_dropdown = page.locator('[data-uia="dropdown-toggle"], .episodeSelector, button:has-text("Season")').first
            await season_dropdown.click(timeout=5000)
            await page.wait_for_timeout(1000)

            # Click the specific season
            await page.locator(f'text="Season {season}"').first.click(timeout=5000)
            await page.wait_for_timeout(2000)
        except Exception as e:
            logger.info("[NF-EPISODE] Season selection: %s", e)

        # Extract episode links from the page
        content = await page.content()

        # Netflix episode URLs: /watch/EPISODE_ID
        episode_ids = re.findall(r'/watch/(\d+)', content)
        # Also look for episode data attributes
        episode_data = re.findall(r'data-episode-id="(\d+)"', content)
        all_ids = episode_ids + episode_data

        logger.info("[NF-EPISODE] Found %d episode IDs on page", len(all_ids))

        # The episodes are listed in order on the page
        # Filter unique IDs preserving order
        seen = set()
        unique_ids = []
        for eid in all_ids:
            if eid not in seen and eid != show_id:
                seen.add(eid)
                unique_ids.append(eid)

        if episode <= len(unique_ids):
            target_id = unique_ids[episode - 1]
            logger.info("[NF-EPISODE] S%dE%d → %s", season, episode, target_id)
        elif unique_ids:
            target_id = unique_ids[0]
            logger.info("[NF-EPISODE] Episode %d not found, using first: %s", episode, target_id)
        else:
            target_id = show_id
            logger.warning("[NF-EPISODE] No episodes found, using show ID")

        await browser.close()
        await pw.stop()
        return target_id

    finally:
        xvfb.terminate()
        try:
            await xvfb.wait()
        except Exception:
            pass
