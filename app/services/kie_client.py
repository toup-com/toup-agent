"""Kie.ai (Nano Banana) image client — PLATFORM-side.

Holds the one shared platform Kie key (settings.kie_api_key) and drives Kie's
async job API for the highest-quality natural image generation and editing:

    generate/edit -> createTask -> poll recordInfo -> fetch result bytes

Nano Banana Pro (`nano-banana-pro`) is a unified generate+edit model: pass an
`image_input` URL to edit, omit it to generate. Source photos for editing are
first uploaded to Kie's file host (kie_upload_base) to obtain a public URL, then
referenced by URL. All calls use a browser User-Agent — Kie fronts its API with
Cloudflare, which 1010-blocks the default python/httpx UA on the upload host.

Only the platform (llm_proxy Kie route) imports this; agents never hold the key.
Every function RAISES KieError on failure so the caller can fall back to OpenAI.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Cloudflare on Kie's hosts rejects the default httpx UA (error 1010).
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# Nano Banana Pro aspect ratios (from Kie docs). Used to map OpenAI-style sizes
# and to preserve a source photo's framing on edit.
_ASPECTS: list[tuple[str, float]] = [
    ("1:1", 1.0), ("2:3", 2 / 3), ("3:2", 3 / 2), ("3:4", 3 / 4), ("4:3", 4 / 3),
    ("4:5", 4 / 5), ("5:4", 5 / 4), ("9:16", 9 / 16), ("16:9", 16 / 9), ("21:9", 21 / 9),
]

# OpenAI-style size string -> Kie aspect ratio (so the generate_image tool's
# existing size enum keeps working transparently).
_SIZE_TO_ASPECT = {
    "1024x1024": "1:1", "1024x1536": "2:3", "1536x1024": "3:2",
}


class KieError(RuntimeError):
    """Any Kie failure — network, task fail, timeout, moderation. Caller falls back."""

    def __init__(self, message: str, *, moderation: bool = False):
        super().__init__(message)
        self.moderation = moderation


@dataclass
class KieResult:
    image_bytes: bytes
    mime: str
    credits_consumed: float          # Kie credits (for our charge conversion)
    model: str


def _aspect_from_size(size: Optional[str]) -> str:
    return _SIZE_TO_ASPECT.get((size or "").strip().lower(), "1:1")


def _nearest_aspect(width: int, height: int) -> str:
    if not width or not height:
        return "1:1"
    ratio = width / height
    return min(_ASPECTS, key=lambda a: abs(a[1] - ratio))[0]


def _headers(json_body: bool = True) -> dict[str, str]:
    h = {"User-Agent": _UA, "Accept": "application/json",
         "Authorization": f"Bearer {settings.kie_api_key}"}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


async def _upload_source(client: httpx.AsyncClient, image_bytes: bytes, mime: str) -> str:
    """Upload source bytes to Kie's file host; return a public downloadUrl."""
    b64 = base64.b64encode(image_bytes).decode()
    ext = "png" if "png" in mime else ("jpg" if ("jpeg" in mime or "jpg" in mime) else "png")
    body = {
        "base64Data": f"data:{mime};base64,{b64}",
        "uploadPath": "images/toup-edit",
        "fileName": f"src.{ext}",
    }
    url = f"{settings.kie_upload_base.rstrip('/')}/api/file-base64-upload"
    r = await client.post(url, json=body, headers=_headers())
    if r.status_code >= 400:
        raise KieError(f"kie upload HTTP {r.status_code}: {r.text[:200]}")
    data = (r.json() or {}).get("data") or {}
    dl = data.get("downloadUrl") or data.get("url") or data.get("fileUrl")
    if not dl:
        raise KieError(f"kie upload returned no downloadUrl: {r.text[:200]}")
    return dl


async def _create_task(client: httpx.AsyncClient, model: str, inp: dict[str, Any]) -> str:
    url = f"{settings.kie_api_base.rstrip('/')}/api/v1/jobs/createTask"
    r = await client.post(url, json={"model": model, "input": inp}, headers=_headers())
    if r.status_code >= 400:
        raise KieError(f"kie createTask HTTP {r.status_code}: {r.text[:200]}")
    body = r.json() or {}
    if body.get("code") != 200:
        raise KieError(f"kie createTask code {body.get('code')}: {body.get('msg')}")
    task_id = (body.get("data") or {}).get("taskId")
    if not task_id:
        raise KieError(f"kie createTask returned no taskId: {r.text[:200]}")
    return task_id


def _looks_like_moderation(msg: str) -> bool:
    m = (msg or "").lower()
    return any(k in m for k in (
        "policy", "safety", "moderat", "nsfw", "prohibited", "not allowed",
        "content violat", "sensitive", "blocked",
    ))


async def _poll(client: httpx.AsyncClient, task_id: str) -> tuple[str, float]:
    """Poll recordInfo until success/fail/timeout; return (result_url, credits)."""
    url = f"{settings.kie_api_base.rstrip('/')}/api/v1/jobs/recordInfo"
    deadline = asyncio.get_event_loop().time() + float(settings.kie_timeout_s)
    interval = float(settings.kie_poll_interval_s)
    while asyncio.get_event_loop().time() < deadline:
        r = await client.get(url, params={"taskId": task_id}, headers=_headers())
        if r.status_code >= 400:
            raise KieError(f"kie recordInfo HTTP {r.status_code}: {r.text[:200]}")
        d = (r.json() or {}).get("data") or {}
        state = d.get("state")
        if state == "success":
            credits = float(d.get("creditsConsumed") or 0.0)
            try:
                result = json.loads(d.get("resultJson") or "{}")
            except (ValueError, TypeError):
                raise KieError(f"kie resultJson unparseable: {str(d.get('resultJson'))[:200]}")
            urls = result.get("resultUrls") or []
            if not urls:
                raise KieError("kie success but no resultUrls")
            return urls[0], credits
        if state == "fail":
            msg = d.get("failMsg") or d.get("failCode") or "unknown"
            raise KieError(f"kie task failed: {msg}", moderation=_looks_like_moderation(str(msg)))
        await asyncio.sleep(interval)
    raise KieError(f"kie task timed out after {settings.kie_timeout_s}s (task {task_id})")


async def _fetch_bytes(client: httpx.AsyncClient, result_url: str) -> bytes:
    # Result URLs are public/presigned; fetch WITHOUT the Kie auth header.
    r = await client.get(result_url, headers={"User-Agent": _UA}, follow_redirects=True)
    if r.status_code >= 400 or not r.content:
        raise KieError(f"kie result fetch HTTP {r.status_code} ({len(r.content)}B)")
    return r.content


async def _run(model: str, inp: dict[str, Any]) -> KieResult:
    if not settings.kie_api_key:
        raise KieError("kie_api_key not configured")
    timeout = httpx.Timeout(float(settings.kie_timeout_s), connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        task_id = await _create_task(client, model, inp)
        result_url, credits = await _poll(client, task_id)
        img = await _fetch_bytes(client, result_url)
    mime = "image/png" if settings.kie_output_format == "png" else "image/jpeg"
    return KieResult(image_bytes=img, mime=mime, credits_consumed=credits, model=model)


async def generate(prompt: str, size: Optional[str] = None) -> KieResult:
    """Text-to-image with Nano Banana Pro."""
    inp = {
        "prompt": prompt,
        "image_size": settings.kie_image_size,
        "aspect_ratio": _aspect_from_size(size),
        "output_format": settings.kie_output_format,
    }
    return await _run(settings.kie_image_model, inp)


async def edit(prompt: str, image_bytes: bytes, mime: str = "image/png") -> KieResult:
    """Edit a user photo with Nano Banana Pro, preserving its framing."""
    aspect = "1:1"
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as im:
            aspect = _nearest_aspect(im.width, im.height)
    except Exception:
        logger.debug("kie edit: could not read source dims, defaulting aspect 1:1", exc_info=True)
    timeout = httpx.Timeout(float(settings.kie_timeout_s), connect=15.0)
    if not settings.kie_api_key:
        raise KieError("kie_api_key not configured")
    async with httpx.AsyncClient(timeout=timeout) as client:
        src_url = await _upload_source(client, image_bytes, mime)
        inp = {
            "prompt": prompt,
            "image_input": [src_url],
            "aspect_ratio": aspect,
            "image_size": settings.kie_image_size,
            "output_format": settings.kie_output_format,
        }
        task_id = await _create_task(client, settings.kie_image_model, inp)
        result_url, credits = await _poll(client, task_id)
        img = await _fetch_bytes(client, result_url)
    out_mime = "image/png" if settings.kie_output_format == "png" else "image/jpeg"
    return KieResult(image_bytes=img, mime=out_mime,
                     credits_consumed=credits, model=settings.kie_image_model)
