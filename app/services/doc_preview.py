"""
Document preview renderer — DOCX/PPTX → PDF via LibreOffice headless.

Why LibreOffice:
    mammoth-HTML is a lossy text extraction — tab stops, page margins,
    numbering, kerning, footers, and many Word features don't survive.
    Every major SaaS with doc previews (Slack, Dropbox, Google Drive,
    Notion, Confluence) uses LibreOffice-to-PDF for faithful rendering.

Flow:
    1. Preview endpoint calls `render_to_pdf(storage_key, mime_type)`.
    2. We check the storage backend for a cached `{key}.preview.pdf`.
    3. If missing, we invoke `libreoffice --headless --convert-to pdf`
       in a dedicated temp dir, read the resulting PDF, and write it
       back to the backend at the cache key.
    4. Return the PDF bytes.

Latency:
    - First conversion: ~3-8s (LibreOffice cold start).
    - Cached conversions: subsecond (disk read).
    - Concurrent requests for the same file share the cached result.

Each conversion uses its own --user-installation dir so multiple
concurrent LibreOffice processes don't collide on config locking.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from typing import Optional

from app.services.file_storage import get_storage_backend

logger = logging.getLogger(__name__)

# Sentinel so we only look up the binary path once per process.
_SOFFICE_PATH: Optional[str] = None

# Per-key asyncio locks — prevent duplicate work when several WS clients
# request the same preview simultaneously (common on page reload).
_CONVERSION_LOCKS: dict[str, asyncio.Lock] = {}


def _cache_key_for(storage_key: str) -> str:
    """Derived key for the converted-PDF cache entry.

    We append `.preview.pdf` so the cache lives beside the source file
    within the same storage backend, inheriting its lifecycle (e.g. a
    future S3 backend's lifecycle rules apply to both).
    """
    return f"{storage_key}.preview.pdf"


def _find_soffice() -> Optional[str]:
    global _SOFFICE_PATH
    if _SOFFICE_PATH is not None:
        return _SOFFICE_PATH or None
    for candidate in ("libreoffice", "soffice", "/usr/bin/libreoffice", "/usr/bin/soffice"):
        found = shutil.which(candidate) if not candidate.startswith("/") else (candidate if os.path.isfile(candidate) else None)
        if found:
            _SOFFICE_PATH = found
            return found
    _SOFFICE_PATH = ""
    return None


async def _run_conversion(input_path: str, out_dir: str, user_dir: str, timeout: float = 60.0) -> Optional[str]:
    """Run `libreoffice --headless --convert-to pdf`. Returns path to the
    generated PDF, or None on failure. Uses a private user-installation
    dir so parallel invocations don't fight over ~/.config/libreoffice."""
    soffice = _find_soffice()
    if not soffice:
        raise RuntimeError(
            "LibreOffice binary not found. The agent container should have "
            "libreoffice-core + libreoffice-writer + libreoffice-impress "
            "installed. Rebuild the agent image or apt-install inside the "
            "running container."
        )

    proc = await asyncio.create_subprocess_exec(
        soffice,
        f"-env:UserInstallation=file://{user_dir}",
        "--headless",
        "--nologo",
        "--nodefault",
        "--nofirststartwizard",
        "--convert-to",
        "pdf",
        "--outdir",
        out_dir,
        input_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        logger.error("LibreOffice timed out (%ss) on %s", timeout, input_path)
        return None

    if proc.returncode != 0:
        logger.error("LibreOffice failed (rc=%s) on %s: %s",
                     proc.returncode, input_path, stderr.decode(errors="replace")[:500])
        return None

    # Output filename: {stem}.pdf in out_dir.
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(out_dir, f"{stem}.pdf")
    if not os.path.isfile(out_path):
        logger.error("LibreOffice reported success but output missing: %s", out_path)
        return None
    return out_path


async def render_to_pdf(storage_key: str) -> Optional[bytes]:
    """Return PDF bytes for the given DOCX/PPTX storage key. Cached.

    Returns None on conversion failure (caller should fall back to a
    download-only preview card). LibreOffice binary missing raises
    RuntimeError — surfaces as 501 in the preview endpoint.
    """
    backend = get_storage_backend()
    cache_key = _cache_key_for(storage_key)

    # Fast path: cached.
    if backend.exists(cache_key):
        with backend.open(cache_key) as f:
            return f.read()

    # Dedup concurrent conversions of the same file.
    lock = _CONVERSION_LOCKS.setdefault(storage_key, asyncio.Lock())
    async with lock:
        # Re-check after acquiring lock — another task may have just finished.
        if backend.exists(cache_key):
            with backend.open(cache_key) as f:
                return f.read()

        if not backend.exists(storage_key):
            logger.warning("Preview requested for missing source: %s", storage_key)
            return None

        src_path = backend.path(storage_key)
        with tempfile.TemporaryDirectory(prefix="soffice_convert_") as work:
            # Private LibreOffice user profile per conversion — avoids the
            # "Another LibreOffice is running" error under concurrency.
            user_dir = os.path.join(work, "uprofile")
            os.makedirs(user_dir, exist_ok=True)
            out_dir = os.path.join(work, "out")
            os.makedirs(out_dir, exist_ok=True)

            out_path = await _run_conversion(src_path, out_dir, user_dir)
            if not out_path:
                return None

            with open(out_path, "rb") as f:
                data = f.read()

        # Persist to cache under the same backend.
        await backend.put(cache_key, data)
        return data
