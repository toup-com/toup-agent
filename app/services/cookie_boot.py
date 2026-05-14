"""Cookie boot decoder for yt-dlp.

If ``YT_DLP_COOKIES_B64`` env var (or ``settings.yt_dlp_cookies_b64``) is
set, decode it to ``/tmp/yt-cookies.txt`` and point ``YT_DLP_COOKIES_PATH``
at it so ``app.api.media_proxy._extract_audio`` picks it up.

R4 (Phase 1 design doc): **never crash boot on cookie misconfiguration**.
Corrupt base64, write-permission error, anything — log a warning and
return cleanly. The proxy still works; extraction just runs without
cookies.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_COOKIE_PATH = "/tmp/yt-cookies.txt"


def install_cookies_from_env(b64_value: str | None, target_path: str = DEFAULT_COOKIE_PATH) -> bool:
    """Decode a base64 cookies.txt blob to disk and point the env at it.

    Returns True iff the file was successfully written and the env var
    was set. Returns False (with a logged warning) on any failure.
    Never raises.
    """
    if not b64_value or not b64_value.strip():
        return False

    raw = b64_value.strip()
    try:
        decoded = base64.b64decode(raw, validate=True)
    except Exception as e:
        logger.warning(
            "[cookie_boot] base64 decode failed (%s); skipping cookie install. "
            "Set a valid Netscape cookies.txt blob in YT_DLP_COOKIES_B64 if you "
            "want yt-dlp to use cookies. Boot continues without cookies.",
            type(e).__name__,
        )
        return False

    # Sanity check: cookies.txt files are ASCII text. If decoded bytes
    # are obviously binary garbage, refuse — saves us from writing a
    # corrupt file that yt-dlp would later choke on with a confusing
    # error.
    try:
        sample = decoded[:512].decode("utf-8")
    except UnicodeDecodeError:
        logger.warning(
            "[cookie_boot] decoded bytes are not UTF-8; skipping cookie install"
        )
        return False
    if not sample.strip():
        logger.warning("[cookie_boot] decoded blob is empty; skipping cookie install")
        return False

    try:
        path = Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(decoded)
        # Cookies are sensitive — readable only by the running process.
        os.chmod(path, 0o600)
    except OSError as e:
        logger.warning(
            "[cookie_boot] failed to write cookies file at %s: %s — boot "
            "continues without cookies",
            target_path, e,
        )
        return False

    os.environ["YT_DLP_COOKIES_PATH"] = str(path)
    logger.info("[cookie_boot] cookies installed at %s (%d bytes)", path, len(decoded))
    return True
