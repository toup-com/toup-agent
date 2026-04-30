"""Helpers for the Baileys / neonize WhatsApp session.

Two concerns live here, isolated from the channel adapter so they can
be unit-tested without spinning up neonize:

* Where the session lives on disk — a per-tenant SQLite file under
  ``/data/whatsapp/``. Survives container restarts (the `/data` volume
  is bind-mounted per-tenant by the bridge), so a Meta retry that
  arrives during the boot window is reconciled correctly.
* Rendering the pairing string neonize emits into a PNG data URL the
  Settings UI modal can drop into an ``<img>`` tag.

Anything that needs to import neonize itself lives in
``whatsapp_baileys.py``. Both modules import from ``whatsapp_helpers``
for ``redact_phone`` so logs never carry a clear E.164.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Default per-tenant session store. Containers mount /data as a
# persistent volume; placing the SQLite file under whatsapp/ keeps it
# isolated from other agent state. Override via env for tests / dev.
DEFAULT_STORE_DIR = "/data/whatsapp"
DEFAULT_STORE_FILENAME = "store.db"


def store_dir() -> Path:
    """Resolve the directory holding ``store.db``. Honours
    ``WHATSAPP_BAILEYS_STORE_DIR`` for dev / tests; defaults to
    ``/data/whatsapp``. Does NOT create the directory — the caller does
    that exactly once at session start so failures surface where the
    user is waiting.
    """
    return Path(os.environ.get("WHATSAPP_BAILEYS_STORE_DIR", DEFAULT_STORE_DIR))


def store_path() -> Path:
    """Full path to the SQLite session store."""
    return store_dir() / DEFAULT_STORE_FILENAME


def ensure_store_dir() -> Path:
    """Create the store directory if missing (chmod 700 — only the
    container's UID needs access). Returns the resolved path.
    """
    p = store_dir()
    p.mkdir(parents=True, exist_ok=True, mode=0o700)
    return p


def session_file_exists() -> bool:
    """``True`` if a previously-paired session is on disk. Used to
    decide whether ``start()`` should resume vs request a fresh QR.
    """
    return store_path().is_file() and store_path().stat().st_size > 0


def clear_session_files() -> None:
    """Remove the session store. Called on logout / forced relink.

    Path-safety: only deletes files inside ``store_dir()``. Symlink
    components are rejected (defensive against a misconfigured volume
    being a symlink to somewhere sensitive).
    """
    base = store_dir().resolve()
    if not base.exists():
        return
    for p in base.iterdir():
        try:
            real = p.resolve(strict=False)
            if not str(real).startswith(str(base)):
                logger.warning(
                    "whatsapp.baileys.clear_session.skipped_outside_dir path=%s",
                    p,
                )
                continue
            if p.is_file():
                p.unlink()
        except Exception as exc:
            logger.warning(
                "whatsapp.baileys.clear_session.unlink_failed path=%s err=%s",
                p, exc,
            )


def render_qr_png_data_url(qr_string: str, box_size: int = 8) -> Optional[str]:
    """Render the pairing string to a base64 PNG ``data:`` URL.

    Returns ``None`` if the ``qrcode`` library isn't installed (e.g. in
    a dev shell) — callers fall back to handing the raw string to a
    terminal-mode renderer.

    ``box_size`` controls the QR module pixel size; 8 yields ~280×280
    which is sharp on retina without bloating the response payload
    (~3–4 KB base64).
    """
    if not qr_string:
        return None
    try:
        import qrcode  # type: ignore
    except ImportError:
        logger.warning(
            "whatsapp.baileys.qr_render.qrcode_not_installed — "
            "pip install qrcode[pil]"
        )
        return None
    try:
        img = qrcode.make(qr_string, box_size=box_size, border=2)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        logger.exception("whatsapp.baileys.qr_render.failed")
        return None


def jid_to_e164(jid: str) -> str:
    """Best-effort conversion of a WhatsApp JID to an E.164 string.

    Handles the two canonical user JID forms emitted by neonize /
    whatsmeow:
      * ``"15555551234@s.whatsapp.net"``
      * ``"15555551234:42@s.whatsapp.net"`` (device-suffixed)

    Returns ``""`` for anything else (group JIDs, status broadcasts,
    LIDs without a server-side mapping). The caller decides what to do
    with empty results — usually drop the message.
    """
    if not jid or "@" not in jid:
        return ""
    user_part, server = jid.split("@", 1)
    if server not in ("s.whatsapp.net", "c.us"):
        return ""
    digits = user_part.split(":", 1)[0]
    if not digits.isdigit():
        return ""
    return f"+{digits}"


def e164_to_jid(e164: str) -> str:
    """Inverse of :func:`jid_to_e164` for outbound sends. Strips the
    leading ``+`` and appends the user-server suffix. Returns ``""``
    on garbage input.
    """
    if not e164:
        return ""
    digits = e164.lstrip("+")
    if not digits.isdigit():
        return ""
    return f"{digits}@s.whatsapp.net"
