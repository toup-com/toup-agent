"""
App manifest loader — reads and validates agentSkill.json from app directories.

Returns None on any failure (missing file, invalid JSON, schema validation error,
SQL param mismatch). Logs structured warnings for each failure mode so broken
manifests are visible in monitoring without breaking the App API response.

Log event names:
- manifest_load_missing (INFO — expected for old apps)
- manifest_load_invalid_json (WARNING — syntax error)
- manifest_load_schema_invalid (WARNING — Pydantic validation failed)
- manifest_unknown_fields (INFO — schema evolved, informational)
"""

import json
import logging
import os
from typing import Optional

from pydantic import ValidationError

from app.models.app_manifest import AppManifest, detect_unknown_fields

logger = logging.getLogger(__name__)


def load_app_manifest(
    app_dir: str,
    app_id: str = "",
) -> Optional[AppManifest]:
    """Load and validate the agentSkill.json manifest from an app directory.

    Returns the parsed AppManifest on success, or None on any failure.
    Never raises — all errors are logged and swallowed.

    Args:
        app_dir: Path to the app's working directory.
        app_id: App ID for log context (optional).
    """
    manifest_path = os.path.join(app_dir, "lib", "agentSkill.json")

    # ── File not found — common for old apps, not alarming ──
    if not os.path.exists(manifest_path):
        logger.info(
            "manifest_load_missing app_id=%s path=%s",
            app_id[:8] if app_id else "?",
            manifest_path,
        )
        return None

    # ── Read and parse JSON ──
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(
            "manifest_load_invalid_json app_id=%s path=%s error=%s",
            app_id[:8] if app_id else "?",
            manifest_path,
            str(e)[:200],
        )
        return None
    except Exception as e:
        logger.warning(
            "manifest_load_invalid_json app_id=%s path=%s error=%s",
            app_id[:8] if app_id else "?",
            manifest_path,
            str(e)[:200],
        )
        return None

    # ── Check for unknown fields (informational, doesn't reject) ──
    if isinstance(raw, dict):
        unknown = detect_unknown_fields(raw)
        if unknown:
            logger.info(
                "manifest_unknown_fields app_id=%s unknown_fields=%s",
                app_id[:8] if app_id else "?",
                unknown,
            )

    # ── Validate against Pydantic schema ──
    try:
        manifest = AppManifest.model_validate(raw)
    except ValidationError as e:
        logger.warning(
            "manifest_load_schema_invalid app_id=%s errors=%s",
            app_id[:8] if app_id else "?",
            str(e)[:500],
        )
        return None
    except Exception as e:
        logger.warning(
            "manifest_load_schema_invalid app_id=%s error=%s",
            app_id[:8] if app_id else "?",
            str(e)[:200],
        )
        return None

    logger.debug(
        "manifest_load_success app_id=%s domain=%s actions=%d",
        app_id[:8] if app_id else "?",
        manifest.domain,
        len(manifest.actions),
    )
    return manifest
