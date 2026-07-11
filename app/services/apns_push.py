"""Direct APNs client for iOS Live Activities (Autopilot phone surface).

Why direct APNs and not Expo's push service: Expo's send API has no
``apns-push-type: liveactivity`` support (expo/expo#43591, closed as
not planned), so Live Activity start/update/end pushes must speak to
Apple directly. Auth is token-based (ES256 JWT signed with the .p8
key) — Apple does not support cert (.p12) auth for Live Activities.

Payload contract: the widget extension in the mobile app decodes
``content-state`` with Swift's default Codable, so the keys here MUST
byte-match ``LiveActivityAttributes.ContentState`` in
expo-live-activity's Swift module (title / subtitle /
timerEndDateInMilliseconds / progress / imageName /
dynamicIslandImageName), and ``attributes-type`` must be the literal
Swift type name ``LiveActivityAttributes``.

Force-quit guarantee: liveactivity pushes are rendered by the widget
extension — the app process is never required. The start payload sets
``"input-push-token": 1`` (iOS 18+) so the push-to-start token doubles
as the update token; we therefore never depend on the app waking up to
report a per-activity token (when it does report one, we use it).

The whole payload (attributes + content-state) must stay under 4KB.
"""
from __future__ import annotations

import base64
import time
from typing import Any, Dict, Optional, Tuple

import httpx
import jwt as pyjwt

from app.config import settings

APNS_HOST_PRODUCTION = "https://api.push.apple.com"
APNS_HOST_SANDBOX = "https://api.sandbox.push.apple.com"

# Swift type name of the ActivityAttributes struct compiled into the
# widget extension — attributes-type in every start payload.
ATTRIBUTES_TYPE = "LiveActivityAttributes"

# Apple requires provider JWTs to be refreshed every 20-60 minutes;
# refresh at 45 to stay clear of both edges.
_JWT_TTL_SECONDS = 45 * 60

# Progress bar tint on the lock-screen card (widget falls back to the
# system tint everywhere the attribute is unset).
_PROGRESS_TINT = "#3B82F6"

_jwt_cache: Dict[str, Any] = {"token": None, "issued_at": 0.0}
_clients: Dict[str, httpx.AsyncClient] = {}


class ApnsNotConfigured(RuntimeError):
    """Raised when a send is attempted without APNs credentials."""


def apns_configured() -> bool:
    return bool(settings.apns_key_b64 and settings.apns_key_id and settings.apns_team_id)


def _provider_jwt() -> str:
    now = time.time()
    if _jwt_cache["token"] and now - _jwt_cache["issued_at"] < _JWT_TTL_SECONDS:
        return _jwt_cache["token"]
    if not apns_configured():
        raise ApnsNotConfigured("apns_key_b64/apns_key_id/apns_team_id not set")
    key_pem = base64.b64decode(settings.apns_key_b64).decode("utf-8")
    token = pyjwt.encode(
        {"iss": settings.apns_team_id, "iat": int(now)},
        key_pem,
        algorithm="ES256",
        headers={"kid": settings.apns_key_id},
    )
    _jwt_cache["token"] = token
    _jwt_cache["issued_at"] = now
    return token


def _client_for(host: str) -> httpx.AsyncClient:
    client = _clients.get(host)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(base_url=host, http2=True, timeout=15.0)
        _clients[host] = client
    return client


def is_token_dead(status_code: int, reason: str) -> bool:
    """True when APNs told us this token will never work again.

    ``BadDeviceToken`` also fires on sandbox/production mismatch — for
    our single-environment-per-row model that's equally terminal.
    """
    if status_code == 410:
        return True
    return reason in {"BadDeviceToken", "Unregistered", "DeviceTokenNotForTopic"}


async def send_live_activity(
    token_hex: str,
    payload: Dict[str, Any],
    *,
    environment: str = "development",
    priority: int = 10,
) -> Tuple[int, str]:
    """POST one liveactivity push. Returns (http_status, apns_reason).

    priority 10 counts against Apple's hourly Live Activity budget;
    use 5 for routine progress deltas (unbudgeted) and 10 only for
    start/end/needs-you moments.
    """
    host = APNS_HOST_SANDBOX if environment == "development" else APNS_HOST_PRODUCTION
    client = _client_for(host)
    resp = await client.post(
        f"/3/device/{token_hex}",
        json=payload,
        headers={
            "authorization": f"bearer {_provider_jwt()}",
            "apns-topic": f"{settings.apns_bundle_id}.push-type.liveactivity",
            "apns-push-type": "liveactivity",
            "apns-priority": str(priority),
        },
    )
    reason = ""
    if resp.status_code != 200:
        try:
            reason = resp.json().get("reason", "")
        except Exception:
            reason = resp.text[:200]
    return resp.status_code, reason


# ── Payload builders (pure — unit-tested against the Swift contract) ──

def _content_state(
    title: str,
    subtitle: Optional[str],
    progress: Optional[float],
    timer_end_ms: Optional[int] = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {"title": title[:80]}
    if subtitle:
        state["subtitle"] = subtitle[:120]
    # Timer wins over discrete progress: the widget renders
    # timerEndDateInMilliseconds as a bar that animates ON-DEVICE with
    # zero pushes — the right surface for bounded quick jobs, while
    # missions push discrete progress values.
    if timer_end_ms:
        state["timerEndDateInMilliseconds"] = int(timer_end_ms)
    elif progress is not None:
        state["progress"] = max(0.0, min(1.0, float(progress)))
    return state


def _alert(alert_title: Optional[str], alert_body: Optional[str]) -> Optional[Dict[str, Any]]:
    if not alert_title:
        return None
    alert: Dict[str, Any] = {"title": alert_title[:120], "sound": "default"}
    if alert_body:
        alert["body"] = alert_body[:400]
    return alert


def build_start_payload(
    *,
    mission_id: str,
    title: str,
    subtitle: Optional[str] = None,
    progress: Optional[float] = 0.0,
    timer_end_ms: Optional[int] = None,
    alert_title: Optional[str] = None,
    alert_body: Optional[str] = None,
    timestamp: Optional[int] = None,
) -> Dict[str, Any]:
    """Push-to-start payload. ``attributes.name`` carries the mission id —
    the app's onTokenReceived listener echoes it back so a reported
    per-activity token can be matched to its live_activities row.

    No ``alert`` kwargs → a silent start: the card appears without a
    banner (used by the self-healing restart path)."""
    aps: Dict[str, Any] = {
        "timestamp": int(timestamp or time.time()),
        "event": "start",
        # iOS 18+: reuse the push-to-start token for updates, so the
        # activity stays updatable even if the app never runs again.
        # Apple leaves multi-activity behavior on a shared token
        # UNDEFINED — live_activity_service enforces at most one
        # platform-driven activity per device (newest wins).
        "input-push-token": 1,
        "attributes-type": ATTRIBUTES_TYPE,
        "attributes": {
            "name": mission_id,
            "progressViewTint": _PROGRESS_TINT,
            "deepLinkUrl": "toup://mission-control",
        },
        "content-state": _content_state(title, subtitle, progress, timer_end_ms),
    }
    alert = _alert(alert_title, alert_body)
    if alert:
        aps["alert"] = alert
    return {"aps": aps}


def build_update_payload(
    *,
    title: str,
    subtitle: Optional[str] = None,
    progress: Optional[float] = None,
    timer_end_ms: Optional[int] = None,
    alert_title: Optional[str] = None,
    alert_body: Optional[str] = None,
    stale_date: Optional[int] = None,
    timestamp: Optional[int] = None,
) -> Dict[str, Any]:
    aps: Dict[str, Any] = {
        "timestamp": int(timestamp or time.time()),
        "event": "update",
        "content-state": _content_state(title, subtitle, progress, timer_end_ms),
    }
    if stale_date:
        aps["stale-date"] = int(stale_date)
    alert = _alert(alert_title, alert_body)
    if alert:
        aps["alert"] = alert
    return {"aps": aps}


def build_end_payload(
    *,
    title: str,
    subtitle: Optional[str] = None,
    progress: Optional[float] = None,
    alert_title: Optional[str] = None,
    alert_body: Optional[str] = None,
    dismissal_date: Optional[int] = None,
    timestamp: Optional[int] = None,
) -> Dict[str, Any]:
    """End payload. Without ``dismissal_date`` the finished card lingers
    on the lock screen (up to 4h) so the user sees the final state."""
    aps: Dict[str, Any] = {
        "timestamp": int(timestamp or time.time()),
        "event": "end",
        "content-state": _content_state(title, subtitle, progress),
    }
    if dismissal_date:
        aps["dismissal-date"] = int(dismissal_date)
    alert = _alert(alert_title, alert_body)
    if alert:
        aps["alert"] = alert
    return {"aps": aps}
