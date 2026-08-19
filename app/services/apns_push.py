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

Alert SOUND is not a free field here — see ``ACTIVITY_KIT_SOUND``. Every
builder in this module emits a liveactivity payload, and a named sound on
one of those is silence on device. Producers pass the tone they chose;
this module drops the name.
"""
from __future__ import annotations

import base64
import logging
import time
from typing import Any, Dict, Optional, Tuple

import httpx
import jwt as pyjwt

from app.config import settings

logger = logging.getLogger(__name__)

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

# The ONLY sound value an ActivityKit alert may carry. Every builder in
# this module produces a ``apns-push-type: liveactivity`` payload, and
# iOS has never honoured a custom named sound on one — the result is
# TOTAL SILENCE, not a fallback (verified on the founder's device
# 2026-07-20/21 with the file present in both the app and the
# widget-extension bundles; Apple forums 718659, unanswered since Oct
# 2022; every production payload ever verified audible uses "default").
#
# So a producer may choose a tone, and this module decides whether the
# wire is allowed to carry the name. Putting the enforcement here rather
# than at each call site is the point: a lane that CAN honour a name
# (a standard ``apns-push-type: alert`` push) is a different builder,
# and until one exists nothing can accidentally silence a card by
# "wiring the picker up properly".
ACTIVITY_KIT_SOUND = "default"


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
    try:
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
    except httpx.HTTPError as exc:
        # Transport failure (timeout, GOAWAY, broken H2 stream…) must
        # surface as a normal error verdict, never a raise — a raise
        # here strands the queue row in 'sending' (2026-07-18: 8h of
        # 10-min retry cycles). Evict the pooled client too: a wedged
        # HTTP/2 connection otherwise fails every subsequent send.
        _clients.pop(host, None)
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001 — best-effort close
            pass
        logger.warning("APNs transport error on %s: %r", host, exc)
        return 599, f"transport:{type(exc).__name__}"
    reason = ""
    if resp.status_code != 200:
        try:
            reason = resp.json().get("reason", "")
        except Exception:
            reason = resp.text[:200]
    return resp.status_code, reason


# ── Payload builders (pure — unit-tested against the Swift contract) ──

# Round 3 (2026-08-18) content-state extras — every key OPTIONAL in the
# Swift ContentState (default Codable ignores unknown keys, so widgets
# built before this ship keep decoding). Names are camelCase like the
# existing keys because Swift's synthesized Codable matches property
# names byte-for-byte:
#
#   jobType     str   verify | search | write | compare | generic — icon
#   stepName    str   the step being worked on ("Done" at completion)
#   stepsDone   int   n of m
#   stepsTotal  int   m
#   percent     int   0-100 (the same value `progress` carries as 0-1;
#                     explicit so the widget's "n/m · 40%" line needs no
#                     arithmetic and no float formatting)
#   preview     str   ≤120 chars of the answer, on completion only
#   chatId      str   deep link: the conversation to open
#   messageId   str   deep link: the assistant message to scroll to
#
# Value caps keep the whole payload inside Apple's 4KB LA budget with
# every field populated (measured worst case ≈ 1.6KB).
from app.services.plain_text import (  # noqa: E402  (stdlib-only module)
    plain_preview as _plain,
    strip_markdown as _strip_md,
)

# Round 8: ``phase`` / ``stepLabel`` / ``jobKind`` are the widget's
# ContentState v2 keys (ios-widget/LiveActivityWidget.swift, 2026-08-18) —
# the app's own local cards write them, and the widget derives its whole
# face from ``phase`` when present, falling back to bar+subtitle inference
# for older payloads. Until now the platform never sent them, so a pushed
# card and a locally-started card for the same job could disagree.
# ``stepName``/``jobType`` stay as-is for the older widget lineage.
_EXTRA_STATE_STR = {"jobType": 16, "stepName": 80, "preview": 120,
                    "chatId": 64, "messageId": 64,
                    "phase": 16, "stepLabel": 80, "jobKind": 16}
_EXTRA_STATE_INT = ("stepsDone", "stepsTotal", "percent")
_KNOWN_PHASES = frozenset({"starting", "running", "completed", "failed", "needs_you"})


def _extra_state(extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Whitelist + cap the Round 3 content-state keys. Anything else,
    or a wrong type, is dropped — never passed through."""
    out: Dict[str, Any] = {}
    if not extra:
        return out
    for key, cap in _EXTRA_STATE_STR.items():
        v = extra.get(key)
        if isinstance(v, str) and v.strip():
            # Round 4 (item 4): the Live Activity renders these verbatim —
            # strip markdown BEFORE the cap (a `**` pair split by the cap
            # would otherwise survive). Ids are never markdown; skipping
            # them keeps _safe_id's output byte-identical.
            if key in ("chatId", "messageId", "jobType", "jobKind", "phase"):
                out[key] = " ".join(v.split())[:cap]
            else:
                out[key] = _plain(v, cap)
    if out.get("phase") not in _KNOWN_PHASES:
        out.pop("phase", None)
    for key in _EXTRA_STATE_INT:
        v = extra.get(key)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[key] = max(0, int(v))
    return out


def _content_state(
    title: str,
    subtitle: Optional[str],
    progress: Optional[float],
    timer_end_ms: Optional[int] = None,
    fired: Optional[bool] = None,
    extra: Optional[Dict[str, Any]] = None,
    timer_start_ms: Optional[int] = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {"title": _plain(title, 80)}
    if subtitle:
        state["subtitle"] = _plain(subtitle, 120)
    # Round 3 extras ride every state — including fired ones, where the
    # deep-link ids are still what a tap needs.
    state.update(_extra_state(extra))
    # fired: alarm-class terminal state (reminder fires). The widget
    # renders a ringing presentation instead of any progress surface,
    # so fired cards carry NEITHER timer NOR progress — a fired
    # reminder shown as a 0%-then-100% bar reads as a stale job card
    # (founder repro 2026-07-22). Optional in the Swift ContentState:
    # old widgets ignore the extra key, old payloads decode nil.
    if fired:
        state["fired"] = True
        state.pop("percent", None)
        return state
    # Timer wins over discrete progress: the widget renders
    # timerEndDateInMilliseconds as a bar that animates ON-DEVICE with
    # zero pushes — the right surface for bounded quick jobs, while
    # missions push discrete progress values.
    if timer_end_ms:
        state["timerEndDateInMilliseconds"] = int(timer_end_ms)
        # The countdown's START — the instant the reminder was SET. Without
        # it the widget's bar starts at view-render time and restarts from
        # zero on every rebuild; with it the lock-screen bar is the same
        # absolute set→fire span the in-app card fills. Optional: old
        # widgets ignore the key.
        if timer_start_ms and timer_start_ms < timer_end_ms:
            state["timerStartDateInMilliseconds"] = int(timer_start_ms)
    elif progress is not None:
        state["progress"] = max(0.0, min(1.0, float(progress)))
        # Keep the two progress spellings consistent when the caller
        # supplied both: the fraction is the one the lane clamps
        # (never-backwards), so it is the source of truth.
        if "percent" in state:
            state["percent"] = int(round(state["progress"] * 100))
    return state


def _valid_hex_color(value: Optional[str]) -> bool:
    """Strict '#RRGGBB' — the widget's hex parser assumes exactly this
    shape; anything looser must be dropped, not passed through."""
    if not isinstance(value, str) or len(value) != 7 or value[0] != "#":
        return False
    try:
        int(value[1:], 16)
    except ValueError:
        return False
    return True


def _alert(
    alert_title: Optional[str], alert_body: Optional[str],
    requested_sound: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build the alert configuration for an ActivityKit payload.

    ``requested_sound`` is the bundled file name the PRODUCER chose (the
    operator's dispatch tone, a reminder's alarm tone). It is recorded and
    then deliberately NOT put on the wire — see ``ACTIVITY_KIT_SOUND``.
    Callers pass what they want rather than pre-flattening it to
    "default", so the rule lives in exactly one place and the payload
    tests can assert it directly.
    """
    if not alert_title:
        return None
    alert: Dict[str, Any] = {
        "title": _plain(alert_title, 120),
        # requested_sound is intentionally unused here. A name lands as
        # silence; "default" is the loudest thing an ActivityKit alert can
        # be, so a chosen tone DEGRADES to the system tone rather than to
        # nothing.
        "sound": ACTIVITY_KIT_SOUND,
    }
    if alert_body:
        # Multi-line surface: keep line breaks, drop the syntax.
        alert["body"] = _strip_md(alert_body)[:400]
    return alert


def build_start_payload(
    *,
    mission_id: str,
    title: str,
    subtitle: Optional[str] = None,
    progress: Optional[float] = 0.0,
    timer_end_ms: Optional[int] = None,
    timer_start_ms: Optional[int] = None,
    alert_title: Optional[str] = None,
    alert_body: Optional[str] = None,
    alert_sound: Optional[str] = None,
    timestamp: Optional[int] = None,
    deep_link: str = "toup://mission-control",
    timer_type: Optional[str] = None,
    orb_color: Optional[str] = None,
    stale_date: Optional[int] = None,
    fired: Optional[bool] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Push-to-start payload. ``attributes.name`` carries the mission id —
    the app's onTokenReceived listener echoes it back so a reported
    per-activity token can be matched to its live_activities row.

    Every start carries an alert configuration — iOS 26 drops alertless
    starts outright. No ``alert`` kwargs → a QUIET start: the alert is
    synthesized from the card content with no sound (used by countdown
    arms and the self-healing restart path)."""
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
            # Where a card tap lands. Attributes are fixed for the
            # activity's lifetime, so the producer picks the target at
            # start (chat turns → the conversation, missions → Mission
            # Control).
            "deepLinkUrl": deep_link,
        },
        "content-state": _content_state(title, subtitle, progress, timer_end_ms, fired, extra, timer_start_ms),
    }
    # Compact Dynamic Island timer style — the widget decodes
    # attributes.timerType ('circular' default | 'digital' mm:ss).
    # Attributes are start-fixed, so it must ride the start push.
    if timer_type in ("circular", "digital"):
        aps["attributes"]["timerType"] = timer_type
    # The user's agent color: the widget draws the orb face (and tints
    # the progress bar) in it, so every user's card matches their
    # in-app agent. Attributes are start-fixed — a color change lands
    # on the NEXT card. Strict '#RRGGBB' only: an unparsable value in
    # Color(hex:) renders black, so anything else is dropped and the
    # widget falls back to the brand default.
    if _valid_hex_color(orb_color):
        aps["attributes"]["orbColor"] = orb_color
        aps["attributes"]["progressViewTint"] = orb_color
    # Zombie-card backstop: if every follow-up push is lost, the card
    # self-marks stale (widget dims it) instead of looking live at
    # 0:00 for hours (founder incident 2026-07-18). Updates refresh it.
    if stale_date:
        aps["stale-date"] = int(stale_date)
    alert = _alert(alert_title, alert_body, alert_sound)
    if alert is None:
        # iOS 26 REJECTS start events with no alert configuration —
        # liveactivitiesd publishes the activity, then SessionCore logs
        # "Received start without an alert configuration" and drops it:
        # APNs 200s, the card never renders (observed on-device
        # 2026-07-18, iOS 26.4.2 — the root cause of every invisible
        # card since silent starts shipped). Every start therefore
        # carries an alert; producer-"silent" starts synthesize it from
        # the card content and omit the sound so the arm stays quiet.
        alert = {"title": title[:120]}
        if subtitle:
            alert["body"] = subtitle[:400]
    aps["alert"] = alert
    return {"aps": aps}


def build_update_payload(
    *,
    title: str,
    subtitle: Optional[str] = None,
    progress: Optional[float] = None,
    timer_end_ms: Optional[int] = None,
    timer_start_ms: Optional[int] = None,
    alert_title: Optional[str] = None,
    alert_body: Optional[str] = None,
    alert_sound: Optional[str] = None,
    stale_date: Optional[int] = None,
    timestamp: Optional[int] = None,
    fired: Optional[bool] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    aps: Dict[str, Any] = {
        "timestamp": int(timestamp or time.time()),
        "event": "update",
        "content-state": _content_state(title, subtitle, progress, timer_end_ms, fired, extra, timer_start_ms),
    }
    if stale_date:
        aps["stale-date"] = int(stale_date)
    alert = _alert(alert_title, alert_body, alert_sound)
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
    alert_sound: Optional[str] = None,
    dismissal_date: Optional[int] = None,
    timestamp: Optional[int] = None,
    fired: Optional[bool] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """End payload. Without ``dismissal_date`` the finished card lingers
    on the lock screen (up to 4h) so the user sees the final state."""
    aps: Dict[str, Any] = {
        "timestamp": int(timestamp or time.time()),
        "event": "end",
        "content-state": _content_state(title, subtitle, progress, fired=fired, extra=extra),
    }
    if dismissal_date:
        aps["dismissal-date"] = int(dismissal_date)
    alert = _alert(alert_title, alert_body, alert_sound)
    if alert:
        aps["alert"] = alert
    return {"aps": aps}
