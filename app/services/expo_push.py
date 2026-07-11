"""Expo push sender (Autopilot PR3).

Thin httpx client for the Expo Push Service — no SDK dependency (the
official expo-server-sdk is Node-only; the Python community one would
add a dep to requirements.platform.txt for what is two POST calls).

API contract (docs.expo.dev/push-notifications, accessed 2026-07-07):
- POST https://exp.host/--/api/v2/push/send      — max 100 messages/req.
  Returns one push TICKET per message, in order. A ticket is
  {"status": "ok", "id": <receipt-id>} or {"status": "error",
  "message": ..., "details": {"error": "DeviceNotRegistered" | ...}}.
- POST https://exp.host/--/api/v2/push/getReceipts — poll ~15 min after
  send with the ticket ids (max 1000/req). A receipt is keyed by ticket
  id with the same ok/error shape; DeviceNotRegistered here means the
  token is dead and MUST stop being sent to (delivery-reputation harm).

Retries: 429/5xx are transient — exponential backoff, small attempt
cap; 4xx (other than 429) are permanent for this batch. Network errors
are transient. The caller (dispatcher) owns row-level retry counting —
this module only retries the HTTP hop.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

EXPO_PUSH_SEND_URL = "https://exp.host/--/api/v2/push/send"
EXPO_PUSH_RECEIPTS_URL = "https://exp.host/--/api/v2/push/getReceipts"

# Android notification channels (importance is FIXED at channel
# creation on-device — the app registers these in PR5; changing
# importance later requires a new channel id, so the taxonomy is
# deliberately small and stable).
ANDROID_CHANNEL_BY_PRIORITY = {
    "high": "approvals",       # needs_input / needs_approval
    "default": "task-complete",
    "low": "digest",
}

# iOS interruption levels per priority tier. time-sensitive is
# reserved for genuinely urgent items (App Review rejects misuse).
IOS_INTERRUPTION_BY_PRIORITY = {
    "high": "timeSensitive",
    "default": "active",
    "low": "passive",
}

_MAX_BATCH = 100
_TRANSIENT_ATTEMPTS = 3


class ExpoPushError(Exception):
    """Permanent, batch-level send failure (non-429 4xx or exhausted
    retries) — the caller decides whether to retry the row later."""


def build_message(
    token: str,
    title: str,
    body: Optional[str],
    data: Optional[Dict[str, Any]],
    priority: str,
) -> Dict[str, Any]:
    """One Expo push message. ``data`` carries the deep-link payload
    the app's response listener feeds into navigate() (PR5)."""
    return {
        "to": token,
        "title": title,
        "body": body or "",
        "data": data or {},
        # Expo priority maps to FCM priority (delivery urgency).
        "priority": "high" if priority == "high" else "default",
        "channelId": ANDROID_CHANNEL_BY_PRIORITY.get(priority, "task-complete"),
        "interruptionLevel": IOS_INTERRUPTION_BY_PRIORITY.get(priority, "active"),
        "sound": "default" if priority != "low" else None,
    }


async def _post_with_retry(url: str, payload: Any) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(_TRANSIENT_ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"transient {resp.status_code}",
                    request=resp.request, response=resp,
                )
            if resp.status_code >= 400:
                raise ExpoPushError(
                    f"expo push {resp.status_code}: {resp.text[:300]}"
                )
            return resp.json()
        except ExpoPushError:
            raise
        except Exception as e:  # noqa: BLE001 — network + transient HTTP
            last_exc = e
            # 0.5s / 2s before attempts 2 and 3 — Expo asks for
            # exponential backoff on 429.
            if attempt < _TRANSIENT_ATTEMPTS - 1:
                await asyncio.sleep(0.5 * (4 ** attempt))
    raise ExpoPushError(f"expo push transient failure after retries: {last_exc}")


async def send_push_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Send ≤100 messages; returns the ticket list (same order as
    input). Raises ExpoPushError on batch-level failure."""
    if not messages:
        return []
    if len(messages) > _MAX_BATCH:
        raise ValueError(f"max {_MAX_BATCH} messages per send")
    body = await _post_with_retry(EXPO_PUSH_SEND_URL, messages)
    tickets = body.get("data")
    if not isinstance(tickets, list) or len(tickets) != len(messages):
        raise ExpoPushError(f"malformed ticket response: {str(body)[:300]}")
    return tickets


async def get_push_receipts(ticket_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch receipts for ticket ids. Missing ids mean 'not ready yet —
    poll again later'; present ids are terminal (ok or error)."""
    if not ticket_ids:
        return {}
    body = await _post_with_retry(EXPO_PUSH_RECEIPTS_URL, {"ids": ticket_ids})
    data = body.get("data")
    return data if isinstance(data, dict) else {}


def is_device_not_registered(ticket_or_receipt: Dict[str, Any]) -> bool:
    return (
        ticket_or_receipt.get("status") == "error"
        and (ticket_or_receipt.get("details") or {}).get("error")
        == "DeviceNotRegistered"
    )
