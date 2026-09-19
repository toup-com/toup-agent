"""Infra alerting — one canonical path to the operator's Telegram.

Law 4 of the bulletproof plan: users are never the monitoring system.
Every self-heal loop that FIRES (reclaimed a stranded user, re-bound a
keyless agent, restarted a sick container) must tell the operator —
healing is good, repeated healing means something upstream broke.

Uses the dedicated infra bot (`INFRA_ALERT_TELEGRAM_TOKEN`), falling
back to the admin bot. Per-category rate limiting keeps a flapping
loop from flooding the chat: the first alert in a window goes out,
repeats within `min_interval_s` are counted and folded into the next
one.
"""

import logging
import time
from typing import Dict, Optional, Tuple

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_LEVEL_PREFIX = {"info": "🔵", "warning": "⚠️", "critical": "🚨"}

# (category, subject) -> [last_sent_epoch, suppressed_count]
#
# Keyed on the SUBJECT as well as the category (2026-09-12 onboarding
# incident, L3-8): with a category-only key, one permanently-stuck account
# blanket-suppressed EVERY other self-heal event fleet-wide for the whole
# window, and the "+N suppressed" line named nobody. `subject=None` keeps the
# old category-wide behaviour for callers with no per-subject identity, so
# existing call sites are unaffected.
_last_sent: Dict[Tuple[str, Optional[str]], list] = {}

# category -> {"start": epoch, "sent": int, "capped": set[str], "carried": set[str]}
#
# The counterweight to per-subject keying. A window per subject is the right
# fix for "one stuck account blanket-suppressed every other self-heal event",
# but it multiplies volume by the number of stuck subjects — and on 13 Sep
# that number was NINE (four ambiguous-ownership accounts plus five
# permanently-MCP-401 tenants), on two replicas. Past
# `infra_alert_category_subject_cap` distinct subjects in one window this
# COLLECTS the subject instead of paging, and the next send in that category
# names every collected subject in a digest line. Nothing is dropped; the
# operator gets one message naming K subjects instead of K messages.
_cat_window: Dict[str, dict] = {}

_DIGEST_NAMES_MAX = 12


def reset_for_tests() -> None:
    """Clear both rate-limit structures. Module state is per-process by
    design (the fleet-wide de-duplication is W1's lease, not this file)."""
    _last_sent.clear()
    _cat_window.clear()


def _telegram_target() -> Tuple[Optional[str], Optional[str]]:
    """The infra bot's (token, chat_id), with the admin bot as fallback.

    ONE resolution, read by both `send_infra_alert` and
    `infra_alerts_configured` — two copies of this pair is how a caller comes
    to believe alerting is configured while the sender disagrees.
    """
    return (
        settings.infra_alert_telegram_token or settings.admin_alert_telegram_token,
        settings.infra_alert_telegram_chat_id or settings.admin_alert_telegram_chat_id,
    )


def infra_alerts_configured() -> bool:
    """Is there anywhere for an alert to GO?

    `send_infra_alert` answers False for three different things —
    unconfigured, rate-limited, refused — and a caller that keeps its own
    retry state has to tell "nobody is listening" from "Telegram said no": the
    first must consume the window silently, the second must be retried. The
    fleet watch in `rollout_service` is the caller that needs this; without it
    a deployment with no Telegram config sits in the retry path forever.
    """
    token, chat_id = _telegram_target()
    return bool(token and chat_id)


def _category_state(category: str, now: float, window_s: float) -> dict:
    st = _cat_window.get(category)
    if st is None:
        st = {"start": now, "sent": 0, "capped": set(), "carried": set()}
        _cat_window[category] = st
        return st
    if now - st["start"] >= window_s:
        # Roll the window. Anything capped in the window we are leaving is
        # still owed to the operator, so it moves to `carried` and rides out
        # on the next confirmed send rather than evaporating.
        st["carried"] |= st["capped"]
        st["capped"] = set()
        st["sent"] = 0
        st["start"] = now
    return st


async def send_infra_alert(
    category: str,
    level: str,
    message: str,
    *,
    subject: Optional[str] = None,
    min_interval_s: int = 600,
) -> bool:
    """Send a Telegram alert, rate-limited per (category, subject).

    `subject` is the thing the alert is ABOUT — a user prefix, a container
    name — so a window for one subject cannot suppress an alert about another.

    Returns True only if the alert was actually DELIVERED (a 2xx from
    Telegram); False if suppressed by the rate limit, unconfigured, or the send
    failed. Never raises.

    A failed send does NOT consume the rate-limit window: `entry[0]` is advanced
    only after a confirmed 2xx (L3-7). Previously the timestamp was written
    before the POST and the POST's status was never checked, so a 429 — the
    canonical result of two replicas posting to the same chat 0.3 s apart —
    silently suppressed the next window against a message nobody received.
    """
    token, chat_id = _telegram_target()
    if not token or not chat_id:
        logger.info("[infra-alert] no telegram config; skipping: %s", message)
        return False

    now = time.time()
    key = (category, subject)
    entry = _last_sent.setdefault(key, [0.0, 0])
    if now - entry[0] < min_interval_s:
        entry[1] += 1
        logger.info(
            "[infra-alert] suppressed (%s/%s, %d in window): %s",
            category, subject or "-", entry[1], message,
        )
        return False

    # Per-category cap on DISTINCT subjects, so per-subject keying cannot
    # become a flood. A category-wide alert (subject=None) is never capped —
    # there is only ever one of it.
    cap = int(getattr(settings, "infra_alert_category_subject_cap", 5) or 0)
    cat = _category_state(category, now, float(min_interval_s))
    if subject is not None and cap > 0 and cat["sent"] >= cap:
        cat["capped"].add(str(subject))
        logger.info(
            "[infra-alert] category cap reached (%s, %d subjects this window) "
            "— collecting %s for the next digest: %s",
            category, cat["sent"], subject, message,
        )
        return False

    # Read the suppressed count but DO NOT consume the window yet — that only
    # happens on a confirmed send below.
    suppressed = entry[1]

    prefix = _LEVEL_PREFIX.get(level, "•")
    subj = f" {subject}" if subject else ""
    body = f"{prefix} [{category}{subj}] {message}"
    if suppressed:
        body += f"\n(+{suppressed} similar suppressed in the last window)"
    carried = sorted(cat["carried"])
    if carried:
        shown = ", ".join(carried[:_DIGEST_NAMES_MAX])
        more = len(carried) - len(carried[:_DIGEST_NAMES_MAX])
        body += (
            f"\n(+{len(carried)} more subject(s) in this category hit the "
            f"per-category cap: {shown}"
            + (f", and {more} more" if more else "")
            + ")"
        )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": body,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
        if resp.status_code >= 300:
            # Non-2xx (429 rate-limit, 5xx, …): leave the window OPEN so the
            # next tick re-sends rather than suppressing against a message
            # Telegram rejected. Do not touch entry[0]/entry[1].
            logger.warning(
                "[infra-alert] telegram returned %s; window NOT consumed: %s",
                resp.status_code, body,
            )
            return False
        # Confirmed delivered — NOW consume the window and clear the count.
        entry[0] = now
        entry[1] = 0
        if subject is not None:
            cat["sent"] += 1
        # The digest rode out on this message; it is no longer owed.
        cat["carried"] = set()
        return True
    except Exception as e:
        logger.warning(
            "[infra-alert] telegram send failed (%r); window NOT consumed", e,
        )
        return False
