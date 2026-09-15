"""
Container Health Monitor — periodically checks managed containers
and alerts admin via Telegram + in-app notification if any go down.

Runs on the platform (Railway), not on the VPS.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import ManagedContainer
from app.services.alerting import send_infra_alert

logger = logging.getLogger(__name__)

# Track consecutive failures per container
_failure_counts: Dict[str, int] = {}
_last_alert: Dict[str, datetime] = {}
ALERT_AFTER_FAILURES = 3  # Alert after 3 consecutive failures (15 min)
ALERT_COOLDOWN = timedelta(hours=1)  # Don't spam — max 1 alert per container per hour

# ── Tenant DB path ────────────────────────────────────────────────
# 2026-08-01: pgbouncer died and every tenant lost its database. Chat 500'd
# fleet-wide for six minutes and NOTHING alerted, because the containers were
# up, Postgres was up, and /agent/health kept returning 200 with
# `status: "healthy"`. The agent's own db_watchdog had already detected it and
# published `db_ok: false` in that same payload — this monitor just never read
# the field. That is the "/agent/health green != chat works" trap, and it is
# the second time it has cost a fleet-wide outage.
#
# db_ok is deliberately NOT folded into `healthy` below. `healthy` drives the
# auto-restart path, and restarting 55 containers because a SHARED component
# (pgbouncer/Postgres) is down is both useless and harmful — it would turn one
# outage into a restart storm. So this is a separate, aggregated signal that
# alerts and never acts.
_db_down_counts: Dict[str, int] = {}
_last_db_alert: Optional[datetime] = None
# One tenant reporting db_ok=false could be its own database; wait a cycle.
# Two or more at once is a shared component — say so immediately, because the
# operator's next move differs completely.
DB_ALERT_AFTER_FAILURES = 2
DB_ALERT_MIN_TENANTS_IMMEDIATE = 2
DB_ALERT_COOLDOWN = timedelta(minutes=30)

# ── Turn path (round N P0, 2026-08-23) ────────────────────────────
# Same contract as the DB path: an agent can be UP and healthy while unable
# to serve a single chat turn (`turn_ready: false` — its pipeline init threw
# and was swallowed). Tracked and alerted, never acted on: for a bad IMAGE
# this hits every recreated tenant at once and a restart storm fixes
# nothing. One consecutive miss alerts — a dead turn path is a total outage
# for that user, and the founder finding it in his own chat at 11:28 PM is
# the incident this block exists to end.
_turn_down_counts: Dict[str, int] = {}
_last_turn_alert: Optional[datetime] = None
TURN_ALERT_AFTER_FAILURES = 1
TURN_ALERT_COOLDOWN = timedelta(minutes=30)


def verdict_from_health_body(data: dict) -> "tuple[bool, Optional[bool], Optional[bool]]":
    """Read `(healthy, db_ok, turn_ready)` out of an /agent/health body.

    Pure, so the decision that mattered on 2026-08-01 is testable without a
    database or an HTTP stack. `db_ok` is a SIBLING of `status`, not nested
    under it — a tenant whose database is unreachable still answers
    `{"status": "healthy", "db_ok": false}`, which is exactly why reading
    only `status` missed a fleet-wide outage.

    `turn_ready` (round N P0, 2026-08-23 — the THIRD "green health while
    every chat fails" arc): the agent-pipeline init threw and was swallowed,
    `_agent_runner` stayed None, and every chat turn answered "Agent not
    available" behind a 200. Same sibling contract as db_ok: tracked and
    alerted, never acted on.

    A missing or non-boolean field is None, meaning "the agent did not
    say" — never "down". Older images and pool-generic boots omit them, and
    treating absence as failure would page forever.
    """
    healthy = data.get("status") in ("healthy", "ok")
    db_ok = data.get("db_ok")
    turn_ready = data.get("turn_ready")
    return (
        healthy,
        (db_ok if isinstance(db_ok, bool) else None),
        (turn_ready if isinstance(turn_ready, bool) else None),
    )


# ── Diagnostic counters (`health_signals`) ────────────────────────
# Process-lifetime ints the agent publishes beside `status`/`db_ok`. LAW 1:
# nothing gates on them — they alert and never act, exactly like db_ok. An
# older image omits the key entirely, which must read as "did not say", never
# as zero, or every pre-rollout container would look like a healed tenant.
_signal_prev: Dict[str, Dict[str, int]] = {}
_orphan_streak: Dict[str, int] = {}
_reply_gap_streak: Dict[str, int] = {}
_media_gap_streak: Dict[str, int] = {}
_future_days_streak: Dict[str, int] = {}
_resolve_fail_streak: Dict[str, int] = {}
_day_chats_5xx_streak: Dict[str, int] = {}


def reset_signal_state_for_tests() -> None:
    """Forget every per-container streak and snapshot. Tests only: the streaks
    count TICKS across `check_all_containers` calls, so a suite that runs
    several scenarios in one process must start each from a cold monitor."""
    for d in (_signal_prev, _orphan_streak, _reply_gap_streak,
              _media_gap_streak, _day_chats_5xx_streak, _LAST_SIGNALS,
              _future_days_streak, _resolve_fail_streak):
        d.clear()

# A lost REPLY is a user-visible hole in their history — page on the first
# one. A lost CARD is a degraded render; wait for weight or persistence.
PERSIST_MEDIA_GAP_BURST = 3
SIGNAL_ALERT_MIN_INTERVAL_S = 1800
# `channel_tag_prefixed_replies` should be exactly zero once the annotation
# rule is in the prompt, so any non-zero deserves a message — but a tenant
# whose model keeps doing it must not page every five minutes.
TAG_LEAK_ALERT_MIN_INTERVAL_S = 6 * 3600


def health_signals_from_body(data: dict) -> Optional[Dict[str, int]]:
    """The `health_signals` object out of an /agent/health body, ints only.

    `None` means the agent did not report the key — an old image, or a
    pool-generic boot. Deliberately NOT folded into
    `verdict_from_health_body`: that function's 3-tuple is the liveness
    contract three call sites and a pinned regression suite read positionally,
    and a diagnostic must not be able to break liveness parsing.
    """
    raw = data.get("health_signals")
    if not isinstance(raw, dict):
        return None
    out: Dict[str, int] = {}
    for k, v in raw.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out[str(k)] = int(v)
    return out


async def _probe_agent_health(
    container: ManagedContainer,
) -> "tuple[bool, Optional[bool], Optional[bool]]":
    """Probe a container's agent health endpoint.

    Returns `(healthy, db_ok, turn_ready)`; the diagnostic counters from the
    same body are read back through `_probe_health_signals`. `healthy` is the
    liveness verdict that drives
    the alert + auto-restart path below. `db_ok` is the agent's own
    db_watchdog verdict on whether its tenant database is reachable, and is
    reported separately — see the note by `_db_down_counts`.
    `db_ok` is None when the agent did not report the field at all.

    Phase 3: URL is the HTTPS subdomain per AgentConfig.agent_url, fronted
    by Caddy on 443. The platform doesn't need bridge mTLS for this check —
    /agent/health is reachable without client cert (the tenant's own agent
    responds publicly; X-Agent-Key only gates authenticated endpoints).
    """
    from app.db.models import AgentConfig
    from app.db.database import async_session_maker
    from sqlalchemy import select

    # Prefer AgentConfig.agent_url (HTTPS subdomain) over the legacy
    # http://{docker_host_ip}:{port} form.
    url: str | None = None
    async with async_session_maker() as db:
        result = await db.execute(
            select(AgentConfig.agent_url).where(AgentConfig.user_id == container.user_id)
        )
        agent_url = result.scalar_one_or_none()
    if agent_url:
        url = f"{agent_url.rstrip('/')}/agent/health"
    elif container.host_port and settings.docker_host_ip:
        # Legacy fallback — only hit during the Phase 3 transition window
        # when AgentConfig rows haven't been populated with HTTPS URLs yet.
        url = f"http://{settings.docker_host_ip}:{container.host_port}/agent/health"
    else:
        # Same 3-wide shape as every other return: the one caller unpacks
        # three, and a fourth value here unwound the whole sweep on the
        # first provisioning row without a port.
        _LAST_SIGNALS[container.id] = None
        return False, None, None

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                body = resp.json()
                # The counters ride the SAME body; stash them for
                # _probe_health_signals so the fold costs no second request.
                _LAST_SIGNALS[container.id] = health_signals_from_body(body)
                return verdict_from_health_body(body)
    except Exception:
        pass
    _LAST_SIGNALS[container.id] = None
    return False, None, None


# One probe body, two readers. `_probe_agent_health` keeps its historical
# (healthy, db_ok, turn_ready) shape — call sites and the 2026-08-01 outage
# regression suite unpack it positionally — and the diagnostic counters are
# handed out by this second accessor. Keyed by container id and consumed on
# read, so a stale snapshot can never be folded twice.
_LAST_SIGNALS: Dict[str, Optional[Dict[str, int]]] = {}


async def _probe_health_signals(
    container: ManagedContainer,
) -> Optional[Dict[str, int]]:
    """The `health_signals` block from the container's last /agent/health
    probe, or None when the image does not publish one (old image) or the
    probe failed. Absent is NOT zero — the fold treats None as "unknown"."""
    return _LAST_SIGNALS.pop(container.id, None)


# Name the contract's tests use for the pure reader.
signals_from_health_body = health_signals_from_body


async def _send_telegram_alert(message: str):
    """Send alert to admin via Telegram."""
    bot_token = settings.admin_alert_telegram_token
    chat_id = settings.admin_alert_telegram_chat_id
    if not bot_token or not chat_id:
        logger.warning("[MONITOR] No Telegram alert config — skipping alert")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
            })
        logger.info("[MONITOR] Telegram alert sent")
    except Exception as e:
        logger.warning(f"[MONITOR] Telegram alert failed: {e}")


async def _store_alert(db: AsyncSession, container: ManagedContainer, message: str):
    """Store alert in container's error_message field for admin panel visibility."""
    await db.execute(
        update(ManagedContainer)
        .where(ManagedContainer.id == container.id)
        .values(error_message=message)
    )
    await db.commit()


async def check_all_containers():
    """Check health of all managed containers. Called periodically."""
    from app.db.database import async_session_maker

    if not settings.docker_host_ip:
        return

    async with async_session_maker() as db:
        result = await db.execute(
            select(ManagedContainer)
            .where(ManagedContainer.status.in_(["running", "provisioning"]))
        )
        containers = result.scalars().all()

        if not containers:
            return

        db_down: list[ManagedContainer] = []
        turn_down: list[ManagedContainer] = []

        _folded_ids: set = set()
        # Alerts the fold decides on are SENT after the loop, outside this
        # session: `send_infra_alert` is a 10 s Telegram POST that does not
        # consume its window on a 429, and awaiting it here pinned a platform
        # DB connection and delayed every liveness verdict behind it.
        signal_alerts: "list[dict]" = []
        for container in containers:
            healthy, db_ok, turn_ready = await _probe_agent_health(container)
            signals = await _probe_health_signals(container)
            key = container.id

            # Diagnostic counters. Folded before the liveness branches so a
            # tenant that is UP and quietly losing rows is still reported —
            # that combination is the whole point of these signals.
            # Once per container per tick: the streak counters below count
            # TICKS, and a container listed twice in one sweep must not read
            # as two consecutive observations.
            if signals is not None and key not in _folded_ids:
                _folded_ids.add(key)
                signal_alerts.extend(_fold_health_signals(container, signals))

            # Turn path — tracked and alerted, never acted on (see the
            # `_turn_down_counts` note). Only an ANSWERING container can
            # report it; a dead one belongs to the liveness path below.
            if turn_ready is False:
                _turn_down_counts[key] = _turn_down_counts.get(key, 0) + 1
                turn_down.append(container)
                logger.error(
                    "[MONITOR] %s reports turn_ready=false (%d consecutive) — "
                    "agent is up but cannot serve a chat turn",
                    container.container_name, _turn_down_counts[key],
                )
            elif turn_ready is True:
                _turn_down_counts.pop(key, None)

            # Tenant DB path — tracked and alerted, never acted on. Only a
            # container that is otherwise ANSWERING can report db_ok; a dead
            # container returns (False, None) and belongs to the liveness
            # path below, not here.
            if db_ok is False:
                _db_down_counts[key] = _db_down_counts.get(key, 0) + 1
                db_down.append(container)
                logger.warning(
                    "[MONITOR] %s reports db_ok=false (%d consecutive) — "
                    "agent is up but its database is unreachable",
                    container.container_name, _db_down_counts[key],
                )
            elif db_ok is True:
                _db_down_counts.pop(key, None)

            if healthy:
                # Reset failure count and clear error
                if key in _failure_counts:
                    if _failure_counts[key] >= ALERT_AFTER_FAILURES:
                        # Was down, now recovered — send recovery alert
                        msg = (
                            f"✅ <b>Container Recovered</b>\n"
                            f"Container: <code>{container.container_name}</code>\n"
                            f"Port: {container.host_port}\n"
                            f"Time: {datetime.utcnow().strftime('%H:%M UTC')}"
                        )
                        await _send_telegram_alert(msg)
                        await _store_alert(db, container, None)
                    del _failure_counts[key]
                # Update status to running if it was something else
                if container.status != "running":
                    container.status = "running"
                    container.error_message = None
                    await db.commit()
            else:
                # Increment failure count
                _failure_counts[key] = _failure_counts.get(key, 0) + 1
                count = _failure_counts[key]

                logger.warning(
                    f"[MONITOR] {container.container_name} unhealthy "
                    f"(failure {count}/{ALERT_AFTER_FAILURES})"
                )

                if count >= ALERT_AFTER_FAILURES:
                    # Check cooldown
                    last = _last_alert.get(key)
                    now = datetime.utcnow()

                    if not last or (now - last) > ALERT_COOLDOWN:
                        _last_alert[key] = now

                        msg = (
                            f"🚨 <b>Container Down</b>\n"
                            f"Container: <code>{container.container_name}</code>\n"
                            f"Port: {container.host_port}\n"
                            f"Failures: {count} consecutive\n"
                            f"Time: {now.strftime('%H:%M UTC')}\n\n"
                            f"Auto-restart attempting..."
                        )
                        await _send_telegram_alert(msg)
                        await _store_alert(db, container, f"Unhealthy since {now.strftime('%H:%M UTC')}")

                        # Attempt auto-restart via SSH
                        try:
                            from app.services.docker_host_service import restart_container
                            await restart_container(db, container.user_id)
                            logger.info(f"[MONITOR] Auto-restarted {container.container_name}")

                            restart_msg = (
                                f"🔄 <b>Auto-Restart Triggered</b>\n"
                                f"Container: <code>{container.container_name}</code>\n"
                                f"Waiting for health check..."
                            )
                            await _send_telegram_alert(restart_msg)
                        except Exception as e:
                            logger.error(f"[MONITOR] Auto-restart failed: {e}")
                            fail_msg = (
                                f"❌ <b>Auto-Restart Failed</b>\n"
                                f"Container: <code>{container.container_name}</code>\n"
                                f"Error: {str(e)[:200]}\n\n"
                                f"Manual intervention required."
                            )
                            await _send_telegram_alert(fail_msg)

                    # Update status.
                    #
                    # A POOL member never leaves running/provisioning through
                    # this writer (2026-09-12, D-2's precondition). Wave 1
                    # closed the door — provision_container now refuses a
                    # pool→named swap for any `recreate` — but this line is
                    # the only realistic way a pool row reaches `error` at
                    # all, and a pool row in `error` is what walks past
                    # `provision_container`'s early return into the named
                    # path, rewriting db_name to toup_agent_<prefix> and
                    # landing an established user on an empty database.
                    # A pool member is repaired IN PLACE — the bridge
                    # re-applies its persisted bind and route after a restart
                    # — so `error` was never the right word for it either.
                    if (container.container_name or "").startswith(
                        "toup-agent-pool-"
                    ):
                        container.error_message = (
                            f"Unhealthy for {count * 5} minutes "
                            "(pool member; row left running for in-place repair)"
                        )
                        await db.commit()
                        continue
                    container.status = "error"
                    container.error_message = f"Unhealthy for {count * 5} minutes"
                    await db.commit()

    await _dispatch_signal_alerts(signal_alerts)
    await _alert_on_db_path(db_down)
    await _alert_on_turn_path(turn_down)
    _prune_signal_state({c.id for c in containers})


def _prune_signal_state(live_ids: "set[str]") -> None:
    """Drop per-container streak/snapshot entries for containers no longer in
    the sweep. Rows retire (named→pool swaps, deleted accounts) and these
    dicts otherwise grow for the platform process's lifetime."""
    for d in (_signal_prev, _orphan_streak, _reply_gap_streak, _media_gap_streak,
              _day_chats_5xx_streak, _future_days_streak, _resolve_fail_streak,
              _LAST_SIGNALS):
        for key in [k for k in d if k not in live_ids]:
            d.pop(key, None)


async def _dispatch_signal_alerts(alerts: "list[dict]") -> None:
    """Send what the fold decided, one Telegram round-trip at a time, after
    the sweep has released its DB session. Never raises."""
    for a in alerts:
        try:
            await send_infra_alert(
                a["category"], a["level"], a["message"],
                subject=a.get("subject"),
                min_interval_s=a.get("min_interval_s", SIGNAL_ALERT_MIN_INTERVAL_S),
            )
        except Exception:
            logger.warning("[MONITOR] signal alert failed: %s", a.get("category"),
                           exc_info=True)


def _fold_health_signals(
    container: ManagedContainer, signals: Dict[str, int]
) -> "list[dict]":
    """Turn one tenant's counter snapshot into at most four alert payloads.

    Pure bookkeeping — it DECIDES and returns; `_dispatch_signal_alerts`
    sends after the sweep, through `send_infra_alert`, which owns the two
    hard parts: per-(category, subject) rate limiting so one stuck tenant
    cannot blanket the fleet, and the per-category distinct-subject cap that
    digests the overflow. Never raises — a monitoring read may not break the
    monitor.

    Cumulative-vs-delta, by category: the persist/orphan gaps are read as
    DIFFERENCES between two monotonic counters (a permanent discrepancy
    should stay visible), while the agent-output and day-chats signals are
    read as this tick's GROWTH — a lifetime counter that once read 1 must
    not page every rate-limit window for the life of the container.
    """
    key = container.id
    subject = (container.user_id or "")[:8] or (container.container_name or "?")
    prev = _signal_prev.get(key) or {}
    _signal_prev[key] = dict(signals)
    out: "list[dict]" = []

    def _now(name: str) -> int:
        return int(signals.get(name, 0) or 0)

    def _grew(name: str) -> int:
        """This tick's increase, clamped at 0 — a container restart resets
        every counter and must not read as a negative delta."""
        return max(0, _now(name) - int(prev.get(name, 0) or 0))

    def _alert(category: str, level: str, message: str, *, min_interval_s: int) -> None:
        out.append(dict(category=category, level=level, message=message,
                        subject=subject, min_interval_s=min_interval_s))

    try:
        # ── persist-gap: a turn completed but its assistant row is missing ──
        turns = _now("turns_completed")
        rows = _now("assistant_rows_written")
        reply_gap = max(0, turns - rows) if turns else 0
        # `turns_completed` advances on entry to _save_messages and
        # `assistant_rows_written` after the commit, so a snapshot taken
        # mid-save reads as a gap of ONE. That is a turn in flight, not a lost
        # reply; paging on it teaches the operator to ignore the category.
        # Two-or-more in one tick, or one that is still there next tick, is.
        if reply_gap:
            _reply_gap_streak[key] = _reply_gap_streak.get(key, 0) + 1
        else:
            _reply_gap_streak.pop(key, None)
        if reply_gap >= 2 or (reply_gap and _reply_gap_streak.get(key, 0) >= 2):
            _alert(
                "persist-gap", "critical",
                f"{container.container_name}: {turns - rows} completed turn(s) "
                f"with no assistant row written (turns={turns} rows={rows}). "
                f"The user's reply is missing from their history.",
                min_interval_s=SIGNAL_ALERT_MIN_INTERVAL_S,
            )

        # ── persist-gap (media): the card was expected and never persisted ──
        media_gap = max(0, _now("media_expected") - _now("media_persisted"))
        prev_gap = max(
            0,
            int(prev.get("media_expected", 0) or 0)
            - int(prev.get("media_persisted", 0) or 0),
        )
        if media_gap:
            _media_gap_streak[key] = _media_gap_streak.get(key, 0) + 1
        else:
            _media_gap_streak.pop(key, None)
        burst = media_gap - prev_gap
        if media_gap and (burst >= PERSIST_MEDIA_GAP_BURST or _media_gap_streak.get(key, 0) >= 2):
            _alert(
                "persist-gap", "warning",
                f"{container.container_name}: {media_gap} media card(s) expected "
                f"and not persisted (+{burst} this tick, "
                f"{_media_gap_streak.get(key, 0)} consecutive ticks). The card is "
                f"invisible on reload.",
                min_interval_s=SIGNAL_ALERT_MIN_INTERVAL_S,
            )

        # ── channel-orphan: inbound claimed, no user row persisted ──
        orphan = max(0, _now("channel_events_claimed") - _now("channel_events_persisted"))
        if orphan:
            _orphan_streak[key] = _orphan_streak.get(key, 0) + 1
        else:
            _orphan_streak.pop(key, None)
        # TWO ticks: a claim and its persist can straddle a 5-minute boundary,
        # and a single-tick alert would fire on every in-flight turn.
        if _orphan_streak.get(key, 0) >= 2:
            _alert(
                "channel-orphan", "warning",
                f"{container.container_name}: {orphan} inbound channel event(s) "
                f"claimed with no user message row, two ticks running. The "
                f"user's message vanished between the adapter and the DB.",
                min_interval_s=SIGNAL_ALERT_MIN_INTERVAL_S,
            )

        # ── agent-output: the model emitted a channel annotation verbatim ──
        # THIS tick's leaks, not the lifetime count: a single leak on Monday
        # must not page every 6 h until the container restarts.
        tag_grew = _grew("channel_tag_prefixed_replies")
        if tag_grew >= 1:
            _alert(
                "agent-output", "warning",
                f"{container.container_name}: {tag_grew} reply(ies) began with a "
                f"[channel h:mmam] tag since the last tick "
                f"(lifetime {_now('channel_tag_prefixed_replies')}). This should "
                f"be exactly zero — the annotation is an input convention and is "
                f"never the agent's own words.",
                min_interval_s=TAG_LEAK_ALERT_MIN_INTERVAL_S,
            )

        # ── day-chats: impossible rows, a failing day index, or a message
        #    written outside any day ──
        future_days = _now("future_dated_day_chats")          # gauge: current state
        future_seen = _grew("future_dated_day_chats_seen")    # counter: new this tick
        if future_days:
            _future_days_streak[key] = _future_days_streak.get(key, 0) + 1
        else:
            _future_days_streak.pop(key, None)
        fivexx_grew = _grew("day_chats_5xx")
        if fivexx_grew:
            _day_chats_5xx_streak[key] = _day_chats_5xx_streak.get(key, 0) + 1
        else:
            _day_chats_5xx_streak.pop(key, None)
        resolve_grew = _grew("day_chat_resolve_failures")
        if resolve_grew:
            _resolve_fail_streak[key] = _resolve_fail_streak.get(key, 0) + 1
        else:
            _resolve_fail_streak.pop(key, None)
        # A NEW impossible row pages at once; a row that nothing repairs pages
        # after three ticks (15 min) — the endpoint heal and the tz-learn
        # trigger both get a chance first; a 5xx or a resolve failure pages
        # when it repeats on consecutive ticks (or two at once).
        if (
            future_seen >= 1
            or _future_days_streak.get(key, 0) >= 3
            or _day_chats_5xx_streak.get(key, 0) >= 2
            or resolve_grew >= 2
            or _resolve_fail_streak.get(key, 0) >= 2
        ):
            _alert(
                "day-chats", "warning",
                f"{container.container_name}: future_dated_day_chats={future_days} "
                f"(+{future_seen} seen this tick, {_future_days_streak.get(key, 0)} "
                f"ticks unrepaired), day-chats 5xx +{fivexx_grew} this tick "
                f"({_day_chats_5xx_streak.get(key, 0)} consecutive), "
                f"day_chat resolve failures +{resolve_grew} "
                f"({_resolve_fail_streak.get(key, 0)} consecutive). A day chat "
                f"dated ahead of the user's local today — or a message written "
                f"outside any day — is invisible to the agent's context AND to "
                f"every recall path.",
                min_interval_s=SIGNAL_ALERT_MIN_INTERVAL_S,
            )
    except Exception:
        logger.warning("[MONITOR] health_signals fold failed for %s",
                       container.container_name, exc_info=True)
    return out


async def _alert_on_turn_path(turn_down: "list[ManagedContainer]") -> None:
    """One aggregated alert for agents that cannot serve a chat turn.

    Mirrors `_alert_on_db_path`: aggregated because the failure this exists
    for (a bad agent image whose init throws) hits every recreated tenant at
    once. Alerts on the FIRST observation — turn_ready=false is a total chat
    outage for that user, already at least one monitor cycle old.
    """
    global _last_turn_alert

    if not turn_down:
        return
    if len(turn_down) < 2:
        only = turn_down[0]
        if _turn_down_counts.get(only.id, 0) < TURN_ALERT_AFTER_FAILURES:
            return

    now = datetime.utcnow()
    if _last_turn_alert and (now - _last_turn_alert) < TURN_ALERT_COOLDOWN:
        return
    _last_turn_alert = now

    names = ", ".join(f"<code>{c.container_name}</code>" for c in turn_down[:3])
    if len(turn_down) > 3:
        names += f" +{len(turn_down) - 3} more"
    many = len(turn_down) >= 2
    verdict = (
        f"<b>{len(turn_down)} tenants at once → almost certainly the current "
        f"agent image.</b> Check the latest rollout; roll back the image. "
        f"Restarting containers will NOT help."
        if many else
        "<b>Single tenant</b> — check its /agent/diagnose (agent_runner "
        "check has the full init traceback)."
    )
    await _send_telegram_alert(
        f"💬 <b>Agent cannot serve chat turns</b>\n"
        f"{names}\n"
        f"Time: {now.strftime('%H:%M UTC')}\n\n"
        f"{verdict}\n\n"
        f"<i>Agents are UP and /agent/health returns 200 — they report "
        f"turn_ready=false. Every chat message answers 'Agent not "
        f"available'.</i>"
    )


async def _alert_on_db_path(db_down: "list[ManagedContainer]") -> None:
    """One aggregated alert for tenants whose database is unreachable.

    Aggregated on purpose: the failure this exists for (pgbouncer down) hits
    every tenant at once, and 55 separate messages would bury the one fact
    that matters. The count IS the diagnosis — several tenants at once means
    a shared component, and the alert says so, because "restart the
    container" is the wrong move there and is what an operator would
    otherwise reach for.
    """
    global _last_db_alert

    if not db_down:
        return

    fleet_wide = len(db_down) >= DB_ALERT_MIN_TENANTS_IMMEDIATE
    # A single tenant may just be its own database; give it one more cycle.
    # Several at once is shared infrastructure — say it now.
    if not fleet_wide:
        only = db_down[0]
        if _db_down_counts.get(only.id, 0) < DB_ALERT_AFTER_FAILURES:
            return

    now = datetime.utcnow()
    if _last_db_alert and (now - _last_db_alert) < DB_ALERT_COOLDOWN:
        return
    _last_db_alert = now

    names = ", ".join(f"<code>{c.container_name}</code>" for c in db_down[:3])
    if len(db_down) > 3:
        names += f" +{len(db_down) - 3} more"

    if fleet_wide:
        verdict = (
            f"<b>{len(db_down)} tenants at once → shared component.</b>\n"
            f"Check pgbouncer first (<code>systemctl is-active pgbouncer</code>, "
            f"port 6432), then Postgres.\n"
            f"<b>Restarting containers will NOT help</b> and risks a restart storm."
        )
    else:
        verdict = (
            "<b>Single tenant</b> — likely its own database, not shared "
            "infrastructure. Check that tenant's DB before touching the fleet."
        )

    await _send_telegram_alert(
        f"🗄️ <b>Tenant DB path unreachable</b>\n"
        f"{names}\n"
        f"Time: {now.strftime('%H:%M UTC')}\n\n"
        f"{verdict}\n\n"
        f"<i>Agents are UP and /agent/health returns 200 — they report "
        f"db_ok=false. Chat is failing for these users.</i>"
    )


async def monitor_loop():
    """Run health checks every 5 minutes forever.

    LEADER-GATED (2026-09-12, L3-1): this loop restarts containers and writes
    `managed_containers.status='error'`, and it ran unelected on both Railway
    replicas. A replica that does not hold the lease skips the tick.
    """
    logger.info("[MONITOR] Container health monitor started (every 5 min)")
    from app.services.infra_lease import acquire_lease, lease_ttl_for
    _ttl = lease_ttl_for(300)
    while True:
        if await acquire_lease("container_monitor", ttl_s=_ttl):
            try:
                await check_all_containers()
            except Exception as e:
                logger.error(f"[MONITOR] Check failed: {e}")
        await asyncio.sleep(300)  # 5 minutes
