"""Anti-abuse guards for the UNLIMITED entitlement (design §7).

Why this exists
===============
Every other account in the system has a natural ceiling: the wallet runs out.
An unlimited account is defined by not having one, so the ceiling has to be
STATED — and the moment it is stated, the question becomes what happens when
somebody reaches it. The founder's answer, taken literally:

    log → alert → pace → refuse

Never a paywall. Never a fake "limit reached". A human using the app by hand
must never reach any tier, and the last two tiers SHIP OFF
(``unlimited_throttle_enabled = False``) because a false positive against a
paying customer is worse than a large bill.

Calibration, from measurement rather than intuition
---------------------------------------------------
================================================  ==========
heaviest real user, ~2 weeks                      140 messages
mean charged chat turn                            1.975 credits
heaviest single turn ever recorded                23 credits
ENTIRE platform, 30 days                          2,003 credits
================================================  ==========

The refuse bars in ``config.py`` sit at 15,000 credits/day (≈ $150 of provider
spend, ≈ 7,600 average turns — roughly 50× the heaviest observed human DAY)
and 2,000 LLM calls/hour (one call every 1.8 s, sustained for an hour). The
NOTICE bars are ~10× the heaviest human day. Neither is reachable by a person
at a keyboard, plus a voice session, plus every automation they own.

Two arms, and the split matters
-------------------------------
* **This module is the FAST arm** — bounded in-process counters, 60-minute
  window. ``railway.json`` runs ``numReplicas 2``, so each process sees
  roughly half the traffic and this arm's effective sensitivity is HALVED. It
  is a coarse tripwire, not the authority, and it must not be read as one.
* **The AUTHORITATIVE arm is cross-replica** — credit-health invariant 5
  (``credit_health_monitor``), which is SQL over ``credit_ledger`` and
  therefore sees every replica. The 24 h and 30 d judgements belong there.

"Charged-equivalent credits" needs no JSON arithmetic anywhere: 1 credit = 1¢
of underlying provider cost by design, so ``underlying_cost_cents`` in the
ledger IS the number, and ``try_charge`` has ``amount_q`` in hand here.

What is deliberately NOT implemented
------------------------------------
The design also lists a concurrent-in-flight-turns signal and a
distinct-channels signal. Neither has an honest hook today: ``try_charge`` is
a POST-hoc report (the provider has already been paid) and nothing in the
system tracks in-flight turns per user. Inventing a tracker to feed a signal
nobody would act on is how a measurement becomes a liability. When a turn
registry exists, ``observe()`` is the place to feed it from.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Deque, Optional

from app.config import settings
from app.services import abuse_metrics
from app.services.abuse_metrics import uidp
from app.services.alerting import send_infra_alert

logger = logging.getLogger(__name__)


# Tiers, in ascending severity. `TIER_OK` is the overwhelmingly common answer
# and costs one dict lookup.
TIER_OK = "ok"
TIER_NOTICE = "notice"       # log only
TIER_ALERT = "alert"         # log + Telegram; nothing user-visible
TIER_PACE = "pace"           # admitted after a bounded delay
TIER_REFUSE = "refuse"       # last resort; truthful "slow down", never a wall

_TIER_RANK = {TIER_OK: 0, TIER_NOTICE: 1, TIER_ALERT: 2, TIER_PACE: 3, TIER_REFUSE: 4}

_HOUR_S = 3600.0
_DAY_S = 86400.0

# Bound the tracker. 512 unlimited accounts is far beyond any plausible
# subscriber count for this tier, and an unbounded dict on a hot path is a
# leak wearing a counter's clothes.
_MAX_TRACKED_USERS = 512


def _cfg(name: str, default):
    return getattr(settings, name, default)


def enabled() -> bool:
    """True iff the pace/refuse tiers may actually act.

    While False the module still RECORDS and still emits
    ``unlimited_throttle_would`` shadow events carrying the tier it would have
    applied — the same shadow-first discipline ``abuse_metrics`` was built for
    ("controls that default OFF emit a SHADOW ``*_would`` event while
    disabled — so the real-world hit / false-positive rate is measurable
    BEFORE the flag is ever flipped").
    """
    return bool(_cfg("unlimited_throttle_enabled", False))


@dataclass
class _UserWindow:
    """One unlimited account's rolling counters.

    ``calls`` holds bare timestamps for the 60-minute call-rate signal;
    ``spend`` holds (timestamp, charged-equivalent credits) for the 24 h cost
    signal. Both are pruned lazily on write, so an idle account costs nothing.
    """
    calls: Deque[float] = field(default_factory=deque)
    spend: Deque[tuple[float, float]] = field(default_factory=deque)
    # Consecutive turns at or past the PACE bar. Ramps the delay, so a brief
    # burst is barely felt and a sustained one is throttled hard.
    pace_turns: int = 0
    # Consecutive turns at or past the REFUSE bar — and the grace is counted
    # against THIS, not against `pace_turns`. That distinction is the whole
    # meaning of "last resort": an account climbing steadily through the pace
    # band for an hour has not spent its grace, because it has not yet done
    # the thing the grace is for.
    refuse_turns: int = 0
    last_tier: str = TIER_OK

    def prune(self, now: float) -> None:
        cutoff_h = now - _HOUR_S
        while self.calls and self.calls[0] < cutoff_h:
            self.calls.popleft()
        cutoff_d = now - _DAY_S
        while self.spend and self.spend[0][0] < cutoff_d:
            self.spend.popleft()

    def credits_24h(self) -> float:
        return sum(c for (_, c) in self.spend)


# LRU-bounded: OrderedDict + move_to_end, evicting the least recently touched.
_users: "OrderedDict[str, _UserWindow]" = OrderedDict()


def _window(user_id: str, *, create: bool) -> Optional[_UserWindow]:
    w = _users.get(user_id)
    if w is None:
        if not create:
            return None
        w = _UserWindow()
        _users[user_id] = w
        while len(_users) > _MAX_TRACKED_USERS:
            _users.popitem(last=False)
    _users.move_to_end(user_id)
    return w


def forget(user_id: str) -> None:
    """Drop one account's window, because its entitlement just ended.

    Nothing else evicts before the 24h/1h cutoffs, so a lapsed, revoked or
    downgraded account kept its counters — and with the throttle enabled its
    next `check_balance` could still be paced or refused with
    REASON_RATE_LIMITED off pre-downgrade traffic. `credit_service`'s own
    comment says "no free, legacy-paid or admin account can ever have an
    entry", and that was true only at WRITE time.

    Called from `credit_service.downgrade_to_free`, which is the one funnel
    every end-of-entitlement path already goes through (Apple EXPIRED /
    REVOKE / REFUND, the reconciler, a revoked or lapsed grant).
    """
    _users.pop(user_id, None)


def reset_for_tests() -> None:
    """Drop all in-process state. Tests only — there is no runtime caller."""
    _users.clear()


def _classify(calls_1h: int, credits_24h: float) -> str:
    """The tier these two numbers earn. Pure; no state, no side effects."""
    refuse = (
        calls_1h >= int(_cfg("unlimited_calls_1h_refuse", 2000))
        or credits_24h >= float(_cfg("unlimited_credits_24h_refuse", 15000.0))
    )
    if refuse:
        return TIER_REFUSE
    pace = (
        calls_1h >= int(_cfg("unlimited_calls_1h_alert", 900))
        or credits_24h >= float(_cfg("unlimited_credits_24h_alert", 5000.0))
    )
    if pace:
        # The alert bar and the pace bar are the same numbers deliberately:
        # past it we both TELL the operator and slow the account down, which
        # is one decision, not two.
        return TIER_PACE
    notice = (
        calls_1h >= int(_cfg("unlimited_calls_1h_notice", 300))
        or credits_24h >= float(_cfg("unlimited_credits_24h_notice", 1500.0))
    )
    return TIER_NOTICE if notice else TIER_OK


def pace_delay_s(tier: str, pace_turns: int) -> float:
    """Seconds to hold a turn before admitting it.

    Ramps with how long the account has been over the pace bar so a brief
    burst is barely felt and a sustained one is throttled hard, capped so a
    delay stays recoverable. Zero for every tier below PACE.
    """
    if _TIER_RANK.get(tier, 0) < _TIER_RANK[TIER_PACE]:
        return 0.0
    cap = float(_cfg("unlimited_pace_max_delay_s", 20.0))
    return min(cap, 0.5 * max(1, pace_turns))


async def observe(
    user_id: str,
    credits_quoted: Decimal | float,
    *,
    event_type: str = "chat_message",
) -> str:
    """Record one served-without-debit event and return the tier it earns.

    Called from ``try_charge``'s unlimited branch AFTER the ledger flush, and
    wrapped by the caller — a measurement that can break a charge is worse
    than no measurement.

    NEVER sleeps and never raises. The pace delay it computes is applied at
    ADMISSION (``check_balance``), not here: this runs inside a transaction
    holding ``SELECT … FOR UPDATE`` on the balance row, and sleeping under a
    row lock would turn a throttle into an outage.
    """
    now = time.time()
    w = _window(user_id, create=True)
    assert w is not None  # create=True
    w.prune(now)
    w.calls.append(now)
    try:
        c = float(credits_quoted)
    except (TypeError, ValueError):
        c = 0.0
    w.spend.append((now, c))

    calls_1h = len(w.calls)
    credits_24h = w.credits_24h()
    tier = _classify(calls_1h, credits_24h)
    rank = _TIER_RANK[tier]
    w.pace_turns = w.pace_turns + 1 if rank >= _TIER_RANK[TIER_PACE] else 0
    w.refuse_turns = w.refuse_turns + 1 if rank >= _TIER_RANK[TIER_REFUSE] else 0
    w.last_tier = tier

    if tier == TIER_OK:
        return tier

    live = enabled()
    # Both a real and a shadow line carry the SAME fields, so the shadow
    # series is directly comparable to what enforcement would have produced.
    abuse_metrics.emit(
        "unlimited_usage_notice" if live else "unlimited_throttle_would",
        uid=uidp(user_id),
        tier=tier,
        calls_1h=calls_1h,
        credits_24h=round(credits_24h, 2),
        pace_turns=w.pace_turns or None,
        refuse_turns=w.refuse_turns or None,
        event_type=event_type,
        enforcing=live,
    )

    if _TIER_RANK[tier] >= _TIER_RANK[TIER_ALERT]:
        # Subject-scoped so one heavy account cannot suppress an alert about
        # another (alerting.py keys the rate limit on (category, subject)).
        await send_infra_alert(
            "unlimited-abuse", "warning",
            f"Unlimited account {uidp(user_id)} is at tier {tier}: "
            f"{calls_1h} LLM call(s) in the last hour, "
            f"{credits_24h:.0f} charged-equivalent credits in 24h "
            f"(this replica only — see credit-health invariant 5 for the "
            f"fleet figure). "
            + (
                f"Pacing is LIVE; turn {w.pace_turns} past the pace bar, "
                f"{w.refuse_turns} past the refuse bar."
                if live else
                "Throttle is OFF — nothing was slowed or refused; this is a "
                "shadow observation."
            ),
            subject=uidp(user_id),
            min_interval_s=3600,
        )
    return tier


def admission_verdict(user_id: str) -> tuple[str, float]:
    """``(tier, delay_seconds)`` for an account about to be admitted.

    A PURE in-memory read — no DB, no I/O — so ``check_balance`` can consult
    it before it touches the database. Returns ``(TIER_OK, 0.0)`` for any user
    this module has never observed, which is every free, legacy-paid and
    unmetered account in the system: only ``try_charge``'s unlimited branch
    ever calls :func:`observe`, so only unlimited accounts have an entry.
    """
    if not enabled():
        return TIER_OK, 0.0
    w = _users.get(user_id)
    if w is None:
        return TIER_OK, 0.0
    w.prune(time.time())
    tier = _classify(len(w.calls), w.credits_24h())
    if _TIER_RANK[tier] < _TIER_RANK[TIER_PACE]:
        return tier, 0.0
    grace = int(_cfg("unlimited_throttle_grace_turns", 50))
    if tier == TIER_REFUSE and w.refuse_turns <= grace:
        # Past the refuse bar but still inside the grace window: pace it.
        # A delay is recoverable and self-limiting; a denial is not, and this
        # is the ordering that makes refusal a genuine last resort rather than
        # a threshold with a dramatic name. The grace counts turns past the
        # REFUSE bar specifically — an account that spent an hour climbing
        # through the pace band has not used any of it.
        return TIER_PACE, pace_delay_s(TIER_PACE, w.pace_turns)
    if tier == TIER_REFUSE:
        return TIER_REFUSE, 0.0
    return TIER_PACE, pace_delay_s(tier, w.pace_turns)


async def apply_admission(user_id: str) -> bool:
    """Pace or refuse one admission. Returns True iff the turn may proceed.

    Sleeping here holds the caller's pooled DB session for up to
    ``unlimited_pace_max_delay_s``. That cost is accepted knowingly: the only
    account that can reach this branch is one making thousands of calls an
    hour, slowing it down is the entire point, and the alternative — refusing
    — is the thing this ladder exists to avoid. The delay is capped for
    exactly that reason.
    """
    tier, delay = admission_verdict(user_id)
    if tier == TIER_REFUSE:
        abuse_metrics.emit(
            "unlimited_throttle_refused", uid=uidp(user_id), tier=tier,
        )
        await send_infra_alert(
            "unlimited-abuse-refusal", "critical",
            f"Unlimited account {uidp(user_id)} was REFUSED a turn by the "
            f"rate ladder after exhausting the pace grace. What they see: on "
            f"web, the credit card with its CTA suppressed (cta_hidden); on "
            f"mobile and voice, a plain error bubble carrying the 'slow down' "
            f"line — App Store build 109 has no cta_hidden, so a "
            f"credit_exhausted frame there would render its hardest paywall, "
            f"and the refusal deliberately does not use that channel. No "
            f"billing language on any of them. "
            f"If this is a real person rather than a runaway loop, the bars in "
            f"config.py are wrong and should be raised — a paying customer "
            f"being told to wait is a worse outcome than a large bill.",
            subject=uidp(user_id),
            min_interval_s=900,
        )
        return False
    if delay > 0:
        abuse_metrics.emit(
            "unlimited_throttle_paced", uid=uidp(user_id),
            tier=tier, delay_s=round(delay, 2),
        )
        await asyncio.sleep(delay)
    return True
