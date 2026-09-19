"""Apple subscription reconciler — the backstop for a push channel that has
already dropped a message.

Why this exists
===============
Everything the platform knows about an Apple subscription arrives by push:
``/subscribe/verify`` when the client asks, and App Store Server Notifications
V2 when Apple decides to tell us. There is no polling, no ``expires_date``
scan, and nothing has ever called ``get_all_subscription_statuses``. Three
facts make that untenable:

* **The channel drops messages.** ``apple_subscriptions`` holds a Sandbox
  Elite row that has read ``status='active'`` for 2.7 months past its
  2026-06-24 expiry with ``last_notification_uuid`` NULL. No notification ever
  arrived, and nothing ever looked.
* **``DID_RENEW`` has never executed in production.** The first three renewals
  land 2026-10-01. Whatever that path does, it will do it unobserved.
* **A handler error is lost forever.** ``iap.py`` acks 200 on any exception
  (deliberately — Apple's retry would replay a non-idempotent
  ``apply_plan_change``), so a raise at renewal is a notification that never
  happened. That endpoint now alerts; this loop is what actually repairs it.

The requirement it closes is "drop back to Free at the end of the paid period
happens automatically, on every channel, with no manual step". The
notification handler is the mechanism; this is the guarantee.

Shape
=====
**Pass 1 — a DB scan, always, no network.** Classifies every
``apple_subscriptions`` row and every ``credit_balances`` row on the Unlimited
plan into the seven classes below.

**Pass 2 — Apple, only for the interesting rows**, bounded by
``apple_reconcile_max_api_calls`` and ordered oldest-``updated_at`` first so
nothing starves. Apple is authoritative; its ``status`` and ``expiresDate``
win over the mirror.

Two rules that are not negotiable
=================================
* **Reconcile against ``expires_date``, never ``status``.** The mirror's
  status column records the last notification we processed, not the truth,
  and the Sandbox Elite is the standing proof that it lies for months at a
  time. ``status`` is only ever an INPUT to classification ("the mirror
  claims active"), never the answer.
* **Sandbox rows are never mutated in production.** They are scanned and
  reported — the Sandbox Elite's drift is exactly the evidence this loop
  exists to surface — but a sandbox subscription renews every few minutes and
  expires in an hour, and letting Apple's test environment move a real
  ``credit_balances`` row is a worse failure than the drift it would fix.
  ``apple_reconcile_sandbox_apply`` (default False) is the deliberate opt-in.
  DEVIATION from design §5.3.4, which expected the first applied pass to
  downgrade the Sandbox Elite; it will now be REPORTED every pass and left
  alone until an operator decides. Write this back into the design.
  That report is ONE aggregate warning, in every mode — see the fold in
  ``_raise_alarms`` — and, because the condition is static rather than
  flapping, it pages at most once per 24 h
  (``SANDBOX_AGGREGATE_MIN_INTERVAL_S``) rather than on every pass.

Idempotency
===========
Every action is a convergence to Apple's state, not a delta. Mirror writes
compare-then-assign. ``activate_subscription``/``downgrade_to_free`` route
through ``apply_plan_change``'s absolute branch for Unlimited transitions,
which is assignment — a second pass computes zero deltas and writes no ledger
rows. Grant lapses are guarded on ``lapsed_at IS NULL``. A converged database
therefore produces a pass with zero writes and zero Apple calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import (
    APPLE_SUB_ACTIVE, APPLE_SUB_BILLING_RETRY, APPLE_SUB_EXPIRED,
    APPLE_SUB_GRACE, APPLE_SUB_REVOKED,
    AppleSubscription, CreditBalance, PlatformSetting, User,
)
from app.db.plan_catalog import LEGACY_SUB_PRODUCT_IDS, UNLIMITED_PLAN_ID
from app.services import apple_iap_service, entitlement
from app.services.abuse_metrics import uidp
from app.services.alerting import send_infra_alert
from app.services.apple_iap_service import (
    APPLE_STATUS_ACTIVE, APPLE_STATUS_BILLING_RETRY, APPLE_STATUS_EXPIRED,
    APPLE_STATUS_GRACE, APPLE_STATUS_REVOKED, IapVerificationError,
)
from app.services.credit_service import PLAN_SOURCE_APPLE, credit_service
from app.services.infra_lease import acquire_lease, lease_ttl_for, release_lease

logger = logging.getLogger(__name__)

ENV_PRODUCTION = "Production"
ENV_SANDBOX = "Sandbox"

LAST_RUN_SETTING_KEY = "billing.apple_reconcile_last_run"
GRANDFATHER_RECEIPT_KEY = "billing.grandfather_unlimited_receipt"

# Classes. Named as constants because they are the summary's keys, the log
# line's vocabulary and the tests' assertions all at once.
CLASS_OVERDUE = "overdue"
CLASS_RESURRECTED = "resurrected"
CLASS_STALE = "stale"
CLASS_ORPHAN = "orphan_entitlement"
CLASS_MISSING = "missing_entitlement"
CLASS_GRANT_LAPSED = "grant_lapsed"
CLASS_CROSSGRADE = "legacy_crossgrade"

# Only these need to ask Apple. `missing_entitlement` is answerable from the
# mirror alone (a live Production row whose expires_date is in the future is
# already the strongest statement the DB can make, and re-asking would spend
# an API call to confirm a row we would act on either way).
_NEEDS_APPLE = (CLASS_OVERDUE, CLASS_RESURRECTED, CLASS_STALE, CLASS_ORPHAN)

# Apple's status enum → our mirror's status column.
_APPLE_STATUS_TO_MIRROR = {
    APPLE_STATUS_ACTIVE: APPLE_SUB_ACTIVE,
    APPLE_STATUS_EXPIRED: APPLE_SUB_EXPIRED,
    APPLE_STATUS_BILLING_RETRY: APPLE_SUB_BILLING_RETRY,
    APPLE_STATUS_GRACE: APPLE_SUB_GRACE,
    APPLE_STATUS_REVOKED: APPLE_SUB_REVOKED,
}
# Apple statuses that still ENTITLE. Billing retry does too: the customer's
# card failed but Apple has not ended the subscription, and _SUB_DOWNGRADE_TYPES
# deliberately waits for EXPIRED. Reading it as "not entitled" here would make
# this loop downgrade people the notification handler is deliberately keeping.
_ENTITLING_APPLE_STATUSES = frozenset({
    APPLE_STATUS_ACTIVE, APPLE_STATUS_GRACE, APPLE_STATUS_BILLING_RETRY,
})
_DEAD_APPLE_STATUSES = frozenset({APPLE_STATUS_EXPIRED, APPLE_STATUS_REVOKED})

# The mirror statuses that CLAIM the subscription is alive. A row claiming one
# of these past its expires_date is `overdue` — a lapse we were never told
# about.
_MIRROR_CLAIMS_ALIVE = frozenset({
    APPLE_SUB_ACTIVE, APPLE_SUB_GRACE, APPLE_SUB_BILLING_RETRY,
})

# An expires_date can legitimately sit a little in the past between Apple's
# renewal and its notification. Only a row that is overdue by more than this
# is evidence of a dropped message.
_OVERDUE_SLACK = timedelta(hours=1)

# The Sandbox aggregate's OWN rate-limit window, handed to `send_infra_alert`
# per alert; every other alert this module raises keeps alerting.py's default.
#
# Why 24 h. The aggregate's condition is STATIC by policy, not flapping: the
# Sandbox Elite has read `status='active'` since its 2026-06-24 expiry and
# `_may_mutate` forbids this loop from ever writing it, so every pass of a
# 6-hourly reconciler re-derives a byte-identical body. At the 600 s default
# that is four identical Telegram messages a day, forever, about a
# TestFlight/Sandbox row nobody needs to act on within hours — and an alarm
# that arrives on a schedule rather than on a change is the one that gets the
# whole channel muted. A day is the longest interval that still guarantees the
# operator sees the row at least once a day, and alerting.py counts the
# suppressed repeats and appends "(+N similar suppressed in the last window)"
# to the next delivered message, so nothing is dropped. It is deliberately NOT
# applied to anything else: a Production finding is evidence of a dropped App
# Store notification, which is the failure this loop exists to catch.
#
# KNOWN RESIDUE: alerting.py's window is module state, per PROCESS by design
# (fleet-wide de-duplication is the infra lease, not that file). A redeploy, a
# replica restart, or the `apple-reconciler` lease moving to the other replica
# starts a fresh window, so an unchanged condition can still page once more
# inside the 24 h. Bounded and acceptable; making it durable means a DB row per
# alert key, which is a bigger mechanism than the noise it would remove.
SANDBOX_AGGREGATE_MIN_INTERVAL_S = 86400


def _cfg(name: str, default):
    return getattr(settings, name, default)


@dataclass
class Finding:
    cls: str
    original_txn: Optional[str] = None
    user_id: Optional[str] = None
    environment: str = ENV_PRODUCTION
    detail: str = ""
    # Filled in pass 2.
    apple: Optional[object] = None
    actions: list[str] = field(default_factory=list)


@dataclass
class ReconcilePass:
    """One pass's result. Returned rather than only logged so the tests can
    assert on it and so ``apply=False`` has something to be read from."""
    ran_at: datetime
    apply: bool
    findings: list[Finding] = field(default_factory=list)
    api_calls: int = 0
    api_failures: int = 0
    api_deferred: int = 0
    applied: int = 0
    would: int = 0
    errors: list[str] = field(default_factory=list)
    # (category, level, message, subject, min_interval_s) — flushed AFTER the
    # commit so a slow Telegram cannot hold a write transaction open.
    #
    # `min_interval_s` is None for "whatever alerting.py's default is", so the
    # cadence of every ordinary alert keeps living in exactly one place; only
    # the Sandbox aggregate sets it (SANDBOX_AGGREGATE_MIN_INTERVAL_S). The
    # fifth element is additive on purpose — every reader in tests/ indexes
    # (`a[0]`, `a[1]`, `a[2]`) rather than unpacking, so widening the tuple
    # cannot break one.
    alerts: list[tuple[str, str, str, Optional[str], Optional[int]]] = field(
        default_factory=list)

    @property
    def classes(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.cls] = out.get(f.cls, 0) + 1
        return out

    def summary(self) -> dict:
        return {
            "ran_at": self.ran_at.isoformat(),
            "apply": self.apply,
            "classes": self.classes,
            "api_calls": self.api_calls,
            "api_failures": self.api_failures,
            "api_deferred": self.api_deferred,
            "applied": self.applied,
            "would": self.would,
            "errors": self.errors[:20],
        }


# ── pass 1: the DB scan ──────────────────────────────────────────────────


async def _grandfathered_txn_allowlist(db: AsyncSession) -> Optional[set[str]]:
    """The original transaction ids the grandfather command actually stamped,
    or None when it has never run.

    None is not an empty set and the difference matters: BEFORE the
    grandfather, every legacy Production row is legitimately unmarked and the
    cross-grade alarm must stay silent. AFTER it, an unmarked legacy row is a
    subscription that appeared or changed outside the cutover — which is
    exactly the "cross-graded down to $9.90 Starter and kept Unlimited" route
    design §3.4 could not close in App Store Connect.
    """
    row = await db.get(PlatformSetting, GRANDFATHER_RECEIPT_KEY)
    if row is None or not (row.value or "").strip():
        return None
    try:
        receipt = json.loads(row.value)
    except Exception as e:
        logger.warning("[apple-reconcile] grandfather receipt unparseable: %s", e)
        return None
    entries = receipt.get("entries") or receipt.get("users") or []
    out = {
        str(e.get("original_transaction_id"))
        for e in entries
        if isinstance(e, dict) and e.get("original_transaction_id")
    }
    return out


async def scan(db: AsyncSession, *, now: Optional[datetime] = None) -> list[Finding]:
    """Classify the whole subscription mirror plus every Unlimited balance.

    Read-only and network-free. This is the half that always runs, including
    when Apple is unreachable — "we could not ask" must still produce a
    readable picture of what the database believes.
    """
    now = now or datetime.utcnow()
    stale_days = int(_cfg("apple_reconcile_stale_days", 40))
    scan_sandbox = bool(_cfg("apple_reconcile_scan_sandbox", True))

    subs = list((await db.execute(select(AppleSubscription))).scalars().all())
    allowlist = await _grandfathered_txn_allowlist(db)
    findings: list[Finding] = []
    live_txns_by_user: dict[str, list[AppleSubscription]] = {}

    for sub in subs:
        env = sub.environment or ENV_PRODUCTION
        if env != ENV_PRODUCTION and not scan_sandbox:
            continue

        expires = sub.expires_date
        status = sub.status

        # A row counts as LIVE for the missing-entitlement class only if its
        # own clock says so AND the mirror has not recorded a terminal event.
        #
        # That second half is the one place `status` is read, and the
        # direction is what makes it safe: a REVOKE (family-sharing removal, a
        # refund) leaves expires_date in the FUTURE while ending the
        # entitlement, so clock-alone would have this loop re-granting the
        # plan the notification handler correctly took away. Status is only
        # ever allowed to argue that something is DEAD, never that it is
        # alive — and a row it calls dead is classified `resurrected` below
        # and put to Apple, who is the authority either way. The terminal set
        # is `entitlement.TERMINAL_APPLE_STATUSES`, shared with the grandfather
        # predicate so the two cannot disagree about what "dead" means.
        if (expires is not None and expires > now
                and status not in entitlement.TERMINAL_APPLE_STATUSES):
            live_txns_by_user.setdefault(sub.user_id, []).append(sub)

        # ── overdue: the mirror claims alive, Apple's own clock says it ended.
        # `expires_date`, never `status` — status is the thing being doubted.
        if (expires is not None
                and expires < now - _OVERDUE_SLACK
                and status in _MIRROR_CLAIMS_ALIVE):
            findings.append(Finding(
                cls=CLASS_OVERDUE, original_txn=sub.original_transaction_id,
                user_id=sub.user_id, environment=env,
                detail=(f"mirror status={status!r} but expires_date="
                        f"{expires.isoformat()} is "
                        f"{(now - expires).days}d in the past"),
            ))
        # ── resurrected: the mirror says dead, the clock says it is not.
        elif (expires is not None and expires > now
                and status in entitlement.TERMINAL_APPLE_STATUSES):
            findings.append(Finding(
                cls=CLASS_RESURRECTED, original_txn=sub.original_transaction_id,
                user_id=sub.user_id, environment=env,
                detail=(f"mirror status={status!r} but expires_date="
                        f"{expires.isoformat()} is in the future"),
            ))
        # ── stale: active and silent for longer than a monthly cycle can be.
        elif (status == APPLE_SUB_ACTIVE
                and sub.updated_at is not None
                and sub.updated_at < now - timedelta(days=stale_days)):
            findings.append(Finding(
                cls=CLASS_STALE, original_txn=sub.original_transaction_id,
                user_id=sub.user_id, environment=env,
                detail=(f"active with no notification since "
                        f"{sub.updated_at.isoformat()} (> {stale_days}d)"),
            ))

        # ── legacy cross-grade. Independent of the classes above: a row can be
        # perfectly converged AND be a subscription that should not exist on a
        # legacy product any more.
        if (env == ENV_PRODUCTION
                and sub.product_id in LEGACY_SUB_PRODUCT_IDS
                and allowlist is not None
                and sub.grandfathered_at is None
                and sub.original_transaction_id not in allowlist):
            findings.append(Finding(
                cls=CLASS_CROSSGRADE, original_txn=sub.original_transaction_id,
                user_id=sub.user_id, environment=env,
                detail=(f"Production row on legacy product {sub.product_id!r} "
                        f"is not in the grandfather receipt"),
            ))
        # The arm above EXCLUDES grandfathered rows, which are the only rows
        # that can cross-grade silently: the txn id never changes, so it stays
        # in the receipt, and the mirror converges to the new product with
        # nothing left disagreeing. The evidence is already in hand — the
        # product they hold versus the product they were grandfathered on.
        elif (env == ENV_PRODUCTION
                and sub.grandfathered_at is not None
                and sub.grandfathered_product_id is not None
                and sub.product_id != sub.grandfathered_product_id):
            findings.append(Finding(
                cls=CLASS_CROSSGRADE, original_txn=sub.original_transaction_id,
                user_id=sub.user_id, environment=env,
                detail=(f"grandfathered on {sub.grandfathered_product_id!r} but "
                        f"now holds {sub.product_id!r} — the grandfather does "
                        f"not follow a product change, so this subscription "
                        f"resolves to its new tier from the next renewal"),
            ))

    # ── missing_entitlement: paid and not entitled. The Parmida class.
    for user_id, rows in live_txns_by_user.items():
        prod_rows = [s for s in rows if (s.environment or ENV_PRODUCTION) == ENV_PRODUCTION]
        if not prod_rows:
            continue
        balance = await db.get(CreditBalance, user_id)
        if balance is None:
            continue
        targets = []
        for s in prod_rows:
            try:
                targets.append((s, apple_iap_service.plan_for_subscription(s.product_id, s)))
            except KeyError:
                logger.warning(
                    "[apple-reconcile] unknown product %r on a live Production "
                    "row for user=%s; skipping", s.product_id, uidp(user_id),
                )
        if not targets:
            continue
        # More than one live Production subscription for one account is not a
        # shape the product can produce (the 409 exclusivity check refuses it),
        # so prefer the one that entitles the MOST rather than inventing a
        # tie-break: Unlimited first, then the newest expiry.
        targets.sort(key=lambda t: (t[1] != UNLIMITED_PLAN_ID,
                                    -(t[0].expires_date or now).timestamp()))
        sub, target_plan = targets[0]
        if balance.plan_id != target_plan:
            findings.append(Finding(
                cls=CLASS_MISSING, original_txn=sub.original_transaction_id,
                user_id=user_id, environment=ENV_PRODUCTION,
                detail=(f"live Production {sub.product_id!r} entitles "
                        f"{target_plan!r} but the balance says "
                        f"{balance.plan_id!r}"),
            ))

    # ── orphan_entitlement: the Unlimited stamp with nothing behind it.
    unlimited_balances = list((await db.execute(
        select(CreditBalance).where(
            CreditBalance.plan_id == UNLIMITED_PLAN_ID,
            CreditBalance.plan_source == PLAN_SOURCE_APPLE,
        )
    )).scalars().all())
    for balance in unlimited_balances:
        user = await db.get(User, balance.user_id)
        if user is not None and getattr(user, "role", None) == "admin":
            # Already unlimited via role; the stamp is redundant, not orphaned.
            continue
        if await entitlement.has_live_apple_entitlement(db, balance.user_id, now=now):
            continue
        if await entitlement.live_grant_for(db, balance.user_id, now=now) is not None:
            continue
        anchor = next(
            (s for s in subs
             if s.user_id == balance.user_id
             and (s.environment or ENV_PRODUCTION) == ENV_PRODUCTION),
            None,
        )
        findings.append(Finding(
            cls=CLASS_ORPHAN,
            original_txn=anchor.original_transaction_id if anchor else None,
            user_id=balance.user_id, environment=ENV_PRODUCTION,
            detail=("balance holds the unlimited plan with plan_source='apple' "
                    "but no live Production subscription and no live grant"),
        ))

    return findings


# ── pass 2 + corrections ─────────────────────────────────────────────────


def _may_mutate(environment: str) -> bool:
    """Sandbox rows are reported, never applied — unless an operator opted in.

    See the module docstring: a sandbox subscription's lifetime is measured in
    minutes, and letting it move a production ``credit_balances`` row trades a
    visible drift for an invisible one.

    Also the key ``_raise_alarms`` folds on, which is why it must stay a
    function of the CONFIGURATION rather than of what a given pass wrote:
    observe-only writes nothing at all, and the alerts still have to tell a
    Sandbox row that was never going to be touched apart from a Production one
    that was.
    """
    if (environment or ENV_PRODUCTION) == ENV_PRODUCTION:
        return True
    return bool(_cfg("apple_reconcile_sandbox_apply", False))


async def _ask_apple(txn: str, environment: str, result: ReconcilePass):
    """One App Store Server API read. Returns Apple's answer, or None for both
    "Apple does not know this transaction" and "we could not ask" — the two
    are distinguished by ``result.api_failures``, never by the return value,
    because acting on either would be acting on nothing."""
    try:
        answer = await apple_iap_service.fetch_subscription_status(txn, environment)
        result.api_calls += 1
        return answer
    except IapVerificationError as e:
        result.api_calls += 1
        result.api_failures += 1
        result.errors.append(f"{txn}: {e}")
        logger.warning(
            "[apple-reconcile] could not read Apple for orig_txn=%s: %s", txn, e,
        )
    except Exception as e:  # pragma: no cover - defensive
        result.api_calls += 1
        result.api_failures += 1
        result.errors.append(f"{txn}: {type(e).__name__}: {e}")
        logger.exception("[apple-reconcile] Apple read blew up")
    return None


async def _converge_mirror(db, sub: AppleSubscription, apple, *, apply: bool) -> list[str]:
    """Compare-then-assign the mirror to Apple's answer. Returns the fields
    that differed (empty on a converged row — that is what makes a second pass
    write nothing).

    ``last_notification_uuid`` is NEVER touched: the sweep is not a
    notification, and overwriting it would defeat the dedup that protects the
    non-idempotent ``apply_plan_change`` from Apple's at-least-once delivery.
    """
    changed: list[str] = []
    want_status = _APPLE_STATUS_TO_MIRROR.get(apple.status)
    if want_status and sub.status != want_status:
        changed.append(f"status {sub.status}→{want_status}")
        if apply:
            sub.status = want_status
    if apple.expires_date is not None and sub.expires_date != apple.expires_date:
        changed.append(f"expires_date {sub.expires_date}→{apple.expires_date}")
        if apply:
            sub.expires_date = apple.expires_date
    if apple.product_id and sub.product_id != apple.product_id:
        changed.append(f"product_id {sub.product_id}→{apple.product_id}")
        if apply:
            sub.product_id = apple.product_id
    if apple.auto_renew_status is not None and sub.auto_renew_status != apple.auto_renew_status:
        changed.append(f"auto_renew {sub.auto_renew_status}→{apple.auto_renew_status}")
        if apply:
            sub.auto_renew_status = apple.auto_renew_status
    if (apple.auto_renew_product_id is not None
            and sub.auto_renew_product_id != apple.auto_renew_product_id):
        changed.append("auto_renew_product_id")
        if apply:
            sub.auto_renew_product_id = apple.auto_renew_product_id
    if changed and apply:
        sub.updated_at = datetime.utcnow()
        await db.flush()
    return changed


async def _entitle(db, finding: Finding, sub, target_plan: str, expires, *, apply: bool) -> bool:
    """Bring the balance up to the plan the subscription entitles."""
    balance = await db.get(CreditBalance, finding.user_id)
    if balance is None:
        return False
    if balance.plan_id == target_plan:
        # Converged on the plan. The only thing left worth correcting is a
        # plan_source that does not say 'apple': NULL on a paid plan reads as
        # stripe server-side and 409-BLOCKS that account from ever buying on
        # iOS, and the mobile client renders no call to action for it.
        if balance.plan_source != PLAN_SOURCE_APPLE:
            finding.actions.append(f"plan_source {balance.plan_source!r}→'apple'")
            if apply:
                balance.plan_source = PLAN_SOURCE_APPLE
                await db.flush()
            return True
        return False
    finding.actions.append(f"entitle {balance.plan_id!r}→{target_plan!r}")
    if apply:
        await credit_service.activate_subscription(
            db, finding.user_id, target_plan, PLAN_SOURCE_APPLE,
            expires or (datetime.utcnow() + timedelta(days=30)),
        )
    return True


async def _de_entitle(db, finding: Finding, reason: str, *, apply: bool) -> bool:
    """Return an account whose subscription Apple says is dead to free — and
    lapse every admin override that subscription was sponsoring, in the same
    transaction, exactly as the notification handler does."""
    balance = await db.get(CreditBalance, finding.user_id)
    if balance is None or balance.plan_id == "free":
        return False
    if (balance.plan_source or None) != PLAN_SOURCE_APPLE:
        # Not ours to end. A Stripe-sourced or legacy-NULL paid plan is not
        # evidence about an Apple subscription, and downgrading it here would
        # be this loop inventing a lapse nobody reported.
        logger.info(
            "[apple-reconcile] user=%s is on %r via plan_source=%r; leaving it "
            "to its own source", uidp(finding.user_id), balance.plan_id,
            balance.plan_source,
        )
        return False
    finding.actions.append(f"downgrade {balance.plan_id!r}→'free' ({reason})")
    if apply:
        await credit_service.downgrade_to_free(db, finding.user_id, reason)
        if finding.original_txn:
            await entitlement.lapse_grants_for_sponsor(
                db, original_txn_id=finding.original_txn, reason=reason,
            )
    return True


async def reconcile_once(
    db: AsyncSession, *, now: Optional[datetime] = None,
    apply: Optional[bool] = None, max_api_calls: Optional[int] = None,
) -> ReconcilePass:
    """One full pass over a caller-supplied session. Does NOT commit.

    ``apply=False`` writes nothing and records every action it WOULD have
    taken on the finding, which is the mode the loop ships in until an
    operator has read a clean pass.
    """
    now = now or datetime.utcnow()
    apply = bool(_cfg("apple_reconcile_apply", False)) if apply is None else apply
    budget = int(max_api_calls if max_api_calls is not None
                 else _cfg("apple_reconcile_max_api_calls", 100))

    result = ReconcilePass(ran_at=now, apply=apply)
    result.findings = await scan(db, now=now)

    # ── pass 2. Oldest first so a permanently-unreadable row cannot starve
    # the rest, and the leftovers roll into the next pass.
    subs_by_txn = {
        s.original_transaction_id: s
        for s in (await db.execute(select(AppleSubscription))).scalars().all()
    }
    need = [f for f in result.findings if f.cls in _NEEDS_APPLE and f.original_txn]

    def _age(txn: str):
        s = subs_by_txn.get(txn)
        stamp = s.updated_at if (s is not None and s.updated_at is not None) else now
        # Production ahead of Sandbox at equal age. Sandbox rows are still
        # READ — an honest "Apple confirms this is dead" is the report — but a
        # noisy sandbox account must never spend the budget a real payer needs.
        env = (s.environment if s is not None else ENV_PRODUCTION) or ENV_PRODUCTION
        return (env != ENV_PRODUCTION, stamp)

    # Budget is spent per SUBSCRIPTION, not per finding: one subscription can
    # legitimately produce two findings at once (an `overdue` mirror row and
    # the `orphan_entitlement` its lapse left behind), and asking Apple the
    # same question twice would halve the budget for no new information.
    wanted: dict[str, str] = {}
    for f in need:
        wanted.setdefault(f.original_txn, f.environment)
    order = sorted(wanted, key=_age)
    if len(order) > budget:
        result.api_deferred = len(order) - budget
        order = order[:budget]

    answers: dict[str, object] = {}
    for txn in order:
        answers[txn] = await _ask_apple(txn, wanted[txn], result)
    for f in need:
        f.apple = answers.get(f.original_txn)

    # ── corrections ──
    #
    # Each finding gets its own SAVEPOINT. Without one the try/except below is
    # a lie on Postgres: the first failed flush aborts the whole transaction,
    # every later finding then raises PendingRollbackError on its first
    # statement, and the pass rolls back wholesale — including corrections that
    # had already succeeded, the summary heartbeat, and the CRITICAL "a paying
    # customer is locked out" alert this loop exists to raise.
    failed_txns: set[str] = set()
    for finding in result.findings:
        try:
            async with db.begin_nested():
                changed = await _apply_finding(
                    db, finding, subs_by_txn, now=now, apply=apply,
                )
        except Exception as e:
            result.errors.append(
                f"{finding.cls}/{finding.original_txn}: {type(e).__name__}: {e}"
            )
            logger.exception("[apple-reconcile] correcting %s failed", finding.cls)
            if finding.original_txn:
                failed_txns.add(finding.original_txn)
            continue
        if not changed:
            continue
        # A Sandbox finding is described but never written, so counting it as
        # `applied` would make the summary — the operator's only proof of what
        # this pass did — say it changed something it deliberately did not.
        if apply and "sandbox:not_applied" not in finding.actions:
            result.applied += 1
        else:
            result.would += 1

    # ── grants: the class that needs no Apple call, and the LAST thing this
    # pass does.
    #
    # It ran FIRST until it was proved wrong. `_sponsor_is_live` reads the
    # apple_subscriptions mirror, and the whole reason this loop exists is that
    # the mirror lies — a dropped DID_RENEW leaves a live subscription reading
    # `expires_date` in the past. Sweeping before pass 2 therefore judged the
    # sponsor against the very row the same pass was about to correct: the
    # grantee was permanently lapsed and dropped to free in the same pass that
    # correctly restored the sponsor, and nothing re-opens a lapsed grant.
    #
    # Order alone is not enough, so `defer_sponsors` carries the subscriptions
    # whose liveness is STILL unresolved: budget-deferred, unreadable at Apple,
    # or (in observe-only) simply not written. A grant sponsored by one of
    # those is skipped this pass and reconsidered next — a late lapse costs
    # nothing, a wrong one is irreversible.
    # `failed_txns` is unioned in whole, not intersected with `wanted`: a
    # correction that RAISED left its subscription in an unknown state
    # whether or not this pass needed to ask Apple about it.
    unresolved = {
        txn for txn in wanted
        if not (apply and answers.get(txn) is not None)
    } | failed_txns
    try:
        swept = await entitlement.sweep_dead_grants(
            db, now=now, apply=apply, defer_sponsors=unresolved,
        )
        for grant_id, why in swept:
            result.findings.append(Finding(
                cls=CLASS_GRANT_LAPSED, detail=f"grant {grant_id}: {why}",
            ))
            if apply:
                result.applied += 1
            else:
                result.would += 1
    except Exception as e:
        result.errors.append(f"grant sweep: {type(e).__name__}: {e}")
        logger.exception("[apple-reconcile] grant sweep failed; continuing")

    _raise_alarms(result)
    return result


async def _apply_finding(
    db, finding: Finding, subs_by_txn: dict, *, now: datetime, apply: bool,
) -> bool:
    """Converge ONE finding. Returns True iff something was (or would be) written."""
    if finding.cls in (CLASS_GRANT_LAPSED, CLASS_CROSSGRADE):
        # The sweep already handled the first; the second is an ALERT, never a
        # mutation — a cross-grade is a customer's own choice at Apple, and the
        # only correct response is to tell a human.
        return False

    may = _may_mutate(finding.environment)
    effective_apply = apply and may
    if apply and not may:
        # Bookkeeping only — it is what keeps `applied` from counting a write
        # that was refused. It is NOT what the alerts fold on: this branch
        # needs `apply`, so the marker does not exist in observe-only and
        # keying the fold on it silenced nothing there.
        finding.actions.append("sandbox:not_applied")

    if finding.cls == CLASS_MISSING:
        sub = subs_by_txn.get(finding.original_txn)
        if sub is None:
            return False
        target = apple_iap_service.plan_for_subscription(sub.product_id, sub)
        return await _entitle(
            db, finding, sub, target, sub.expires_date, apply=effective_apply,
        )

    apple = finding.apple
    if apple is None:
        # We could not read Apple (or Apple does not know the transaction).
        # Reporting is the whole action: acting on a mirror we already decided
        # not to trust is how a reconciler invents the drift it is meant to fix.
        return False

    sub = subs_by_txn.get(finding.original_txn)
    if sub is not None:
        drift = await _converge_mirror(db, sub, apple, apply=effective_apply)
        if drift:
            finding.actions.append("mirror: " + ", ".join(drift))

    if apple.status in _DEAD_APPLE_STATUSES:
        reason = f"apple:reconcile:{_APPLE_STATUS_TO_MIRROR.get(apple.status, apple.status)}"
        if finding.user_id:
            await _de_entitle(db, finding, reason, apply=effective_apply)
    elif apple.status in _ENTITLING_APPLE_STATUSES and finding.user_id and sub is not None:
        product = apple.product_id or sub.product_id
        try:
            target = apple_iap_service.plan_for_subscription(product, sub)
        except KeyError:
            logger.warning(
                "[apple-reconcile] Apple reports unknown product %r for "
                "orig_txn=%s; not entitling", product, finding.original_txn,
            )
            target = None
        if target is not None:
            await _entitle(
                db, finding, sub, target,
                apple.expires_date or sub.expires_date, apply=effective_apply,
            )

    return bool(finding.actions) and finding.actions != ["sandbox:not_applied"]


# ── alerts ───────────────────────────────────────────────────────────────


def _raise_alarms(result: ReconcilePass) -> None:
    """Queue this pass's alerts. Sent after the commit, never inside it.

    Subjects are per-account so one noisy subscription cannot suppress an
    alert about another (alerting.py's (category, subject) window).
    """
    def _add(category, level, message, subject=None, min_interval_s=None):
        result.alerts.append((category, level, message, subject, min_interval_s))

    def _acts(acted: list[str]) -> str:
        """The action list, in the tense of the mode that produced it.

        This replaces a trailing " [observe-only: nothing was written]" tag,
        which had every body assert a completed action in its first clause and
        deny it one sentence later. An alarm an operator must read twice to
        learn whether anything happened is an alarm that gets muted.
        """
        joined = "; ".join(acted)
        if result.apply:
            return f"Actions: {joined}."
        return f"Proposed actions (observe-only — nothing was written): {joined}."

    for f in result.findings:
        acted = [a for a in f.actions if a != "sandbox:not_applied"]
        if not _may_mutate(f.environment):
            # Folded into the one aggregate warning below instead. Firing
            # `apple-reconcile-downgrade` CRITICAL here would page an operator
            # about a downgrade that deliberately did not happen — an alarm
            # describing an action rather than a fact, which is how a channel
            # gets muted.
            #
            # Keyed on the configuration FACT, never on `_apply_finding`'s
            # "sandbox:not_applied" marker: that marker is written only under
            # `if apply and not may`, so in the observe-only mode production
            # runs the fold was UNREACHABLE — a Sandbox row took the CRITICAL
            # arm below on every pass where Apple answered that the
            # subscription was dead, claiming a downgrade nothing was ever
            # going to write.
            #
            # It cannot swallow `apple-legacy-crossgrade`, which is an alert
            # rather than a refused write: `scan` emits that class only for
            # ENV_PRODUCTION, so it never reaches here. Keep that true if the
            # scan's arms ever widen.
            continue
        if f.cls == CLASS_CROSSGRADE:
            _add("apple-legacy-crossgrade", "critical",
                 f"{f.detail}. Either a new legacy purchase, or a subscriber "
                 f"changing product inside the group from iOS Settings — which "
                 f"is a repricing nobody in this system approved, in a group "
                 f"whose four legacy products can never be taken off sale. "
                 f"The entitlement is already correct (a grandfather does not "
                 f"follow a product change); what needs a human is whether the "
                 f"new price is one we meant to offer them. Check "
                 f"original_transaction_id={f.original_txn}.",
                 uidp(f.user_id))
        elif f.cls == CLASS_MISSING and acted:
            _add("apple-reconcile-missing-entitlement", "critical",
                 f"A LIVE Apple payer was not entitled: {f.detail} "
                 f"(user={uidp(f.user_id)}, orig_txn={f.original_txn}). "
                 f"{_acts(acted)} This is a paying customer "
                 f"locked out of what they bought.",
                 uidp(f.user_id))
        elif any(a.startswith("downgrade ") for a in acted):
            # "had to" asserts a write. In observe-only there was none, so the
            # verb is conditional rather than contradicted by a tag at the end
            # of its own sentence.
            verb = "had to downgrade" if result.apply else "would have to downgrade"
            _add("apple-reconcile-downgrade", "critical",
                 f"The reconciler {verb} a {f.environment} payer: "
                 f"{f.detail} (user={uidp(f.user_id)}, orig_txn={f.original_txn}). "
                 f"{_acts(acted)} A downgrade reaching this "
                 f"loop means the App Store notification channel dropped the "
                 f"lapse message — the failure this reconciler exists to catch.",
                 uidp(f.user_id))
        elif f.cls == CLASS_ORPHAN and not acted:
            # Deliberately NOT corrected. An unlimited stamp with no Apple row
            # at all cannot be checked against Apple — there is no transaction
            # to ask about — and stripping a paid-looking entitlement on the
            # strength of an ABSENCE is exactly the class of destructive
            # inference this loop must not make. Report it; let a human read
            # it. (When there IS a row, Apple decides and the downgrade arm
            # above fires instead.)
            _add("apple-reconcile-drift", "warning",
                 f"Unlimited entitlement with nothing behind it: {f.detail} "
                 f"(user={uidp(f.user_id)}). NOT corrected — there is no Apple "
                 f"transaction to reconcile against, so this needs a human: "
                 f"either the account should hold a grant, or the stamp is "
                 f"stale and should be revoked by hand.",
                 uidp(f.user_id))
        elif f.cls == CLASS_STALE:
            _add("apple-reconcile-stale", "warning",
                 f"Apple subscription orig_txn={f.original_txn} "
                 f"(user={uidp(f.user_id)}, {f.environment}): {f.detail}. A "
                 f"monthly subscription that has produced no notification for "
                 f"over a cycle is either silently dead or a channel that has "
                 f"stopped delivering.",
                 uidp(f.user_id))
        elif acted:
            _add("apple-reconcile-drift", "warning",
                 f"{f.cls}: {f.detail} (user={uidp(f.user_id)}, "
                 f"orig_txn={f.original_txn}). {_acts(acted)}",
                 uidp(f.user_id))

    # Same key as the fold above, for the same reason: a marker-keyed list was
    # empty in observe-only, so the rows the fold removed from the per-finding
    # arms were reported by NOTHING once the fold started working there.
    sandbox_skipped = [f for f in result.findings
                       if not _may_mutate(f.environment)]
    if sandbox_skipped:
        # This line is the ONLY thing an operator ever reads about these rows,
        # so it may not name a cause it does not know. A folded finding is in
        # one of three states and only the first is the sandbox policy's doing:
        # a correction was determined and refused (`acted` non-empty); Apple
        # returned no answer for the transaction this pass (`f.apple is None`,
        # which `_apply_finding` treats as "reporting is the whole action");
        # or Apple answered and there was nothing to converge. One sentence
        # blaming all three on "a sandbox subscription must never move a
        # production balance" sends an operator to flip an opt-in that cannot
        # change the last two — and, when every Apple call failed, contradicts
        # the `apple-reconcile-unreachable` warning raised by the same pass.
        refused, unread, converged = [], [], []
        for f in sandbox_skipped:
            if [a for a in f.actions if a != "sandbox:not_applied"]:
                refused.append(f)
            elif f.apple is None:
                unread.append(f)
            else:
                converged.append(f)

        def _txns(group) -> str:
            return ", ".join(str(f.original_txn) for f in group[:5])

        parts = [f"{len(sandbox_skipped)} drifted Sandbox row(s) were "
                 f"REPORTED, not corrected."]
        if refused:
            # Observe-only needs both switches, and an operator sent to flip
            # one would watch the next pass change nothing and conclude the
            # opt-in is broken.
            opt_in = ("apple_reconcile_sandbox_apply=true" if result.apply else
                      "apple_reconcile_apply=true and "
                      "apple_reconcile_sandbox_apply=true")
            parts.append(
                f"{len(refused)} row(s) had a correction determined and "
                f"refused — a sandbox subscription must never move a "
                f"production balance: {_txns(refused)}. Set {opt_in} to let "
                f"this loop clean them.")
        if unread:
            parts.append(
                f"{len(unread)} row(s) could not be evaluated — Apple "
                f"returned no answer for them this pass (unreadable, unknown "
                f"to Apple, or past this pass's API budget), so no correction "
                f"is known: {_txns(unread)}.")
        if converged:
            parts.append(
                f"{len(converged)} row(s) were read at Apple and produced no "
                f"correction to make: {_txns(converged)}.")
        # The one alert in this module with a window of its own: an unchanged
        # Sandbox condition pages at most once a day rather than on every pass.
        # See SANDBOX_AGGREGATE_MIN_INTERVAL_S for why, and for the per-process
        # residue.
        _add("apple-reconcile-drift", "warning", " ".join(parts), "sandbox",
             SANDBOX_AGGREGATE_MIN_INTERVAL_S)

    # "Every Apple call failed" is a DIFFERENT fact from "nothing to fix", and
    # a reconciler that cannot tell them apart reports health it did not check.
    if result.api_calls and result.api_failures >= result.api_calls:
        _add("apple-reconcile-unreachable", "warning",
             f"All {result.api_calls} App Store Server API call(s) failed this "
             f"pass — the reconciler could not look. Errors: "
             f"{'; '.join(result.errors[:3])}")


async def _flush_alerts(result: ReconcilePass) -> None:
    for category, level, message, subject, min_interval_s in result.alerts:
        # Forward a window only when the alert asked for one; `min_interval_s`
        # has a default in alerting.py and passing None would override it with
        # nothing rather than fall back to it.
        window = {} if min_interval_s is None else {"min_interval_s": min_interval_s}
        try:
            await send_infra_alert(
                category, level, message, subject=subject, **window,
            )
        except Exception as e:  # pragma: no cover - alerting is best-effort
            logger.warning("[apple-reconcile] alert %s failed: %s", category, e)


# ── the loop ─────────────────────────────────────────────────────────────


async def run_reconcile_pass() -> ReconcilePass:
    """One pass in its own session, committed, with the summary persisted and
    the alerts flushed afterwards.

    Alerts are flushed in a `finally`: a pass that raised on commit is exactly
    the pass whose findings an operator most needs to see, and the alarms were
    already computed from what the scan found, not from what was written.
    """
    result: Optional[ReconcilePass] = None
    try:
        async with async_session_maker() as db:
            result = await reconcile_once(db)
            try:
                await _store_summary(db, result)
                await db.commit()
            except Exception:
                await db.rollback()
                raise
    finally:
        if result is not None:
            await _flush_alerts(result)
            logger.info("[apple-reconcile] %s", result.summary())
    return result


async def _store_summary(db: AsyncSession, result: ReconcilePass) -> None:
    """Persist the pass summary. Also the operator's proof the loop is alive —
    a `ran_at` that stopped moving is the failure mode a monitor cannot see
    from the absence of alerts.

    Its own savepoint: a heartbeat that cannot be written must not roll back
    the billing corrections that were.
    """
    blob = json.dumps(result.summary(), sort_keys=True)
    try:
        async with db.begin_nested():
            row = await db.get(PlatformSetting, LAST_RUN_SETTING_KEY)
            if row is None:
                db.add(PlatformSetting(key=LAST_RUN_SETTING_KEY, value=blob))
            else:
                row.value = blob
                row.updated_at = datetime.utcnow()
            await db.flush()
    except Exception:
        logger.warning(
            "[apple-reconcile] could not persist the pass summary", exc_info=True,
        )


async def apple_reconcile_loop() -> None:
    """Forever loop; start via asyncio.create_task in the lifespan."""
    interval = max(3600, int(_cfg("apple_reconcile_interval_s", 21600)))
    logger.info(
        "[apple-reconcile] reconciler started (interval=%ss, apply=%s)",
        interval, _cfg("apple_reconcile_apply", False),
    )
    lease_ttl = lease_ttl_for(interval)
    while True:
        try:
            # Sleep first: a deploy is the worst moment to page, and at boot
            # the mirror holds whatever the previous replica left behind.
            await asyncio.sleep(interval)
            # ONE runner fleet-wide. railway.json is numReplicas 2 and this
            # loop moves real money: two replicas racing the same finding
            # write two plan_change ledger rows for one event, and on the
            # PRORATED arm of apply_plan_change they apply the delta twice.
            # `_lock_balance`'s FOR UPDATE does not save it — the second
            # replica is a different process reading its own snapshot. Same
            # lease primitive the 2026-09-12 incident's other loops use.
            if not await acquire_lease("apple-reconciler", ttl_s=lease_ttl):
                logger.info(
                    "[apple-reconcile] another replica holds the lease — "
                    "skipping this tick",
                )
                continue
            await run_reconcile_pass()
        except asyncio.CancelledError:
            # Hand the lease back so the surviving replica takes the next tick
            # immediately instead of waiting out the TTL.
            await release_lease("apple-reconciler")
            raise
        except Exception:
            # One bad pass must not silence the reconciler for the process
            # lifetime, and it is never swallowed quietly either.
            logger.exception("[apple-reconcile] pass failed; continuing")
