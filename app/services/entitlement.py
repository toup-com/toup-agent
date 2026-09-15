"""Who holds the UNLIMITED entitlement, and what is behind it.

Two layers, and the split is the most important thing in this file.

**The charge path** (``credit_service.try_charge`` / ``reserve``) reads
``credit_service._entitlement_is_unlimited(user, balance)`` — role == 'admin',
or ``credit_balances.plan_id == 'unlimited'``. Nothing more. That is a pure
function over two objects the caller already holds, so it costs no query and
does not widen the ``SELECT … FOR UPDATE`` the charge runs under.

**This module** is the CONTROLLER side: it answers *why* an account should or
should not hold the stamp, by looking at Apple's subscription mirror and the
admin-override table. It is read by the grandfather command, the Apple
reconciler, and the admin endpoints — never by ``try_charge``. The balance row
is the single materialisation of the entitlement; everything here decides what
to WRITE there.

That direction is deliberate and must stay one-way: ``credit_service`` does
not import this module. ``iap.py``, the reconciler and the admin router import
``entitlement``, which calls ``credit_service`` to apply the result. A
hot-path oracle would let the balance row say ``unlimited`` while the gate
quietly disagreed, which is the split-brain class this repo has been bitten by
before.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import inspect as sa_inspect, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    APPLE_SUB_EXPIRED,
    APPLE_SUB_GRACE,
    APPLE_SUB_REVOKED,
    SPONSOR_KIND_APPLE,
    SPONSOR_KIND_STRIPE,
    AppleSubscription,
    CreditBalance,
    UnlimitedGrant,
    User,
)
from app.db.plan_catalog import (
    LEGACY_SUB_PRODUCT_IDS,
    UNLIMITED_PLAN_ID,
    UNLIMITED_SUB_PRODUCT_ID,
)
from app.services import abuse_metrics
from app.services.abuse_metrics import uidp
from app.services.credit_service import PLAN_SOURCE_APPLE, PLAN_SOURCE_STRIPE

logger = logging.getLogger(__name__)

_GRANTS_TABLE = "unlimited_grants"


# Apple's maximum billing grace period. `apple_subscriptions` has no
# gracePeriodExpiresDate column, so a row stuck at status='grace' would
# otherwise become a permanent comp; this bounds it.
_GRACE_MAX_DAYS = 16

# Status may argue that a subscription is DEAD. It may never argue that one is
# ALIVE — that is `expires_date`'s job, and the Sandbox Elite (status='active'
# 2.7 months past expiry, last_notification_uuid NULL) is the standing proof.
#
# The dead direction is load-bearing and was missing. On REFUND and REVOKE the
# handler writes status='expired'/'revoked' but only overwrites expires_date
# `if expires_date is not None` — and Apple's REFUND transaction carries the
# ORIGINAL expiresDate, so the row keeps a FUTURE clock. A clock-only predicate
# therefore reads a refunded subscription as a live payer. Concretely: a
# customer locked out at 0.02 credits asks Apple for a refund, and the
# grandfather command run that afternoon hands them 1,000,000 credits
# indefinitely, with Apple's money already returned.
#
# `apple_reconciler.scan` imports this set so the two cannot drift on WHICH
# statuses are terminal.
TERMINAL_APPLE_STATUSES = frozenset({APPLE_SUB_EXPIRED, APPLE_SUB_REVOKED})

# The plan ids sold before Unlimited. Used only by the Stripe candidate arm.
LEGACY_PLAN_IDS = ("starter", "builder", "pro", "elite")


@dataclass
class GrandfatherCandidate:
    user: User
    sub: AppleSubscription
    balance: CreditBalance


# ── Apple ────────────────────────────────────────────────────────────────


async def has_live_apple_entitlement(
    db: AsyncSession, user_id: str, *, now: Optional[datetime] = None,
) -> Optional[AppleSubscription]:
    """The user's live PRODUCTION Apple subscription row, or None.

    ``status`` is deliberately NEVER the primary test. The Sandbox Elite row
    (user ``e4cad7f5``) has read ``status='active'`` for 2.7 months past its
    2026-06-24 expiry with ``last_notification_uuid`` NULL — the column records
    the last notification we processed, not the truth, and no notification ever
    arrived. ``expires_date`` is Apple's own clock and is what we reconcile
    against.

    The GRACE arm is the one exception and it is narrow: Apple's billing grace
    period legitimately keeps entitlement past ``expires_date``, and we honour
    ``status == 'grace'`` for at most ``_GRACE_MAX_DAYS`` past it.

    Status IS read in the other direction — see ``TERMINAL_APPLE_STATUSES``.
    A REFUND or REVOKE leaves ``expires_date`` in the FUTURE while ending the
    entitlement, so a clock-only reading of this row would report a refunded
    subscription as live: the orphan-entitlement backstop would skip the very
    user whose downgrade was lost, and ``create_grant`` would refuse an admin
    comp on the grounds of a subscription Apple has revoked.

    ``expires_date`` is a naive UTC column (written from ``datetime.utcnow()``),
    so ``now`` must be naive UTC too.
    """
    now = now or datetime.utcnow()
    rows = (await db.execute(
        select(AppleSubscription).where(
            AppleSubscription.user_id == user_id,
            AppleSubscription.environment == "Production",
            AppleSubscription.product_id.in_(
                LEGACY_SUB_PRODUCT_IDS | {UNLIMITED_SUB_PRODUCT_ID}
            ),
            AppleSubscription.expires_date.is_not(None),
        )
    )).scalars().all()
    for sub in rows:
        if sub.status in TERMINAL_APPLE_STATUSES:
            continue
        if sub.expires_date > now:
            return sub
        if (sub.status == APPLE_SUB_GRACE
                and sub.expires_date > now - timedelta(days=_GRACE_MAX_DAYS)):
            return sub
    return None


async def select_legacy_apple_payers(
    db: AsyncSession, *, now: Optional[datetime] = None,
) -> list[GrandfatherCandidate]:
    """Every account that PAID Apple for a legacy tier and is still inside a
    period it paid for. The grandfather command's selection.

    (It is NOT the reconciler's live-set query, which an earlier version of
    this docstring claimed. The reconciler scans every row rather than one
    user's, and its live test is inlined in ``apple_reconciler.scan``. What
    the two DO share is ``TERMINAL_APPLE_STATUSES``, so they cannot drift on
    the only judgement both make.)

    Five conditions, and each one excludes something real:

    1. ``environment = 'Production'`` — the only predicate in the database
       that distinguishes a paid entitlement from a granted one. Excludes the
       Sandbox Elite.
    2. ``product_id IN (the four legacy ids)`` — pinned explicitly so a future
       product cannot be swept in by accident.
    3. + 4. ``expires_date > now`` — a second, independent reason the Sandbox
       row is excluded (it lapsed 2026-06-24). ``status`` is never allowed to
       argue that a row is ALIVE.
    5. ``status NOT IN (expired, revoked)`` — status arguing DEAD, which it
       may. A REFUND keeps the original future ``expires_date``, so without
       this a refunded customer is selected as a payer and handed 1,000,000
       credits. See ``TERMINAL_APPLE_STATUSES``.

    Also excluded, by having no Apple row at all: ``nariman@toup.ai``'s
    alembic-054 Builder grant, the two App Review comps, and the three admin
    accounts (already unlimited via ``role == 'admin'``).

    An auto-renew-OFF subscriber inside their paid window IS selected — they
    are still inside a period they paid for, and Apple's ``EXPIRED`` returns
    them to free through the ordinary downgrade path with no special case.

    The ``credit_balances`` join is OUTER on purpose. It used to be INNER, so
    a payer with no balance row vanished from the selection with no log at
    all — a silent skip on the one command whose entire job is "grant the
    payers". There are no such rows today; if one appears it is raised, not
    dropped.
    """
    now = now or datetime.utcnow()
    rows = (await db.execute(
        select(User, AppleSubscription, CreditBalance)
        .join(AppleSubscription, AppleSubscription.user_id == User.id)
        .outerjoin(CreditBalance, CreditBalance.user_id == User.id)
        .where(
            AppleSubscription.environment == "Production",
            AppleSubscription.product_id.in_(LEGACY_SUB_PRODUCT_IDS),
            AppleSubscription.expires_date.is_not(None),
            AppleSubscription.expires_date > now,
            AppleSubscription.status.notin_(tuple(TERMINAL_APPLE_STATUSES)),
        )
        .order_by(User.email)
    )).all()
    walletless = [u.email for (u, _s, b) in rows if b is None]
    if walletless:
        raise RuntimeError(
            f"legacy Apple payer(s) with no credit_balances row: {walletless}. "
            f"Refusing to select a partial population — create the balance "
            f"row(s) first, then re-run."
        )
    return [GrandfatherCandidate(user=u, sub=s, balance=b) for (u, s, b) in rows]


# ── Stripe ───────────────────────────────────────────────────────────────


async def select_legacy_stripe_payers(db: AsyncSession) -> list[tuple[User, CreditBalance]]:
    """CANDIDATES ONLY — never applied without operator confirmation.

    There is no Stripe subscription mirror in this schema, so the database
    genuinely cannot tell a Stripe payer from an alembic-054 bundle grant —
    which is exactly what ``nariman@toup.ai`` (871bac24: builder,
    ``plan_source`` NULL, no Stripe customer, clock-renewing) is. Writing a
    confident predicate here would be writing fiction.

    So this arm additionally requires a Stripe customer id, excludes admins,
    and is gated behind ``--include-stripe`` at the command with every hit
    printed for a human to confirm against the Stripe dashboard.

    With today's data it returns ZERO rows. That is the expected output.
    """
    return list((await db.execute(
        select(User, CreditBalance)
        .join(CreditBalance, CreditBalance.user_id == User.id)
        .where(
            CreditBalance.plan_id.in_(LEGACY_PLAN_IDS),
            or_(CreditBalance.plan_source.is_(None),
                CreditBalance.plan_source == PLAN_SOURCE_STRIPE),
            User.stripe_customer_id.is_not(None),
            User.role != "admin",
        )
        .order_by(User.email)
    )).all())


# ── The admin override (unlimited_grants) ────────────────────────────────
#
# The table is a CONTROLLER of the entitlement, never an oracle on the charge
# path — `try_charge` reads `credit_balances.plan_id` and nothing else. Three
# reasons, all in the module docstring; the shortest is that the balance row
# is the single materialisation of the entitlement, so when a grant ends the
# thing that must change is that row, and every surface then agrees by
# construction.
#
# A grant is LIVE at time T iff ALL of:
#   1. revoked_at IS NULL AND lapsed_at IS NULL   — nobody ended it;
#   2. expires_at IS NULL OR expires_at > T       — its own backstop;
#   3. the SPONSOR is live at T                   — see _sponsor_is_live,
#      which fails CLOSED on a missing sponsor row.


class GrantError(Exception):
    """A grant/revoke the caller asked for cannot be honoured.

    Carries an HTTP-ish ``status`` so the admin router and the CLI can render
    the SAME refusal without either re-deriving the rules.
    """

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


# Cached per process. `unlimited_grants` arrives with alembic 105, and the
# window in which this code exists without the table is a deploy that has not
# yet migrated.
#
# Probed rather than caught: swallowing a ProgrammingError leaves a Postgres
# transaction ABORTED, so every subsequent statement on that session fails
# with InFailedSqlTransaction — the "graceful degradation" would take the
# reconciler pass down with it, one statement later and nowhere near the
# cause. A read-only introspection call costs one round trip, once.
_grants_table_present_cache: bool = False


async def _grants_table_present(db: AsyncSession) -> bool:
    global _grants_table_present_cache
    # ONLY the positive answer is cached. A cached "absent" would latch for the
    # life of the process, so a replica that happened to boot while the deploy's
    # migration was still running would answer "no override" for every account
    # until someone restarted it — silently, with the table sitting right there.
    # Re-probing costs one round trip on an environment that has no table, which
    # is the environment where nothing else is happening either.
    if _grants_table_present_cache:
        return True
    try:
        conn = await db.connection()
        names = await conn.run_sync(
            lambda sync_conn: sa_inspect(sync_conn).get_table_names()
        )
    except Exception as e:  # pragma: no cover - introspection should not fail
        logger.warning("[entitlement] could not introspect for %s: %s", _GRANTS_TABLE, e)
        return False
    if _GRANTS_TABLE not in set(names):
        logger.info(
            "[entitlement] %s absent (alembic 105 not applied here); "
            "treating every account as having no override", _GRANTS_TABLE,
        )
        return False
    _grants_table_present_cache = True
    return True


async def live_grant_for(
    db: AsyncSession, user_id: str, *, now: Optional[datetime] = None,
) -> Optional[UnlimitedGrant]:
    """The user's LIVE unlimited grant, or None."""
    if not await _grants_table_present(db):
        return None
    now = now or datetime.utcnow()
    rows = (await db.execute(
        select(UnlimitedGrant).where(
            UnlimitedGrant.granted_to_user_id == user_id,
            UnlimitedGrant.revoked_at.is_(None),
            UnlimitedGrant.lapsed_at.is_(None),
        ).order_by(UnlimitedGrant.granted_at.desc())
    )).scalars().all()
    for grant in rows:
        if grant.expires_at is not None and grant.expires_at <= now:
            continue
        if not await _sponsor_is_live(db, grant, now=now):
            continue
        return grant
    return None


async def open_grant_for(
    db: AsyncSession, user_id: str,
) -> Optional[UnlimitedGrant]:
    """The user's OPEN grant — neither revoked nor lapsed — regardless of
    whether its sponsor is still paying.

    Distinct from :func:`live_grant_for` on purpose. The one-live-grant
    invariant (and the partial unique index behind it) is about OPEN rows: a
    grant whose sponsor has quietly lapsed is still an open row that the
    reconciler is about to close, and issuing a second one on top of it would
    hit the index and 500 instead of the router's 409.
    """
    if not await _grants_table_present(db):
        return None
    return (await db.execute(
        select(UnlimitedGrant).where(
            UnlimitedGrant.granted_to_user_id == user_id,
            UnlimitedGrant.revoked_at.is_(None),
            UnlimitedGrant.lapsed_at.is_(None),
        )
    )).scalars().first()


async def grants_sponsored_by(
    db: AsyncSession, *, original_txn_id: Optional[str] = None,
    sponsor_user_id: Optional[str] = None,
) -> list[UnlimitedGrant]:
    """Every still-OPEN grant sponsored by an Apple transaction or a user.

    The lapse path reads this in the SAME transaction as the sponsor's own
    downgrade, which is what makes "lapses automatically with it, no manual
    step" true rather than aspirational.
    """
    if original_txn_id is None and sponsor_user_id is None:
        return []
    if not await _grants_table_present(db):
        return []
    clauses = [UnlimitedGrant.revoked_at.is_(None), UnlimitedGrant.lapsed_at.is_(None)]
    if original_txn_id is not None:
        clauses.append(UnlimitedGrant.sponsor_original_txn_id == original_txn_id)
    if sponsor_user_id is not None:
        clauses.append(UnlimitedGrant.sponsor_user_id == sponsor_user_id)
    return list((await db.execute(select(UnlimitedGrant).where(*clauses))).scalars().all())


async def list_grants(
    db: AsyncSession, *, user_id: Optional[str] = None,
    live: Optional[bool] = None, now: Optional[datetime] = None,
) -> list[UnlimitedGrant]:
    """Grant history, newest first. Nothing is filtered out by default — a
    revoked or lapsed grant is the part of the record an operator most often
    needs to read.
    """
    if not await _grants_table_present(db):
        return []
    now = now or datetime.utcnow()
    q = select(UnlimitedGrant).order_by(UnlimitedGrant.granted_at.desc())
    if user_id is not None:
        q = q.where(UnlimitedGrant.granted_to_user_id == user_id)
    rows = list((await db.execute(q)).scalars().all())
    if live is None:
        return rows
    out = []
    for g in rows:
        is_live = (
            g.revoked_at is None and g.lapsed_at is None
            and (g.expires_at is None or g.expires_at > now)
            and await _sponsor_is_live(db, g, now=now)
        )
        if is_live == live:
            out.append(g)
    return out


async def _sponsor_is_live(db: AsyncSession, grant, *, now: datetime) -> bool:
    """Is the thing paying for this grant still paying?

    Apple: fail CLOSED on a missing row. ``apple_subscriptions.user_id`` is
    ON DELETE CASCADE, so deleting the sponsor's account destroys the
    subscription row — and a grant whose sponsor no longer exists must end,
    not persist because we could not find a reason to stop it.
    """
    kind = getattr(grant, "sponsor_kind", None)
    if kind == SPONSOR_KIND_APPLE:
        txn = getattr(grant, "sponsor_original_txn_id", None)
        if not txn:
            return False
        sub = (await db.execute(
            select(AppleSubscription).where(
                AppleSubscription.original_transaction_id == txn,
                AppleSubscription.environment == "Production",
            )
        )).scalar_one_or_none()
        if sub is None or sub.expires_date is None:
            return False
        if sub.expires_date > now:
            return True
        return (sub.status == APPLE_SUB_GRACE
                and sub.expires_date > now - timedelta(days=_GRACE_MAX_DAYS))
    if kind == SPONSOR_KIND_STRIPE:
        # The only signal this schema offers: there is no Stripe subscription
        # mirror to consult.
        uid = getattr(grant, "sponsor_user_id", None)
        if not uid:
            return False
        bal = await db.get(CreditBalance, uid)
        if bal is None or bal.plan_id == "free":
            return False
        return (bal.plan_source or None) in (None, PLAN_SOURCE_STRIPE)
    return False


# ── Writing the stamp ────────────────────────────────────────────────────
#
# Exactly one implementation of grant and revoke. The admin endpoints and
# `python -m app.scripts.unlimited_grant` are both thin front ends over these
# two functions — a second copy of the rules is a second set of rules.


async def _sponsor_period_end(db: AsyncSession, grant) -> datetime:
    """When the grantee's entitlement window should end if nothing renews it.

    Apple's ``expires_date`` for an Apple sponsor; the sponsor's own period
    for a Stripe one. It is a WINDOW, not the lapse rule: the lapse rule is
    the sponsor's own lifecycle (``_sponsor_is_live`` + the notification
    hook). This only stops the grantee's balance from sitting on a period
    that outlives its sponsor's by months if every other path fails.
    """
    if grant.sponsor_kind == SPONSOR_KIND_APPLE and grant.sponsor_original_txn_id:
        sub = (await db.execute(
            select(AppleSubscription).where(
                AppleSubscription.original_transaction_id
                == grant.sponsor_original_txn_id,
                AppleSubscription.environment == "Production",
            )
        )).scalar_one_or_none()
        if sub is not None and sub.expires_date is not None:
            return sub.expires_date
    if grant.sponsor_user_id:
        bal = await db.get(CreditBalance, grant.sponsor_user_id)
        if bal is not None:
            return bal.period_end
    return datetime.utcnow() + timedelta(days=30)


async def create_grant(
    db: AsyncSession, *, granted_to_user_id: str, sponsor_kind: str,
    sponsor_user_id: Optional[str] = None,
    sponsor_original_txn_id: Optional[str] = None,
    sponsor_stripe_sub_id: Optional[str] = None,
    reason: str, expires_at: Optional[datetime] = None,
    granted_by_user_id: str, granted_by_email: str,
    now: Optional[datetime] = None, notify: bool = True,
) -> UnlimitedGrant:
    """Issue an admin override and stamp the entitlement on the grantee.

    Refuses (never silently repairs) when:
      404  the grantee does not exist
      409  the grantee already holds an OPEN grant
      422  the sponsor is not identified, is not live right now, is the
           grantee themselves, or the grantee is already paying for
           something of their own that this would overwrite

    The last one is the important refusal. ``apply_plan_change`` ASSIGNS on
    the Unlimited arm, and revoke returns the grantee to FREE — so granting
    over an account that holds its own paid plan would silently replace that
    plan and then, at revoke, hand the person a free wallet instead of the
    tier they are still being charged for.
    """
    from app.services.credit_service import credit_service

    if not await _grants_table_present(db):
        raise GrantError(503, f"{_GRANTS_TABLE} does not exist here (alembic 105 not applied)")
    now = now or datetime.utcnow()
    reason = (reason or "").strip()
    if not reason:
        raise GrantError(422, "reason is required — a comp with no recorded why is not auditable")
    if sponsor_kind not in (SPONSOR_KIND_APPLE, SPONSOR_KIND_STRIPE):
        raise GrantError(422, f"sponsor_kind must be one of apple|stripe (got {sponsor_kind!r})")

    grantee = await db.get(User, granted_to_user_id)
    if grantee is None:
        raise GrantError(404, f"no such user {granted_to_user_id}")

    if sponsor_kind == SPONSOR_KIND_APPLE:
        if not sponsor_original_txn_id:
            raise GrantError(422, "an apple sponsor needs sponsor_original_transaction_id")
        sub = (await db.execute(
            select(AppleSubscription).where(
                AppleSubscription.original_transaction_id == sponsor_original_txn_id,
                AppleSubscription.environment == "Production",
            )
        )).scalar_one_or_none()
        if sub is None:
            raise GrantError(
                422,
                f"no PRODUCTION apple_subscriptions row for "
                f"original_transaction_id={sponsor_original_txn_id}",
            )
        sponsor_user_id = sponsor_user_id or sub.user_id
    elif not sponsor_user_id:
        raise GrantError(422, "a stripe sponsor needs sponsor_user_id")

    if sponsor_user_id == granted_to_user_id:
        raise GrantError(
            422,
            "a grant is a LINK from one account's subscription to another; "
            "sponsoring yourself entitles nothing",
        )

    if await open_grant_for(db, granted_to_user_id) is not None:
        raise GrantError(409, f"user {granted_to_user_id} already holds an open grant")

    balance = await credit_service.get_or_create_balance(db, granted_to_user_id)
    if balance.plan_id not in ("free", UNLIMITED_PLAN_ID):
        raise GrantError(
            422,
            f"grantee is on plan {balance.plan_id!r} — a grant would overwrite it "
            f"and revoke would return them to free rather than to that plan",
        )
    if await has_live_apple_entitlement(db, granted_to_user_id, now=now) is not None:
        raise GrantError(
            422, "grantee has their own live Apple subscription; nothing to comp",
        )

    grant = UnlimitedGrant(
        granted_to_user_id=granted_to_user_id,
        sponsor_kind=sponsor_kind,
        sponsor_user_id=sponsor_user_id,
        sponsor_original_txn_id=sponsor_original_txn_id,
        sponsor_stripe_sub_id=sponsor_stripe_sub_id,
        reason=reason,
        expires_at=expires_at,
        granted_by_user_id=granted_by_user_id,
        granted_by_email=granted_by_email,
        granted_at=now,
        created_at=now,
        updated_at=now,
    )
    db.add(grant)
    await db.flush()

    # Sponsor liveness is checked AFTER the row exists so the refusal message
    # can name the sponsor's actual state, and BEFORE any entitlement is
    # stamped so a dead sponsor never produces one.
    if not await _sponsor_is_live(db, grant, now=now):
        raise GrantError(
            422,
            f"sponsor ({sponsor_kind}) is not live right now — a grant cannot "
            f"outlive a subscription it never overlapped",
        )

    # plan_source MIRRORS the sponsor's source. Not cosmetic: NULL or 'stripe'
    # renders as 'web' in get_balance_view, and iOS + 'web' matches none of the
    # shipped client's three UI branches — the grantee would see no plan
    # surface at all. 'apple' renders "Subscribed — manage in Settings", which
    # is substantively true and carries no false purchase affordance.
    source = PLAN_SOURCE_APPLE if sponsor_kind == SPONSOR_KIND_APPLE else PLAN_SOURCE_STRIPE
    await credit_service.activate_subscription(
        db, granted_to_user_id, UNLIMITED_PLAN_ID, source,
        await _sponsor_period_end(db, grant),
    )
    logger.info(
        "[entitlement] grant %s: user=%s sponsor=%s/%s by=%s",
        grant.id, uidp(granted_to_user_id), sponsor_kind,
        sponsor_original_txn_id or uidp(sponsor_user_id), granted_by_email,
    )
    # `notify=False` is for a caller that is going to ROLL BACK — the CLI's dry
    # run. Announcing a comp that never happened trains the operator to ignore
    # the channel. The alert is otherwise sent before the caller's commit,
    # which can over-report on a failed commit; that trade is deliberate, since
    # an alert that never fires is the worse failure here.
    if notify:
        abuse_metrics.emit(
            "unlimited_grant_created", uid=uidp(granted_to_user_id),
            grant=grant.id, sponsor_kind=sponsor_kind,
            sponsor=uidp(sponsor_user_id),
        )
        await _alert(
            "warning",
            f"UNLIMITED granted to {uidp(granted_to_user_id)} "
            f"(sponsor {sponsor_kind} {uidp(sponsor_user_id)}) by {granted_by_email}: "
            f"{reason}",
            subject=uidp(granted_to_user_id),
        )
    return grant


async def revoke_grant(
    db: AsyncSession, grant_id: str, *, revoked_by_user_id: str,
    revoked_by_email: str, revoked_reason: str,
    now: Optional[datetime] = None, notify: bool = True,
) -> UnlimitedGrant:
    """End a grant by operator decision. Idempotent: revoking an already-dead
    grant returns it unchanged rather than rewriting how it died.
    """
    if not await _grants_table_present(db):
        raise GrantError(503, f"{_GRANTS_TABLE} does not exist here (alembic 105 not applied)")
    now = now or datetime.utcnow()
    grant = await db.get(UnlimitedGrant, grant_id)
    if grant is None:
        raise GrantError(404, f"no such grant {grant_id}")
    if grant.revoked_at is not None or grant.lapsed_at is not None:
        return grant

    grant.revoked_at = now
    grant.revoked_by_user_id = revoked_by_user_id
    grant.revoked_by_email = revoked_by_email
    grant.revoked_reason = (revoked_reason or "").strip() or "revoked"
    grant.updated_at = now
    await db.flush()

    await _end_entitlement(db, grant, reason=f"grant:revoked:{grant.id}", now=now)
    logger.info(
        "[entitlement] grant %s revoked: user=%s by=%s",
        grant.id, uidp(grant.granted_to_user_id), revoked_by_email,
    )
    if notify:  # see create_grant
        abuse_metrics.emit(
            "unlimited_grant_revoked", uid=uidp(grant.granted_to_user_id),
            grant=grant.id,
        )
        await _alert(
            "warning",
            f"UNLIMITED revoked from {uidp(grant.granted_to_user_id)} by "
            f"{revoked_by_email}: {grant.revoked_reason}",
            subject=uidp(grant.granted_to_user_id),
        )
    return grant


async def lapse_grants_for_sponsor(
    db: AsyncSession, *, original_txn_id: Optional[str] = None,
    sponsor_user_id: Optional[str] = None, reason: str,
    now: Optional[datetime] = None,
) -> list[UnlimitedGrant]:
    """End every open grant this sponsor was paying for, and return each
    grantee to free.

    Called from the Apple notification handler in the SAME transaction as the
    sponsor's own ``downgrade_to_free`` — that synchronicity is what satisfies
    "lapses automatically with it, no manual step, on every channel". The
    reconciler calls it too, as the backstop for the channel that has already
    dropped a message once.

    It deliberately does NOT consult ``_sponsor_is_live``: at the notification
    call site the mirror row still carries the OLD ``expires_date`` (the upsert
    runs afterwards), so a liveness re-check there would read the dying
    subscription as alive and lapse nothing.
    """
    now = now or datetime.utcnow()
    grants = await grants_sponsored_by(
        db, original_txn_id=original_txn_id, sponsor_user_id=sponsor_user_id,
    )
    for grant in grants:
        grant.lapsed_at = now
        grant.lapsed_reason = (reason or "sponsor_ended")[:120]
        grant.updated_at = now
    if grants:
        await db.flush()
    for grant in grants:
        await _end_entitlement(db, grant, reason=f"grant:lapsed:{reason}", now=now)
        logger.info(
            "[entitlement] grant %s lapsed (%s): user=%s",
            grant.id, grant.lapsed_reason, uidp(grant.granted_to_user_id),
        )
        abuse_metrics.emit(
            "unlimited_grant_lapsed", uid=uidp(grant.granted_to_user_id),
            grant=grant.id, reason=grant.lapsed_reason,
        )
        await _alert(
            "warning",
            f"UNLIMITED lapsed for {uidp(grant.granted_to_user_id)} — sponsor "
            f"ended ({grant.lapsed_reason})",
            subject=uidp(grant.granted_to_user_id),
        )
    return grants


async def sweep_dead_grants(
    db: AsyncSession, *, now: Optional[datetime] = None, apply: bool = True,
    defer_sponsors: Optional[set] = None,
) -> list[tuple[str, str]]:
    """Close every OPEN grant whose reason to exist is gone. Returns
    ``[(grant_id, why)]``.

    Two cases the notification hook structurally cannot cover, and one it
    already has but must not be trusted for alone:

    * ``expires_at`` passed — a backstop by definition has no notification.
    * the sponsor's ``apple_subscriptions`` row is GONE (their account was
      deleted; ``user_id`` is ON DELETE CASCADE). Apple sends nothing when a
      Toup account is deleted, so nothing else would ever notice.
    * the sponsor's own lapse notification never arrived. The Sandbox Elite
      has read ``status='active'`` for 2.7 months past its expiry with
      ``last_notification_uuid`` NULL — this channel has already dropped one.

    ``apply=False`` reports what it WOULD close and writes nothing, which is
    the mode the Apple reconciler runs in until an operator has read a clean
    pass (design §5.1).

    ``defer_sponsors`` is a set of Apple ``original_transaction_id``s whose
    mirror state the CALLER knows to be unresolved right now — deferred by the
    reconciler's API budget, unreadable at Apple, or simply not yet written in
    observe-only mode. `_sponsor_is_live` reads that mirror, and the mirror
    lying is the reason the reconciler exists, so judging a grant against an
    unresolved row is how a dropped DID_RENEW permanently lapses a LIVE
    sponsor's grant. Those grants are skipped and reconsidered next pass. The
    ``expires_at`` arm is deliberately NOT deferred: it is our own clock, not
    Apple's, and nothing about it is in doubt.
    """
    if not await _grants_table_present(db):
        return []
    now = now or datetime.utcnow()
    open_rows = (await db.execute(
        select(UnlimitedGrant).where(
            UnlimitedGrant.revoked_at.is_(None),
            UnlimitedGrant.lapsed_at.is_(None),
        )
    )).scalars().all()

    deferred = {str(t) for t in (defer_sponsors or ())}
    out: list[tuple[str, str]] = []
    for grant in open_rows:
        if grant.expires_at is not None and grant.expires_at <= now:
            why = "expires_at"
        elif (grant.sponsor_original_txn_id
                and str(grant.sponsor_original_txn_id) in deferred):
            logger.info(
                "[entitlement] grant %s: sponsor %s is unresolved this pass — "
                "deferring the liveness judgement",
                grant.id, grant.sponsor_original_txn_id,
            )
            continue
        elif not await _sponsor_is_live(db, grant, now=now):
            why = f"sponsor_dead:{grant.sponsor_kind}"
        else:
            continue
        out.append((grant.id, why))
        if not apply:
            logger.info(
                "[entitlement] would lapse grant %s (%s) for user=%s",
                grant.id, why, uidp(grant.granted_to_user_id),
            )
            continue
        grant.lapsed_at = now
        grant.lapsed_reason = why[:120]
        grant.updated_at = now
        await db.flush()
        await _end_entitlement(db, grant, reason=f"grant:lapsed:{why}", now=now)
        logger.info(
            "[entitlement] grant %s lapsed by sweep (%s): user=%s",
            grant.id, why, uidp(grant.granted_to_user_id),
        )
        abuse_metrics.emit(
            "unlimited_grant_lapsed", uid=uidp(grant.granted_to_user_id),
            grant=grant.id, reason=why,
        )
        await _alert(
            "warning",
            f"UNLIMITED lapsed for {uidp(grant.granted_to_user_id)} — {why}",
            subject=uidp(grant.granted_to_user_id),
        )
    return out


async def _end_entitlement(
    db: AsyncSession, grant, *, reason: str, now: datetime,
) -> bool:
    """Return the grantee to free, unless they hold the entitlement for a
    reason this grant was not providing.

    Two guards, and both have teeth:

    * **The balance must actually be on the Unlimited plan.** Otherwise a
      revoke would downgrade whatever plan the account has since moved to.
    * **The grantee must not qualify independently** — their own live Apple
      subscription, or a second open grant that outlives this one. Downgrading
      a paying customer because a comp ended is the worst outcome available
      here.
    """
    from app.services.credit_service import credit_service

    user_id = grant.granted_to_user_id
    balance = await db.get(CreditBalance, user_id)
    if balance is None or balance.plan_id != UNLIMITED_PLAN_ID:
        return False

    sub = await has_live_apple_entitlement(db, user_id, now=now)
    if sub is not None:
        logger.info(
            "[entitlement] grant %s ended but user=%s has their own live Apple "
            "subscription; leaving the entitlement in place",
            grant.id, uidp(user_id),
        )
        return False
    other = await live_grant_for(db, user_id, now=now)
    if other is not None and other.id != grant.id:
        logger.info(
            "[entitlement] grant %s ended but user=%s holds live grant %s; "
            "leaving the entitlement in place", grant.id, uidp(user_id), other.id,
        )
        return False

    await credit_service.downgrade_to_free(db, user_id, reason)
    return True


async def _alert(level: str, message: str, *, subject: str) -> None:
    """A comped account is a thing the operator should SEE happen — including
    when it is the operator doing it. Never allowed to fail a mutation.
    """
    try:
        from app.services.alerting import send_infra_alert
        await send_infra_alert("unlimited-grant", level, message, subject=subject)
    except Exception as e:  # pragma: no cover - alerting is best-effort
        logger.warning("[entitlement] alert failed: %s", e)

# ── The controller-side answer ───────────────────────────────────────────


async def resolve_unlimited(
    db: AsyncSession, user_id: str, *, now: Optional[datetime] = None,
) -> tuple[bool, str]:
    """``(should_hold_unlimited, reason)`` for one account, from first
    principles rather than from the stamp.

    Reasons, in the order they are checked:

      ``admin``       role == 'admin' — already unlimited via
                      ``_is_unlimited_user``; nothing to stamp.
      ``apple``       a live PRODUCTION Apple subscription whose product
                      resolves to the Unlimited plan — i.e. the Unlimited
                      product, or a legacy product on a GRANDFATHERED
                      subscription.
      ``grant``       a live admin override.
      ``plan``        the balance already says ``unlimited`` and nothing above
                      explains it. This is the reconciler's ``orphan_entitlement``
                      class, NOT a justification — the caller decides.
      ``none``        no entitlement.

    This is the reconciler's and the admin router's question. It is NOT the
    charge path's question; that one is answered by
    ``credit_service._entitlement_is_unlimited`` off the locked balance row.
    """
    from app.services.apple_iap_service import plan_for_subscription

    now = now or datetime.utcnow()
    user = await db.get(User, user_id)
    if user is not None and getattr(user, "role", None) == "admin":
        return (True, "admin")

    sub = await has_live_apple_entitlement(db, user_id, now=now)
    if sub is not None:
        try:
            if plan_for_subscription(sub.product_id, sub) == UNLIMITED_PLAN_ID:
                return (True, "apple")
        except KeyError:
            logger.warning(
                "[entitlement] unknown product %r on a live Production row for "
                "user=%s — treating as not-unlimited", sub.product_id, user_id,
            )

    if await live_grant_for(db, user_id, now=now) is not None:
        return (True, "grant")

    balance = await db.get(CreditBalance, user_id)
    if getattr(balance, "plan_id", None) == UNLIMITED_PLAN_ID:
        return (True, "plan")
    return (False, "none")
