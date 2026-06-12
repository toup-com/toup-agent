"""StoreKit In-App Purchase — consumable credit-pack endpoints.

POST /iap/apple/verify        — client posts a StoreKit2 transactionId after
                                a purchase; we verify it with Apple and grant
                                the mapped credits to the user's non-expiring
                                purchased wallet. Idempotent on transaction_id.
POST /iap/apple/notifications — App Store Server Notifications V2 webhook
                                (no user auth; JWS-verified). On REFUND /
                                REFUND_REVERSED for a known consumable, claw
                                back / restore the granted purchased credits.

See app/services/apple_iap_service.py for the verification surface and the
server-authoritative product → credits map. The whole feature is inert until
the IAP key is configured (verify → 503).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db
from app.db.models import (
    APPLE_SUB_ACTIVE, APPLE_SUB_BILLING_RETRY, APPLE_SUB_EXPIRED,
    APPLE_SUB_GRACE, APPLE_SUB_REVOKED,
    AppleSubscription, CreditLedger, User,
)
from app.services import apple_iap_service
from app.services.apple_iap_service import (
    APPLE_SUB_PRODUCT_TO_PLAN, PRODUCT_CREDITS, IapVerificationError,
)
from app.services.credit_service import credit_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/iap", tags=["iap"])


# ── /apple/verify ───────────────────────────────────────────────────


class AppleVerifyRequest(BaseModel):
    transaction_id: str
    product_id: str
    environment: Optional[str] = "Production"  # "Production" | "Sandbox" | "Xcode"


class AppleVerifyResponse(BaseModel):
    ok: bool
    product_id: str
    credits_granted: int
    message_credits_remaining: float
    purchased_credits_remaining: float
    already_redeemed: bool


@router.post("/apple/verify", response_model=AppleVerifyResponse)
async def apple_verify(
    body: AppleVerifyRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AppleVerifyResponse:
    """Verify a StoreKit purchase and grant its credits. Idempotent."""
    if not apple_iap_service.iap_configured():
        # Client treats 503 as "purchases temporarily unavailable".
        raise HTTPException(503, "In-App Purchases are not configured on the server.")

    if body.product_id not in PRODUCT_CREDITS:
        raise HTTPException(400, f"Unknown product: {body.product_id!r}")

    try:
        verified = await apple_iap_service.verify_transaction(
            body.transaction_id, body.environment or "Production",
        )
    except IapVerificationError as e:
        logger.warning("[iap] verify failed user=%s txn=%s: %s",
                       current_user.id, body.transaction_id, e)
        raise HTTPException(422, f"Receipt verification failed: {e}")

    # Defend against a client mislabeling the product: trust the decoded
    # transaction's product id, not the client-supplied one.
    if verified.product_id != body.product_id:
        logger.warning(
            "[iap] product mismatch user=%s client=%s verified=%s",
            current_user.id, body.product_id, verified.product_id,
        )
        raise HTTPException(422, "Product id does not match the verified transaction.")

    balance_after, already_redeemed = await credit_service.grant_purchased(
        db, current_user.id, verified.credits,
        idempotency_key=verified.transaction_id,
        metadata={
            "product_id": verified.product_id,
            "original_transaction_id": verified.original_transaction_id,
            "environment": verified.environment,
            "surface": "iap_apple_verify",
        },
    )
    view = await credit_service.get_balance_view(db, current_user.id)
    await db.commit()

    return AppleVerifyResponse(
        ok=True,
        product_id=verified.product_id,
        credits_granted=int(verified.credits),
        message_credits_remaining=float(view.message_credits_remaining),
        purchased_credits_remaining=float(view.purchased_credits_remaining),
        already_redeemed=already_redeemed,
    )


# ── /apple/subscribe/verify ──────────────────────────────────────────


class AppleSubscribeRequest(BaseModel):
    transaction_id: str
    product_id: str
    environment: Optional[str] = "Production"  # "Production" | "Sandbox" | "Xcode"


class AppleSubscribeResponse(BaseModel):
    ok: bool
    plan_id: str
    plan_source: str  # always 'apple' here
    renews_at: Optional[str] = None
    already_active: bool


@router.post("/apple/subscribe/verify", response_model=AppleSubscribeResponse)
async def apple_subscribe_verify(
    body: AppleSubscribeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AppleSubscribeResponse:
    """Verify an Apple auto-renewable subscription purchase and activate the
    matching plan. Idempotent on ``original_transaction_id``.

    Reconciliation: a user has at most one active paid subscription. If the
    user already has an active STRIPE plan we refuse with 409 (the mobile UI
    hides Subscribe in that state; this is the server-side race backstop).
    """
    if not apple_iap_service.iap_configured():
        raise HTTPException(503, "In-App Purchases are not configured on the server.")

    if body.product_id not in APPLE_SUB_PRODUCT_TO_PLAN:
        raise HTTPException(400, f"Unknown product: {body.product_id!r}")

    # Reconciliation (§3a): never auto-grant a second sub over an active Stripe
    # one. Same-platform ('apple') is NOT a conflict — it's the normal
    # re-verify / upgrade path handled by the idempotent activate below.
    src = await credit_service.active_paid_source(db, current_user.id)
    if src == "stripe":
        logger.warning(
            "[iap] apple subscribe blocked — user=%s already has active stripe sub",
            current_user.id,
        )
        raise HTTPException(409, detail={
            "code": "subscription_exists_other_platform",
            "message": "You already have a subscription via the web. Manage it at toup.ai/account.",
        })

    try:
        verified = await apple_iap_service.verify_subscription_transaction(
            body.transaction_id, body.environment or "Production",
        )
    except IapVerificationError as e:
        logger.warning("[iap] subscription verify failed user=%s txn=%s: %s",
                       current_user.id, body.transaction_id, e)
        raise HTTPException(422, f"Receipt verification failed: {e}")

    # Defend against a client mislabeling the product.
    if verified.product_id != body.product_id:
        logger.warning(
            "[iap] subscription product mismatch user=%s client=%s verified=%s",
            current_user.id, body.product_id, verified.product_id,
        )
        raise HTTPException(422, "Product id does not match the verified transaction.")

    # Idempotent upsert keyed on the lifecycle anchor.
    sub = (await db.execute(
        select(AppleSubscription).where(
            AppleSubscription.original_transaction_id == verified.original_transaction_id
        )
    )).scalar_one_or_none()

    already_active = False
    if (
        sub is not None
        and sub.status == APPLE_SUB_ACTIVE
        and sub.expires_date is not None
        and verified.expires_date is not None
        and sub.expires_date >= verified.expires_date
    ):
        # A verify replay / restore-purchases tap must not re-grant.
        already_active = True
    else:
        if sub is None:
            sub = AppleSubscription(
                user_id=current_user.id,
                original_transaction_id=verified.original_transaction_id,
                product_id=verified.product_id,
                plan_id=verified.plan_id,
                status=APPLE_SUB_ACTIVE,
                expires_date=verified.expires_date,
                auto_renew_status=True,
                environment=verified.environment,
            )
            db.add(sub)
        else:
            sub.user_id = current_user.id
            sub.product_id = verified.product_id
            sub.plan_id = verified.plan_id
            sub.status = APPLE_SUB_ACTIVE
            sub.expires_date = verified.expires_date
            sub.environment = verified.environment
        await credit_service.activate_subscription(
            db, current_user.id, verified.plan_id, "apple",
            verified.expires_date or _fallback_period_end(),
        )

    await db.commit()

    return AppleSubscribeResponse(
        ok=True,
        plan_id=verified.plan_id,
        plan_source="apple",
        renews_at=verified.expires_date.isoformat() if verified.expires_date else None,
        already_active=already_active,
    )


def _fallback_period_end() -> datetime:
    """Apple should always send expiresDate, but never persist a NULL period
    window — fall back to a 30-day month so the renewal guard still has a
    sensible bound if Apple omits it."""
    return datetime.utcnow() + timedelta(days=30)


# ── /apple/notifications (App Store Server Notifications V2) ──────────


@router.post("/apple/notifications")
async def apple_notifications(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """App Store Server Notifications V2 webhook. No user auth — JWS-verified.

    Best-effort. Returns 200 for anything handled (so Apple stops retrying)
    and 400 ONLY for an unverifiable signature. Never 500 on malformed input.
    On REFUND / REFUND_REVERSED for a known consumable transaction we claw
    back / restore the granted purchased credits.
    """
    if not apple_iap_service.iap_configured():
        # Nothing to verify against; ack so Apple doesn't hammer retries.
        logger.info("[iap] notification received but IAP not configured; acking")
        return Response(status_code=200)

    try:
        raw = await request.body()
        payload = raw.decode("utf-8") if raw else ""
        import json
        signed_payload = json.loads(payload).get("signedPayload", "")
    except Exception as e:
        logger.warning("[iap] notification body unparseable: %s", e)
        return Response(status_code=200)  # malformed → ack, don't 500/retry

    if not signed_payload:
        logger.warning("[iap] notification missing signedPayload")
        return Response(status_code=200)

    try:
        decoded = await apple_iap_service.verify_and_decode_notification(signed_payload)
    except IapVerificationError as e:
        # Unverifiable signature is the one case we return 400 for.
        logger.warning("[iap] notification signature unverifiable: %s", e)
        return Response(status_code=400)
    except Exception as e:
        logger.warning("[iap] notification verify unexpected error: %s", e)
        return Response(status_code=200)

    try:
        await _handle_notification(db, decoded)
        await db.commit()
    except Exception as e:
        # A handler error must not turn into a 500 retry storm — log + ack.
        logger.exception("[iap] notification handler error: %s", e)
        try:
            await db.rollback()
        except Exception:
            pass
    return Response(status_code=200)


def _decode_inner_transaction(signed_txn: str):
    """Decode a notification's inner signedTransactionInfo (already chain-
    verified as part of the notification). Returns the decoded txn or None.

    Reuses the dual-env verifier path; notifications carry no env hint.
    """
    from appstoreserverlibrary.models.Environment import Environment
    last_err = None
    for env in (Environment.PRODUCTION, Environment.SANDBOX):
        try:
            return apple_iap_service._verify_and_decode_transaction(env, signed_txn)
        except IapVerificationError as e:
            last_err = e
            continue
    logger.warning("[iap] could not decode notification transaction: %s", last_err)
    return None


def _decode_inner_renewal_info(signed_renewal_info: Optional[str]):
    """Decode a notification's inner signedRenewalInfo, or None if absent."""
    if not signed_renewal_info:
        return None
    from appstoreserverlibrary.models.Environment import Environment
    last_err = None
    for env in (Environment.PRODUCTION, Environment.SANDBOX):
        try:
            return apple_iap_service._verify_and_decode_renewal_info(env, signed_renewal_info)
        except IapVerificationError as e:
            last_err = e
            continue
    logger.warning("[iap] could not decode notification renewal info: %s", last_err)
    return None


async def _handle_notification(db: AsyncSession, decoded) -> None:
    """Dispatch an App Store Server Notification V2 by product class.

    Consumable credit packs (``PRODUCT_CREDITS``) keep the existing REFUND /
    REFUND_REVERSED clawback/regrant path, UNCHANGED. Subscription products
    (``APPLE_SUB_PRODUCT_TO_PLAN``) route to the lifecycle handler.
    """
    data = getattr(decoded, "data", None)
    signed_txn = getattr(data, "signedTransactionInfo", None) if data is not None else None
    if not signed_txn:
        # Some notification types (e.g. TEST, EXTERNAL_PURCHASE_TOKEN) carry no
        # transaction — nothing actionable.
        return

    decoded_txn = _decode_inner_transaction(signed_txn)
    if decoded_txn is None:
        return

    product_id = getattr(decoded_txn, "productId", None)

    if product_id in APPLE_SUB_PRODUCT_TO_PLAN:
        renewal_info = _decode_inner_renewal_info(
            getattr(data, "signedRenewalInfo", None) if data is not None else None
        )
        await _handle_subscription_notification(db, decoded, decoded_txn, renewal_info)
        return

    if product_id in PRODUCT_CREDITS:
        await _handle_consumable_refund_notification(db, decoded, decoded_txn)
        return

    # Unknown product — log + ignore (don't 500, don't act).
    logger.info("[iap] notification for unrecognised product %r; ignoring", product_id)


async def _handle_consumable_refund_notification(db: AsyncSession, decoded, decoded_txn) -> None:
    """Original consumable credit-pack path: REFUND clawback / REFUND_REVERSED
    regrant. Behaviour is byte-for-byte the pre-subscription logic."""
    notification_type = getattr(decoded, "notificationType", None)
    type_str = getattr(notification_type, "value", str(notification_type or ""))
    if type_str not in ("REFUND", "REFUND_REVERSED"):
        return  # We only care about consumable refunds.

    product_id = getattr(decoded_txn, "productId", None)
    transaction_id = str(getattr(decoded_txn, "transactionId", "") or "")
    if not transaction_id:
        return

    # Find the original grant ledger row (idempotency_key == transaction_id)
    # to recover which user it belongs to. Refund the credits we minted.
    grant_row = (await db.execute(
        select(CreditLedger).where(CreditLedger.idempotency_key == transaction_id)
    )).scalars().first()
    if grant_row is None:
        logger.info("[iap] %s for txn=%s but no grant ledger row found; skipping",
                    type_str, transaction_id)
        return

    user_id = grant_row.user_id
    credits = PRODUCT_CREDITS[product_id]

    if type_str == "REFUND":
        await credit_service.clawback_purchased(
            db, user_id, credits,
            idempotency_key=f"{transaction_id}:refund",
            metadata={"product_id": product_id, "notification": "REFUND"},
        )
        logger.info("[iap] clawed back %s credits for user=%s txn=%s (REFUND)",
                    credits, user_id, transaction_id)
    else:  # REFUND_REVERSED — Apple reinstated the purchase; re-grant.
        await credit_service.grant_purchased(
            db, user_id, credits,
            idempotency_key=f"{transaction_id}:refund_reversed",
            metadata={"product_id": product_id, "notification": "REFUND_REVERSED"},
        )
        logger.info("[iap] re-granted %s credits for user=%s txn=%s (REFUND_REVERSED)",
                    credits, user_id, transaction_id)


# notificationType values that, when seen for a subscription product, end the
# entitlement and require a downgrade to free. NEVER includes
# AUTO_RENEW_DISABLED or DID_FAIL_TO_RENEW/GRACE_PERIOD — those keep the user
# active through their paid window.
_SUB_DOWNGRADE_TYPES = frozenset({
    "EXPIRED", "GRACE_PERIOD_EXPIRED", "REVOKE", "REFUND",
})


async def _handle_subscription_notification(db: AsyncSession, decoded, decoded_txn, renewal_info) -> None:
    """Apply an App Store Server Notification V2 to a subscription's lifecycle.

    Implements the notificationType→action table (mig-063 §4b) with the strict
    downgrade discipline: only EXPIRED / GRACE_PERIOD_EXPIRED / REVOKE / REFUND
    downgrade. Deduped on ``notificationUUID`` (stored on the
    apple_subscriptions row) so a replayed notification is a no-op — this
    protects the non-idempotent apply_plan_change from Apple's at-least-once
    delivery.
    """
    notification_type = getattr(decoded, "notificationType", None)
    type_str = getattr(notification_type, "value", str(notification_type or ""))
    subtype = getattr(decoded, "subtype", None)
    subtype_str = getattr(subtype, "value", str(subtype or "")) if subtype is not None else ""
    notif_uuid = getattr(decoded, "notificationUUID", None)

    product_id = getattr(decoded_txn, "productId", None)
    plan_id = APPLE_SUB_PRODUCT_TO_PLAN[product_id]
    original_txn = getattr(decoded_txn, "originalTransactionId", None)
    if not original_txn:
        logger.warning("[iap] sub notification %s without originalTransactionId; skipping", type_str)
        return
    original_txn = str(original_txn)

    expires_date = apple_iap_service.ms_to_datetime(getattr(decoded_txn, "expiresDate", None))
    grace_expires = apple_iap_service.ms_to_datetime(
        getattr(renewal_info, "gracePeriodExpiresDate", None)
    ) if renewal_info is not None else None
    environment = getattr(decoded_txn, "environment", None)
    env_str = getattr(environment, "value", str(environment or "Production"))

    # auto_renew + pending-downgrade target from the renewal info.
    auto_renew_status = True
    auto_renew_product_id = None
    if renewal_info is not None:
        ars = getattr(renewal_info, "autoRenewStatus", None)
        # AutoRenewStatus.ON == 1, OFF == 0.
        ars_val = getattr(ars, "value", ars)
        if ars_val is not None:
            auto_renew_status = bool(ars_val)
        auto_renew_product_id = getattr(renewal_info, "autoRenewProductId", None)

    # Locate / create the lifecycle row (keyed on the stable original txn id).
    sub = (await db.execute(
        select(AppleSubscription).where(
            AppleSubscription.original_transaction_id == original_txn
        )
    )).scalar_one_or_none()

    # Dedup: a replayed notification (same UUID on the same row) is a no-op.
    if sub is not None and notif_uuid and sub.last_notification_uuid == notif_uuid:
        logger.info("[iap] sub notification %s replay (uuid=%s); skipping", type_str, notif_uuid)
        return

    # We can only act on credits when we know the user. For a brand-new
    # SUBSCRIBED we may not have a row yet — without a user we can't grant.
    user_id = sub.user_id if sub is not None else None

    # ── credit action by notificationType / subtype ──
    if type_str in ("SUBSCRIBED", "OFFER_REDEEMED"):
        if user_id is not None:
            await credit_service.activate_subscription(
                db, user_id, plan_id, "apple", expires_date or _fallback_period_end(),
            )
        else:
            # No prior row → no user mapping. The /subscribe/verify call (which
            # carries the authed user) is the authoritative activation; the
            # SUBSCRIBED notification then just mirrors state. Log so a missing
            # verify is visible to support.
            logger.warning(
                "[iap] SUBSCRIBED for orig_txn=%s with no local row/user; "
                "awaiting /subscribe/verify to map the user", original_txn,
            )

    elif type_str == "DID_RENEW":
        if user_id is not None:
            await credit_service._apple_renew(
                db, user_id, plan_id, expires_date or _fallback_period_end(),
            )

    elif type_str == "DID_CHANGE_RENEWAL_PREF":
        if subtype_str == "UPGRADE" and user_id is not None:
            # Immediate: a new txn was issued; switch plan now, prorated.
            await credit_service.activate_subscription(
                db, user_id, plan_id, "apple", expires_date or _fallback_period_end(),
            )
        # DOWNGRADE (and any other subtype): deferred — store the pending
        # target only; the plan changes at the next DID_RENEW. No credit change.

    elif type_str == "DID_CHANGE_RENEWAL_STATUS":
        # AUTO_RENEW_ENABLED / AUTO_RENEW_DISABLED — keep ACTIVE through
        # expiresDate; only the auto_renew flag changes. NO downgrade.
        pass

    elif type_str == "DID_FAIL_TO_RENEW":
        # GRACE_PERIOD → keep active through gracePeriodExpiresDate; otherwise
        # at-risk (billing retry). Either way NO downgrade — we wait for
        # EXPIRED / GRACE_PERIOD_EXPIRED.
        pass

    elif type_str in _SUB_DOWNGRADE_TYPES:
        if user_id is not None:
            reason = f"apple:{type_str.lower()}"
            if type_str == "EXPIRED" and subtype_str:
                reason = f"apple:expired:{subtype_str.lower()}"
            await credit_service.downgrade_to_free(db, user_id, reason)

    elif type_str == "REFUND_REVERSED":
        # Re-grant the lapsed sub if the user is currently free and the txn is
        # still within its expires window.
        if user_id is not None:
            psrc = await credit_service.active_paid_source(db, user_id)
            within_window = (
                expires_date is not None and expires_date >= datetime.utcnow()
            )
            if psrc is None and within_window:
                await credit_service.activate_subscription(
                    db, user_id, plan_id, "apple", expires_date,
                )
    # All other types (PRICE_INCREASE, METADATA_UPDATE, RENEWAL_EXTENDED,
    # MIGRATION, CONSUMPTION_REQUEST, TEST, …) → no entitlement change.

    # ── upsert the lifecycle mirror regardless of whether credits changed ──
    new_status = _sub_status_for(type_str, subtype_str, grace_expires)
    if sub is None:
        # We have no user mapping for a fresh notification (e.g. SUBSCRIBED that
        # arrives before /subscribe/verify). Persist a mirror row anchored on
        # the original txn so the eventual verify can find/update it; user_id
        # is required (FK), so skip persistence when unknown.
        if user_id is None:
            return
        sub = AppleSubscription(
            user_id=user_id,
            original_transaction_id=original_txn,
            product_id=product_id,
            plan_id=plan_id,
            status=new_status,
            expires_date=expires_date,
            auto_renew_status=auto_renew_status,
            auto_renew_product_id=auto_renew_product_id,
            environment=env_str,
            last_notification_uuid=notif_uuid,
        )
        db.add(sub)
    else:
        sub.product_id = product_id
        sub.plan_id = plan_id
        sub.status = new_status
        if expires_date is not None:
            sub.expires_date = expires_date
        sub.auto_renew_status = auto_renew_status
        if auto_renew_product_id is not None:
            sub.auto_renew_product_id = auto_renew_product_id
        sub.environment = env_str
        sub.last_notification_uuid = notif_uuid
    await db.flush()


def _sub_status_for(type_str: str, subtype_str: str, grace_expires) -> str:
    """Map a notificationType/subtype to the apple_subscriptions.status mirror."""
    if type_str in ("EXPIRED", "GRACE_PERIOD_EXPIRED"):
        return APPLE_SUB_EXPIRED
    if type_str == "REVOKE":
        return APPLE_SUB_REVOKED
    if type_str == "REFUND":
        return APPLE_SUB_EXPIRED
    if type_str == "DID_FAIL_TO_RENEW":
        return APPLE_SUB_GRACE if subtype_str == "GRACE_PERIOD" else APPLE_SUB_BILLING_RETRY
    # SUBSCRIBED / DID_RENEW / OFFER_REDEEMED / renewal-pref/status changes /
    # REFUND_REVERSED → active.
    return APPLE_SUB_ACTIVE
