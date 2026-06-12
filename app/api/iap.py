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
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db
from app.db.models import CreditLedger, User
from app.services import apple_iap_service
from app.services.apple_iap_service import (
    PRODUCT_CREDITS, IapVerificationError,
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


async def _handle_notification(db: AsyncSession, decoded) -> None:
    """Act on REFUND / REFUND_REVERSED for a known consumable transaction."""
    notification_type = getattr(decoded, "notificationType", None)
    type_str = getattr(notification_type, "value", str(notification_type or ""))
    if type_str not in ("REFUND", "REFUND_REVERSED"):
        return  # We only care about consumable refunds.

    data = getattr(decoded, "data", None)
    signed_txn = getattr(data, "signedTransactionInfo", None) if data is not None else None
    if not signed_txn:
        logger.info("[iap] %s notification without signedTransactionInfo; skipping", type_str)
        return

    # Decode the inner transaction (already chain-verified as part of the
    # notification). Reuse the same verifier path.
    from appstoreserverlibrary.models.Environment import Environment
    decoded_txn = None
    last_err = None
    for env in (Environment.PRODUCTION, Environment.SANDBOX):
        try:
            decoded_txn = apple_iap_service._verify_and_decode_transaction(env, signed_txn)
            break
        except IapVerificationError as e:
            last_err = e
            continue
    if decoded_txn is None:
        logger.warning("[iap] could not decode refund transaction: %s", last_err)
        return

    product_id = getattr(decoded_txn, "productId", None)
    if product_id not in PRODUCT_CREDITS:
        return  # Not one of our credit packs — ignore (could be a sub, etc.)

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
