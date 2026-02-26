"""
Stripe integration helpers for VPS checkout sessions.
"""

import logging
from typing import Optional

import stripe

from app.config import settings

logger = logging.getLogger(__name__)

# Map plan IDs to Stripe Price IDs from config
PLAN_PRICE_MAP = {
    "starter": "stripe_starter_price_id",
    "standard": "stripe_standard_price_id",
    "pro": "stripe_pro_price_id",
}


def _get_stripe_client() -> stripe.StripeClient:
    if not settings.stripe_secret_key:
        raise ValueError("STRIPE_SECRET_KEY is not configured")
    return stripe.StripeClient(settings.stripe_secret_key)


def create_checkout_session(
    user_id: str,
    user_email: str,
    plan_id: str,
    vps_instance_id: str,
    success_url: str,
    cancel_url: str,
) -> dict:
    """
    Create a Stripe Checkout Session for a VPS subscription.

    Returns the session dict with at least:
        { "id": "cs_...", "url": "https://checkout.stripe.com/..." }

    The vps_instance_id is stored in session metadata so the webhook
    can look up the right VPSInstance record.
    """
    client = _get_stripe_client()

    price_id_attr = PLAN_PRICE_MAP.get(plan_id)
    if not price_id_attr:
        raise ValueError(f"Unknown plan_id: {plan_id}")

    stripe_price_id: str = getattr(settings, price_id_attr, "")

    if not stripe_price_id:
        raise ValueError(
            f"Stripe price ID for plan '{plan_id}' is not configured "
            f"(set {price_id_attr.upper()} in environment)"
        )

    session = client.checkout.sessions.create(
        params={
            "mode": "subscription",
            "line_items": [{"price": stripe_price_id, "quantity": 1}],
            "customer_email": user_email,
            "success_url": success_url,
            "cancel_url": cancel_url,
            "metadata": {
                "user_id": user_id,
                "vps_instance_id": vps_instance_id,
                "plan_id": plan_id,
            },
            "subscription_data": {
                "metadata": {
                    "user_id": user_id,
                    "vps_instance_id": vps_instance_id,
                }
            },
        }
    )

    return {"id": session.id, "url": session.url}


def create_bundle_checkout_session(
    user_id: str,
    user_email: str,
    success_url: str,
    cancel_url: str,
) -> dict:
    """
    Create a Stripe Checkout Session for the $40/mo LLM bundle subscription.
    """
    client = _get_stripe_client()

    price_id = settings.stripe_llm_bundle_price_id
    if not price_id:
        raise ValueError(
            "STRIPE_LLM_BUNDLE_PRICE_ID is not configured"
        )

    session = client.checkout.sessions.create(
        params={
            "mode": "subscription",
            "line_items": [{"price": price_id, "quantity": 1}],
            "customer_email": user_email,
            "success_url": success_url,
            "cancel_url": cancel_url,
            "metadata": {
                "user_id": user_id,
                "type": "llm_bundle",
            },
            "subscription_data": {
                "metadata": {
                    "user_id": user_id,
                    "type": "llm_bundle",
                }
            },
        }
    )

    return {"id": session.id, "url": session.url}


def verify_webhook(payload: bytes, sig_header: str) -> Optional[dict]:
    """
    Verify a Stripe webhook signature and return the parsed event dict,
    or None if the signature is invalid.
    """
    webhook_secret = settings.stripe_webhook_secret
    if not webhook_secret:
        logger.warning("STRIPE_WEBHOOK_SECRET not set — skipping signature verification")
        import json
        return json.loads(payload)

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
        return dict(event)
    except stripe.error.SignatureVerificationError:
        logger.warning("Stripe webhook signature verification failed")
        return None
    except Exception as exc:
        logger.exception("Error parsing Stripe webhook: %s", exc)
        return None
