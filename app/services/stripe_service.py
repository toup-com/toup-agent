"""
Stripe integration — checkout sessions, webhook verification,
customer management, billing portal, and subscription helpers.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import stripe

from app.config import settings

logger = logging.getLogger(__name__)

# Pin Stripe API version explicitly. Default-version drift through wheel
# upgrades is what removed `invoice.payment_intent` in Basil 2025-03-31 and
# silently broke create-subscription in prod. Keep the contract loud in code.
# When bumping, audit changelogs for affected fields and update tests + skill.
stripe.api_version = "2025-03-31.basil"

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


# ── Customer management ──────────────────────────────────────────────


def get_or_create_customer(user_id: str, email: str, name: Optional[str] = None) -> str:
    """
    Return an existing Stripe customer ID for the email, or create one.
    Always tags metadata.user_id so we can reconcile later.
    """
    client = _get_stripe_client()

    # Search by email first
    existing = client.customers.list(params={"email": email, "limit": 1})
    if existing.data:
        cust = existing.data[0]
        # Ensure our user_id is in metadata. NB: stripe-python 15.x exposes
        # `cust.metadata` as a StripeObject, not a dict — calling `.get(...)`
        # raises `AttributeError: get`. Use `getattr` so the lookup works on
        # both StripeObject (via __getattr__) and plain dicts.
        if getattr(cust.metadata, "user_id", None) != user_id:
            client.customers.update(
                cust.id,
                params={"metadata": {"user_id": user_id}},
            )
        return cust.id

    # Create new customer
    cust = client.customers.create(
        params={
            "email": email,
            "name": name or "",
            "metadata": {"user_id": user_id},
        }
    )
    return cust.id


def create_subscription_with_intent(
    customer_id: str,
    price_id: str,
    user_id: str,
) -> dict:
    """
    Create a Subscription with payment_behavior='default_incomplete'.
    Returns the client_secret for the frontend PaymentElement to confirm,
    plus the subscription ID.
    """
    client = _get_stripe_client()
    sub = client.subscriptions.create(
        params={
            "customer": customer_id,
            "items": [{"price": price_id}],
            "payment_behavior": "default_incomplete",
            "payment_settings": {
                "save_default_payment_method": "on_subscription",
            },
            "metadata": {
                "user_id": user_id,
                "type": "llm_bundle",
            },
            "expand": ["latest_invoice.confirmation_secret"],
        }
    )
    return {
        "subscription_id": sub.id,
        "client_secret": sub.latest_invoice.confirmation_secret.client_secret,
        "status": sub.status,
    }


def get_active_subscription_for_customer(customer_id: str, price_id: str) -> Optional[dict]:
    """Check if the customer already has an active subscription for this price."""
    try:
        client = _get_stripe_client()
        subs = client.subscriptions.list(
            params={
                "customer": customer_id,
                "price": price_id,
                "status": "active",
                "limit": 1,
            }
        )
        if subs.data:
            return {"subscription_id": subs.data[0].id, "status": "active"}
        # Also check incomplete (payment pending)
        subs_inc = client.subscriptions.list(
            params={
                "customer": customer_id,
                "price": price_id,
                "status": "incomplete",
                "limit": 1,
                "expand": ["data.latest_invoice.confirmation_secret"],
            }
        )
        if subs_inc.data:
            cs = subs_inc.data[0].latest_invoice.confirmation_secret
            return {
                "subscription_id": subs_inc.data[0].id,
                "client_secret": cs.client_secret if cs else None,
                "status": "incomplete",
            }
        return None
    except Exception as exc:
        logger.warning("Failed to check existing subscriptions: %s", exc)
        return None


def delete_customer(customer_id: str) -> bool:
    """Delete a Stripe Customer. Cascade-cancels all of their subscriptions
    as a Stripe-side side effect (per Stripe docs).

    Returns True if the call succeeded (or the customer was already deleted),
    False on any other Stripe error. Idempotent — calling twice is safe.

    Used by admin user-delete to wipe Stripe state alongside platform DB.
    Without this, deleting a user from the admin panel leaves an orphan
    Stripe customer + active subscription that keeps charging.
    """
    try:
        client = _get_stripe_client()
        client.customers.delete(customer_id)
        return True
    except stripe.error.InvalidRequestError as exc:
        # 'No such customer' or 'already deleted' — treat as idempotent success.
        if "No such customer" in str(exc) or "already" in str(exc).lower():
            return True
        logger.warning("Stripe delete_customer InvalidRequest %s: %s", customer_id, exc)
        return False
    except Exception as exc:
        logger.warning("Stripe delete_customer failed %s: %s", customer_id, exc)
        return False


def find_customers_by_email(email: str) -> list[str]:
    """Return all Stripe Customer IDs matching the given email. Used as a
    defense-in-depth fallback by admin user-delete: even if `User.stripe_
    customer_id` was somehow not synced, we still find + delete any orphan
    customer whose email matches. Returns [] on lookup error.
    """
    try:
        client = _get_stripe_client()
        result = client.customers.list(params={"email": email, "limit": 10})
        return [c.id for c in result.data]
    except Exception as exc:
        logger.warning("Stripe find_customers_by_email failed %s: %s", email, exc)
        return []


def create_billing_portal_session(customer_id: str, return_url: str) -> str:
    """Create a Stripe Customer Portal session. Returns the portal URL."""
    client = _get_stripe_client()
    session = client.billing_portal.sessions.create(
        params={
            "customer": customer_id,
            "return_url": return_url,
        }
    )
    return session.url


# ── Subscription helpers ─────────────────────────────────────────────


def get_subscription(subscription_id: str) -> Optional[dict]:
    """Retrieve a subscription from Stripe. Returns None on error."""
    try:
        client = _get_stripe_client()
        sub = client.subscriptions.retrieve(subscription_id)
        return {
            "id": sub.id,
            "status": sub.status,
            "current_period_end": datetime.fromtimestamp(
                sub.current_period_end, tz=timezone.utc
            ),
            "cancel_at_period_end": sub.cancel_at_period_end,
            "customer": sub.customer,
        }
    except Exception as exc:
        logger.warning("Failed to retrieve subscription %s: %s", subscription_id, exc)
        return None


def get_price(price_id: str) -> Optional[dict]:
    """Retrieve a Stripe Price. Returns dict with unit_amount, currency, nickname."""
    try:
        client = _get_stripe_client()
        price = client.prices.retrieve(price_id, params={"expand": ["product"]})
        product = price.product
        return {
            "id": price.id,
            "unit_amount": price.unit_amount,
            "currency": price.currency,
            "interval": price.recurring.interval if price.recurring else None,
            "product_name": product.name if hasattr(product, "name") else "",
        }
    except Exception as exc:
        logger.warning("Failed to retrieve price %s: %s", price_id, exc)
        return None


# ── Checkout sessions ────────────────────────────────────────────────


def create_checkout_session(
    user_id: str,
    user_email: str,
    plan_id: str,
    vps_instance_id: str,
    success_url: str,
    cancel_url: str,
    stripe_customer_id: Optional[str] = None,
) -> dict:
    """
    Create a Stripe Checkout Session for a VPS subscription.

    Returns the session dict with at least:
        { "id": "cs_...", "url": "https://checkout.stripe.com/..." }
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

    params: dict = {
        "mode": "subscription",
        "line_items": [{"price": stripe_price_id, "quantity": 1}],
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
    if stripe_customer_id:
        params["customer"] = stripe_customer_id
    else:
        params["customer_email"] = user_email

    session = client.checkout.sessions.create(params=params)
    return {"id": session.id, "url": session.url, "customer": session.customer}


def get_stripe_price_for_plan(plan_id: str) -> str:
    """Return the Stripe Price ID for a VPS plan."""
    attr = PLAN_PRICE_MAP.get(plan_id)
    if not attr:
        raise ValueError(f"Unknown plan_id: {plan_id}")
    price_id: str = getattr(settings, attr, "")
    if not price_id:
        raise ValueError(
            f"Stripe price ID for plan '{plan_id}' is not configured "
            f"(set {attr.upper()} in environment)"
        )
    return price_id


def create_combined_checkout_session(
    user_id: str,
    user_email: str,
    line_items: list[dict],
    metadata: dict,
    success_url: str,
    cancel_url: str,
    stripe_customer_id: Optional[str] = None,
) -> dict:
    """
    Create a Stripe Checkout Session with multiple subscription line items.
    Used for combined VPS + LLM Bundle + Managed Supabase checkout.
    """
    client = _get_stripe_client()
    params: dict = {
        "mode": "subscription",
        "line_items": line_items,
        "success_url": success_url,
        "cancel_url": cancel_url,
        "metadata": metadata,
        "subscription_data": {"metadata": metadata},
    }
    if stripe_customer_id:
        params["customer"] = stripe_customer_id
    else:
        params["customer_email"] = user_email

    session = client.checkout.sessions.create(params=params)
    return {"id": session.id, "url": session.url, "customer": session.customer}


# ── Webhook verification ─────────────────────────────────────────────


def verify_webhook(payload: bytes, sig_header: str) -> Optional[dict]:
    """
    Verify a Stripe webhook signature and return the parsed event dict,
    or None if the signature is invalid.

    Raises ValueError if STRIPE_WEBHOOK_SECRET is not configured —
    unsigned payloads are never accepted.
    """
    webhook_secret = settings.stripe_webhook_secret
    if not webhook_secret:
        raise ValueError(
            "STRIPE_WEBHOOK_SECRET is not configured — "
            "cannot process webhooks without signature verification"
        )

    try:
        # construct_event validates the signature and raises on failure.
        # We don't use its return value — newer stripe-python versions
        # return a StripeObject that dict()'s inconsistently across SDK
        # versions. Parse the verified payload back to a plain dict ourselves.
        stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except stripe.error.SignatureVerificationError:
        logger.warning("Stripe webhook signature verification failed")
        return None
    except Exception as exc:
        logger.exception("Error verifying Stripe webhook: %s", exc)
        return None

    try:
        import json
        raw = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload
        return json.loads(raw)
    except Exception as exc:
        logger.exception("Error parsing Stripe webhook payload: %s", exc)
        return None


# ── Revenue aggregation (admin ROI dashboard) ──────────────────────


def sum_invoice_revenue_for_customer(customer_id: str, since_unix: int) -> dict:
    """Sum paid invoice revenue for a Stripe customer since `since_unix`.

    Returns a dict with `total_usd` (float, rounded to 4 dp) and `currency_breakdown`
    keyed by ISO currency code. Currency normalization to USD is intentionally
    skipped — admin dashboard surfaces the per-currency split so mixed-region
    revenue (e.g. CAD payments alongside USD) doesn't get silently lossy-converted.

    For ROI math we use the USD-major bucket; CAD revenue is reported separately
    so the admin can apply their own FX rate. (Toup's pricing is USD-anchored;
    CAD is just a localized display of the same product.)

    Returns {"total_usd": 0.0, "currency_breakdown": {}} on any error so the
    caller never has to handle exceptions — admin overview must not 500
    just because Stripe is briefly unreachable.
    """
    if not settings.stripe_secret_key or not customer_id:
        return {"total_usd": 0.0, "currency_breakdown": {}}
    try:
        breakdown: dict[str, float] = {}
        # Stripe paginates at 100/page. For our scale (≤1k invoices/customer),
        # one page is plenty; the auto_paging_iter handles edge cases anyway.
        invoices = stripe.Invoice.list(
            customer=customer_id,
            status="paid",
            created={"gte": since_unix},
            limit=100,
            api_key=settings.stripe_secret_key,
        )
        for inv in invoices.auto_paging_iter():
            currency = (inv.get("currency") or "usd").lower()
            paid_minor = inv.get("amount_paid") or 0  # cents
            breakdown[currency] = round(breakdown.get(currency, 0.0) + paid_minor / 100.0, 4)
        # USD-major bucket (CAD reported separately for admin's FX call).
        total_usd = breakdown.get("usd", 0.0)
        return {"total_usd": total_usd, "currency_breakdown": breakdown}
    except Exception as exc:
        logger.warning("Stripe invoice list failed for %s: %s", customer_id, exc)
        return {"total_usd": 0.0, "currency_breakdown": {}}
