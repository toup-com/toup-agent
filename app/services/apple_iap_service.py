"""StoreKit In-App Purchase — consumable credit-pack receipt validation.

This is the SERVER-AUTHORITATIVE source of how many credits each product
grants. The client's credit numbers are display-only; only this module mints
credits, via :data:`PRODUCT_CREDITS`.

Two responsibilities, both built on Apple's official
``app-store-server-library`` (verified against v3.1.2):

1. ``verify_transaction`` — given a StoreKit2 ``transactionId`` and an
   environment hint, call the App Store Server API's ``get_transaction_info``
   to fetch the signed transaction, then ``verify_and_decode_signed_transaction``
   to validate Apple's JWS cert chain and decode the payload. Asserts the
   bundle id, that the product is a known consumable, and returns the credit
   amount from :data:`PRODUCT_CREDITS`.
2. ``verify_and_decode_notification`` — verify+decode an App Store Server
   Notification V2 ``signedPayload`` (for the refund webhook).

Inert-until-configured discipline (mirrors apple_auth.py): if the IAP key
isn't fully set, :func:`iap_configured` returns False and callers (the
endpoint) return 503. All network / verification failures raise
:class:`IapVerificationError`, which the endpoint maps to 422.

The library's clients are synchronous (``requests``); the async wrappers here
push the blocking calls onto a thread via ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

from app.config import settings
from app.db.plan_catalog import (
    LEGACY_SUB_PRODUCT_IDS,
    UNLIMITED_PLAN_ID,
    UNLIMITED_SUB_PRODUCT_ID,
)

logger = logging.getLogger(__name__)


# Server-authoritative product → credits map. Product IDs are IMMUTABLE once
# created in App Store Connect (see the shared IAP contract). This is the ONLY
# place credit amounts are minted from.
PRODUCT_CREDITS: dict[str, Decimal] = {
    "ai.toup.app.credits.small":  Decimal("500"),
    "ai.toup.app.credits.medium": Decimal("1200"),
    "ai.toup.app.credits.large":  Decimal("2800"),
    "ai.toup.app.credits.xl":     Decimal("7500"),
}


# Auto-renewable subscription products → the subscription_plans row the product
# was SOLD AS. This map is intentionally DISJOINT from PRODUCT_CREDITS
# (consumables): a product id belongs to exactly one map, and the notification
# handler branches on which. It is the membership test (iap.py) and the
# historical record of what each product is — it is NOT the entitlement answer.
# Call :func:`plan_for_subscription` for that.
#
# ⚠️ THE FOUR LEGACY IDS MUST NEVER BE REMOVED FROM THIS DICT. Removing one
#    makes that subscriber's DID_RENEW fall through to "notification for
#    unrecognised product; ignoring" plus a 200 ack — no grant, no downgrade,
#    no retry, no alert, and Apple never redelivers. They also stay on sale in
#    App Store Connect forever: "remove from sale" stops RENEWALS too, and
#    EXPIRED is in _SUB_DOWNGRADE_TYPES, so removing them downgrades the
#    people who are still paying.
APPLE_SUB_PRODUCT_TO_PLAN: dict[str, str] = {
    "ai.toup.app.sub.starter": "starter",
    "ai.toup.app.sub.builder": "builder",
    "ai.toup.app.sub.pro":     "pro",
    "ai.toup.app.sub.elite":   "elite",
    UNLIMITED_SUB_PRODUCT_ID:  UNLIMITED_PLAN_ID,
}


def plan_for_subscription(product_id: str, sub=None) -> str:
    """The plan a subscription ENTITLES right now — the entitlement resolver.

    * The Unlimited product always entitles Unlimited.
    * A legacy product entitles Unlimited **iff this subscription was
      grandfathered ON THAT SAME PRODUCT** — recorded on the
      ``apple_subscriptions`` row by ``app.scripts.grandfather_unlimited``,
      never inferred from the product.
    * A legacy product on a subscription that was never grandfathered, or that
      was grandfathered on a DIFFERENT product, entitles the tier it was sold
      as.

    The product half is load-bearing, and reading only ``grandfathered_at``
    was a live money leak. The four legacy products must stay on sale in App
    Store Connect forever (removing one stops its renewals, and EXPIRED is in
    _SUB_DOWNGRADE_TYPES), so Apple keeps showing all of them in the
    customer's own Manage Subscriptions. A cross-grade inside a subscription
    group keeps the SAME originalTransactionId, so a grandfathered Builder
    payer who switches to Starter arrives here at the next DID_RENEW on the
    same row with ``grandfathered_at`` still set — and, on the stamp alone,
    kept Unlimited for CAD 9.90 permanently and unlogged. That is precisely
    the route this column exists to close, and until now nothing read it.

    A NULL ``grandfathered_product_id`` counts as a match: a row stamped by
    hand, or before that column existed, must not be silently demoted.

    Forfeiting on a product change is deliberately the answer in BOTH
    directions, including an upgrade. The grandfather is "you keep Unlimited
    at the price you were already paying"; a different product is a different
    price, and the honest response to one is to hand it to a human — the
    reconciler pages `apple-legacy-crossgrade` on exactly this row — not to
    guess from `subscription_plans.price_cents`, which is the abandoned web
    ladder and does not describe what Apple charges.

    This is the function that stops the 2026-10-01 renewal from reverting the
    grandfather: ``DID_RENEW`` re-derives the plan from the PRODUCT ID and
    reapplies it, so stamping ``plan_id='unlimited'`` on the balance alone
    would silently be undone the first time Apple renewed a legacy product.
    The grandfathered payer's ``DID_RENEW`` resolves to ``unlimited`` here and
    ``_apple_renew`` re-grants the Unlimited allowance instead.

    Robustness property worth keeping: if this resolver ships but the
    grandfather command has NOT run, ``grandfathered_at`` is NULL, the legacy
    product resolves to its legacy tier, and behaviour is byte-for-byte
    today's. The grandfather is the only thing that changes the answer.

    ``sub`` is None only for a brand-new SUBSCRIBED that arrives before
    ``/subscribe/verify`` has created the row — and a brand-new subscription
    cannot be grandfathered, so resolving by product alone is correct there.
    """
    if product_id == UNLIMITED_SUB_PRODUCT_ID:
        return UNLIMITED_PLAN_ID
    if product_id in LEGACY_SUB_PRODUCT_IDS:
        if sub is not None and getattr(sub, "grandfathered_at", None) is not None:
            stamped = getattr(sub, "grandfathered_product_id", None)
            if stamped is None or stamped == product_id:
                return UNLIMITED_PLAN_ID
            logger.warning(
                "[iap] subscription %s was grandfathered on %s but now holds "
                "%s — the grandfather does not follow a product change; "
                "resolving to the tier it now holds",
                getattr(sub, "original_transaction_id", "?"), stamped, product_id,
            )
        return APPLE_SUB_PRODUCT_TO_PLAN[product_id]
    raise KeyError(product_id)


_ROOT_CERTS_DIR = Path(__file__).resolve().parent / "apple_root_certs"


class IapVerificationError(Exception):
    """Raised when an IAP transaction / notification fails verification.

    The endpoint maps this to HTTP 422 (verification failed / bundle mismatch
    / not consumable / unknown product).
    """


@dataclass
class VerifiedTransaction:
    transaction_id: str
    original_transaction_id: Optional[str]
    product_id: str
    environment: str
    credits: Decimal


@dataclass
class VerifiedSubscription:
    transaction_id: str
    original_transaction_id: str
    product_id: str
    environment: str
    expires_date: Optional[datetime]
    plan_id: str


def _library_available() -> bool:
    """True iff Apple's app-store-server-library is importable.

    Defensive: the library installs via requirements.platform.txt. If a build ever
    ships without it, every verify call would ImportError -> HTTP 500. Folding this
    into iap_configured() degrades that to a clean 503 (inert) matching the
    "purchases temporarily unavailable" contract, never a hard error.
    """
    import importlib.util
    return importlib.util.find_spec("appstoreserverlibrary") is not None


def iap_configured() -> bool:
    """True iff the IAP key (key_id + issuer_id + private_key) is fully set AND
    Apple's server library is importable.

    Mirrors apple_auth.siwa_revocation_configured(): when any piece is missing
    the module is provably inert and the endpoint returns 503.
    """
    return bool(
        (settings.apple_iap_key_id or "").strip()
        and (settings.apple_iap_issuer_id or "").strip()
        and (settings.apple_iap_private_key or "").strip()
        and _library_available()
    )


def _signing_key() -> str:
    # Railway / .env single-line values store the .p8 PEM with literal "\n";
    # normalise back to real newlines (mirror apple_auth._private_key_pem).
    return (settings.apple_iap_private_key or "").replace("\\n", "\n").strip()


def _load_root_certificates() -> list[bytes]:
    """Read every *.cer (DER) in apple_root_certs/ as raw bytes.

    The library's SignedDataVerifier needs Apple's root CA certs to validate
    the JWS chain. Operators must drop AppleRootCA-G3.cer and
    AppleRootCA-G2.cer here (downloaded from
    https://www.apple.com/certificateauthority/). Loading by glob means
    adding a future root is just a file drop, no code change.
    """
    if not _ROOT_CERTS_DIR.is_dir():
        raise IapVerificationError(
            f"Apple root cert dir missing: {_ROOT_CERTS_DIR} — operator must "
            "drop AppleRootCA-G3.cer / AppleRootCA-G2.cer there."
        )
    certs = [p.read_bytes() for p in sorted(_ROOT_CERTS_DIR.glob("*.cer"))]
    if not certs:
        raise IapVerificationError(
            f"No *.cer root certificates in {_ROOT_CERTS_DIR} — download them "
            "from https://www.apple.com/certificateauthority/."
        )
    return certs


def _env_from_hint(hint: Optional[str]):
    """Map the wire environment hint to the library's Environment enum.

    Defaults to Production. Unknown / empty → Production.
    """
    from appstoreserverlibrary.models.Environment import Environment

    h = (hint or "").strip().lower()
    if h == "sandbox":
        return Environment.SANDBOX
    if h == "xcode":
        return Environment.XCODE
    return Environment.PRODUCTION


def _build_api_client(environment):
    from appstoreserverlibrary.api_client import AppStoreServerAPIClient

    return AppStoreServerAPIClient(
        signing_key=_signing_key().encode("utf-8"),
        key_id=settings.apple_iap_key_id,
        issuer_id=settings.apple_iap_issuer_id,
        bundle_id=settings.apple_iap_bundle_id,
        environment=environment,
    )


def _build_verifier(environment):
    from appstoreserverlibrary.signed_data_verifier import SignedDataVerifier

    app_apple_id: Optional[int] = None
    raw = (settings.apple_iap_app_apple_id or "").strip()
    if raw:
        try:
            app_apple_id = int(raw)
        except ValueError:
            app_apple_id = None
    return SignedDataVerifier(
        root_certificates=_load_root_certificates(),
        # enable_online_checks=True asks the verifier to confirm the signing
        # cert hasn't been revoked (OCSP). It's the production-safe default.
        enable_online_checks=True,
        environment=environment,
        bundle_id=settings.apple_iap_bundle_id,
        app_apple_id=app_apple_id,
    )


def _fetch_signed_transaction(environment, transaction_id: str) -> Optional[str]:
    """Blocking: call get_transaction_info; return the signed JWS or None.

    Returns None when Apple reports the transaction isn't found in this
    environment (so the caller can retry the other env). Re-raises any other
    APIException as IapVerificationError.
    """
    from appstoreserverlibrary.api_client import AppStoreServerAPIClient, APIException

    client: AppStoreServerAPIClient = _build_api_client(environment)
    try:
        resp = client.get_transaction_info(transaction_id)
    except APIException as e:
        # A genuinely-unknown txn in this env shows up as a not-found-ish
        # APIException; treat as "try the other env" rather than a hard fail.
        logger.info(
            "[apple-iap] get_transaction_info APIException env=%s txn=%s: %s",
            environment, transaction_id, e,
        )
        return None
    except Exception as e:  # network / TLS / unexpected
        raise IapVerificationError(
            f"App Store Server API call failed: {e}"
        ) from e
    return getattr(resp, "signedTransactionInfo", None)


async def verify_transaction(
    transaction_id: str, environment_hint: str,
) -> VerifiedTransaction:
    """Fetch + verify a StoreKit transaction; return its credit grant.

    Picks the env from the hint (default Production), fetches the signed
    transaction; if not found there (or the hint was Sandbox) retries the
    OTHER env. Then cryptographically verifies the JWS against Apple's root
    chain, asserts bundle id + known consumable product, and returns the
    server-authoritative credit amount.
    """
    if not iap_configured():
        raise IapVerificationError("IAP not configured on server")
    if not transaction_id:
        raise IapVerificationError("missing transaction_id")

    primary = _env_from_hint(environment_hint)
    from appstoreserverlibrary.models.Environment import Environment

    # Try the hinted env first, then the other production/sandbox env.
    tried = [primary]
    if primary == Environment.SANDBOX:
        tried.append(Environment.PRODUCTION)
    elif primary == Environment.PRODUCTION:
        tried.append(Environment.SANDBOX)

    signed: Optional[str] = None
    used_env = primary
    for env in tried:
        signed = await asyncio.to_thread(
            _fetch_signed_transaction, env, transaction_id,
        )
        if signed:
            used_env = env
            break
    if not signed:
        raise IapVerificationError(
            f"transaction {transaction_id} not found in any environment"
        )

    decoded = await asyncio.to_thread(
        _verify_and_decode_transaction, used_env, signed,
    )

    decoded_bundle = getattr(decoded, "bundleId", None)
    if decoded_bundle != settings.apple_iap_bundle_id:
        raise IapVerificationError(
            f"bundle mismatch: {decoded_bundle!r} != {settings.apple_iap_bundle_id!r}"
        )

    product_id = getattr(decoded, "productId", None)
    if product_id not in PRODUCT_CREDITS:
        raise IapVerificationError(f"unknown product: {product_id!r}")

    # Must be a consumable credit pack — never mint for a subscription / NC.
    from appstoreserverlibrary.models.Type import Type

    ptype = getattr(decoded, "type", None)
    if ptype is not None and ptype != Type.CONSUMABLE:
        raise IapVerificationError(f"product {product_id!r} is not consumable ({ptype})")

    return VerifiedTransaction(
        transaction_id=str(getattr(decoded, "transactionId", transaction_id)),
        original_transaction_id=(
            str(getattr(decoded, "originalTransactionId", None))
            if getattr(decoded, "originalTransactionId", None) is not None else None
        ),
        product_id=product_id,
        environment=getattr(used_env, "value", str(used_env)),
        credits=PRODUCT_CREDITS[product_id],
    )


def ms_to_datetime(ms: Optional[int]) -> Optional[datetime]:
    """Apple delivers all dates as epoch MILLISECONDS. Convert to a naive UTC
    ``datetime`` (matching how credit_balances stores period bounds)."""
    if ms is None:
        return None
    return datetime.utcfromtimestamp(int(ms) / 1000)


async def verify_subscription_transaction(
    transaction_id: str, environment_hint: str,
) -> VerifiedSubscription:
    """Fetch + verify a StoreKit AUTO-RENEWABLE SUBSCRIPTION transaction.

    Mirrors :func:`verify_transaction` (same dual-env fetch + JWS chain
    verification + bundle assertion) but asserts the decoded ``type`` is
    AUTO_RENEWABLE_SUBSCRIPTION (not CONSUMABLE), resolves the product to its
    plan via :data:`APPLE_SUB_PRODUCT_TO_PLAN`, and returns the subscription's
    ``expires_date`` + ``plan_id`` instead of a credit amount.
    """
    if not iap_configured():
        raise IapVerificationError("IAP not configured on server")
    if not transaction_id:
        raise IapVerificationError("missing transaction_id")

    from appstoreserverlibrary.models.Environment import Environment

    primary = _env_from_hint(environment_hint)
    tried = [primary]
    if primary == Environment.SANDBOX:
        tried.append(Environment.PRODUCTION)
    elif primary == Environment.PRODUCTION:
        tried.append(Environment.SANDBOX)

    signed: Optional[str] = None
    used_env = primary
    for env in tried:
        signed = await asyncio.to_thread(
            _fetch_signed_transaction, env, transaction_id,
        )
        if signed:
            used_env = env
            break
    if not signed:
        raise IapVerificationError(
            f"transaction {transaction_id} not found in any environment"
        )

    decoded = await asyncio.to_thread(
        _verify_and_decode_transaction, used_env, signed,
    )

    decoded_bundle = getattr(decoded, "bundleId", None)
    if decoded_bundle != settings.apple_iap_bundle_id:
        raise IapVerificationError(
            f"bundle mismatch: {decoded_bundle!r} != {settings.apple_iap_bundle_id!r}"
        )

    product_id = getattr(decoded, "productId", None)
    if product_id not in APPLE_SUB_PRODUCT_TO_PLAN:
        raise IapVerificationError(f"unknown subscription product: {product_id!r}")

    # Must be an auto-renewable subscription — never activate a plan for a
    # consumable / NC transaction.
    from appstoreserverlibrary.models.Type import Type

    ptype = getattr(decoded, "type", None)
    if ptype is not None and ptype != Type.AUTO_RENEWABLE_SUBSCRIPTION:
        raise IapVerificationError(
            f"product {product_id!r} is not an auto-renewable subscription ({ptype})"
        )

    original_txn = getattr(decoded, "originalTransactionId", None)
    if not original_txn:
        # The lifecycle anchor is mandatory for subscriptions.
        raise IapVerificationError("subscription transaction missing originalTransactionId")

    return VerifiedSubscription(
        transaction_id=str(getattr(decoded, "transactionId", transaction_id)),
        original_transaction_id=str(original_txn),
        product_id=product_id,
        environment=getattr(used_env, "value", str(used_env)),
        expires_date=ms_to_datetime(getattr(decoded, "expiresDate", None)),
        # The CATALOGUE plan — what this product was sold as. This function
        # never sees the apple_subscriptions row, so it cannot know whether
        # the subscription was grandfathered. The verify route re-resolves
        # through plan_for_subscription() once it has the row in hand.
        plan_id=APPLE_SUB_PRODUCT_TO_PLAN[product_id],
    )


# ── Apple as the oracle: get_all_subscription_statuses ───────────────────
#
# Everything above is driven BY Apple (a client verify, a pushed
# notification). This is the one call we make to ASK Apple what the truth is,
# and it exists because the push channel has already dropped a message: the
# Sandbox Elite row has read status='active' for 2.7 months past its expiry
# with last_notification_uuid NULL, and DID_RENEW has never executed in
# production. See app/services/apple_reconciler.py.


# Apple's Status enum from get_all_subscription_statuses. Kept as raw ints
# here so this module needs no import from app.db; the reconciler maps them
# onto the apple_subscriptions.status mirror.
#   1 active · 2 expired · 3 billing retry · 4 billing grace period · 5 revoked
APPLE_STATUS_ACTIVE = 1
APPLE_STATUS_EXPIRED = 2
APPLE_STATUS_BILLING_RETRY = 3
APPLE_STATUS_GRACE = 4
APPLE_STATUS_REVOKED = 5


@dataclass
class AppleSubscriptionStatus:
    """Apple's own answer for one subscription.

    ``status`` is Apple's; ``expires_date`` / ``auto_renew_*`` come from the
    signed transaction + renewal-info payloads carried alongside it, decoded
    through the same JWS chain verification every other path uses. A field is
    None when Apple did not send it or its payload would not decode — the
    caller must treat None as "unknown", never as "changed to nothing".
    """
    original_transaction_id: str
    environment: str
    status: int
    product_id: Optional[str] = None
    expires_date: Optional[datetime] = None
    auto_renew_status: Optional[bool] = None
    auto_renew_product_id: Optional[str] = None


def _fetch_subscription_statuses(environment, original_txn: str):
    """Blocking: call get_all_subscription_statuses. None ⇒ Apple does not
    know this transaction in this environment (the caller may try the other).

    Mirrors ``_fetch_signed_transaction``'s error discipline exactly: an
    APIException is "not here", anything else is a hard failure the caller
    must be able to tell apart from "nothing to fix".
    """
    from appstoreserverlibrary.api_client import AppStoreServerAPIClient, APIException

    client: AppStoreServerAPIClient = _build_api_client(environment)
    try:
        return client.get_all_subscription_statuses(original_txn)
    except APIException as e:
        logger.info(
            "[apple-iap] get_all_subscription_statuses APIException env=%s "
            "orig_txn=%s: %s", environment, original_txn, e,
        )
        return None
    except Exception as e:  # network / TLS / unexpected
        raise IapVerificationError(
            f"App Store Server API call failed: {e}"
        ) from e


async def fetch_subscription_status(
    original_transaction_id: str, environment_hint: str,
) -> Optional[AppleSubscriptionStatus]:
    """Ask Apple for the current state of one subscription.

    Queries ONLY the environment it is handed. That is deliberate and differs
    from ``verify_transaction``'s dual-env fallback: the reconciler already
    knows which environment each mirror row belongs to, and a Production
    lookup that silently fell back to Sandbox would let sandbox test data
    decide a real customer's plan.

    Returns None when Apple does not know the transaction there. Raises
    :class:`IapVerificationError` on a network/API failure, so a caller can
    tell "Apple says nothing is wrong" from "we could not ask".
    """
    if not iap_configured():
        raise IapVerificationError("IAP not configured on server")
    if not original_transaction_id:
        raise IapVerificationError("missing original_transaction_id")

    env = _env_from_hint(environment_hint)
    resp = await asyncio.to_thread(
        _fetch_subscription_statuses, env, original_transaction_id,
    )
    if resp is None:
        return None

    item = _last_transaction_for(resp, original_transaction_id)
    if item is None:
        logger.info(
            "[apple-iap] no lastTransactions entry for orig_txn=%s env=%s",
            original_transaction_id, env,
        )
        return None

    status_val = getattr(item, "status", None)
    status_int = getattr(status_val, "value", status_val)
    if status_int is None:
        logger.warning(
            "[apple-iap] status response for orig_txn=%s carries no status",
            original_transaction_id,
        )
        return None

    out = AppleSubscriptionStatus(
        original_transaction_id=original_transaction_id,
        environment=getattr(env, "value", str(env)),
        status=int(status_int),
    )

    # The signed payloads are optional in the response and each decodes
    # independently. A decode failure must degrade to "unknown" for that
    # field, never fail the whole read — Apple's STATUS is the part this
    # reconciler cannot do without, and it is already in hand.
    signed_txn = getattr(item, "signedTransactionInfo", None)
    if signed_txn:
        try:
            txn = await asyncio.to_thread(
                _verify_and_decode_transaction, env, signed_txn,
            )
            out.product_id = getattr(txn, "productId", None)
            out.expires_date = ms_to_datetime(getattr(txn, "expiresDate", None))
        except Exception as e:
            logger.warning(
                "[apple-iap] could not decode status txn for orig_txn=%s: %s",
                original_transaction_id, e,
            )

    signed_renewal = getattr(item, "signedRenewalInfo", None)
    if signed_renewal:
        try:
            info = await asyncio.to_thread(
                _verify_and_decode_renewal_info, env, signed_renewal,
            )
            ars = getattr(info, "autoRenewStatus", None)
            ars_val = getattr(ars, "value", ars)
            if ars_val is not None:
                out.auto_renew_status = bool(ars_val)
            out.auto_renew_product_id = getattr(info, "autoRenewProductId", None)
        except Exception as e:
            logger.warning(
                "[apple-iap] could not decode status renewal info for "
                "orig_txn=%s: %s", original_transaction_id, e,
            )
    return out


def _last_transaction_for(resp, original_transaction_id: str):
    """Find this subscription's lastTransactions entry in a StatusResponse.

    The response is grouped by subscription group and each group carries every
    subscription in it, so a match on ``originalTransactionId`` is required —
    taking ``data[0].lastTransactions[0]`` would read a DIFFERENT subscription
    of the same customer whenever they hold more than one.
    """
    for group in (getattr(resp, "data", None) or []):
        for item in (getattr(group, "lastTransactions", None) or []):
            if str(getattr(item, "originalTransactionId", "")) == str(original_transaction_id):
                return item
    return None


def _verify_and_decode_transaction(environment, signed_transaction: str):
    from appstoreserverlibrary.signed_data_verifier import VerificationException

    verifier = _build_verifier(environment)
    try:
        return verifier.verify_and_decode_signed_transaction(signed_transaction)
    except VerificationException as e:
        raise IapVerificationError(f"signed transaction verification failed: {e}") from e
    except Exception as e:
        raise IapVerificationError(f"signed transaction decode failed: {e}") from e


def _verify_and_decode_renewal_info(environment, signed_renewal_info: str):
    """Verify + decode a notification's ``signedRenewalInfo`` JWS.

    Returns the library's ``JWSRenewalInfoDecodedPayload`` (autoRenewStatus,
    autoRenewProductId, gracePeriodExpiresDate, expirationIntent, …). Same
    chain verification path as the transaction decode.
    """
    from appstoreserverlibrary.signed_data_verifier import VerificationException

    verifier = _build_verifier(environment)
    try:
        return verifier.verify_and_decode_renewal_info(signed_renewal_info)
    except VerificationException as e:
        raise IapVerificationError(f"signed renewal info verification failed: {e}") from e
    except Exception as e:
        raise IapVerificationError(f"signed renewal info decode failed: {e}") from e


async def verify_and_decode_notification(signed_payload: str):
    """Verify + decode an App Store Server Notification V2 signedPayload.

    Returns the library's decoded ResponseBodyV2DecodedPayload. Raises
    IapVerificationError on an unverifiable signature so the endpoint can
    return 400. Notifications don't carry an env hint, so we verify against
    Production first then Sandbox.
    """
    if not iap_configured():
        raise IapVerificationError("IAP not configured on server")
    if not signed_payload:
        raise IapVerificationError("missing signed payload")

    from appstoreserverlibrary.models.Environment import Environment

    last_err: Optional[Exception] = None
    for env in (Environment.PRODUCTION, Environment.SANDBOX):
        try:
            return await asyncio.to_thread(
                _verify_and_decode_notification_sync, env, signed_payload,
            )
        except IapVerificationError as e:
            last_err = e
            continue
    raise IapVerificationError(
        f"notification verification failed in all environments: {last_err}"
    )


def _verify_and_decode_notification_sync(environment, signed_payload: str):
    from appstoreserverlibrary.signed_data_verifier import VerificationException

    verifier = _build_verifier(environment)
    try:
        return verifier.verify_and_decode_notification(signed_payload)
    except VerificationException as e:
        raise IapVerificationError(f"notification verification failed: {e}") from e
    except Exception as e:
        raise IapVerificationError(f"notification decode failed: {e}") from e
