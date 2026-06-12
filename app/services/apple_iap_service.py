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


# Auto-renewable subscription products → the subscription_plans row id they map
# to. Apple reuses the SAME plan rows Stripe uses (no new plan rows). This map
# is intentionally DISJOINT from PRODUCT_CREDITS (consumables): a product id
# belongs to exactly one map, and the notification handler branches on which.
APPLE_SUB_PRODUCT_TO_PLAN: dict[str, str] = {
    "ai.toup.app.sub.starter": "starter",
    "ai.toup.app.sub.builder": "builder",
    "ai.toup.app.sub.pro":     "pro",
    "ai.toup.app.sub.elite":   "elite",
}


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


def iap_configured() -> bool:
    """True iff the IAP key (key_id + issuer_id + private_key) is fully set.

    Mirrors apple_auth.siwa_revocation_configured(): when any piece is missing
    the module is provably inert and the endpoint returns 503.
    """
    return bool(
        (settings.apple_iap_key_id or "").strip()
        and (settings.apple_iap_issuer_id or "").strip()
        and (settings.apple_iap_private_key or "").strip()
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
        plan_id=APPLE_SUB_PRODUCT_TO_PLAN[product_id],
    )


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
