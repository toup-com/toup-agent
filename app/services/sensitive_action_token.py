"""Fresh-JWT sensitive-action token issuance + replay-defended verification.

Why this exists
---------------
POST /auth/delete-account is irrevocable. The previous design required
the user to re-enter their password — fine on web, awkward on mobile
(SecureStore doesn't surface the plaintext password and we don't want
to store it). The new design splits authentication into two steps:

  1. Client calls POST /auth/reauth using its existing access token.
     The server issues a SHORT-LIVED token whose claims include
     `purpose: "delete_account"` and a unique `jti`.
  2. Client calls POST /auth/delete-account passing the short token
     in the X-Sensitive-Action-Token header. The server validates
     signature + expiry + purpose + user binding, then atomically
     INSERTs the jti into sensitive_action_redemptions. Replay of
     the same jti hits the PK conflict and is rejected.

The token is a JWT signed with the same `settings.jwt_secret` /
`settings.jwt_algorithm` as access tokens — one fewer secret to
rotate. TTL defaults to 5 minutes; long enough that a user can
confirm a destructive action without re-clicking, short enough that
a leaked token is harmless.

Reusable for any future destructive endpoint (rotate API keys,
change billing email, etc.) — call sites pass a different
SensitiveActionPurpose value.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Optional
import time
import uuid

from fastapi import HTTPException, status
from jose import jwt, JWTError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import SensitiveActionRedemption


class SensitiveActionPurpose(StrEnum):
    """The kinds of destructive actions a sensitive token can authorize.

    Tokens are minted FOR a specific purpose and rejected when used
    against any other purpose. Adding a new sensitive endpoint =
    adding a new value here + the endpoint that calls
    `verify_sensitive_action_token(..., expected_purpose=NEW)`.
    """

    DELETE_ACCOUNT = "delete_account"


# Default TTL — 5 minutes. The product reasoning: long enough for a
# user to read the confirmation, type into a destructive form, and
# tap Confirm without the token expiring; short enough that a token
# captured on the wire (HTTPS-MITM scenario) is unusable.
DEFAULT_TTL_SECONDS = 300

# Hard upper bound on caller-supplied TTL. A future endpoint MUST NOT
# accidentally mint a 24-hour sensitive token.
MAX_TTL_SECONDS = 600

# Distinct "kind" claim so an access token can never be misused as a
# sensitive-action token (and vice versa). decode_access_token only
# accepts payloads without this claim; verify_sensitive_action_token
# only accepts payloads WITH this claim.
TOKEN_KIND_CLAIM = "kind"
TOKEN_KIND_VALUE = "sensitive_action"


def issue_sensitive_action_token(
    user_id: str,
    purpose: SensitiveActionPurpose,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> tuple[str, datetime]:
    """Mint a single-use sensitive-action token.

    Returns (jwt_string, expiry_datetime). The expiry is returned so
    the caller can persist it on the redemption row at verify time
    without re-decoding the token.

    Args:
        user_id: the user this token authorizes the action for.
        purpose: which destructive operation this token unlocks.
        ttl_seconds: lifetime; clamped to [60, MAX_TTL_SECONDS].
    """
    ttl = max(60, min(ttl_seconds, MAX_TTL_SECONDS))
    # Use unix-epoch timestamps directly. `datetime.utcnow().timestamp()`
    # silently treats the naive value as local time and shifts by the
    # process's tz offset — caused a real bug in early development where
    # a "10s expired" token decoded as 4h-in-the-future on EDT. Stick to
    # `time.time()` (true UTC unix epoch) for JWT claims.
    now_ts = int(time.time())
    exp_ts = now_ts + ttl
    # `exp` datetime returned to the caller is timezone-aware UTC so it
    # round-trips cleanly to `expires_at` on the redemption row (which
    # SQLAlchemy stores as naive UTC — we drop tzinfo at write time).
    exp = datetime.fromtimestamp(exp_ts, tz=timezone.utc)
    jti = str(uuid.uuid4())
    payload = {
        "sub": user_id,
        "purpose": purpose.value,
        TOKEN_KIND_CLAIM: TOKEN_KIND_VALUE,
        "iat": now_ts,
        "exp": exp_ts,
        "jti": jti,
    }
    token = jwt.encode(
        payload,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    return token, exp


async def verify_sensitive_action_token(
    db: AsyncSession,
    token: str,
    *,
    expected_user_id: str,
    expected_purpose: SensitiveActionPurpose,
) -> None:
    """Validate a sensitive-action token AND atomically mark its jti
    as redeemed.

    Raises HTTPException(401) on any of:
      - empty / malformed token
      - signature mismatch
      - expired
      - missing or wrong `kind` claim (rejects access tokens)
      - sub != expected_user_id
      - purpose != expected_purpose
      - jti already redeemed (replay)

    On success the row is left in `sensitive_action_redemptions` and
    the function returns None. The redemption is durable — survives
    process restarts, so a token can never be replayed even across a
    deploy mid-action.

    The PK insert IS the redeem. We intentionally don't read-then-
    write because that opens a TOCTOU window — two concurrent verifies
    of the same jti could both pass the SELECT before either INSERT.
    Letting the INSERT race and catching IntegrityError is correct
    Postgres concurrency.
    """
    if not token:
        _raise_invalid("missing token")

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError as e:
        _raise_invalid(f"token decode failed: {e!s}")

    if payload.get(TOKEN_KIND_CLAIM) != TOKEN_KIND_VALUE:
        # An access token (or any other JWT we issue) lacks the
        # sensitive-action `kind` claim. Reject explicitly so a stolen
        # access token can't be used here.
        _raise_invalid("not a sensitive-action token")

    if payload.get("sub") != expected_user_id:
        _raise_invalid("token sub mismatch")

    if payload.get("purpose") != expected_purpose.value:
        _raise_invalid("token purpose mismatch")

    jti: Optional[str] = payload.get("jti")
    if not jti:
        _raise_invalid("token missing jti")

    exp_ts = payload.get("exp")
    if not exp_ts:
        _raise_invalid("token missing exp")
    # jose's default jwt.decode does NOT verify `exp` — the early
    # implementation assumed it did and shipped a token-expiry bypass.
    # Compare against unix epoch directly to avoid the naive-datetime
    # tz-offset trap (see issue_sensitive_action_token comment).
    if int(exp_ts) <= int(time.time()):
        _raise_invalid("token expired")
    # Stored on the redemption row as a naive UTC datetime so it
    # behaves consistently with every other DateTime column in the
    # platform DB.
    expires_at = datetime.utcfromtimestamp(int(exp_ts))

    # Atomic redeem. PK collision means already-redeemed → replay.
    redemption = SensitiveActionRedemption(
        jti=jti,
        user_id=expected_user_id,
        purpose=expected_purpose.value,
        expires_at=expires_at,
    )
    db.add(redemption)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        _raise_invalid("token already redeemed")
    # Caller commits as part of the larger destructive operation; we
    # only flush here so the constraint check happens now and the
    # caller's commit is the durable point.


def _raise_invalid(detail: str) -> None:
    """Single chokepoint for all 401s — the public-facing message is
    intentionally generic ('Invalid or expired sensitive-action
    token') so an attacker probing the endpoint can't distinguish
    between a wrong-purpose token, a replay, and a forgery. The
    specific reason goes to the server log only.
    """
    import logging
    logging.getLogger(__name__).info("[SAT] reject: %s", detail)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired sensitive-action token",
    )
