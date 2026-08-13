"""Dual-accept rotation for ROLLOUT_SECRET and JWT_SECRET (Last Mile L-3).

Both secrets rotate with the same two-phase pattern: verify {new, old},
sign/send new only, then drop old. Before this, both were single-accept:

  * ROLLOUT_SECRET — a rotation had an unavoidable mismatch window that
    could 401 an in-flight build's rollout-notify (the exact 08-10
    deploy-path outage class). The closeout STAGED this rotation rather
    than risk the window unattended.
  * JWT_SECRET — a rotation instantly invalidated every live session
    (mass logout, no transition). Worse, the regression trap: several
    post-decode revocation checks (password_changed_at vs iat, jti)
    caught JWTError and `pass`ed, relying on upstream validation —
    per-site single-secret decode during a rotation would have SKIPPED
    those checks for old-secret tokens.

Every failure-mode test here was proven RED against the pre-fix tree.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException
from jose import jwt


@pytest.fixture
def rotation_settings():
    """Snapshot/restore the secrets this file mutates."""
    from app.config import settings
    prev = (
        settings.rollout_secret,
        settings.rollout_secret_previous,
        settings.jwt_secret,
        settings.jwt_secret_previous,
    )
    yield settings
    (
        settings.rollout_secret,
        settings.rollout_secret_previous,
        settings.jwt_secret,
        settings.jwt_secret_previous,
    ) = prev


# ── ROLLOUT_SECRET ──────────────────────────────────────────────────


def _verify(header):
    from app.api.admin.rollouts import _verify_rollout_secret
    return _verify_rollout_secret(header)


def test_old_secret_accepted_while_previous_is_set(rotation_settings):
    """The rotation window: platform already carries the new secret, an
    in-flight build still sends the old one. MUTATION: revert to
    single-accept → 401 → red. Proven red on the pre-fix tree."""
    rotation_settings.rollout_secret = "new-rollout-secret"
    rotation_settings.rollout_secret_previous = "old-rollout-secret"
    _verify("old-rollout-secret")  # must not raise


def test_new_secret_accepted(rotation_settings):
    rotation_settings.rollout_secret = "new-rollout-secret"
    rotation_settings.rollout_secret_previous = "old-rollout-secret"
    _verify("new-rollout-secret")  # must not raise


def test_garbage_rejected_401(rotation_settings):
    rotation_settings.rollout_secret = "new-rollout-secret"
    rotation_settings.rollout_secret_previous = "old-rollout-secret"
    with pytest.raises(HTTPException) as e:
        _verify("neither-of-them")
    assert e.value.status_code == 401


def test_neither_configured_503(rotation_settings):
    rotation_settings.rollout_secret = ""
    rotation_settings.rollout_secret_previous = ""
    with pytest.raises(HTTPException) as e:
        _verify("anything")
    assert e.value.status_code == 503


def test_empty_previous_never_becomes_an_acceptable_comparand(
    rotation_settings,
):
    """previous='' must not turn an empty/missing header into a pass —
    guarding empties is load-bearing in the candidate list."""
    rotation_settings.rollout_secret = "new-rollout-secret"
    rotation_settings.rollout_secret_previous = ""
    for bad in ("", None):
        with pytest.raises(HTTPException) as e:
            _verify(bad)
        assert e.value.status_code == 401


def test_rotation_completes_old_rejected_once_previous_cleared(
    rotation_settings,
):
    rotation_settings.rollout_secret = "new-rollout-secret"
    rotation_settings.rollout_secret_previous = ""
    with pytest.raises(HTTPException) as e:
        _verify("old-rollout-secret")
    assert e.value.status_code == 401


# ── JWT_SECRET ──────────────────────────────────────────────────────


def _mint(secret: str, **extra) -> str:
    from app.config import settings
    claims = {
        "sub": "user-under-test",
        "exp": datetime.utcnow() + timedelta(minutes=30),
        "iat": datetime.utcnow(),
        "jti": "jti-under-test",
    }
    claims.update(extra)
    return jwt.encode(claims, secret, algorithm=settings.jwt_algorithm)


def test_old_secret_token_validates_during_rotation(rotation_settings):
    """A session minted before the flip survives the rotation window.
    MUTATION: revert decode_access_token to single-secret → None → red.
    Proven red on the pre-fix tree."""
    from app.services.auth_service import decode_access_token

    token = _mint("old-jwt-secret")
    rotation_settings.jwt_secret = "new-jwt-secret"
    rotation_settings.jwt_secret_previous = "old-jwt-secret"
    assert decode_access_token(token) == "user-under-test"


def test_new_secret_token_validates(rotation_settings):
    from app.services.auth_service import decode_access_token

    rotation_settings.jwt_secret = "new-jwt-secret"
    rotation_settings.jwt_secret_previous = "old-jwt-secret"
    assert decode_access_token(_mint("new-jwt-secret")) == "user-under-test"


def test_rotation_completes_old_token_rejected_once_previous_cleared(
    rotation_settings,
):
    from app.services.auth_service import decode_access_token

    token = _mint("old-jwt-secret")
    rotation_settings.jwt_secret = "new-jwt-secret"
    rotation_settings.jwt_secret_previous = ""
    assert decode_access_token(token) is None


def test_signing_stays_new_only(rotation_settings):
    """create_access_token must mint with the CURRENT secret — a token
    minted during the window must outlive the previous being cleared."""
    from app.services.auth_service import (
        create_access_token, decode_access_token,
    )

    rotation_settings.jwt_secret = "new-jwt-secret"
    rotation_settings.jwt_secret_previous = "old-jwt-secret"
    token = create_access_token("user-under-test")
    rotation_settings.jwt_secret_previous = ""
    assert decode_access_token(token) == "user-under-test"


def test_no_decode_site_bypasses_the_helper():
    """THE regression trap: a site that still calls jwt.decode against
    settings.jwt_secret directly catches JWTError and `pass`es — during
    a rotation it would silently SKIP its revocation check for
    old-secret tokens (password_changed_at, jti). Every platform decode
    goes through decode_platform_jwt; signing (jwt.encode) is exempt.
    MUTATION: restore any single-secret decode → red. Proven red on the
    pre-fix tree."""
    import pathlib
    import re

    backend = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for f in (backend / "app").rglob("*.py"):
        src = f.read_text()
        for m in re.finditer(r"jwt\.decode\([^)]*settings\.jwt_secret", src):
            offenders.append(f"{f.relative_to(backend)}: {m.group(0)[:60]}")
    allowed = {"app/services/auth_service.py"}
    real = [o for o in offenders if o.split(":")[0] not in allowed]
    assert not real, (
        "direct jwt.decode against settings.jwt_secret outside "
        f"decode_platform_jwt: {real}"
    )


def test_scoped_preview_token_still_rejected_from_general_auth(
    rotation_settings,
):
    """Dual-accept must not loosen the scope gate: an app_preview-scoped
    token — under EITHER secret — is still rejected by general auth."""
    from app.services.auth_service import decode_access_token

    rotation_settings.jwt_secret = "new-jwt-secret"
    rotation_settings.jwt_secret_previous = "old-jwt-secret"
    for secret in ("new-jwt-secret", "old-jwt-secret"):
        scoped = _mint(secret, scope="app_preview", app_id="42")
        assert decode_access_token(scoped) is None
