"""Fernet-based encryption for streaming_credentials.password at rest.

The key never leaves app memory — we encrypt/decrypt in Python, store
ciphertext in a plain Text column. Rotation supported via MultiFernet:
set `platform_encryption_key` to the new primary and
`platform_encryption_key_previous` to the retiring one (comma-separated
for multi-step rotations); decrypt tries primary first, then each previous
in turn.

A Fernet token is base64 and starts with `gAAAAAB`, which is a reliable
sentinel to distinguish encrypted from legacy plaintext during the
one-time data migration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from app.config import settings

# Fernet tokens are base64; they always begin with this marker.
# Used by the migration to skip already-encrypted rows and by
# decrypt_str() to decide whether a legacy plaintext row is sitting in
# the column (defensive — post-migration this should never happen).
_FERNET_PREFIX = "gAAAAAB"


def _parse_keys() -> List[str]:
    primary = (settings.platform_encryption_key or "").strip()
    if not primary:
        return []
    keys = [primary]
    prev_raw = (settings.platform_encryption_key_previous or "").strip()
    if prev_raw:
        keys.extend(k.strip() for k in prev_raw.split(",") if k.strip())
    return keys


@lru_cache(maxsize=1)
def _multi_fernet() -> MultiFernet:
    keys = _parse_keys()
    if not keys:
        raise RuntimeError(
            "platform_encryption_key is not set — streaming credential "
            "encryption cannot initialize. Set PLATFORM_ENCRYPTION_KEY."
        )
    fernets = [Fernet(k.encode()) for k in keys]
    return MultiFernet(fernets)


def assert_configured() -> None:
    """Call at platform startup. Fails fast if the key is missing in prod.

    Agent-mode processes do not encrypt streaming credentials (that table
    lives on the platform), so this is only called from platform_main's
    lifespan — never from agent_main.
    """
    _ = _multi_fernet()  # raises RuntimeError if no key


def encrypt_str(plaintext: str) -> str:
    if plaintext is None:
        raise ValueError("encrypt_str: plaintext is None")
    token = _multi_fernet().encrypt(plaintext.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_str(token: str) -> str:
    if token is None:
        raise ValueError("decrypt_str: token is None")
    try:
        return _multi_fernet().decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        # Legacy plaintext row (pre-migration). Tell the caller so it can
        # handle gracefully; in practice the data migration eliminates this.
        if not token.startswith(_FERNET_PREFIX):
            raise InvalidToken("row is legacy plaintext, not Fernet ciphertext")
        raise


def looks_encrypted(value: str) -> bool:
    return isinstance(value, str) and value.startswith(_FERNET_PREFIX)


def rotate_token(token: str) -> str:
    """Re-encrypt a token with the primary key (used by the rotation script).

    MultiFernet.rotate() decrypts with any known key and re-encrypts with
    the primary. Cheap, idempotent if already on primary.
    """
    rotated = _multi_fernet().rotate(token.encode("utf-8"))
    return rotated.decode("utf-8")
