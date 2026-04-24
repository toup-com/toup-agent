"""Shared encrypted-save helper for streaming_credentials.

Vault CP4.3. Two write paths converge here:

  1. `PUT /api/streaming/credentials` — user-facing web/mobile UI
  2. `credential_confirm_response` WS frame — chat-save flow (Layer A
      intercept in ws_chat_proxy)

Both call `save_encrypted_credential(...)`. The function calls
`encrypt_str()` exactly once, commits, and drops the local plaintext
reference before returning. There is no other code path that writes to
streaming_credentials without going through here.

Password lifetime in this module: one local variable in one stack frame,
reassigned to "" immediately after encrypt_str() returns the ciphertext.
Python strings are immutable so this isn't a true zeroize — but it drops
the reference before the next await yields, which is the best Python
affords.

No logging of email or password in this module. The caller can log a
key-name-only audit line before calling; this function never does.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.credential_crypto import encrypt_str


async def save_encrypted_credential(
    db: AsyncSession,
    user_id: str,
    channel: str,
    email: str,
    password: str,
) -> None:
    """Upsert (user_id, channel) into streaming_credentials with Fernet
    ciphertext in the password column. Commits the session before return.
    """
    from app.db.models import StreamingCredential

    # Encrypt first, then drop the plaintext reference. Anything after
    # this line must not read `password`.
    ciphertext = encrypt_str(password)
    password = ""  # noqa: F841 — reassignment intentionally drops reference

    email_clean = email.strip()

    result = await db.execute(
        select(StreamingCredential).where(
            and_(
                StreamingCredential.user_id == user_id,
                StreamingCredential.channel == channel,
            )
        )
    )
    cred = result.scalar_one_or_none()

    if cred is not None:
        cred.email = email_clean
        cred.password = ciphertext
        cred.updated_at = datetime.utcnow()
    else:
        cred = StreamingCredential(
            id=str(uuid.uuid4()),
            user_id=user_id,
            channel=channel,
            email=email_clean,
            password=ciphertext,
        )
        db.add(cred)

    await db.commit()
