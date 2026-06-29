"""Free-credit grant eligibility tombstone (Sybil / multi-account resistance).

One row per CANONICAL email identity that has ever received the one-time
free-credit grant. Deliberately decoupled from ``users``:

* The PRIMARY KEY is ``canonical_email_hash`` (see
  :mod:`app.services.email_canonical`). The DB uniqueness of the PK makes
  the grant idempotent per human identity — Gmail dot/``+alias`` variants
  collapse to one hash — and makes concurrent signups RACE-SAFE: two
  parallel INSERTs of the same hash, one wins and the other hits the
  constraint (the grant path reads that as "already claimed").

* There is intentionally **NO ForeignKey to users**. This row is the
  deletion-durable tombstone: it MUST survive ``DELETE FROM users`` so
  that delete-account → re-register cannot farm a fresh grant. Because it
  carries no FK, the users-table cascade never reaches it, and
  ``user_deletion._wipe_user_data_tables`` never lists it. ``first_user_id``
  is a plain string (the account that first claimed the grant), never
  cascaded or nulled by deletion.

Only a SHA-256 hash of the canonical email is stored — never the raw
address — so a deleted user's email is not retained here.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class GrantEligibility(Base):
    __tablename__ = "grant_eligibility"

    # SHA-256 hex of the canonical email (64 chars). PK ⇒ unique ⇒ the
    # race-safe idempotency key for the one-time free grant.
    canonical_email_hash: Mapped[str] = mapped_column(String(64), primary_key=True)

    granted_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )

    # The account that first claimed the grant for this identity. Plain
    # String, NOT a ForeignKey — the tombstone must outlive the user row
    # (see module docstring). Nullable so a future hard-anonymization pass
    # could clear it without dropping the tombstone.
    first_user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
