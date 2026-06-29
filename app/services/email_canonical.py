"""Email canonicalization for free-credit grant identity.

This is the *grant-eligibility* identity key — deliberately separate from
the login identity (``users.email``, which we never rewrite). Two email
strings that a single human controls map to the same canonical form:

* Gmail / Googlemail ignore dots in the local part and treat
  ``user@googlemail.com`` and ``user@gmail.com`` as the same mailbox.
* ``+tag`` subaddressing (``user+anything@…``) is delivered to ``user@…``
  on every major provider that supports it (Gmail, Outlook, Fastmail,
  iCloud, ProtonMail, …).

Collapsing these is what stops one inbox from minting unlimited
free-grant-eligible accounts. We store only a SHA-256 *hash* of the
canonical form (see :class:`app.db.models.GrantEligibility`) so no raw
email — least of all a deleted user's — is retained in the tombstone.

NOTE: canonicalization is intentionally conservative. For non-Gmail
domains we only lowercase, trim, and strip ``+tag`` — we do NOT touch
dots (most providers treat ``a.b@`` and ``ab@`` as different mailboxes).
The goal is zero false-positives (never collapse two *different* humans),
accepting that a determined attacker can still use genuinely distinct
addresses — those are handled by the other controls (verification gate,
disposable-domain blocklist, rate limits).
"""

from __future__ import annotations

import hashlib

# Domains where the local part ignores dots and the two domains alias.
_GMAIL_DOMAINS = {"gmail.com", "googlemail.com"}
_GMAIL_CANONICAL_DOMAIN = "gmail.com"

# Providers KNOWN to deliver ``user+tag@`` to ``user@`` (subaddressing).
# This is a CONSERVATIVE allowlist by design: an unlisted domain keeps its
# ``+tag`` so two genuinely-distinct accounts there can NEVER be merged into
# one grant identity. A false-merge denies a real user their grant, which is
# worse than missing a +tag alias on an obscure domain. Extend deliberately.
_SUBADDRESS_DOMAINS = _GMAIL_DOMAINS | {
    "outlook.com", "hotmail.com", "live.com", "msn.com",
    "yahoo.com", "ymail.com", "rocketmail.com",
    "icloud.com", "me.com", "mac.com",
    "fastmail.com", "fastmail.fm",
    "proton.me", "protonmail.com", "pm.me",
    "hey.com", "zoho.com", "yandex.com",
}


def canonicalize_email(email: str) -> str:
    """Return the canonical form of ``email`` for grant-identity dedupe.

    Pure, deterministic, and dependency-free so the signup hot path and
    the backfill migration compute identical hashes. Falls back to the
    lowercased/trimmed input for anything that doesn't parse as an email.

    Subaddressing (``+tag``) is stripped ONLY for providers in
    ``_SUBADDRESS_DOMAINS``, and dots ONLY for Gmail/Googlemail — so a
    domain we don't recognise is never collapsed and two distinct humans
    there are never merged (see :data:`_SUBADDRESS_DOMAINS`).
    """
    e = (email or "").strip().lower()
    if "@" not in e:
        return e
    local, _, domain = e.rpartition("@")
    if not local or not domain:
        return e

    base_local = local
    # Strip +tag subaddress only for providers that actually route it. Keep
    # the pre-strip local as a fallback so a pathological ``+tag@domain``
    # (empty base local) never collapses to ``@domain``.
    if domain in _SUBADDRESS_DOMAINS:
        base_local = local.split("+", 1)[0] or local

    if domain in _GMAIL_DOMAINS:
        base_local = base_local.replace(".", "") or base_local
        domain = _GMAIL_CANONICAL_DOMAIN

    return f"{base_local}@{domain}"


def canonical_email_hash(email: str) -> str:
    """SHA-256 hex digest of the canonical email. The value stored in the
    grant-eligibility tombstone — opaque, fixed-width (64 chars), and
    privacy-preserving (no raw address is recoverable)."""
    canon = canonicalize_email(email)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()
