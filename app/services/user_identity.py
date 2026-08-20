"""Who the owner is — the identity resolver the memory writer asks first.

Round 8 had no such thing. The only self-check in the memory tree was
`memory_taxonomy._is_self_entity`, literally `name.lower() == "user"`, and
it was render-only — so `people/nariman`, `people/nariman-hosseini` and
`people/user` were three separate People files about the account owner, and
nothing could merge them (taxonomy-entities recon, Q1). v3 §2.2 makes the
answer a precondition of every `people/*` op: a fact whose subject resolves
to the owner belongs in `you/profile`, not in a file about a stranger with
the owner's name.

This generalizes `MemoryService._user_aliases`, which had exactly one
consumer (the relationship mirror's gate) and lived as a private method on
a service the v3 write path never touches. Same three sources — `users.name`,
its first token, the email local part — and the same two hard-won rules:

* **Cache on SUCCESS only.** Caching the empty list from a failed lookup
  pinned the old gate into its strict mode for the life of the service
  instance, and no later call could recover it: the retry never happened
  because the cache said "asked already". A failure costs one lookup.
* **Fail SOFT, and say when the identity is unknown.** Fresh tenants carry
  placeholder identities ("Agent Owner", "<hex>@agent.local"). Treating a
  placeholder as the owner's name attaches every fact to a person file
  called "Agent Owner"; treating "unknown" as "definitely not the owner"
  drops real facts on new tenants. `known` reports which case you are in
  and the caller decides.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# How a person refers to themselves, or is referred to, regardless of tenant.
# Mirrors memory_gate._SELF_REFERENTS; kept local so this module stays
# importable from the writer without dragging the row-era gate in.
SELF_REFERENTS: FrozenSet[str] = frozenset({
    "user", "the user", "me", "i", "my", "myself", "self", "owner",
    "the owner", "account owner",
})

# Names that identify nobody. A tenant boots with users.name = "Agent Owner"
# and an "<hex>@agent.local" email; both are provisioning artefacts.
PLACEHOLDER_ALIASES: FrozenSet[str] = frozenset({
    "agent owner", "agent", "owner", "user", "toup user", "new user",
    "unknown", "none", "test user",
})

_MIN_KNOWN_ALIAS_LEN = 3
_CACHE_TTL_SEC = 300.0
_cache: Dict[str, Tuple[float, "UserIdentity"]] = {}


def _norm(text: Optional[str]) -> str:
    """Casefold + strip accents-noise and possessives, for comparison only."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).casefold().strip()
    text = re.sub(r"['’]s\b", "", text)
    return re.sub(r"\s+", " ", text).strip(" .,;:!?")


@dataclass(frozen=True)
class UserIdentity:
    """The owner's names, normalized for comparison."""

    user_id: str
    display_name: Optional[str] = None
    #: Normalized tenant-specific aliases (real name, first token, email local).
    aliases: FrozenSet[str] = field(default_factory=frozenset)
    #: False when every alias we hold is a placeholder — see the module note.
    known: bool = False

    def is_self(self, name: Optional[str]) -> bool:
        """True when `name` names the account owner.

        Generic self-referents always match — they are identity-free and
        cannot be anyone else. A tenant alias matches only as the WHOLE
        string: "Nariman's brother" is the brother, and filing his facts
        under the owner is the mirror image of the bug this module exists
        to fix.
        """
        n = _norm(name)
        if not n:
            return False
        if n in SELF_REFERENTS:
            return True
        if n in self.aliases:
            return True
        # "user account", "the user" — a leading self-referent still names
        # the owner. A leading ALIAS does not: "Nariman Hosseini" is the
        # owner, "Nariman Tajik" is somebody else.
        first = n.split(" ", 1)[0]
        return first in SELF_REFERENTS


async def resolve_user_identity(db: AsyncSession, user_id: str) -> UserIdentity:
    """The owner's identity, cached for `_CACHE_TTL_SEC` on success.

    Never raises: an unreachable users row yields `known=False`, which the
    writer treats as "cannot prove this is the owner" rather than as proof
    of the opposite.
    """
    now = time.monotonic()
    hit = _cache.get(user_id)
    if hit and now - hit[0] < _CACHE_TTL_SEC:
        return hit[1]

    name: Optional[str] = None
    email: Optional[str] = None
    try:
        from app.db.models.user import User

        row = (await db.execute(
            select(User.name, User.email).where(User.id == user_id)
        )).first()
        if row:
            name, email = row
    except Exception as exc:
        # One lookup lost, not the turn — and nothing cached, so the next
        # call retries instead of inheriting a permanent "unknown".
        logger.warning(
            "[user_identity] lookup failed for user=%s; the writer runs "
            "without owner aliases this call: %s", str(user_id)[:8], exc,
        )
        return UserIdentity(user_id=user_id)

    aliases: set[str] = set()
    display = (name or "").strip() or None
    if display:
        aliases.add(_norm(display))
        first = display.split()[0] if display.split() else ""
        if first:
            aliases.add(_norm(first))
    if email:
        local = email.split("@")[0].strip()
        if local:
            aliases.add(_norm(local))
    aliases.discard("")

    known = any(
        alias not in PLACEHOLDER_ALIASES
        and not re.fullmatch(r"[0-9a-f]{6,}", alias)
        # A one- or two-character alias identifies nobody. It arrives from
        # a short email local part ("n@toup.ai") and would flip `known` on
        # for a tenant whose users.name is still the "Agent Owner"
        # placeholder — which is exactly the case this flag exists to
        # report. Short aliases stay in the MATCH set; they just cannot be
        # the sole evidence that we know who the owner is.
        and len(alias) >= _MIN_KNOWN_ALIAS_LEN
        for alias in aliases
    )
    identity = UserIdentity(
        user_id=user_id,
        display_name=display,
        aliases=frozenset(aliases),
        known=known,
    )
    _cache[user_id] = (now, identity)
    return identity


def forget_cached_identity(user_id: Optional[str] = None) -> None:
    """Drop the cache — for a rename, and for tests."""
    if user_id is None:
        _cache.clear()
    else:
        _cache.pop(user_id, None)
