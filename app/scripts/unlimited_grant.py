"""Grant / revoke / list the UNLIMITED admin override, from a shell.

An argparse front end over ``app.services.entitlement`` — the SAME functions
``POST/GET/DELETE /api/admin/unlimited-grants`` call. Nothing about who may
sponsor whom, when a grant is refused, or what happens to the entitlement
when one ends is decided here; duplicating any of it would be duplicating the
rules.

Why both a CLI and an endpoint: the endpoint is how it will be done from the
admin panel, and the CLI is how it gets done at 2am from a Railway shell with
no panel deployed. Both authenticate as a real admin — the CLI takes
``--as-admin <email>`` and REFUSES an address whose ``users.role`` is not
``admin``, so the audit trail names a person either way.

    python -m app.scripts.unlimited_grant list [--user <email|id>] [--live]

    python -m app.scripts.unlimited_grant grant \\
        --user parmida.isazadeh@gmail.com \\
        --sponsor-apple-txn 2000000912345678 \\
        --reason "second account, rides her Builder sub" \\
        --as-admin nariman@toup.ai [--expires 2027-01-01] [--apply]

    python -m app.scripts.unlimited_grant revoke <grant-id> \\
        --reason "no longer needed" --as-admin nariman@toup.ai [--apply]

``grant`` and ``revoke`` are DRY RUN by default. Without ``--apply`` the
command runs the whole thing inside a transaction it then rolls back, so the
refusals you would have hit are the refusals it reports.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import Optional

from sqlalchemy import func, select

from app.db.database import async_session_maker
from app.db.models import User
from app.services import entitlement
from app.services.entitlement import GrantError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("unlimited_grant")


async def _find_user(db, ident: str) -> User:
    """By id or by (case-insensitive) email. Refuses on neither."""
    user = await db.get(User, ident)
    if user is not None:
        return user
    user = (await db.execute(
        select(User).where(func.lower(User.email) == ident.strip().lower())
    )).scalars().first()
    if user is None:
        raise SystemExit(f"✗ no user matching {ident!r}")
    return user


async def _admin(db, email: str) -> User:
    user = await _find_user(db, email)
    if getattr(user, "role", None) != "admin":
        # Not a formality. The grant row's granted_by_* is the audit trail, and
        # an entitlement handed out under a non-admin's name is an entitlement
        # nobody is accountable for.
        raise SystemExit(f"✗ {user.email} is role={user.role!r}, not admin")
    return user


def _fmt(g, *, live: Optional[bool] = None) -> str:
    if g.revoked_at is not None:
        state = f"REVOKED {g.revoked_at:%Y-%m-%d} by {g.revoked_by_email} ({g.revoked_reason})"
    elif g.lapsed_at is not None:
        state = f"LAPSED  {g.lapsed_at:%Y-%m-%d} ({g.lapsed_reason})"
    elif live is False:
        state = "OPEN but NOT live (sponsor dead or expires_at passed)"
    else:
        state = "LIVE"
    sponsor = g.sponsor_original_txn_id or g.sponsor_user_id or "?"
    return (
        f"  {g.id}\n"
        f"    grantee   {g.granted_to_user_id}\n"
        f"    sponsor   {g.sponsor_kind}:{sponsor}\n"
        f"    reason    {g.reason}\n"
        f"    granted   {g.granted_at:%Y-%m-%d %H:%M} by {g.granted_by_email}\n"
        f"    expires   {g.expires_at or '—'}\n"
        f"    state     {state}"
    )


async def cmd_list(args) -> int:
    async with async_session_maker() as db:
        user_id = None
        if args.user:
            user_id = (await _find_user(db, args.user)).id
        live = True if args.live else None
        grants = await entitlement.list_grants(db, user_id=user_id, live=live)
        if not grants:
            print("(no grants)")
            return 0
        now = datetime.utcnow()
        for g in grants:
            is_live = (
                g.revoked_at is None and g.lapsed_at is None
                and (g.expires_at is None or g.expires_at > now)
                and await entitlement._sponsor_is_live(db, g, now=now)
            )
            print(_fmt(g, live=is_live))
        return 0


async def cmd_grant(args) -> int:
    async with async_session_maker() as db:
        admin = await _admin(db, args.as_admin)
        grantee = await _find_user(db, args.user)
        expires = (
            datetime.fromisoformat(args.expires) if args.expires else None
        )
        try:
            grant = await entitlement.create_grant(
                db,
                granted_to_user_id=grantee.id,
                sponsor_kind="apple" if args.sponsor_apple_txn else "stripe",
                sponsor_original_txn_id=args.sponsor_apple_txn,
                sponsor_user_id=(
                    (await _find_user(db, args.sponsor_user)).id
                    if args.sponsor_user else None
                ),
                sponsor_stripe_sub_id=args.sponsor_stripe_sub,
                reason=args.reason,
                expires_at=expires,
                granted_by_user_id=admin.id,
                granted_by_email=admin.email,
                # A dry run must not announce a comp it is about to roll back.
                notify=args.apply,
            )
        except GrantError as e:
            await db.rollback()
            print(f"✗ {e.status}: {e.detail}")
            return 1
        print(_fmt(grant))
        if args.apply:
            await db.commit()
            print(f"✓ APPLIED — {grantee.email} is now on unlimited")
        else:
            await db.rollback()
            print("DRY RUN — nothing written. Re-run with --apply.")
        return 0


async def cmd_revoke(args) -> int:
    async with async_session_maker() as db:
        admin = await _admin(db, args.as_admin)
        try:
            grant = await entitlement.revoke_grant(
                db, args.grant_id,
                revoked_by_user_id=admin.id,
                revoked_by_email=admin.email,
                revoked_reason=args.reason,
                notify=args.apply,
            )
        except GrantError as e:
            await db.rollback()
            print(f"✗ {e.status}: {e.detail}")
            return 1
        print(_fmt(grant))
        if args.apply:
            await db.commit()
            print("✓ APPLIED")
        else:
            await db.rollback()
            print("DRY RUN — nothing written. Re-run with --apply.")
        return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app.scripts.unlimited_grant")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="grant history (revoked and lapsed included)")
    pl.add_argument("--user", help="email or user id")
    pl.add_argument("--live", action="store_true", help="only grants live right now")
    pl.set_defaults(fn=cmd_list)

    pg = sub.add_parser("grant", help="comp an account onto unlimited")
    pg.add_argument("--user", required=True, help="grantee email or user id")
    pg.add_argument("--sponsor-apple-txn", help="the sponsor's original_transaction_id")
    pg.add_argument("--sponsor-user", help="the sponsor's email or user id (stripe)")
    pg.add_argument("--sponsor-stripe-sub", help="the sponsor's Stripe subscription id")
    pg.add_argument("--reason", required=True)
    pg.add_argument("--expires", help="ISO datetime backstop, independent of the sponsor")
    pg.add_argument("--as-admin", required=True, help="the admin's email — recorded on the row")
    pg.add_argument("--apply", action="store_true")
    pg.set_defaults(fn=cmd_grant)

    pr = sub.add_parser("revoke", help="end a grant (the row is kept)")
    pr.add_argument("grant_id")
    pr.add_argument("--reason", required=True)
    pr.add_argument("--as-admin", required=True)
    pr.add_argument("--apply", action="store_true")
    pr.set_defaults(fn=cmd_revoke)
    return p


async def run(argv: Optional[list[str]] = None) -> int:
    """The whole command, minus the event loop.

    Separate from :func:`main` so tests can await it inside pytest-asyncio's
    already-running loop — ``asyncio.run`` raises there, and a CLI whose only
    entry point is ``asyncio.run`` is a CLI no async test can exercise.
    """
    args = build_parser().parse_args(argv)
    if args.cmd == "grant" and not (args.sponsor_apple_txn or args.sponsor_user):
        print("✗ a grant needs a sponsor: --sponsor-apple-txn or --sponsor-user")
        return 2
    return await args.fn(args)


def main(argv: Optional[list[str]] = None) -> int:
    return asyncio.run(run(argv))


if __name__ == "__main__":
    sys.exit(main())
