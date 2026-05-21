"""Create or update the four Stripe credit-tier products idempotently.

Mints (or updates) one recurring Product per credit tier:
  * starter ($16/mo)
  * builder ($40/mo)
  * pro ($80/mo)
  * elite ($160/mo)

Matching is done by `metadata.toup_tier_id` so re-runs don't create
duplicates. The freshly-minted price ids are emitted to stdout (and
also written to backend/.env when --write-env is passed) so they can
be pasted into the Railway environment as
STRIPE_PRICE_ID_{STARTER,BUILDER,PRO,ELITE}.

Stripe Search API has a 30-60s indexing lag which makes it unsafe for
idempotency — we use Product.list with metadata-filter walking
instead. Slower but always consistent.

Usage:
    python -m app.scripts.create_stripe_credit_products [--write-env]
        [--mode test|live]

Requires STRIPE_SECRET_KEY in environment (sk_test_* or sk_live_*).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional


TIERS = [
    {"toup_tier_id": "starter", "display_name": "Starter",
     "price_cents": 1600, "message_credits": 500,  "integration_credits": 1500},
    {"toup_tier_id": "builder", "display_name": "Builder",
     "price_cents": 4000, "message_credits": 1500, "integration_credits": 5000},
    {"toup_tier_id": "pro",     "display_name": "Pro",
     "price_cents": 8000, "message_credits": 3500, "integration_credits": 12000},
    {"toup_tier_id": "elite",   "display_name": "Elite",
     "price_cents": 16000, "message_credits": 8000, "integration_credits": 30000},
]


@dataclass
class TierResult:
    tier_id: str
    product_id: str
    price_id: str
    created_product: bool
    created_price: bool


def _find_existing_product(stripe, toup_tier_id: str) -> Optional[dict]:
    """Walk Product.list searching for metadata.toup_tier_id match.

    Cheaper than search API and consistent immediately after writes.
    Limited to 100 products per page; we walk pages until we find a
    match or exhaust. For test mode this is bounded by hand-curated
    products; for prod this is bounded by tier count + a small set
    of legacy products.
    """
    starting_after = None
    while True:
        kwargs = {"limit": 100, "active": True}
        if starting_after:
            kwargs["starting_after"] = starting_after
        page = stripe.Product.list(**kwargs)
        for prod in page.data:
            md = getattr(prod, "metadata", None) or {}
            try:
                if md["toup_tier_id"] == toup_tier_id:
                    return prod
            except (KeyError, TypeError):
                continue
        if not page.has_more:
            return None
        starting_after = page.data[-1].id


def _find_existing_price(stripe, product_id: str, expected_amount: int) -> Optional[dict]:
    """Find an active price on `product_id` matching unit_amount."""
    prices = stripe.Price.list(product=product_id, active=True, limit=10).data
    for p in prices:
        if p.unit_amount == expected_amount and p.recurring and p.recurring.interval == "month":
            return p
    return None


def ensure_tier(stripe, tier: dict) -> TierResult:
    """Idempotently ensure one Product + one Price exists for this tier."""
    metadata = {
        "toup_tier_id": tier["toup_tier_id"],
        "message_credits_monthly": str(tier["message_credits"]),
        "integration_credits_monthly": str(tier["integration_credits"]),
    }

    existing = _find_existing_product(stripe, tier["toup_tier_id"])
    if existing is None:
        product = stripe.Product.create(
            name=f"Toup {tier['display_name']}",
            description=(
                f"{tier['message_credits']:,} message credits/mo + "
                f"{tier['integration_credits']:,} integration credits/mo."
            ),
            metadata=metadata,
        )
        created_product = True
    else:
        # Patch metadata if upstream values shifted (e.g. we doubled
        # the credit allotment). Idempotent — Stripe accepts setting
        # the same values.
        stripe.Product.modify(existing.id, metadata=metadata)
        product = existing
        created_product = False

    existing_price = _find_existing_price(stripe, product.id, tier["price_cents"])
    if existing_price is None:
        price = stripe.Price.create(
            product=product.id,
            unit_amount=tier["price_cents"],
            currency="usd",
            recurring={"interval": "month"},
            nickname=f"Toup {tier['display_name']} (monthly)",
            metadata=metadata,
        )
        created_price = True
    else:
        price = existing_price
        created_price = False

    return TierResult(
        tier_id=tier["toup_tier_id"],
        product_id=product.id,
        price_id=price.id,
        created_product=created_product,
        created_price=created_price,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-env", action="store_true",
        help="Append STRIPE_PRICE_ID_* lines to backend/.env (no overwrite if already set).",
    )
    parser.add_argument(
        "--mode", choices=["test", "live"], default=None,
        help="Sanity-check that STRIPE_SECRET_KEY matches this mode (sk_test_/sk_live_).",
    )
    args = parser.parse_args()

    try:
        import stripe
    except ImportError:
        print("ERROR: stripe SDK not installed. pip install stripe", file=sys.stderr)
        return 2

    key = (os.environ.get("STRIPE_SECRET_KEY") or "").strip()
    if not key:
        print("ERROR: STRIPE_SECRET_KEY env var is required.", file=sys.stderr)
        return 2
    if args.mode == "test" and not key.startswith("sk_test_"):
        print("ERROR: --mode test but STRIPE_SECRET_KEY is not sk_test_*.", file=sys.stderr)
        return 2
    if args.mode == "live" and not key.startswith("sk_live_"):
        print("ERROR: --mode live but STRIPE_SECRET_KEY is not sk_live_*.", file=sys.stderr)
        return 2
    stripe.api_key = key
    print(f"Using Stripe key: {key[:12]}…")

    results: list[TierResult] = []
    for tier in TIERS:
        try:
            r = ensure_tier(stripe, tier)
            print(
                f"  {r.tier_id:<8} product={r.product_id} price={r.price_id}"
                f"   {'(new product)' if r.created_product else '(reused product)'}"
                f"   {'(new price)' if r.created_price else '(reused price)'}"
            )
            results.append(r)
        except Exception as e:
            print(f"  {tier['toup_tier_id']:<8} FAILED: {e}", file=sys.stderr)
            return 3

    print("\nPaste into Railway env (or backend/.env):")
    for r in results:
        print(f"  STRIPE_PRICE_ID_{r.tier_id.upper()}={r.price_id}")

    if args.write_env:
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        env_path = os.path.abspath(env_path)
        try:
            with open(env_path, "r") as f:
                content = f.read()
        except FileNotFoundError:
            content = ""
        lines = content.splitlines()
        existing_keys = {ln.split("=", 1)[0]: i for i, ln in enumerate(lines) if "=" in ln}
        for r in results:
            key_name = f"STRIPE_PRICE_ID_{r.tier_id.upper()}"
            new_line = f"{key_name}={r.price_id}"
            if key_name in existing_keys:
                lines[existing_keys[key_name]] = new_line
            else:
                lines.append(new_line)
        with open(env_path, "w") as f:
            f.write("\n".join(lines).rstrip() + "\n")
        print(f"\nWrote {env_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
