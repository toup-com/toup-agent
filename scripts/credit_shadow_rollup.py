"""Roll `[CREDIT-SHADOW]` log lines up into the two numbers that decide whether
to switch the credit cap on.

    share of spend that would have been denied
    distinct users affected, and how many hit the cap in a normal day

Usage
-----
    # from backend/
    python -m scripts.credit_shadow_rollup /path/to/agent.log
    docker logs toup-agent-<id> 2>&1 | python -m scripts.credit_shadow_rollup
    ssh prod 'journalctl -u platform-api --since -7d' | \
        python -m scripts.credit_shadow_rollup --json

Options
-------
    --json                 machine-readable output
    --include-unlimited    count admin/unlimited accounts (excluded by default:
                           they are never charged, so their "would-deny" lines
                           describe a refusal that would never bill anyone)
    --type TYPE            restrict to one ledger event_type (repeatable), e.g.
                           `--type chat_message` to exclude image generation

Reading the output
------------------
Every figure is labelled MEASURED or MODELLED, and the distinction is not
cosmetic:

* MEASURED — "of the credits that were really spent, this share was spent by a
  user whose next pre-flight would have been refused". That is a fact about
  observed traffic, read straight off the lines.

* MODELLED — "enforcement would have saved that much". It would not, exactly.
  A denial changes the future: the denied turn never spends, so `used_today`
  stops climbing and the following turns are judged against different state.
  The measured share is the right first-order estimate and, because a user at
  their cap stays at their cap for the rest of the local day, it is close to
  tight — but it is a model, not a receipt.

The script REFUSES to guess. A line whose schema version it does not recognise,
or that is missing a field, is skipped and counted in `lines_skipped` rather
than being read with a zero in place of the field it could not find.

Parsing lives in `app/credit_shadow.py` — the same module that writes the
lines. Do not reimplement it here; a reader that reimplements a writer is how
you get a plausible, uniform, fictional answer.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Iterator, Optional

# Allow both `python -m scripts.credit_shadow_rollup` from backend/ and
# `python backend/scripts/credit_shadow_rollup.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.credit_shadow import (  # noqa: E402
    DECISION_DENY, SHADOW_SCHEMA_VERSION, parse_line,
)


def _iter_lines(paths: list[str]) -> Iterator[str]:
    if not paths:
        yield from sys.stdin
        return
    for path in paths:
        with open(path, "r", errors="replace") as handle:
            yield from handle


def _pct(part: Decimal, whole: Decimal) -> Optional[float]:
    if whole == 0:
        return None
    return float(part / whole * 100)


def rollup(
    lines: Iterable[str], *,
    include_unlimited: bool = False,
    event_types: Optional[set[str]] = None,
) -> dict:
    """Aggregate shadow lines. Pure — takes an iterable of strings.

    Returns a dict of plain Python types (Decimals for credit amounts) so the
    tests can assert on the numbers rather than on formatted output.
    """
    parsed = 0
    skipped = 0
    filtered = 0
    spend_total = Decimal("0")
    spend_denied = Decimal("0")
    users_seen: set[str] = set()
    users_denied: set[str] = set()
    per_day_users: dict[str, set[str]] = defaultdict(set)
    per_day_denied_users: dict[str, set[str]] = defaultdict(set)
    per_type: dict[str, dict] = defaultdict(
        lambda: {"charges": 0, "spend": Decimal("0"), "denied_spend": Decimal("0")}
    )
    regimes: dict[str, int] = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)

    for raw in lines:
        record = parse_line(raw)
        if record is None:
            # Not a shadow line at all, or a shadow line this build cannot
            # read. Only the second case matters, and it is loud in the
            # summary rather than silently averaged in.
            skipped += 1
            continue
        parsed += 1
        if record["unlimited"] and not include_unlimited:
            filtered += 1
            continue
        if event_types is not None and record["type"] not in event_types:
            filtered += 1
            continue

        amount = record["amount"]
        denied = record["decision"] == DECISION_DENY
        user = record["user"]
        day = record["day"]

        spend_total += amount
        users_seen.add(user)
        per_day_users[day].add(user)
        bucket = per_type[record["type"]]
        bucket["charges"] += 1
        bucket["spend"] += amount
        regimes[
            f"enforcement={int(record['enforcement'])}"
            f" cap_admission={int(record['cap_admission'])}"
        ] += 1
        if denied:
            spend_denied += amount
            users_denied.add(user)
            per_day_denied_users[day].add(user)
            bucket["denied_spend"] += amount
            reasons[record["reason"] or "-"] += 1

    # "How many would hit the cap in a normal day": distinct users with at
    # least one would-deny, counted per USER-LOCAL day (the cap rolls on the
    # user's day, not the server's), then summarised across the days observed.
    # Days with observations but no denials count as zero — dropping them would
    # inflate the typical day.
    daily_denied_counts = [
        len(per_day_denied_users.get(day, set())) for day in sorted(per_day_users)
    ]

    return {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "lines_parsed": parsed,
        "lines_skipped": skipped,
        "lines_filtered_out": filtered,
        "charges": sum(v["charges"] for v in per_type.values()),
        "spend_credits": spend_total,
        "would_deny_spend_credits": spend_denied,
        "would_deny_spend_pct": _pct(spend_denied, spend_total),
        "distinct_users": len(users_seen),
        "distinct_users_would_deny": len(users_denied),
        "distinct_users_would_deny_pct": (
            float(len(users_denied) / len(users_seen) * 100) if users_seen else None
        ),
        "days_observed": len(per_day_users),
        "users_hitting_cap_per_day": {
            "median": (statistics.median(daily_denied_counts)
                       if daily_denied_counts else None),
            "mean": (round(statistics.fmean(daily_denied_counts), 2)
                     if daily_denied_counts else None),
            "max": max(daily_denied_counts) if daily_denied_counts else None,
            "series": dict(zip(sorted(per_day_users), daily_denied_counts)),
        },
        "by_event_type": {k: dict(v) for k, v in sorted(per_type.items())},
        "flag_regimes": dict(sorted(regimes.items())),
        "would_deny_reasons": dict(sorted(reasons.items())),
    }


def _render(result: dict) -> str:
    out: list[str] = []
    add = out.append
    add(f"[CREDIT-SHADOW ROLLUP] schema=v{result['schema_version']} "
        f"lines_parsed={result['lines_parsed']} "
        f"lines_skipped={result['lines_skipped']} "
        f"lines_filtered_out={result['lines_filtered_out']}")
    if result["lines_parsed"] == 0:
        add("")
        add("  No shadow lines found. Either the flag "
            "(CREDIT_SHADOW_ADMISSION_LOGGING) is off, or these logs predate it.")
        return "\n".join(out)

    pct = result["would_deny_spend_pct"]
    add("")
    add("MEASURED — read straight off the observed lines")
    add(f"  charges observed                {result['charges']}")
    add(f"  credits spent                   {result['spend_credits']:.4f}")
    add(f"  credits behind a would-deny     {result['would_deny_spend_credits']:.4f}"
        + (f"  ({pct:.1f}% of spend)" if pct is not None else ""))
    add(f"  distinct users                  {result['distinct_users']}")
    upct = result["distinct_users_would_deny_pct"]
    add(f"  users with >=1 would-deny       {result['distinct_users_would_deny']}"
        + (f"  ({upct:.1f}% of users)" if upct is not None else ""))
    add(f"  user-local days observed        {result['days_observed']}")
    cap = result["users_hitting_cap_per_day"]
    add(f"  users hitting the cap per day   median={cap['median']} "
        f"mean={cap['mean']} max={cap['max']}")

    add("")
    add("MODELLED — what enforcement would have SAVED is not the same number")
    add("  A denial changes what happens next: the denied turn never spends, so")
    add("  used_today stops climbing and later turns face different state. Treat")
    add("  the measured share as a first-order estimate, not a receipt.")

    add("")
    add("FLAG REGIME OF THE OBSERVED LINES")
    for regime, count in result["flag_regimes"].items():
        add(f"  {regime}: {count}")
    if any(not k.startswith("enforcement=0") for k in result["flag_regimes"]):
        add("  WARNING: some lines were written with enforcement ON. For those,")
        add("  a would-deny turn may have been refused for real, so its spend is")
        add("  not 'served free' — do not mix the two regimes in one figure.")

    if result["would_deny_reasons"]:
        add("")
        add("WOULD-DENY REASONS")
        for reason, count in result["would_deny_reasons"].items():
            add(f"  {reason}: {count}")

    add("")
    add("BY EVENT TYPE")
    for name, stats in result["by_event_type"].items():
        share = _pct(stats["denied_spend"], stats["spend"])
        add(f"  {name}: charges={stats['charges']} spend={stats['spend']:.4f} "
            f"would_deny_spend={stats['denied_spend']:.4f}"
            + (f" ({share:.1f}%)" if share is not None else ""))
    return "\n".join(out)


def _jsonable(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Roll [CREDIT-SHADOW] log lines into denial-rate numbers.",
    )
    parser.add_argument("files", nargs="*", help="log files (default: stdin)")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--include-unlimited", action="store_true",
        help="include admin/unlimited accounts (they are never charged)",
    )
    parser.add_argument(
        "--type", action="append", dest="types", metavar="EVENT_TYPE",
        help="restrict to this ledger event_type; repeatable",
    )
    args = parser.parse_args(argv)

    result = rollup(
        _iter_lines(args.files),
        include_unlimited=args.include_unlimited,
        event_types=set(args.types) if args.types else None,
    )
    if args.as_json:
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    else:
        print(_render(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
