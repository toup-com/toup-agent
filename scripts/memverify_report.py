#!/usr/bin/env python3
"""Print the memory verification headline numbers into the CI job summary.

`memory_verify.py` writes a machine-readable result file per run under
artifacts/memverify/. Everything in it is already correct; nothing ever read
it. This turns the three gated numbers into something a human sees on the run
page, which is the difference between a metric and a number in a file.

Exits 0 even when there is no artifact: this runs with `if: always()` so that a
failed suite still publishes its numbers, and it must not convert a suite
failure into a confusing second failure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = REPO / "artifacts" / "memverify"

HEADLINE = (
    ("recall_pct", "recall — must_store", "%"),
    ("precision_pct", "precision — must_not_store", "%"),
    ("unlabeled_rate_pct", "unlabeled — no usable taxonomy label", "%"),
    ("non_canonical_rate_pct", "…of which non-canonical", "%"),
    ("catch_all_rate_pct", "…of which catch-all", "%"),
)


def latest_result() -> Path | None:
    if not ARTIFACTS.is_dir():
        return None
    runs = sorted(
        (p for p in ARTIFACTS.glob("*.json") if not p.name.endswith("-summary.json")),
        key=lambda p: p.stat().st_mtime,
    )
    return runs[-1] if runs else None


def main() -> int:
    path = latest_result()
    if path is None:
        print("no memverify result artifact found — nothing to report")
        return 0

    data = json.loads(path.read_text())
    metrics = data.get("metrics", {})
    totals = data.get("totals", {})

    lines = [
        "## Agent memory verification",
        "",
        f"run `{data.get('run_id', '?')}` — "
        f"{totals.get('passed', '?')} passed, {totals.get('failed', '?')} failed, "
        f"{totals.get('skipped', '?')} skipped",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, label, unit in HEADLINE:
        if key in metrics:
            lines.append(f"| {label} | {metrics[key]}{unit} |")
    lines += [
        "",
        f"stored rows sampled: {metrics.get('stored_rows_total', '?')} · "
        f"facts {metrics.get('recall_facts_found', '?')}/"
        f"{metrics.get('recall_facts_total', '?')} · "
        f"junk {metrics.get('junk_stored_count', '?')}/"
        f"{metrics.get('junk_markers_total', '?')} markers",
        "",
        "Gated against `backend/tests/memverify/baseline.json`.",
    ]
    body = "\n".join(lines)

    print(body)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(body + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
