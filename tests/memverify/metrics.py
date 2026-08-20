"""The headline numbers of the memory verification suite, and the committed
baseline they are gated against.

Why this module is separate from the tests that use it
------------------------------------------------------
Everything here is a pure function over plain data. That is deliberate: the
suite it summarises can only run against a live Postgres with a real
OPENAI_API_KEY (see conftest.py), so if the metric DEFINITIONS lived inside
those tests they would be as unrunnable as the suite. Keeping them here lets
`tests/test_memverify_metrics.py` — an ordinary unit test that runs in CI's
sqlite sweep — pin the arithmetic and the baseline comparison with fixtures.

This module must import nothing heavier than `app.memory_files` (a leaf
module: stdlib only). In particular it must never import `.conftest` or
`.pipeline`.

The four numbers (v3)
---------------------
1. ``capture_pct``      — of the durable facts the corpus says MUST be in a
                          file, what fraction are. Denominator: the
                          `must_capture` markers of the CAPTURE + SENSITIVE
                          scenarios.

2. ``precision_pct``    — of the `must_reject` markers across the WHOLE
                          labeled corpus, what fraction stayed out. Counted
                          over every scenario, not just the JUNK ones,
                          because a capture scenario that also stores
                          garbage is still a precision failure.

3. ``lint_clean_pct``   — of the bullets the writer actually produced, what
                          fraction pass `bullet_problem`; descriptions are
                          folded in the same way. This is the metric that
                          replaces `unlabeled_rate_pct`.

4. ``misroute_pct``     — of the markers that WERE captured, what fraction
                          landed in the wrong file.

Why lint_clean and misroute exist, and what they replace
--------------------------------------------------------
The row-era headline was `unlabeled_rate_pct`: the share of stored rows
carrying no usable taxonomy label. It measured a real failure — 28% of the
founder's rows rendered as "Other" — but its unit was a per-row `category`
column, and v3 has no such column. Rows are not the product; files are.

The two failures that replace it are the two the file model can have and the
row model could not:

* a bullet that is not in the house voice ("You use an Android phone", a
  UUID, `max_results=1`) — the product's TEXT is the thing the user reads,
  so a lint failure is a visible defect, not a metadata one;
* a fact filed under the wrong subject — root cause #3, and structurally
  invisible to the row corpus, which could only ask "is this text stored
  somewhere".

Both are baselined at 0. Unlike `unlabeled_rate_pct` (whose 8.0 ceiling was
a measured tolerance for an LLM's labelling), these are CONTRACTS: the lint
is deterministic and runs inside the writer's own validator, so a non-zero
lint rate means something bypassed `validate_ops`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

BASELINE_PATH = Path(__file__).with_name("baseline.json")

# The metrics that gate the build. Every one of these must be present in both
# the measured summary and baseline.json, or `check()` reports a violation —
# so a metric cannot be quietly dropped from either side.
GATED_METRICS: Tuple[str, ...] = (
    "capture_pct",
    "precision_pct",
    "lint_clean_pct",
    "misroute_pct",
)

# The count each gated metric divides by. A gate whose denominator is zero is
# vacuous — "100% capture of nothing" and "0% junk out of no markers" both
# read as a pass — so `check()` treats a zero denominator as a violation
# rather than trusting the rate.
DENOMINATOR_OF: Dict[str, str] = {
    "capture_pct": "capture_markers_total",
    "precision_pct": "junk_markers_total",
    "lint_clean_pct": "lint_units_total",
    "misroute_pct": "captured_markers_total",
}

_EPS = 1e-9


# ── Per-scenario input ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ScenarioCounts:
    """One labeled scenario's contribution, as plain data.

    Captured while the scenario runs: the labeled corpus executes once per
    session and the suite's autouse fixture TRUNCATEs the memory tables
    before each test, so by the time an assertion body runs the files
    themselves are gone. `ScenarioResult` carries these out as data for
    exactly that reason.
    """

    id: str
    capture_total: int = 0
    capture_found: int = 0
    misrouted: int = 0
    reject_total: int = 0
    reject_hit: int = 0
    bullets_total: int = 0
    bullet_problems: int = 0
    descriptions_total: int = 0
    description_problems: int = 0
    forbidden_slugs: int = 0
    cardinality_violations: int = 0
    counts_toward_capture: bool = False


def _pct(part: int, whole: int) -> float:
    return 0.0 if whole <= 0 else round(100.0 * part / whole, 2)


def summarize(scenarios: Sequence[ScenarioCounts]) -> Dict[str, Any]:
    """Compute every reported number from the labeled run."""
    cap_total = sum(s.capture_total for s in scenarios if s.counts_toward_capture)
    cap_found = sum(s.capture_found for s in scenarios if s.counts_toward_capture)
    captured_all = sum(s.capture_found for s in scenarios)
    misrouted = sum(s.misrouted for s in scenarios)

    junk_total = sum(s.reject_total for s in scenarios)
    junk_hit = sum(s.reject_hit for s in scenarios)

    bullets = sum(s.bullets_total for s in scenarios)
    bullet_bad = sum(s.bullet_problems for s in scenarios)
    descs = sum(s.descriptions_total for s in scenarios)
    desc_bad = sum(s.description_problems for s in scenarios)
    lint_units = bullets + descs
    lint_bad = bullet_bad + desc_bad

    return {
        # 1. capture
        "capture_pct": _pct(cap_found, cap_total),
        "capture_markers_total": cap_total,
        "capture_markers_found": cap_found,
        # 2. precision
        "precision_pct": _pct(junk_total - junk_hit, junk_total),
        "junk_markers_total": junk_total,
        "junk_stored_count": junk_hit,
        # 3. lint
        "lint_clean_pct": _pct(lint_units - lint_bad, lint_units),
        "lint_units_total": lint_units,
        "lint_problems": lint_bad,
        "bullets_total": bullets,
        "bullet_problems": bullet_bad,
        "descriptions_total": descs,
        "description_problems": desc_bad,
        # 4. routing
        "misroute_pct": _pct(misrouted, captured_all),
        "captured_markers_total": captured_all,
        "misrouted_count": misrouted,
        # Structural violations — not rates, but they must be visible in the
        # artifact: a forbidden `people/<owner>` file is root cause #3 back.
        "forbidden_slugs": sum(s.forbidden_slugs for s in scenarios),
        "cardinality_violations": sum(s.cardinality_violations for s in scenarios),
    }


# ── Baseline comparison ──────────────────────────────────────────────────

@dataclass(frozen=True)
class Violation:
    metric: str
    kind: str  # "min" | "max" | "unrecorded" | "vacuous" | "missing"
    measured: Optional[float] = None
    bound: Optional[float] = None
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - formatting only
        if self.kind == "unrecorded":
            return (
                f"{self.metric}: no baseline recorded. Measured {self.measured}. "
                f"Record it by setting this in {BASELINE_PATH.name}:\n"
                f'      "{self.metric}": {{"max": {self.measured}, '
                f'"source": "measured", "recorded_at": "<run id>"}}'
            )
        if self.kind in ("vacuous", "missing"):
            return f"{self.metric}: {self.detail}"
        cmp = ">=" if self.kind == "min" else "<="
        return (
            f"{self.metric}: measured {self.measured}, baseline requires "
            f"{cmp} {self.bound}{(' — ' + self.detail) if self.detail else ''}"
        )


def load_baseline(path: Optional[Path] = None) -> Dict[str, Any]:
    data = json.loads((path or BASELINE_PATH).read_text())
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path or BASELINE_PATH}: missing a 'metrics' object")
    return metrics


def check(
    measured: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    gated: Sequence[str] = GATED_METRICS,
) -> List[Violation]:
    """Compare a summary against the committed baseline.

    A baseline entry is `{"min": <n>}` or `{"max": <n>}`. A bound of `null`
    means the number has never been measured — reported as a violation, not
    tolerated, because the whole point of this file is that a number nobody
    has looked at must not read as a pass.
    """
    out: List[Violation] = []
    for name in gated:
        if name not in measured:
            out.append(
                Violation(name, "missing", detail="not present in the measured summary")
            )
            continue
        entry = baseline.get(name)
        if not isinstance(entry, dict) or not ({"min", "max"} & set(entry)):
            out.append(
                Violation(
                    name, "missing", measured=measured[name],
                    detail=(
                        f"no baseline entry (measured {measured[name]}); every "
                        "gated metric needs a 'min' or 'max' in baseline.json"
                    ),
                )
            )
            continue

        denom_key = DENOMINATOR_OF.get(name)
        if denom_key is not None and not measured.get(denom_key):
            out.append(
                Violation(
                    name, "vacuous", measured=measured[name],
                    detail=(
                        f"{denom_key} is 0, so this rate is vacuous — it would "
                        "read as a pass whatever the system did"
                    ),
                )
            )
            continue

        kind = "min" if "min" in entry else "max"
        bound = entry[kind]
        value = float(measured[name])
        if bound is None:
            out.append(Violation(name, "unrecorded", measured=value))
            continue
        bound = float(bound)
        if kind == "min" and value < bound - _EPS:
            out.append(Violation(name, "min", value, bound, entry.get("note", "")))
        elif kind == "max" and value > bound + _EPS:
            out.append(Violation(name, "max", value, bound, entry.get("note", "")))
    return out


def format_report(measured: Mapping[str, Any]) -> str:
    """The four headline numbers, for a job summary or a console line."""
    return (
        "memory verification — headline numbers\n"
        f"  capture    (must_capture)    {measured.get('capture_pct')}%  "
        f"({measured.get('capture_markers_found')}/"
        f"{measured.get('capture_markers_total')} facts)\n"
        f"  precision  (must_reject)     {measured.get('precision_pct')}%  "
        f"({measured.get('junk_stored_count')} junk of "
        f"{measured.get('junk_markers_total')} markers)\n"
        f"  lint clean (voice + shape)   {measured.get('lint_clean_pct')}%  "
        f"({measured.get('lint_problems')}/{measured.get('lint_units_total')} "
        f"units bad; bullets {measured.get('bullets_total')}, "
        f"descriptions {measured.get('descriptions_total')})\n"
        f"  misroute   (wrong file)      {measured.get('misroute_pct')}%  "
        f"({measured.get('misrouted_count')}/"
        f"{measured.get('captured_markers_total')} captured)\n"
        f"  structural: {measured.get('forbidden_slugs')} forbidden file(s), "
        f"{measured.get('cardinality_violations')} cardinality violation(s)"
    )
