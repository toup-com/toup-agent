"""The headline numbers of the memory verification suite, and the committed
baseline they are gated against.

Why this module is separate from the tests that use it
------------------------------------------------------
Everything here is a pure function over plain data. That is deliberate: the
suite it summarises can only run against a live Postgres+pgvector with a real
OPENAI_API_KEY (see conftest.py), so if the metric DEFINITIONS lived inside
those tests they would be as unrunnable as the suite. Keeping them here lets
`tests/test_memverify_metrics.py` — an ordinary unit test that runs in CI's
sqlite sweep — pin the arithmetic, the classification rules and the baseline
comparison with fixtures.

This module must therefore import nothing heavier than `app.memory_taxonomy`
(a leaf module that imports only `enum` and `re`). In particular it must never
import `.conftest` or `.pipeline`.

The three numbers
-----------------
1. ``recall_pct``     — of the durable facts the corpus says MUST be stored,
                        what fraction actually reached the store. Denominator:
                        the `must_store` markers of the CAPTURE scenarios,
                        exactly as test_a_capture_recall.py counts them.

2. ``precision_pct``  — of the `must_not_store` markers across the WHOLE
                        labeled corpus, what fraction stayed out. This is the
                        junk rate, and it reuses test_b_precision_junk.py's
                        definition verbatim: a marker is "junk stored" when
                        `Marker.found_in()` matches an active memory. Counted
                        over ALL_LABELED, not just the JUNK scenarios, because
                        a capture scenario that also stores garbage is still a
                        precision failure (test_b's own comment).

3. ``unlabeled_rate_pct`` — of the rows actually stored, what fraction carries
                        no usable taxonomy label. Defined below.

Why unlabeled_rate exists
-------------------------
Recall and precision both ask "is the right SET of rows in the store". Neither
can see the failure that motivated this whole program: rows that are stored,
are individually defensible, and are nevertheless useless because the app
cannot say what they ARE. The 2026-07-29 audit found 28% of the founder's
production rows rendering as "Other" — see the module docstring of
`app/memory_taxonomy.py`. That is invisible to every public memory benchmark,
because a benchmark scores answers, not the label on the row that produced
them. It is not cosmetic either: `hybrid_search` ANDs a `category.in_(...)`
filter for classified queries, so a row in the catch-all bucket is unreachable
by any query the classifier has an opinion about.

A stored row is UNLABELED when either of these holds:

  (a) NON-CANONICAL — `category` is NULL/blank, or the value in the column is
      not a member of the canonical enum for the row's `brain_type`. The app's
      label map has no entry, so it renders "Other". Every production write
      path normalises (memory_service.py:536, memory_extractor.py:872 and
      :1131, agent_reflection.py:168), so a non-canonical row means some path
      bypassed `normalize_category` — a structural defect, not a quality
      judgement. Baselined at 0.

  (b) CATCH-ALL — the value IS canonical but is the bucket
      `normalize_category` returns when it recognises nothing: `other` for the
      USER brain, `domain_knowledge` for the AGENT brain. The write succeeded
      and the extractor produced no usable label, which downstream is
      indistinguishable from (a).

The WORK brain is exempt from rule (b): `WorkCategory` has exactly one member
(`process`), so it cannot express a label distinction and counting every work
row as unlabeled would be an artefact of the enum's size rather than a finding.
Rule (a) still applies to it.

Both sub-rates are reported alongside the composite, so a regression tells you
which half moved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

BASELINE_PATH = Path(__file__).with_name("baseline.json")

# What `normalize_category` returns for a value it does not recognise, per
# brain. Kept as data rather than probed at runtime so the classification is
# readable; `test_memverify_metrics.py` asserts it still matches the taxonomy.
CATCH_ALL_BY_BRAIN: Dict[str, str] = {
    "user": "other",
    "agent": "domain_knowledge",
}

# Brains whose enum is too small to carry a label distinction — rule (b) is not
# applied to them. See the module docstring.
NO_CATCH_ALL_BRAINS: Tuple[str, ...] = ("work",)

LABELED = "labeled"
NON_CANONICAL = "non_canonical"
CATCH_ALL = "catch_all"

# The metrics that gate the build. Every one of these must be present in both
# the measured summary and baseline.json, or `check()` reports a violation —
# so a metric cannot be quietly dropped from either side.
GATED_METRICS: Tuple[str, ...] = (
    "recall_pct",
    "precision_pct",
    "unlabeled_rate_pct",
    "non_canonical_rate_pct",
)

# The count each gated metric divides by. A gate whose denominator is zero is
# vacuous — "100% recall of nothing" and "0% junk out of no markers" both read
# as a pass — so `check()` treats a zero denominator as a violation rather than
# trusting the rate.
DENOMINATOR_OF: Dict[str, str] = {
    "recall_pct": "recall_facts_total",
    "precision_pct": "junk_markers_total",
    "unlabeled_rate_pct": "stored_rows_total",
    "non_canonical_rate_pct": "stored_rows_total",
}

_EPS = 1e-9


# ── Label classification ─────────────────────────────────────────────────


def _canon_key(value: Optional[str]) -> str:
    """The same folding `normalize_category` applies before lookup."""
    return (value or "").strip().lower().replace(" ", "_").replace("-", "_")


def label_verdict(category: Optional[str], brain_type: Optional[str]) -> str:
    """Classify one stored row's taxonomy label.

    Returns one of LABELED / NON_CANONICAL / CATCH_ALL.
    """
    from app.memory_taxonomy import normalize_category

    brain = _canon_key(brain_type) or "user"
    raw = _canon_key(category)
    if not raw:
        return NON_CANONICAL

    canonical = normalize_category(raw, brain_type=brain)
    if canonical != raw:
        # Either an alias ("schedule" -> "active_task") or an unrecognised
        # value that fell back. Both mean the value SITTING IN THE COLUMN is
        # not one the app can label.
        return NON_CANONICAL

    if brain in NO_CATCH_ALL_BRAINS:
        return LABELED
    return CATCH_ALL if canonical == CATCH_ALL_BY_BRAIN.get(brain) else LABELED


def is_unlabeled(category: Optional[str], brain_type: Optional[str]) -> bool:
    return label_verdict(category, brain_type) != LABELED


# ── Per-scenario input ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ScenarioCounts:
    """One labeled scenario's contribution, as plain data.

    `labels` is every ACTIVE row the scenario left in the store, as
    (category, brain_type) pairs. It has to be captured while the scenario
    runs: the labeled corpus executes once per session and the suite's autouse
    fixture TRUNCATEs the memory tables before each test, so by the time an
    assertion body runs the rows themselves are gone. `ScenarioResult` carries
    them out as data for exactly this reason.
    """

    id: str
    must_store_total: int = 0
    must_store_found: int = 0
    must_not_store_total: int = 0
    must_not_store_hit: int = 0
    labels: Tuple[Tuple[str, str], ...] = ()
    counts_toward_recall: bool = False


def _pct(part: int, whole: int) -> float:
    return 0.0 if whole <= 0 else round(100.0 * part / whole, 2)


def summarize(scenarios: Sequence[ScenarioCounts]) -> Dict[str, Any]:
    """Compute every reported number from the labeled run."""
    recall_total = sum(s.must_store_total for s in scenarios if s.counts_toward_recall)
    recall_found = sum(s.must_store_found for s in scenarios if s.counts_toward_recall)

    junk_total = sum(s.must_not_store_total for s in scenarios)
    junk_hit = sum(s.must_not_store_hit for s in scenarios)

    verdicts = [label_verdict(cat, brain) for s in scenarios for (cat, brain) in s.labels]
    stored = len(verdicts)
    non_canonical = sum(1 for v in verdicts if v == NON_CANONICAL)
    catch_all = sum(1 for v in verdicts if v == CATCH_ALL)

    return {
        # 1. recall / must_store
        "recall_pct": _pct(recall_found, recall_total),
        "recall_facts_total": recall_total,
        "recall_facts_found": recall_found,
        # 2. precision / must_not_store (the junk rate)
        "precision_pct": _pct(junk_total - junk_hit, junk_total),
        "junk_markers_total": junk_total,
        "junk_stored_count": junk_hit,
        # 3. unlabeled
        "unlabeled_rate_pct": _pct(non_canonical + catch_all, stored),
        "non_canonical_rate_pct": _pct(non_canonical, stored),
        "catch_all_rate_pct": _pct(catch_all, stored),
        "stored_rows_total": stored,
        "unlabeled_rows": non_canonical + catch_all,
        "non_canonical_rows": non_canonical,
        "catch_all_rows": catch_all,
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
        if self.kind == "vacuous":
            return f"{self.metric}: {self.detail}"
        if self.kind == "missing":
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
    means the number has never been measured — that is reported as a
    violation, not tolerated, because the whole point of this file is that a
    number nobody has looked at must not read as a pass.
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
                    name,
                    "missing",
                    measured=measured[name],
                    detail=(
                        f"no baseline entry (measured {measured[name]}); every gated "
                        "metric needs a 'min' or 'max' in baseline.json"
                    ),
                )
            )
            continue

        denom_key = DENOMINATOR_OF.get(name)
        if denom_key is not None and not measured.get(denom_key):
            out.append(
                Violation(
                    name,
                    "vacuous",
                    measured=measured[name],
                    detail=(
                        f"{denom_key} is 0, so this rate is vacuous — it would read as "
                        "a pass whatever the system did"
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
    """The three headline numbers, for a job summary or a console line."""
    return (
        "memory verification — headline numbers\n"
        f"  recall     (must_store)      {measured.get('recall_pct')}%  "
        f"({measured.get('recall_facts_found')}/{measured.get('recall_facts_total')} facts)\n"
        f"  precision  (must_not_store)  {measured.get('precision_pct')}%  "
        f"({measured.get('junk_stored_count')} junk of "
        f"{measured.get('junk_markers_total')} markers)\n"
        f"  unlabeled  (no usable label) {measured.get('unlabeled_rate_pct')}%  "
        f"({measured.get('unlabeled_rows')}/{measured.get('stored_rows_total')} stored rows"
        f"; non-canonical {measured.get('non_canonical_rows')}"
        f", catch-all {measured.get('catch_all_rows')})"
    )
