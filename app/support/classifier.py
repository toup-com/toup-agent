"""Classify a reported issue: BUG | MISSING_FEATURE | UNCLEAR.

Only BUG proceeds to the fix loop. MISSING_FEATURE (works-as-built, the
thing was never implemented — e.g. an unbuilt server path) and UNCLEAR are
surfaced to the admin and parked. We NEVER "fix" an unbuilt feature.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.support.enums import SupportClassification
from app.support.llm import ask_json

_SYSTEM = (
    "You triage incoming problem reports for the Toup platform. Classify each "
    "report into EXACTLY ONE of:\n"
    "  - \"bug\": broken behavior in code that is supposed to work (regressions, "
    "errors, wrong output, crashes).\n"
    "  - \"missing_feature\": the behavior was never built / works-as-designed but "
    "the user wants something that does not exist yet (e.g. an unimplemented "
    "server path, a feature request). These must NOT be auto-fixed.\n"
    "  - \"unclear\": not enough information to tell, or ambiguous.\n"
    "Be conservative: if a report describes desired NEW capability rather than a "
    "broken existing one, it is missing_feature, not bug. If you cannot tell, use "
    "unclear. Output JSON: {\"classification\": \"bug|missing_feature|unclear\", "
    "\"confidence\": 0.0-1.0, \"rationale\": \"one or two sentences\"}."
)


@dataclass
class ClassificationResult:
    classification: SupportClassification
    confidence: float
    rationale: str


def _coerce(value: str) -> SupportClassification:
    v = (value or "").strip().lower()
    for c in SupportClassification:
        if v == c.value:
            return c
    if "bug" in v:
        return SupportClassification.BUG
    if "feature" in v or "missing" in v:
        return SupportClassification.MISSING_FEATURE
    return SupportClassification.UNCLEAR


async def classify(*, user_id: str, raw_report: str, repro_info: str = "") -> ClassificationResult:
    prompt = f"REPORT:\n{raw_report}\n"
    if repro_info:
        prompt += f"\nREPRO / EXTRA INFO:\n{repro_info}\n"
    data = await ask_json(
        user_id=user_id,
        operation_type="system.support.classify",
        system=_SYSTEM,
        prompt=prompt,
        max_tokens=400,
    )
    if not data:
        # LLM unavailable / unparseable → UNCLEAR (safe: parks, never fixes).
        return ClassificationResult(
            classification=SupportClassification.UNCLEAR,
            confidence=0.0,
            rationale="Classifier LLM unavailable or returned no parseable result; parked for admin review.",
        )
    return ClassificationResult(
        classification=_coerce(str(data.get("classification", "unclear"))),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        rationale=str(data.get("rationale", "")).strip(),
    )
