"""W2.0 — unit tests for the behavioral eval harness's pure logic.

The harness itself (scripts/eval/behavioral_suite.py) is deliberately not a
backend package — it runs host-side / in-container with stdlib only. These
tests load it by file path and pin the evaluators, the [PERF] log annotation,
and the compare gate with canned fixtures. No network, no DB.
"""
import importlib.util
from pathlib import Path

import pytest

_SUITE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "behavioral_suite.py"
_spec = importlib.util.spec_from_file_location("behavioral_suite", _SUITE_PATH)
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)


# ── evaluators ─────────────────────────────────────────────────────────

def test_contains_any_case_insensitive_and_empty_keywords():
    assert bs.contains_any("The capital is PARIS.", ["paris"])
    assert bs.contains_any("zephyr-ab12 is my boat", ["Zephyr-AB12"])
    assert not bs.contains_any("no match here", ["mars"])
    assert not bs.contains_any("anything", ["", None] if None else [""])


def test_has_button_markers():
    assert bs.has_button_markers("Pick one: [[button:Yes|y]]")
    assert bs.has_button_markers("[[BUTTON:Upper|x]]")
    assert not bs.has_button_markers("plain reply, [[Label]] is fine here")
    assert not bs.has_button_markers("")


def test_parse_iso_z_docker_nanoseconds_offsets_and_garbage():
    # docker logs -t emits RFC3339 with nanoseconds
    ts = bs._parse_iso_z("2026-07-27T22:24:36.235789403Z")
    assert ts is not None
    # trailing offset form
    ts2 = bs._parse_iso_z("2026-07-27T22:24:36.235789+00:00")
    assert ts2 is not None and abs(ts - ts2) < 0.001
    assert bs._parse_iso_z("not a timestamp") is None
    assert bs._parse_iso_z("") is None


def test_find_new_reminder_excludes_known_and_matches_any_field():
    rows = [
        {"id": "old1", "reminder_text": "water the ferns eval-xyz"},
        {"id": "new1", "name": "Water ferns", "prompt_text": "eval-XYZ water"},
        {"id": "new2", "reminder_text": "unrelated"},
    ]
    hit = bs.find_new_reminder(rows, known_ids={"old1"}, keyword="eval-xyz")
    assert hit and hit["id"] == "new1"
    assert bs.find_new_reminder(rows, known_ids={"old1", "new1"}, keyword="eval-xyz") is None
    assert bs.find_new_reminder([], set(), "kw") is None


def test_files_newer_than(tmp_path):
    old = tmp_path / "old.pdf"
    old.write_text("x")
    import os
    os.utime(old, (1000, 1000))
    new = tmp_path / "sub" / "new.pdf"
    new.parent.mkdir()
    new.write_text("y")
    found = bs.files_newer_than(str(tmp_path), since_epoch=2000)
    assert any(p.endswith("new.pdf") for p in found)
    assert not any(p.endswith("old.pdf") for p in found)


# ── [PERF] log parsing + annotation ────────────────────────────────────

_PERF_LINE = (
    "2026-07-27T22:24:36.235789403Z [PERF] cache_read=22144 cache_creation=0 "
    "input=22663 output=17 model=gpt-5.5 provider=openai"
)


def test_parse_perf_log_accepts_real_shape_and_skips_noise():
    lines = [
        _PERF_LINE,
        "2026-07-27T22:24:37.0Z [PERF] hybrid_search: 200ms — 3 results",  # not main-runner
        "[PERF] cache_read=1 cache_creation=0 input=2 output=3 model=m",   # no timestamp
        "2026-07-27T22:25:00.1Z [PERF] cache_read=0 cache_creation=0 input=100 output=5 model=gpt-4o-mini",
    ]
    events = bs.parse_perf_log(lines)
    assert len(events) == 2
    assert events[0]["cache_read"] == 22144
    assert events[0]["input"] == 22663
    assert events[0]["provider"] == "openai"
    assert events[1]["provider"] == ""  # provider optional


def _mk_results(turn_windows):
    scenarios = []
    for i, (start, end) in enumerate(turn_windows):
        scenarios.append({
            "id": f"s{i}",
            "pass": True,
            "turns": [{"started_epoch": start, "ended_epoch": end, "tokens_input": 10}],
        })
    scenarios.append({"id": "token_telemetry", "pass": None, "turns": []})
    return {"scenarios": scenarios}


def test_annotate_results_assigns_events_and_fills_telemetry():
    base_ts = bs._parse_iso_z("2026-07-27T22:24:36.2Z")
    results = _mk_results([(base_ts - 1, base_ts + 1), (base_ts + 100, base_ts + 101)])
    events = bs.parse_perf_log([_PERF_LINE])
    bs.annotate_results(results, events, slack=2.0)
    s0, s1, tele_sc = results["scenarios"]
    assert s0["perf_input_sum"] == 22663 and s0["perf_cached_sum"] == 22144
    assert s1["perf_input_sum"] is None  # no event in its window
    assert results["telemetry"]["turns_matched"] == 1
    assert results["telemetry"]["cache_hit_ratio"] == round(22144 / 22663, 4)
    # token_telemetry passes only when EVERY turn matched
    assert tele_sc["pass"] is False


def test_annotate_results_all_matched_passes_telemetry():
    base_ts = bs._parse_iso_z("2026-07-27T22:24:36.2Z")
    results = _mk_results([(base_ts - 1, base_ts + 1)])
    bs.annotate_results(results, bs.parse_perf_log([_PERF_LINE]), slack=2.0)
    tele_sc = results["scenarios"][-1]
    assert tele_sc["pass"] is True


def test_scenario_input_tokens_prefers_perf_then_response():
    assert bs.scenario_input_tokens({"perf_input_sum": 500, "turns": [{"tokens_input": 1}]}) == ("perf", 500)
    assert bs.scenario_input_tokens({"turns": [{"tokens_input": 3}, {"tokens_input": 4}]}) == ("response", 7)
    assert bs.scenario_input_tokens({}) == ("none", None)


# ── compare gate ───────────────────────────────────────────────────────

def _res(passes, tokens, source="perf"):
    scenarios = []
    for sid, ok in passes.items():
        sc = {"id": sid, "pass": ok, "turns": [{"tokens_input": tokens.get(sid, 0)}]}
        if source == "perf" and sid in tokens:
            sc["perf_input_sum"] = tokens[sid]
        scenarios.append(sc)
    return {"scenarios": scenarios}


def test_compare_pass_to_fail_is_exit_1():
    base = _res({"qa": True}, {"qa": 1000})
    cur = _res({"qa": False}, {"qa": 1000})
    code, lines = bs.compare_results(base, cur)
    assert code == bs.EXIT_SCENARIO_FAIL
    assert any("REGRESSION" in line for line in lines)


def test_compare_missing_scenario_that_passed_is_exit_1():
    base = _res({"qa": True, "mem": True}, {"qa": 1000, "mem": 1000})
    cur = _res({"qa": True}, {"qa": 1000})
    code, _ = bs.compare_results(base, cur)
    assert code == bs.EXIT_SCENARIO_FAIL


def test_compare_token_regression_needs_pct_and_min_delta():
    base = _res({"qa": True}, {"qa": 10_000})
    worse = _res({"qa": True}, {"qa": 12_500})  # +25%, +2500 → gate
    code, lines = bs.compare_results(base, worse, token_tolerance=0.15, token_min_delta=500)
    assert code == bs.EXIT_TOKEN_REGRESSION
    assert any("TOKEN REGRESSION" in line for line in lines)
    small = _res({"qa": True}, {"qa": 10_300})  # +3% → no gate
    assert bs.compare_results(base, small)[0] == bs.EXIT_OK
    # big % but under min-delta
    tiny_base = _res({"qa": True}, {"qa": 100})
    tiny_cur = _res({"qa": True}, {"qa": 200})  # +100% but +100 < 500
    assert bs.compare_results(tiny_base, tiny_cur, token_min_delta=500)[0] == bs.EXIT_OK


def test_compare_source_mismatch_reported_not_gated():
    base = _res({"qa": True}, {"qa": 10_000}, source="perf")
    cur = _res({"qa": True}, {"qa": 20_000}, source="response")
    code, lines = bs.compare_results(base, cur)
    assert code == bs.EXIT_OK
    assert any("source mismatch" in line for line in lines)


def test_compare_improvement_and_new_scenario_not_gated():
    base = _res({"qa": False}, {"qa": 1000})
    cur = _res({"qa": True, "extra": True}, {"qa": 900, "extra": 50})
    code, lines = bs.compare_results(base, cur)
    assert code == bs.EXIT_OK
    assert any("improved" in line for line in lines)
    assert any("new scenario" in line for line in lines)
