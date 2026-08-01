"""W3 step 1 (remediation) — unit tests for the memory-quality harness's pure logic.

The harness itself (scripts/eval/memory_quality_suite.py) is deliberately not
a backend package — it runs in-container with stdlib only (httpx/asyncpg are
lazy live-run imports). These tests load it by file path (same pattern as
test_behavioral_suite_eval.py) and pin the scorers, the scenario-pack
invariants, the compare gate, the canary identity gate, and the cleanup-SQL
ordering with canned fixtures. No network, no DB, no live stack imports.
"""
import importlib.util
from pathlib import Path

_SUITE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "eval" / "memory_quality_suite.py"
)
_spec = importlib.util.spec_from_file_location("memory_quality_suite", _SUITE_PATH)
mq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mq)


# ── query scoring ──────────────────────────────────────────────────────

def test_score_query_expected_token_case_insensitive():
    q = mq.score_query("The code is MEMQA-ab12-lock-ff01.", ["memqa-ab12-lock-ff01"], [])
    assert q["pass"] is True and q["problems"] == []


def test_score_query_missing_expected_fails_with_problem():
    q = mq.score_query("I can't find that.", ["memqa-ab12-lock-ff01"], [])
    assert q["pass"] is False
    assert any("missing" in p for p in q["problems"])


def test_score_query_update_conflict_semantics():
    # Stale value back = the D-mem-A shape = fail.
    assert mq.score_query("Your spot is OLD-1.", ["NEW-1"], ["OLD-1"])["pass"] is False
    # Both mentioned = ambiguous recall = fail (documented in README).
    both = mq.score_query("It changed from OLD-1 to NEW-1.", ["NEW-1"], ["OLD-1"])
    assert both["pass"] is False
    assert any("forbidden" in p for p in both["problems"])
    # Only the new value = pass.
    assert mq.score_query("Your spot is NEW-1.", ["NEW-1"], ["OLD-1"])["pass"] is True


def test_score_query_negative_recall_absence():
    ok = mq.score_query(
        "I don't have any record of an aquarium passphrase.",
        [], ["memqa-ab12"], require_absence=True,
    )
    assert ok["pass"] is True and ok["absence_cue"] is True
    # Confidently fabricated answer, no absence cue → fail.
    fab = mq.score_query("Your passphrase is fishtank42.", [], ["memqa-ab12"],
                         require_absence=True)
    assert fab["pass"] is False and fab["absence_cue"] is False
    # Honest wording but leaking a seeded token → still a fail (bleed).
    leak = mq.score_query("I don't have that, though memqa-ab12-lock rings a bell.",
                          [], ["memqa-ab12"], require_absence=True)
    assert leak["pass"] is False and leak["forbidden_found"] == ["memqa-ab12"]


def test_absence_cue_matches_unicode_apostrophe():
    assert mq.has_absence_cue("I don’t have that on record.")
    assert mq.has_absence_cue("No record of it.")
    assert not mq.has_absence_cue("Here it is: swordfish.")


# ── aggregation math ───────────────────────────────────────────────────

def test_category_score_and_empty():
    assert mq.category_score([{"pass": True}, {"pass": False}]) == (0.5, 1, 2)
    assert mq.category_score([]) == (0.0, 0, 0)


def test_aggregate_overall_is_macro_average():
    # Each category weighs the same regardless of query count.
    overall = mq.aggregate_overall([
        {"score": 1.0, "passed": 4, "total": 4},
        {"score": 0.0, "passed": 0, "total": 1},
    ])
    assert overall == {"score": 0.5, "passed": 4, "total": 5}
    assert mq.aggregate_overall([]) == {"score": 0.0, "passed": 0, "total": 0}


# ── scenario pack invariants ───────────────────────────────────────────

def test_build_scenarios_has_all_six_categories_in_order():
    scen = mq.build_scenarios("memqa-test1234")
    assert [s["id"] for s in scen] == mq.CATEGORY_IDS
    assert len(mq.CATEGORY_IDS) == 6


def test_build_scenarios_every_store_is_marker_scoped():
    marker = "memqa-test1234"
    for s in mq.build_scenarios(marker):
        for st in s["steps"]:
            if st["kind"] == "store":
                assert marker in st["message"], f"unmarked store in {s['id']}"
                assert st["save"] is True  # extraction only sees saved turns
            else:
                assert st.get("expect_all") or st.get("require_absence")


def test_refusal_is_not_an_absence_cue():
    # A capability refusal and an honest absence are different things. The
    # 2026-08-01 run conflated them: the models refused to store passphrases,
    # the token never reached memory, and the suite reported a MEMORY defect.
    assert mq.looks_like_refusal("I can't save passphrases. Use a password manager.")
    assert mq.looks_like_refusal("I can’t retain Priya’s desk access code.")
    assert mq.looks_like_refusal("I cannot store account numbers.")
    assert not mq.looks_like_refusal("I don't have that stored.")
    assert not mq.looks_like_refusal("Saved.")
    # ...and an honest absence must still score as one.
    assert mq.score_query("I don't have that stored.", [], ["memqa-z"],
                          require_absence=True)["pass"]


def test_no_seeded_fact_is_credential_shaped():
    """Regression guard for the 2026-08-01 invalid run.

    Both gpt-5.5 and gpt-5.6-terra refuse to store credential-shaped values
    ("I can't save passphrases -- keep it in a password manager"). A refusal
    reaches the scorer as a missing token, so a credential-shaped seed makes
    the suite report a memory failure for something that never got near
    memory. Seeds must stay ordinary identifiers.
    """
    banned = ("passphrase", "password", "door code", "access code",
              "account number", "membership number", "security word")
    for s in mq.build_scenarios("memqa-test1234"):
        for st in s["steps"]:
            if st["kind"] != "store":
                continue
            low = st["message"].lower()
            for b in banned:
                assert b not in low, f"{s['id']}/{st['label']} seeds a '{b}'"


def test_cleanup_sweeps_the_transcript_not_just_memories():
    """Regression guard: a seeded fact lives in `messages` too.

    Sweeping only `memories` leaves every marker token retrievable from
    conversation history, so a later run can answer with an earlier run's
    value. Measured 2026-08-01: 0 marker memories, 55 marker messages, and a
    temporal_ordering failure that was really the previous run's bicycle.
    """
    labels = [lbl for lbl, _t, _s in mq.MESSAGE_CLEANUP_STATEMENTS]
    tables = [t for _l, t, _s in mq.MESSAGE_CLEANUP_STATEMENTS]
    assert "messages" in tables
    # The nullable back-reference must be unlinked BEFORE the delete, or the
    # delete trips memories.source_message_id (no ON DELETE in this subsystem).
    assert labels.index("memories.source_message_id (unlink)") < labels.index("messages")
    assert "ILIKE $1" in mq.MESSAGE_ID_LOOKUP_SQL
    for _l, _t, sql in mq.MESSAGE_CLEANUP_STATEMENTS:
        assert "::text[]" in sql, "ids are TEXT columns — casts are required"


def test_build_scenarios_update_tokens_share_shape_but_differ():
    # v1/v2 must be near-identical restatements (kestrel-dbf7 vs kestrel-13b4
    # shape) so the >=0.90 auto-duplicate path (D-mem-A) is really exercised.
    up = next(s for s in mq.build_scenarios("memqa-test1234") if s["id"] == "update_conflict")
    assert up["old_token"] != up["new_token"]
    assert up["old_token"].rsplit("-", 1)[0] == up["new_token"].rsplit("-", 1)[0]
    recall = [st for st in up["steps"] if st["kind"] == "query"][0]
    assert recall["expect_all"] == [up["new_token"]]
    assert recall["forbid"] == [up["old_token"]]


def test_build_scenarios_dry_replies_score_perfect():
    # The canned ideal replies must satisfy their own checks — this is what
    # --dry-run runs through the full pipeline.
    for s in mq.build_scenarios("memqa-test1234"):
        for st in s["steps"]:
            if st["kind"] != "query":
                continue
            q = mq.score_query(
                st["dry_reply"], st.get("expect_all") or [], st.get("forbid") or [],
                require_absence=bool(st.get("require_absence")),
            )
            assert q["pass"], f"dry reply fails its own check: {s['id']}/{st['label']}"


def test_build_scenarios_negative_query_never_contains_marker():
    # Any marker occurrence in the negative reply must be bleed from memory,
    # so the query itself must not hand the marker to the model.
    marker = "memqa-test1234"
    neg = next(s for s in mq.build_scenarios(marker) if s["id"] == "negative_recall")
    q = neg["steps"][0]
    assert marker not in q["message"]
    assert q["forbid"] == [marker] and q["require_absence"] is True


# ── compare gate ───────────────────────────────────────────────────────

def _res(scores: dict, overall: float) -> dict:
    return {
        "categories": [{"id": cid, "score": s} for cid, s in scores.items()],
        "overall": {"score": overall},
    }


def test_compare_regression_beyond_threshold_is_exit_1():
    base = _res({"a": 1.0, "b": 0.5}, 0.75)
    cur = _res({"a": 0.5, "b": 0.5}, 0.5)
    code, lines = mq.compare_memory_results(base, cur, regression_threshold=0.10)
    assert code == mq.EXIT_REGRESSION
    assert any("REGRESSION" in ln for ln in lines)


def test_compare_drop_within_threshold_is_exit_0():
    base = _res({"a": 1.0}, 1.0)
    cur = _res({"a": 0.95}, 0.95)
    assert mq.compare_memory_results(base, cur, regression_threshold=0.10)[0] == mq.EXIT_OK
    # exactly at the threshold boundary → not gated
    edge = _res({"a": 0.9}, 0.9)
    assert mq.compare_memory_results(base, edge, regression_threshold=0.10)[0] == mq.EXIT_OK


def test_compare_missing_category_is_exit_1():
    base = _res({"a": 1.0, "b": 1.0}, 1.0)
    cur = _res({"a": 1.0}, 1.0)
    assert mq.compare_memory_results(base, cur)[0] == mq.EXIT_REGRESSION


def test_compare_improvement_and_new_category_not_gated():
    base = _res({"a": 0.0}, 0.0)
    cur = _res({"a": 1.0, "new_cat": 0.0}, 0.5)
    code, lines = mq.compare_memory_results(base, cur)
    assert code == mq.EXIT_OK
    assert any("improved" in ln for ln in lines)
    assert any("not gated" in ln for ln in lines)


# ── canary identity gate ───────────────────────────────────────────────

def test_identity_gate_canary_passes():
    ok, detail = mq.check_canary_identity("533354ce-0000-4000-8000-000000000000")
    assert ok and "canary" in detail


def test_identity_gate_unknown_and_missing_refused():
    assert mq.check_canary_identity("871bac24-x")[0] is False
    assert mq.check_canary_identity("")[0] is False
    assert mq.check_canary_identity(None)[0] is False


def test_identity_gate_override_is_exact_match_only():
    assert mq.check_canary_identity("871bac24-x", override="871bac24-x")[0] is True
    assert mq.check_canary_identity("871bac24-x", override="871bac24-y")[0] is False


def test_identity_gate_real_user_refused_even_with_override():
    # 2739b5c6 is the real beta user from incident 2026-07-28 — never.
    ok, detail = mq.check_canary_identity("2739b5c6-real", override="2739b5c6-real")
    assert ok is False and "REAL USER" in detail


# ── cleanup SQL contract ───────────────────────────────────────────────

def test_normalize_db_url_strips_sqlalchemy_driver():
    assert mq.normalize_db_url("postgresql+asyncpg://u:p@h:5432/db") == "postgresql://u:p@h:5432/db"
    assert mq.normalize_db_url("postgres+asyncpg://u@h/db") == "postgres://u@h/db"
    assert mq.normalize_db_url("postgresql://u@h/db") == "postgresql://u@h/db"
    assert mq.normalize_db_url("") == ""


def test_cleanup_statements_fk_children_before_parent():
    labels = [label for label, _t, _s in mq.CLEANUP_STATEMENTS]
    assert labels[-1] == "memories"  # parent DELETE strictly last
    for child in ("memory_events", "entity_links", "memory_relationships"):
        assert labels.index(child) < labels.index("memories")


def test_cleanup_statements_use_text_array_casts():
    # ids are TEXT (String(36)) columns, not native uuid — every ANY() must cast.
    for _label, _table, sql in mq.CLEANUP_STATEMENTS:
        assert "::text[]" in sql, f"missing ::text[] cast: {sql}"


def test_cleanup_covers_both_relationship_directions_and_nullable_links():
    sqls = {label: sql for label, _t, sql in mq.CLEANUP_STATEMENTS}
    assert "source_id" in sqls["memory_relationships"]
    assert "target_id" in sqls["memory_relationships"]
    # nullable references are unlinked, not deleted
    assert sqls["document_chunks (unlink)"].startswith("UPDATE")
    assert sqls["media (unlink)"].startswith("UPDATE")
    assert sqls["memories.superseded_by (unlink)"].startswith("UPDATE")


def test_cleanup_lookup_matches_all_content_columns():
    for col in ("content", "canonical_content", "summary"):
        assert col in mq.MEMORY_ID_LOOKUP_SQL
    assert "ILIKE" in mq.MEMORY_ID_LOOKUP_SQL


def test_cleanup_pattern_is_marker_scoped():
    # The sweep must only ever match harness-marked rows.
    assert mq.CLEANUP_PATTERN == "%memqa-%"
    assert mq.MARKER_PREFIX in mq.CLEANUP_PATTERN


# ── review-round pins (2026-07-29): hedged hallucinations + order-aware ──

def test_negative_recall_fails_hedged_hallucination():
    q = mq.score_query(
        "No problem — I believe it is swordfish-99, though I'm not 100% sure.",
        [], ["memqa-z"], require_absence=True,
    )
    assert not q["pass"]
    assert q["fabricated"] == ["swordfish-99"]


def test_negative_recall_requires_phrase_level_cue():
    assert not mq.score_query("No.", [], ["memqa-z"], require_absence=True)["pass"]
    assert mq.score_query("I have no record of that passphrase.", [], ["memqa-z"],
                          require_absence=True)["pass"]
    # hyphenated prose without a digit is never an answer-shaped value
    assert mq.score_query("I don't have that — double-check your notes.", [],
                          ["memqa-z"], require_absence=True)["pass"]


def test_temporal_query_first_is_order_aware():
    ok = mq.score_query("You told me AAA-11 first; BBB-22 came later.",
                        ["AAA-11"], [], order_before=("AAA-11", "BBB-22"))
    assert ok["pass"] and ok["order_ok"]
    bad = mq.score_query("BBB-22 then AAA-11.",
                        ["AAA-11"], [], order_before=("AAA-11", "BBB-22"))
    assert not bad["pass"] and not bad["order_ok"]


def test_scored_query_count_is_pinned():
    # Docs say 7 scored queries; keep them honest.
    scen = mq.build_scenarios("memqa-pin")
    queries = [st for s in scen for st in s["steps"] if st["kind"] == "query"]
    assert len(queries) == 7
