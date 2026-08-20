# memverify — the memory eval set (v3)

The release bar for the memory system. Real writer, real model, real
Postgres, labels written before the run.

    make memory-verify        # one clean run
    make memory-verify-3      # three consecutive greens — the bar
    make memory-verify-ci     # the deterministic lane, sqlite, no key

`scripts/memory_verify.py` provisions a throwaway database, refuses any DSN
that is not obviously disposable, exports the env before conftest imports,
and writes `artifacts/memverify/<stamp>.json`. Its mechanics are unchanged
from the row era; what changed is what the suite MEASURES.

## STATUS, 2026-08-20: not yet run against a live model

The harness runs end to end — provisioning, schema, 33 labeled scenarios, 2
belt runs, the artifact — but every model call returns **HTTP 401** from
OpenAI: the key in `backend/.env` is expired or revoked. So no number in
`baseline.json` is a measurement of v3 behaviour, and `misroute_pct` (the one
bound that can only come from a measurement) is `null`.

That run was not wasted. It proved the anti-vacuity design: 46 tests passed
on a completely dead writer — every junk assertion is trivially satisfied
when nothing is written — and the tests that exist to catch exactly that
failed loudly (`test_o_baseline::test_the_writer_actually_wrote_something`,
`::test_no_scenario_errored`, and every capture assertion). A suite that went
green there would have been the real defect.

To finish: export a working key and run `make memory-verify-3`. The first run
will FAIL on `misroute_pct` with "no baseline recorded" and print the JSON to
paste back — the same way this file was populated for the row era on
2026-08-07.

## What the unit is now

A **file body**. Every assertion reads `memory_files.body_md` after the real
`memory_curator.curate_turn` has run:

* a `Capture` marker is satisfied when all its tokens appear in ONE BULLET,
  **and that bullet is in the labeled file**;
* a `Reject` marker is violated when its tokens appear together in ANY body;
* every bullet goes through `bullet_problem`, every description through
  `DESCRIPTION_RE`.

Round 8's corpus could only ask "is this text in some row". That cannot see
a fact filed under the wrong subject, which was root cause #3, and it cannot
tell a correctly merged file from a duplicated one — the axis a file rewrite
changes on purpose.

## The four headline numbers

| metric | asks | bound |
|---|---|---|
| `capture_pct` | of the labeled durable facts, how many are in a file | min 100 (contract) |
| `precision_pct` | of the junk markers, how many stayed out | min 100 (contract) |
| `lint_clean_pct` | of the bullets + descriptions written, how many pass lint | min 100 (contract) |
| `misroute_pct` | of the facts captured, how many landed in the wrong file | max, **unrecorded** — see STATUS |

`lint_clean_pct` and `misroute_pct` replace the row era's
`unlabeled_rate_pct`, whose unit was a per-row `category` column that no
longer exists. See the module docstring of `metrics.py` for why those two
are the failures the file model can have and the row model could not.

Every gated metric has a denominator in `DENOMINATOR_OF`, and a **zero
denominator is a violation, not a pass** — "100% capture of nothing" reads
as green otherwise.

## The belt run

A scenario carrying `Turn.injected` is executed TWICE:

* **clean** — the writer gets what the user typed, which is what production
  does (`agent_runner` passes `display_user_message`);
* **`[dirty]`** — the writer gets the string ws_chat actually builds, the
  `[SYSTEM: The track "…"]` line and all.

Both must refuse, and the two failures mean different things. Clean-fails
means the durability rules are wrong. Dirty-fails-only means the rules
depend on someone upstream having stripped the injection first, i.e. the
product is one argument away from root cause #1. The structural half — that
the runner passes the clean copy — is pinned separately and cheaply in
`tests/test_curator_producers.py`.

## Categories

| kept | what it asserts |
|---|---|
| **A** capture | the labeled facts are in the labeled files; merge-don't-append; contradiction resolves; Farsi byte-exact; the owner never gets a `people/` file; one person, one file; every write left a change line |
| **B** precision | every bad memory from the dispatch's Section 2, refused — clean and dirty; a trivial turn writes nothing; a one-off never mints a file |
| **G** isolation | two users' files never cross; a placeholder identity still captures; `forget_everything` is scoped |
| **J** injection | pasted content cannot command the writer (`delete_file` is a real op) and is not believed as a fact |
| **K** privacy | the never-store tier, with the metformin POSITIVE that keeps it honest |
| **O** baseline | the four numbers against `baseline.json`, with anti-vacuous denominators |
| **Z** smoke | the harness is on the real stack with the v3 tables |

### Deleted, with the reason

| gone | why it has no v3 counterpart |
|---|---|
| **C** retrieval | asserted on `get_core_facts` + `hybrid_search` rows composing the injected context. v3 injects Profile + Current context + Learned + a described index + two whole files, through one loader and one renderer; that assembly is pinned byte-wise by `tests/agent/test_system_prompt_assembly.py` and `tests/agent/test_voice_context_parity.py`, which run in the ordinary sweep and need no key. |
| **D** persistence | `test_nothing_is_silently_overwritten` required four distinct facts to accumulate as four ROWS. Under files they legitimately become four bullets of one file, and the test cannot tell that from loss. The property that survives — a restated fact does not vanish — is A's merge test. |
| **E** updates/forget | supersede lineage (`superseded_by`, `is_active`) is row machinery. Newest-wins is now a CONTENT property and is A's contradiction test; `forget_all` is G's scoping test. |
| **F** dedup | every assertion was a ROW COUNT (`len(rows) == 1` for five phrasings). The file model's answer to the same question is "one bullet, rewritten", which is A's merge test. |
| **H** scale | seeded 1000 unfiled rows with `deduplicate=False` and measured a bounded injected context. Under files there is no such state to seed, and the injection budget is enforced deterministically by `truncate_body`, unit-tested in `tests/test_memory_files.py`. |
| **I** concurrency | rapid-fire writes losing nothing, and duplicate writes collapsing to one row — both row-count assertions over the dedup advisory lock, which is deleted. |
| **L** resilience | the capture outbox, when it parked serialized `MemoryCreate` rows. The outbox survives with a different payload (the TURN) and is covered far more thoroughly by `tests/test_curator_never_loses_a_fact.py` (11 tests: raise → park → replay → the fact lands; unparseable reply; a rejection is a DECISION not a loss; `apply_ops` atomic under a mid-batch crash; poison-turn abandonment; stale-turn drop; the one-per-turn replay cap) plus the runner probe in `tests/test_curator_producers.py`. A live-stack version would cost a model call per retry to test what a scripted one tests exactly. |
| **M** performance | measured extraction latency per row and the size of the injected row set. Neither has a v3 unit, and a latency assertion against a shared API is a flake generator. |
| **N** soak | 500 generated conversations through the row extractor. Worth rebuilding once the v3 numbers have a few weeks of history; rebuilding it now would baseline noise. |

`tests/RETIRED_WITH_MEMORY_V3.md` carries the same accounting for the
backend unit tests.

## The CI lane

`make memory-verify-ci` runs the DETERMINISTIC half on sqlite with no key:
the lint and description rules, the ops engine's validation and scoping, the
identity resolver, the turn pre-gates, and the curator driven by a scripted
fake LLM. It has real teeth — the validator is where a bad op is actually
stopped — and it is the lane that runs on a fork PR where the secret is
absent. `scripts/memverify_ci_guard.sh` decides whether the full suite is
required and is unchanged.
