# Tests retired with memory v3 (2026-08-20, WS-2)

Contract: `docs/memory/rebuild-2026-08-v3.md`. Nothing here was deleted to
make a suite green. Each line says what the file protected and where that
protection lives now — or why it has no v3 counterpart because its SUBJECT
no longer exists.

The rule this file exists to enforce: **a test may only be deleted with its
subject.** If the behaviour still ships, the test moves; it does not vanish.

## Deleted because the SERVICE was deleted

| file | tests | subject | why it has no counterpart |
|---|---:|---|---|
| `test_decay.py` | 25 | `decay_service` | Ebbinghaus strength per row. `strength` is not a product concept any more: no payload carries it, nothing ranks by it, and the file body is the unit. |
| `test_consolidation.py` | 11 | `consolidation_service` | The ADDITIVE pass — it wrote a new semantic row and retired nothing ("0 rows rewritten or retired" on the 2026-08-10 dry run). v3 merges IN PLACE via the `rewrite` op, covered by `test_memory_file_ops.py`. Was already quarantined in COVERAGE_DEBT.txt. |
| `test_active_task.py` | 8 | `active_task_service` | The `working` file's producer: 7 regexes over the user's message, `db.add(Memory(...))` directly, importance 0.9, always injected. There is no Working-on surface in v3; a standing arrangement is ONE line in `you/profile` and the curator's durability rules cover it (`test_memory_curator.py::test_the_durability_rules_name_the_dispatch_s_own_bad_memories`). |
| `test_active_task_gate_regressions.py` | 16 | `detect_active_tasks` | Same subject. Its corpus of "I need to finish X" strings is exactly the class v3 declines to store; the eval set carries the interesting ones as REJECT fixtures. |
| `test_dedup_adjudicates_every_candidate.py` | 21 | `MemoryDedupService` | Row-level adjudication (contradiction > merge > duplicate > new) over a global similarity window. v3's merge is the writer choosing `rewrite` over `add` against the file body it was shown, measured by the eval set's merge-don't-append fixture rather than by row counts. |
| `test_destructive_verdict_confirmation.py` | 12 | `MemoryDedupService` | Same — but see the note below: two of its tests pinned a PROPERTY that has a v3 successor. **Also:** this file was RED on `main` before v3 (round 8 added `self._routing_unavailable` in `__init__`, and three tests build the service with `__new__`). It is resolved by retiring it WITH the service, which is what the do-not-break list asked for. |
| `test_memory_store_embedding_provider.py` | 10 | `memory_store` → dedup → embeddings | `memory_store` is now an instruction to the curator; it computes no embedding and writes no row. What survives is "the tool must say plainly when nothing was written", pinned in `test_curator_producers.py`. |
| `test_portrait_cache_ttl.py` | — | `user_portrait_service` | The portrait block is gone from the prompt (v3 §3.1); WS-1's `test_system_prompt_assembly.py` asserts its absence. |
| `test_retrieval_reinforcement_is_cite_gated.py` | — | `DecayService.reinforce_memory` | Reinforcement raised `strength`. Both are gone; `retrieval_feedback.reinforce_cited_memories` is a documented no-op stub because `app/api/feedback.py` still routes to the service. |

## Deleted because the ROUTES were deleted

| file | tests | why |
|---|---:|---|
| `test_api.py` (partial — 15 of 26) | 15 | A driver for the round-8 ROW API: `POST /memories`, `GET /memories/{id}`, `/search`, `/category/{c}`, `/{id}/events`, `/{id}/reinforce`, and the strength / memory_level / emotional_salience / "enhanced fields" response assertions — plus `POST /admin/decay` and `POST /admin/consolidate`. All nine routes are deleted (§1.1, §4), so there is nothing left to drive. **The file is NOT deleted:** 11 tests have a live subject (auth, ingest, stats, `/agent/store` + `/agent/recall`, and two pure-function service tests) and stay. `test_admin_memory_health_requires_auth` was REPAIRED rather than retired — it asked for `/api/admin/memory-health`, a path that never existed (the route has always carried `{user_id}`), so it got a router 404 and asserted 401, meaning it had never once tested authentication. **Its 8 remaining reds all predate this rebuild:** 6 are `create_user` → `no such table: credit_balances` (the file's own quarantine reason in COVERAGE_DEBT.txt; `auth_service.py` is byte-identical to 91c45ee9), 1 is a missing `sentence_transformers`, and 1 is the rule-based extractor's own assertion (that method is byte-identical to 91c45ee9 too — v3 deleted only its LLM half). The `auth_headers` fixture was rewired to mint a token directly: it took one from `POST /api/auth/demo`, which has been gated off since 2026-08-09 (c83c575b, "demo login is an open door"), so the fixture was asserting a security hole and raising in setup. `test_demo_login` now asserts the shipped behaviour — 404 while disabled — instead of the behaviour that PR removed. |

## Deleted because the WRITE PATH was replaced

| file | tests | why |
|---|---:|---|
| `test_extraction_fanout.py` | 21 | The `max_memories=15` fanout: extract → build 15 `MemoryCreate` → person-name lists → one batched dedup → entity upsert → relationship mirror. Every stage is gone. The two properties worth keeping moved: "a trivial turn costs no model call" is `test_memory_curator.py::test_a_trivial_turn_costs_no_model_call`, and "the connection is released before the LLM round trip" is now inside `curate_turn` (`await db.commit()` before the model call) with the same #407 comment. |
| `test_memory_write_path.py` | 9 | Round-8 file routing (`_same_task` variant matching, standing-arrangement leases, `smart_create` file routing, supersede inheriting `file_position`) plus three SOURCE probes that all three prompts speak SECOND PERSON. v3's voice is subjectless third person and is enforced by `bullet_problem`, pinned in `test_memory_files.py`. |
| `app/services/memory_file_service.py` + `app/services/memory_file_migration.py` (MODULES, not tests) | — | The round-8 row-routing pair, deleted together once WS-5 had read the legacy file assignments out of them. `memory_file_service` did category→slug routing for `memories` rows; `memory_file_migration` was its ONLY importer and had none of its own — a closed dead cycle, which is the shape an "is anything importing this" check finds and a "does anything mention this" grep does not. Both are now in `test_curator_producers.py`'s `DELETED_MODULES`. |
| `test_memory_file_migration.py` | 4 | Tested `normalize_working_leases` (three legacy lease shapes) and `_needs_curation`. The lease repair is deleted — it PROMOTED junk to permanent whenever the content matched a bare `every day|daily|routine` regex. Its one surviving assertion, the agent scheduler's cron-not-interval registration, is re-pinned in `test_agent_memory_maintenance.py`. |
| `test_memory_supersede_and_explicit_capture.py` | 40 | D-mem-A supersede lineage over rows, plus D-mem-C's explicit-remember predicate. The predicate itself (`is_explicit_remember_request`) still ships and still has a consumer (`query_intent`'s recall boost); what died is the extractor path that consumed it. Newest-wins is now a CONTENT property of the file body and is a positive fixture in the eval set (`P07`). |
| `test_dropped_span_promotion_gate.py` | 5 | The A8-6 promotion path still exists and still respects `disable_post_processing`; it now calls the curator. Re-pinned in `test_curator_producers.py::test_the_dropped_span_promotion_buckets_by_EXACT_role`, which additionally catches the role-bucketing defect this file never looked at (a `role=='tool'` message became the assistant half of a synthetic extraction). |
| `test_memory_extraction_retry.py` | 7 | `_complete_json_with_retry` MOVED to `memory_curator` with the model call it protected. Re-pinned in `test_memory_curator.py::test_one_transient_blip_is_retried_before_it_becomes_a_failure`. |

### The one property that outlived its mechanism

`test_destructive_verdict_confirmation.py` was mostly about the dedup
service's adjudication, but two of its tests were not:
`test_provider_failure_during_confirmation_keeps_both_facts` and
`test_confirmation_raising_keeps_both_facts` pinned **a failure inside the
write path must never silently lose a fact**.

Its successor is **`tests/test_curator_never_loses_a_fact.py`** (11 tests),
which is strictly broader because v3's writer runs fire-and-forget AFTER the
reply — there is no request to fail and no user watching. It covers the four
outcomes and separates the one that is a LOSS from the three that are not:

1. the model call RAISES → the turn is parked, a later turn replays it, and
   the fact lands;
2. the reply is UNPARSEABLE → handled identically (a malformed reply is not
   "the model decided nothing");
3. the validator REJECTS everything twice → a DECISION, not a loss: the turn
   is NOT parked (that would retry a refusal forever at one model call each)
   and the reasons are returned;
4. `apply_ops` raises MID-BATCH → nothing is half-written: no body without
   its change line, no half-created file, no orphan change row.

Plus the outbox's own liabilities: a poison turn is abandoned with its
reason recorded, a turn older than `MAX_AGE_HOURS` is dropped rather than
replayed with a stale relative date, the replay is capped at one per turn,
and a park on a poisoned session does not double-write.

## How the sixth writer stayed hidden through TWO verification passes

Worth writing down, because "the sweep is green" is what the next person
will trust, and it was wrong twice for two different reasons.

**1. The reporting filter swallowed mixed results.** My sweep harness piped
pytest's summary line through
`grep -E "failed|error" | grep -v "PydanticDeprecated\|passed,"`. A clean
file prints `299 passed, 917 warnings`; a FAILING file prints
`1 failed, 15 passed, 98 warnings`. Both contain `passed,`, so the exclusion
meant to hide clean files hid every **partially** failing one too — a file
had to be 100% red to appear. `test_ingestion_routes_to_tenant.py` was in
the sweep SET all along (verified: 299 files, target present); its failure
was filtered out of the report. Fixed by keying on pytest's **exit code**
instead of grepping its prose: `/tmp/sweep_rc.sh` in the round that found
this. Never derive a pass/fail from a summary line when the process already
told you.

**2. The sweep set was narrower than the tree.** It subtracted the files
named by a workflow step, on the theory that a named step already covers
them. It does — in CI. Locally that just means nobody runs them, and a red
one is invisible. The honest local set is everything not excused by
`COVERAGE_DEBT.txt`: **365 files**, not 299.

**3. The sweep used the wrong `DATABASE_URL`.** It ran
`sqlite+aiosqlite:///:memory:` while CI's job uses
`sqlite+aiosqlite:///file::memory:?cache=shared&uri=true`. With plain
`:memory:` every engine gets its own private database, so multi-engine
behaviour differs from CI's. `tests/test_sqlite_test_infra.py` exists to
catch precisely this and duly failed in both lanes — a red that was about my
harness, not the code. It is green under CI's URL.

None of the three is a memory bug. All three are reasons a green sweep
meant less than it looked like it did.

## The sixth writer, found after the fact

`app/api/ingest.py` survived the convergence pass. `POST /ingest/message`
and `POST /ingest/conversation` called the extractor's RULE-BASED half and
wrote through `MemoryService.create_memory` directly, with
`extract_memories` defaulting to True, on a router mounted by BOTH
entrypoints — and it was still minting `MemoryCategory.ACTIVE_TASK` rows,
the category whose entire surface v3 deleted.

It matched none of the retired module names, so nothing caught it. It cannot
feed a memory FILE, so it was never a correctness bug in the product; it was
a **truth bug in the invariant**, and the invariant is the headline of the
rebuild.

Both extraction blocks are severed and the message storage kept — the same
treatment REST `/chat` got. Callers checked first: `frontend/src/store.ts`'s
`ingestDemoMessage` (zero call sites) and `backend/app/scripts/demo_mode.py`
(an operator script). No mobile caller, no production client.

The lesson is in the guard, not just the fix.
`test_curator_producers.py` now walks the **AST** of every module under
`app/` for calls that can put a row in `memories` — `create_memory`,
`smart_create_memory*`, `extract_memories*`, `store_active_task`, and direct
`Memory(...)` construction — against an allowlist keyed
`path -> (count, reason)`. Three properties, each mutation-tested:

* a NEW write site anywhere fails (mutation: a `create_memory` call added
  under a comment saying extraction is severed — caught);
* a new write site inside an **already-allowlisted** file fails, because the
  count is pinned (mutation: a second `Memory(...)` in `memory_service.py`,
  which a file-level allowlist let through — caught only after tightening);
* the allowlist may not go stale (mutation: a listed file stops writing —
  caught).

AST rather than grep on purpose: this repo is now full of retirement
comments naming these very functions, and a text search drowns in its own
documentation. `ast` does not see comments or docstrings at all.

The four remaining sites are an audit, not an exemption:
`memory_service.py` (the one direct construction, inside `create_memory`),
`documents.py` (the surviving document/media leg, §3.4), `agent.py`
(`POST /api/agent/store` — an EXPLICIT-store API, so not a producer in the
sense the invariant means, but residue: its rows are unreachable from every
v3 surface and its only declared client has zero call sites), and
`seed_data.py` (a manual demo seeder with no route and no client).

`agent.py` is kept deliberately rather than severed: an explicit-store API is
the kind of surface someone integrates against without leaving a call site in
this repo, so its removal should be a change whose whole diff is its removal.
Two independent passes (WS-2 and the voice workstream) found nothing calling
it. **The trap for whoever deletes it:** `test_agent_recall` seeds itself by
POSTing to `/agent/store` (`test_api.py:202`), so the route and that test go
together — recall needs its own seeding first.

## The two READERS of the deleted route (found by the voice workstream)

Neither is a producer, so the write-site AST scan could not see them. Both
were live on every voice call.

**Four `GET /api/memories` reads in `ws_realtime.py`.** Two in the
warm-context `asyncio.gather`, two in onboarding finalize, each pair being
`brain_type=agent` + `brain_type=user`. The route is deleted, so all four
404'd, `_vps_api` folded each to `None`, and the `_missing` list logged
`agent_brain,user_brain` on every voice call forever — the exact "looks like
the user has nothing" failure that list was built in response to on
2026-07-31.

Deleted rather than re-pointed, and the reason is about the ROLLOUT: the
route and the data migrate together, per tenant. An agent on the old image
renders rows; one on the new image renders files. So any agent that answers
at all answers consistently with its own storage state, and fleet skew
mid-rollout stays self-consistent per tenant. A platform-side reader is the
one thing that breaks that — it would render whichever shape the PLATFORM
believes in against whatever the TENANT actually migrated to, which is a
guaranteed mismatch for the whole rollout window.

The residual case (agent reachable, `/internal/voice-context` failing for a
bug or a timeout) is not answered with a second reader. It is made
**countable**: `voice_ctx_hollow=1`, a stable grep token logged beside the
`_missing` block, so "how often are we serving hollow context" is answerable
from the trail without a repro. Same principle `_missing` was built on.

Onboarding finalize was rewired rather than deleted: the USER half reads
`you/profile` (one read of a route that exists — it compiles a profile
DOCUMENT, not a voice prompt), and the AGENT half — its name and personality,
which are SOUL and which v3 does not touch — arrives as `finalize_onboarding`
parameters instead of being recovered by searching agent-brain rows for the
substring `"my name is"`.

Also fixed while in there: `_instructions_step` built the legacy
instructions **unconditionally** and discarded them whenever the agent path
succeeded — the common case now that the flag is on. Built lazily; still
eager under `want_shadow`, because comparing is the shadow's whole job. No
latency figure is claimed: the argument is that the work is provably unused,
and nobody has measured the cost.

## Four prompts taught a tool signature that no longer exists

`memory_store(brain_type=…, category=…)` — in both copies of the voice
onboarding script (`voice_context.render_onboarding` and the relay's legacy
copy) and both text-chat scripts (`agent_runner`'s onboarding section and
ws_chat's `[SYSTEM: ONBOARDING]` line). v3's `memory_store` takes `content`
alone, so a brand-new user's first session had the model calling the tool
with parameters the schema rejects — in the ONE flow whose purpose is to
capture their name.

It presents as the user finishing onboarding with nothing stored, and the
agent forgetting them immediately after. Nobody traces that to a tool schema.

The agent-soul half was a separate bug hiding inside it. Those rows were
written and **deliberately never read**: `agent_runner.py:4735` filters
`agent_soul` out of the agent-brain injection, and `memory_taxonomy.py:183`
records that they are Soul-owned. Only the voice finalize path ever compiled
them, via a substring search. So text-chat onboarding never had a way to
persist the agent's name and still does not — `AgentConfig.agent_name` is
owned by `PUT /api/soul` and there is no agent-side tool. The text scripts
now say so instead of faking it; the voice path persists it properly through
`finalize_onboarding`. **Flagged as a product gap, not fixed here.**

Guarded by `test_curator_producers.py`: one test scans every string constant
under `app/` for the retired signature, and one PARSES the onboarding prose
for the tool calls it demonstrates and checks each parameter name against the
LIVE schema the model is handed. Mutation-tested both ways (putting
`category=` back, and routing the agent name through `memory_store`) — three
tests catch it.

## Rewritten in place, not deleted

* `tests/memverify/` — rebuilt for the file model. See `tests/memverify/README.md`
  for the per-category disposition (D/F/H/I/M lost their subject; A/B/G/J/K
  were re-expressed against file bodies).
* `test_decay_schedule_survives_restarts.py` — DELETED. It pinned that the
  `memory_decay` job used a CronTrigger rather than an IntervalTrigger,
  because an interval's first fire is measured from scheduler start and the
  fleet is recreated more often than that. The job is gone; the LESSON is
  re-pinned on the surviving slot by
  `test_agent_memory_maintenance.py::test_agent_main_registers_the_memory_jobs_behind_flag`,
  which asserts the cron trigger and that the retired ids are absent.
* `test_relationship_embed_position_and_secret_gate.py` — two of five tests
  retired inline (the double-embedding and the transaction-ordering
  properties, both of which existed only for the Memory MIRROR). The
  never-store screen and the graph-edge-survives test stay: the screen is
  reachable from a prompt via the MCP `entity_relationship_create` tool.
* `test_channel_formatting_rules.py` — its
  `run_retrieval_feedback_analysis`-by-name duplicate check became a general
  "no function in `scheduled_tasks.py` is defined twice", which is the class
  of defect the original found.
* `test_agent_runner_subagent_params.py` — was quarantined in
  COVERAGE_DEBT.txt as "fails for a reason not yet triaged", so it ran in
  NEITHER sweep. Triaged: two failures were stale source probes that predate
  memory v3 (it asserted the `asyncio.create_task(...)` spawn form while the
  runner uses `_spawn_background(...)` — not cosmetic, since a bare
  `create_task` keeps no strong reference and the GC can collect the whole
  post-processing block mid-flight; and it matched an import by TEXT, which
  a reflow to a parenthesised two-name form broke). The third,
  `test_existing_call_sites_do_not_pass_new_params`, WAS caused by v3: it
  asserted the dead "Phase 3 ships invisible" invariant, i.e. that nobody
  passes `disable_post_processing=True` — which is now precisely what every
  synthetic-prompt runner must do. Replaced by the live invariant: a turn
  whose `user_message` is machine-authored passes it, over an AST-derived,
  reason-carrying caller audit that fails if the set grows OR shrinks.
  17/17 green under `RUN_MODE=agent`, and moved out of quarantine into the
  agent-mode sweep so it is genuinely covered again.
* `test_ingestion_routes_to_tenant.py` — REWRITTEN, not retired, and the
  clearest case of the rule this file exists for. Its
  `test_control_message_ingest_still_works_and_is_retrievable` monkeypatched
  `ingest_mod.get_memory_extractor` with a stub and asserted
  `memories_extracted == 1` plus a `Memory` row containing "Falcon". Half its
  subject died with the sixth writer; the other half — a turn goes in, the
  conversation and both messages are stored, the tenant was not called — is
  exactly what a control is for and still ships. So the control half is kept
  verbatim and the memory half now asserts the new truth: zero rows,
  **parametrized over `extract_memories` True AND False**, because the
  request field is still in the schema and a test that only passed `False`
  would read "the caller asked for nothing and got nothing" while extraction
  quietly came back. The stub and the monkeypatch went with the symbol.
  A sibling control was ADDED at the same time:
  `/ingest/conversation` had only a proxy test, and v3 restructured that
  handler's message loop when the extraction block came out — three messages
  in, three rows out, zero memory, with an odd count so a pairing-loop
  regression cannot hide. Both mutation-tested (a returning writer is caught
  by six tests across three files; a dropped trailing message by the new
  control).
* `test_memory_retrieval_filters.py::test_ingest_applies_the_full_gate_not_just_the_backstop`
  → `test_the_ingest_surface_extracts_nothing_at_all`. It pinned that both
  `/ingest` extraction sites called `memory_gate_reason` BEFORE
  `create_memory`, checking the order — right while those sites existed. The
  replacement is a strictly stronger statement ("there are no extraction
  sites") that no later reordering can satisfy, asserted on the AST because
  the module now carries a long comment explaining what was severed and a
  grep would match its own explanation. It keeps an anti-vacuity clause: the
  two handlers must still exist, or deleting the module would pass.
* `test_memory_gate_regressions.py`, `test_memory_junk_gate.py`,
  `test_memory_taxonomy_and_ttl.py`, `test_memory_gate_cross_script.py`,
  `test_subagent_context_isolation.py`, `test_background_connection_leak.py`,
  `test_memory_read_path_hardening.py`, `test_memory_retrieval_filters.py`,
  `test_memory_upsert_and_enum.py`, `test_mcp_memory_search_tenant.py`,
  `test_agent_memory_maintenance.py`, `test_hybrid_search_honours_limit.py` —
  each keeps the half whose subject survives and drops the half whose subject
  does not; the drops are annotated in the file.

## New

* `test_curator_producers.py` — the structural guard that every producer
  converges on the curator or is severed, and that nothing imports a retired
  service.
* `test_memory_curator.py` — extended with the turn curator: the two-slot
  prompt labelling, the durability rules naming the dispatch's own bad
  memories, the pre-gate table (skip AND must-not-skip), the transient retry,
  and today-in-the-user's-timezone.
