"""Every producer converges on the curator, or is severed (v3 §2.1).

Round 8 had five independent writers, each with its own idea of the gate,
and the gate could only ever be as strong as the weakest one. This file is
the structural guard that there is now exactly one — and that the two
mistakes which made the old fanout dangerous cannot come back:

1. **The writer is handed the CLEAN text.** ws_chat rewrites `user_message`
   (the fast-media `[SYSTEM: The track "…"]` line with a scraped YouTube
   title, Chrome page context, a reply quote) and passes the clean copy as
   `display_user_message`. Round 8 gave persistence the clean one and the
   extractor the dirty one, and every provenance rule downstream measured
   overlap against that dirty string — so the injection disarmed all three
   at once. That is root cause #1 of the whole rebuild.

2. **A machine-authored prompt is not a user turn.** Three synthetic
   runners omitted `disable_post_processing`, so "[Scheduled task: Gmail
   briefing] … max_results=1" was mined as something the user said, every
   day on a schedule.

These are SOURCE probes on purpose. Both defects are about which ARGUMENT
reaches a call, on paths that are fire-and-forget after the reply — there
is no return value to assert on, and a behavioural test would have to stand
up the whole runner to observe an argument it could read directly.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

def code_of(path: pathlib.Path) -> str:
    """Source with COMMENTS AND DOCSTRINGS REMOVED.

    Every probe below asks "does this still RUN?", and this file explains at
    length what was deleted and why — naming `_extract_voice_memories`,
    `memory_create` and friends in prose. A probe that greps raw source
    fails on its own explanation, and the fix people reach for is to delete
    the explanation. So the probes read code only.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            body.pop(0)
            if not body:
                body.append(ast.Pass())
    return ast.unparse(ast.fix_missing_locations(tree))


BACKEND = pathlib.Path(__file__).resolve().parent.parent
RUNNER = BACKEND / "app/agent/agent_runner.py"
WS_REALTIME = BACKEND / "app/api/ws_realtime.py"
API_V1 = BACKEND / "app/api/api_v1.py"
TOOL_EXEC = BACKEND / "app/agent/tool_executor.py"
TOOL_DEFS = BACKEND / "app/agent/tool_definitions.py"
REFLECTION = BACKEND / "app/services/agent_reflection.py"
MCP = BACKEND / "app/mcp_server.py"


# ── 1. The chat producer ──────────────────────────────────────────────

def test_the_runner_calls_the_curator_and_nothing_else():
    src = code_of(RUNNER)
    assert "memory_curator.curate_turn(" in src
    # The retired fanout, by name. Each of these was a writer.
    for gone in (
        "extract_memories_with_llm",
        "detect_active_tasks",
        "store_active_task",
        "decay_expired_tasks",
        "expire_stale_memories",
        "smart_create_memories",
        "MemoryDedupService",
        "UserPortraitService",
        "extract_relationships_with_llm",
    ):
        assert gone not in src, f"{gone} still runs in the runner"


def test_the_curator_is_handed_display_user_message_not_the_rewritten_one():
    """Root cause #1. Asserted on the AST, so a reflow cannot fake it."""
    tree = ast.parse(RUNNER.read_text())
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "curate_turn"
    ]
    assert calls, "no curate_turn call in agent_runner"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "user_text" in kw, "curate_turn must be called by keyword"
        source = ast.unparse(kw["user_text"])
        assert "user_message" not in source or "display_user_message" in source, (
            "curate_turn was handed the rewritten `user_message`; ws_chat "
            "puts a scraped YouTube title in that string"
        )


def test_the_clean_text_is_bound_from_display_user_message():
    src = RUNNER.read_text()
    assert "_curator_user_text = display_user_message or user_message" in src, (
        "the fallback must be display_user_message FIRST — a caller that "
        "does not rewrite passes only user_message, and that is fine, but "
        "the order decides which one wins when both are present"
    )


def test_the_failed_turn_is_parked_not_dropped():
    src = RUNNER.read_text()
    assert "record_turn_failure(" in src
    assert "replay_pending(" in src


def test_the_dropped_span_promotion_buckets_by_EXACT_role():
    """A role=='tool' message is a RAW TOOL RESULT.

    Round 8 bucketed with `user if role=='user' else assistant`, so tool
    output became the assistant half of a synthetic extraction. The v3
    prompt tells the model the user block is its only source of facts, so a
    tool result reaching either block is a lie to the model.
    """
    src = RUNNER.read_text()
    assert '(user_parts if _dm.get("role") == "user" else asst_parts)' not in src
    assert 'elif _role == "assistant":' in src


# ── 2. Synthetic runners ──────────────────────────────────────────────

@pytest.mark.parametrize("path,marker", [
    ("app/agent/cron_service.py", "[Scheduled task:"),
    ("app/agent/heartbeat_service.py", "[Heartbeat]"),
    ("app/agent/subagent.py", "[Background Task:"),
])
def test_a_machine_authored_prompt_disables_post_processing(path, marker):
    src = (BACKEND / path).read_text()
    assert marker in src, f"{path} no longer builds {marker!r} — re-point this test"
    assert "disable_post_processing=True" in src, (
        f"{path} mines its OWN prompt as a user turn — this is where the "
        "routine-prompt rows in the founder's brain came from"
    )


# ── 3. Voice ──────────────────────────────────────────────────────────

def test_the_voice_tunnel_is_severed():
    src = code_of(WS_REALTIME)
    assert "_extract_voice_memories" not in src
    assert "get_memory_extractor" not in src, (
        "the platform must not run an extractor of its own — the curator "
        "runs agent-side, where the memory lives"
    )
    assert 'send_tool_call(user_id, "memory_store"' not in src
    assert "_curate_voice_turn" in src


def test_the_voice_seam_is_a_real_agent_route():
    """The relay is platform-side and the writer is agent-side, so the seam
    has to be a hop. It cannot be `/internal/agent-turn`: that is called with
    save=False (→ disable_post_processing) precisely because `think`'s task
    string is the realtime model's synthesis, not the user's words."""
    relay = WS_REALTIME.read_text()
    assert "/api/v1/internal/curate-turn" in relay
    route = API_V1.read_text()
    assert '@router.post("/internal/curate-turn"' in route
    assert "memory_curator.curate_turn(" in route
    # Same auth and same run-mode guard as its sibling /internal/voice-context.
    at = route.index('@router.post("/internal/curate-turn"')
    body = route[at: at + 3000]
    assert 'settings.run_mode != "agent"' in body
    assert 'request.headers.get("X-Agent-Key"' in body


def test_the_voice_write_gets_a_writer_sized_timeout():
    """`_vps_api`'s default is 15 s, tuned for the identity/memory READS. A
    curator call is an LLM round trip on the agent; at 15 s it times out
    mid-write and the turn is silently lost."""
    relay = WS_REALTIME.read_text()
    at = relay.index("/api/v1/internal/curate-turn")
    assert "timeout=60.0" in relay[at: at + 800]


# ── 4. The model-callable tools ───────────────────────────────────────

def test_memory_store_routes_through_the_curator():
    src = code_of(TOOL_EXEC)
    at = src.index("async def _tool_memory_store")
    body = src[at: src.index("async def _tool_memory_delete")]
    assert "memory_curator.instruct_global(" in body
    assert "explicit_save" not in body, (
        "`explicit_save=True` was not a hint — it turned OFF three gate "
        "rules, and this tool passed it unconditionally"
    )
    assert "MemoryDedupService" not in body


def test_memory_delete_forgets_by_DESCRIPTION_not_by_id():
    """There are no memory ids in the product any more. The round-8 schema
    told the model to fetch one from memory_search, which now returns file
    slugs and snippets."""
    src = code_of(TOOL_EXEC)
    at = src.index("async def _tool_memory_delete")
    body = src[at: at + 2000]
    assert "memory_curator.instruct_global(" in body
    assert "delete_memory(" not in body

    defs = TOOL_DEFS.read_text()
    at = defs.index('"name": "memory_delete"')
    schema = defs[at: at + 1200]
    assert '"required": ["content"]' in schema
    assert "memory_id" not in schema


def test_memory_store_no_longer_asks_the_model_for_a_category():
    """v3 has no per-row taxonomy; the curator picks the FILE. A schema that
    still asked for a category was asking the model to choose from an enum
    whose values route nothing."""
    defs = TOOL_DEFS.read_text()
    at = defs.index('"name": "memory_store"')
    schema = defs[at: defs.index('"name": "memory_delete"')]
    assert '"required": ["content"]' in schema
    for gone in ('"category"', '"brain_type"', '"importance"'):
        assert gone not in schema, f"memory_store still takes {gone}"


# ── 5. agent_reflection ───────────────────────────────────────────────

def test_reflection_writes_the_learned_FILE_through_the_curator():
    src = code_of(REFLECTION)
    assert "memory_curator.instruct_file(" in src
    assert "LEARNED_SLUG" in src
    assert "MemoryDedupService" not in src
    assert "MemoryCreate" not in src
    # The cheap gate is the point of the module and must survive: a
    # reflection call on every turn is a third LLM call on every turn.
    assert "def should_reflect(" in src


def test_reflection_makes_ONE_call_for_the_turns_notes():
    """Three separate instructs would let the second and third dedupe
    against a file the first had already changed — the ordering hazard the
    ops engine's single-walk simulation exists to remove."""
    src = code_of(REFLECTION)
    at = src.index("async def store_agent_reflections")
    body = src[at: src.index("async def _resolve_tenant_api_key")]
    assert body.count("instruct_file(") == 1
    assert "for n in notes" in body or "for n in notes)" in body


def test_automation_facts_project_only_through_the_curator():
    """R29: `automations/facts.py` owns the fact TABLE; its brain half
    is a projection through the sanctioned entries only — the curator
    (instruct_file / instruct_global / memory_notes) plus ONE
    deterministic create_file walk for the fixed topic files. Never a
    raw row write, never `disable_post_processing`."""
    src = code_of(BACKEND / "app/agent/automations/facts.py")
    assert "memory_curator.instruct_global(" in src
    assert "memory_curator.instruct_file(" in src
    assert "record_automation_fact(" in src
    # The only direct ops use is the deterministic topic-file create —
    # the validate→apply walk every writer uses, no LLM.
    assert src.count("ops.apply_ops(") == 1
    assert "'op': 'create_file'" in src  # code_of unparses to single quotes
    # Severed paths: no row-store writes, no post-processing bypass.
    assert "Memory(" not in src
    assert "create_memory" not in src
    assert "disable_post_processing" not in src


# ── 6. MCP ────────────────────────────────────────────────────────────

def test_no_mcp_tool_proxies_a_deleted_route():
    src = code_of(MCP)
    for gone in (
        "MemorySearchRequest",     # the row search schema
        "memory_create",           # wrote rows, and told the model to store
        "memory_update",           # routine/trigger memories by ref_kind
        "memory_list",
        "ref_kind",
        "_proxy_memory_write_to_tenant",
    ):
        assert gone not in src, f"mcp_server still references {gone}"
    # What replaced them.
    assert "async def memory_search(" in src
    assert "async def memory_files(" in src
    assert "async def memory_remember(" in src
    # The write proxies to the v3 global-instruct route, not to a row route.
    assert '\'instruct\'' in src or '"instruct"' in src


def test_the_mcp_write_never_falls_back_to_the_platform_session():
    """`memory_files` is AGENT_ONLY. "Succeeding" against the platform DB is
    how a user comes to believe something was saved that was not."""
    src = MCP.read_text()
    at = src.index("async def memory_remember(")
    body = src[at: src.index("# ── Session Tools")]
    assert "isn't reachable" in body
    assert "MemoryService(" not in body


# ── 7. The relationship mirror ────────────────────────────────────────

def test_the_relationship_mirror_row_is_gone_but_the_graph_edge_is_not():
    src = code_of(BACKEND / "app/services/memory_service.py")
    at = src.index("async def store_entity_relationship")
    body = src[at: src.index("async def _upsert_entity")]
    # 59 of the 73 junk rows on the founder's tenant came through here.
    assert "rel_memory" not in body
    assert "EntityLink(" not in body
    assert 'source_type="entity_extraction"' not in body
    # The graph survives: MCP entity_search / graph_traverse and
    # app/api/graph.py read it.
    assert "EntityRelationship(" in body
    assert "_upsert_entity(user_id, source_name" in body
    # The never-store screen survives and still aborts the WHOLE write:
    # `entities.name` stores the string verbatim.
    assert "sensitive_content_reason(" in body


# ── 8. The REST chat legacy ───────────────────────────────────────────

@pytest.mark.parametrize("path", ["app/api/chat.py", "app/modules/chat/router.py"])
def test_the_legacy_rest_chat_writes_no_memory(path):
    src = code_of(BACKEND / path)
    assert "extract_memories_with_llm" not in src
    assert "MemoryDedupService" not in src


# ── 9. Nothing imports a deleted module ───────────────────────────────

DELETED_MODULES = (
    "memory_dedup_service",
    "decay_service",
    "consolidation_service",
    "active_task_service",
    "memory_expiry",
    "user_portrait_service",
    # The round-8 ROW-routing pair, retired together once WS-5 had read the
    # legacy file assignments out of them. `memory_file_service` did
    # category→slug routing for `memories` rows; `memory_file_migration` was
    # its only importer and had none of its own — a closed dead cycle, which
    # is precisely the shape a "does anything import it" check finds and a
    # "does anything reference it" grep does not.
    "memory_file_service",
    "memory_file_migration",
)


def test_no_module_under_app_imports_a_retired_service():
    offenders = []
    for path in (BACKEND / "app").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            elif isinstance(node, ast.Import):
                mod = " ".join(a.name for a in node.names)
            if mod and any(d in mod for d in DELETED_MODULES):
                offenders.append(f"{path.relative_to(BACKEND)}:{node.lineno} {mod}")
    assert not offenders, "imports of retired services: " + "; ".join(offenders)


def test_the_retired_service_files_are_actually_gone():
    for name in DELETED_MODULES:
        assert not (BACKEND / f"app/services/{name}.py").exists(), name


def test_the_never_store_tier_survives_the_gate_being_cut():
    """§2.3 keeps exactly one rule from the row-era gate, and it lives in a
    v3-owned module so that cutting `memory_gate` cannot strand it."""
    from app.services.memory_secrets import sensitive_content_reason

    assert sensitive_content_reason("my card is 4111 1111 1111 1111")
    # Self-documenting on purpose. The PREFIX has to be real (`sk-ant-` is
    # what the pattern keys on), so the body says what it is instead — a
    # reader of this line, or of secret_scan_allowlist.txt, can tell a
    # fixture from a real key that somebody waved through.
    assert sensitive_content_reason("token sk-ant-EXAMPLE-NOT-A-REAL-CREDENTIAL")
    # The discriminating POSITIVE: a medication is a durable health fact.
    assert sensitive_content_reason("takes metformin 500mg twice daily") is None

    ops_src = (BACKEND / "app/services/memory_file_ops.py").read_text()
    assert "from app.services.memory_secrets import sensitive_content_reason" in ops_src
    cur_src = (BACKEND / "app/services/memory_curator.py").read_text()
    assert "from app.services.memory_secrets import sensitive_content_reason" in cur_src


# ── 10. THE INVARIANT ITSELF, over the whole of app/ ──────────────────
#
# Everything above names a producer it already knows about. That is how a
# SIXTH writer survived the convergence pass: `app/api/ingest.py` called the
# extractor's RULE-BASED half and wrote through `MemoryService.create_memory`
# directly, so it matched none of the retired module names the grep looked
# for — and it was still minting `MemoryCategory.ACTIVE_TASK` rows, the
# category whose entire surface v3 deleted, on a route mounted by BOTH
# entrypoints with `extract_memories` defaulting to True.
#
# It cannot feed a memory FILE, so it was never a correctness bug in the
# product. It was a truth bug in the claim, and the claim is the headline of
# the rebuild.
#
# So this walks the AST of every module under `app/` and finds every call
# that can put a ROW in `memories` or that runs an extractor, whatever it is
# imported as. AST, not grep: this repo is now full of retirement comments
# naming these very functions, and a text search drowns in its own
# documentation. `ast` does not see comments or docstrings at all.
#
# The allowlist is an AUDIT, not an escape hatch: each entry states why that
# call site is legitimate, and the test fails if the set GROWS **or
# SHRINKS**, so a residue cannot quietly widen and cannot be silently
# forgotten once it is cut.

#: Calls that can create a `memories` row, or that mine text for one.
_WRITE_CALLS = frozenset({
    "create_memory",
    "smart_create_memory",
    "smart_create_memories",
    "extract_memories",
    "extract_memories_with_llm",
    "extract_relationships_with_llm",
    "store_active_task",
})

#: Every legitimate site: `path -> (how many, why)`.
#:
#: The COUNT is the load-bearing half. An allowlist keyed on the file alone
#: exempts the file, so a NEW write site inside an already-listed module —
#: which is precisely how the relationship mirror lived next to
#: `create_memory` for months — passes unnoticed. Mutation-tested: adding a
#: second `Memory(...)` to memory_service.py survives a file-level allowlist
#: and fails this one.
_ALLOWED_WRITE_SITES = {
    "app/services/memory_service.py": (1,
        "Holds the ONE direct `Memory(...)` construction in the tree, and it "
        "is inside `create_memory` itself. The relationship mirror used to "
        "be a second one — a row INSERTed straight onto the session, past "
        "every gate — which is why direct construction is scanned at all."
    ),
    "app/api/documents.py": (2,
        "The ONE surviving row reader/writer (v3 §3.4): uploads and "
        "transcripts keep their embedding pipeline and are reachable only "
        "through memory_search's document leg. They appear in no memory UI."
    ),
    "app/api/agent.py": (1,
        "POST /api/agent/store — an EXPLICIT-store API: the caller names the "
        "rows, nothing is inferred from a conversation, so it is not a "
        "producer in the sense this invariant is about. It is nonetheless "
        "residue: the rows it writes are unreachable from every v3 surface. "
        "Confirmed unreferenced twice (WS-2 and the voice workstream): one "
        "route at app/api/agent.py:27, one dead frontend wrapper at "
        "frontend/src/api.ts (`agent.store`, zero call sites), and two "
        "tests. Kept rather than severed on purpose — an explicit-store API "
        "is the kind of surface someone integrates against WITHOUT leaving a "
        "call site in this repo, so its removal should be a change whose "
        "whole diff is its removal, not a side effect of a memory sweep. "
        "WHEN YOU DO DELETE IT: `test_agent_recall` seeds itself by POSTing "
        "to /agent/store (tests/test_api.py:202), so the two go together — "
        "give recall its own seeding FIRST, or you will find this out "
        "halfway through the delete."
    ),
    "app/scripts/seed_data.py": (2,
        "A manual demo seeder for the old 3D-brain demo (`demo@toup.local`). "
        "No route, no client, run by hand. Severing its extraction would "
        "leave a script that pretends to seed."
    ),
    "app/agent/automations/memory.py": (1,
        "The automations engine's working-state row (R28 contract §6): ONE "
        "row per automation, ref_kind='automation', upserted after each "
        "terminal run so templates can read {{memory.<key>}} at fire time. "
        "It is the current_context.py precedent applied to a row — direct, "
        "deterministic, no curator, no LLM — and INVISIBLE to the brain by "
        "construction: not a memory file, unreachable from load_brain / "
        "search_files / the Memory UI; only the engine and GET "
        "/api/automations/{id}/memory read it, and deleting the automation "
        "deletes it. The invariant this audit protects — conversational "
        "facts reach the brain only through the curator — is untouched: an "
        "automation's BRAIN facts go through memory_curator.instruct_file "
        "(automations/memory_notes.py), never through this row."
    ),
}


def _row_write_sites() -> dict:
    """{relative path: [(lineno, call name)]} for every row-write call."""
    found: dict = {}
    for path in sorted((BACKEND / "app").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover
            continue
        rel = str(path.relative_to(BACKEND))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                continue
            if name in _WRITE_CALLS:
                found.setdefault(rel, []).append((node.lineno, name))
            # Constructing the model directly bypasses `create_memory` and
            # every gate with it. That is exactly what the relationship
            # mirror did: `self.db.add(Memory(...))`, 59 of the 73 junk rows
            # on the founder's tenant. Scanning only for the FUNCTION would
            # miss its return.
            elif isinstance(func, ast.Name) and func.id == "Memory":
                found.setdefault(rel, []).append((node.lineno, "Memory(...)"))
    return found


def test_no_module_outside_the_audited_set_writes_a_memory_row():
    """The invariant, stated over the code rather than over a list of names."""
    sites = _row_write_sites()
    unexpected = {
        rel: calls for rel, calls in sites.items()
        if rel not in _ALLOWED_WRITE_SITES
    }
    assert not unexpected, (
        "a module writes `memories` rows outside the audited set — v3 has ONE "
        "writer, `memory_curator`, and it writes FILES:\n"
        + "\n".join(f"  {rel}: {calls}" for rel, calls in sorted(unexpected.items()))
        + "\n\nIf the site is legitimate, add it to _ALLOWED_WRITE_SITES with "
        "the reason. If it is not, sever it the way app/api/ingest.py was."
    )

    # …and no ALLOWLISTED module grew a new one.
    grew = {
        rel: (len(calls), _ALLOWED_WRITE_SITES[rel][0], calls)
        for rel, calls in sites.items()
        if rel in _ALLOWED_WRITE_SITES and len(calls) != _ALLOWED_WRITE_SITES[rel][0]
    }
    assert not grew, (
        "an audited module's write-site COUNT changed — a file-level "
        "exemption is how a second writer lives next to a legitimate one:\n"
        + "\n".join(
            f"  {rel}: found {got}, audited {want} -> {calls}"
            for rel, (got, want, calls) in sorted(grew.items())
        )
    )


def test_the_allowlist_has_not_gone_stale():
    """It must SHRINK when a residue is cut, or the audit rots into a list of
    files that used to matter."""
    sites = _row_write_sites()
    stale = sorted(set(_ALLOWED_WRITE_SITES) - set(sites))
    assert not stale, (
        f"{stale} no longer write memory rows — remove them from "
        "_ALLOWED_WRITE_SITES so the exception list stays an audit"
    )


def test_every_allowed_site_states_a_reason():
    for rel, (count, reason) in _ALLOWED_WRITE_SITES.items():
        assert count >= 1, f"{rel}: an audited site with no sites"
        assert len(reason) > 40, f"{rel}: an exception without a reason is a hole"


def test_the_ingest_routes_write_no_memory():
    """The sixth writer, specifically.

    Both routes still STORE the conversation — that half is what an external
    ingestion client is importing, and severing it would break a working
    feature to fix a different one. What must be gone is the extraction.
    """
    src = code_of(BACKEND / "app/api/ingest.py")
    for gone in (
        "get_memory_extractor",
        "extract_memories(",
        "MemoryService",
        "create_memory",
        "memory_gate_reason",
        "_get_or_create_entity",
        "EntityLink",
    ):
        assert gone not in src, f"app/api/ingest.py still references {gone}"

    # …and the storage half is intact. Without this, deleting the whole
    # module would pass the assertions above.
    for kept in ("Conversation(", "Message(", "embed_to_json", "db.commit()"):
        assert kept in src, f"ingest lost its message storage: {kept} is gone"


def test_the_ingest_response_shape_is_unchanged_and_honest():
    """An existing client must not break, and must not be told about rows
    that were not written."""
    from app.schemas import IngestResponse

    fields = set(IngestResponse.model_fields)
    assert {"conversation_id", "messages_ingested", "memories_extracted",
            "entities_extracted", "memories"} <= fields

    src = code_of(BACKEND / "app/api/ingest.py")
    assert src.count("memories_extracted=0") == 2, (
        "both routes must report 0 memories extracted, honestly"
    )
    assert src.count("entities_extracted=0") == 2


def test_the_chat_routes_keep_no_dead_extractor_binding():
    """`memory_extractor = get_memory_extractor()` outlived the extraction it
    fed — an unused local that kept a half-retired module imported on a live
    request path."""
    for rel in ("app/api/chat.py", "app/modules/chat/router.py"):
        src = code_of(BACKEND / rel)
        assert "get_memory_extractor" not in src, f"{rel} still binds the extractor"


# ── 11. No prompt may teach a tool signature that no longer exists ────
#
# [d] of the integration round. The voice ONBOARDING script instructed the
# model, verbatim:
#
#     memory_store(brain_type='user', category='identity', content='…')
#     memory_store(brain_type='agent', category='agent_soul', content='…')
#
# v3's `memory_store` takes `content` alone. So a brand-new user's very first
# session — the ONE flow where the agent is explicitly told to store things —
# had the model calling the tool with parameters the schema rejects. Four
# prompts carried it: both copies of the voice script (agent-side
# `voice_context.render_onboarding` and the relay's legacy copy) and both
# text-chat scripts (`agent_runner`'s onboarding section and ws_chat's
# `[SYSTEM: ONBOARDING]` line).
#
# A prompt is not type-checked, so nothing failed loudly — the model just got
# a rejection at run time, in front of a first-time user. This asserts on the
# STRING CONSTANTS in the AST rather than on the source text, for the same
# reason the write-site scan does: the modules now carry comments quoting the
# old signature to explain what was removed.

def _prompt_strings(path: pathlib.Path):
    """Every string literal in a module, with its line number."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:  # pragma: no cover
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


#: Docstrings that legitimately quote the retired signature to explain it.
_EXPLAINS_THE_RETIREMENT = ("v3 REWIRING", "learned` file's producer")


def test_no_prompt_teaches_the_retired_memory_store_signature():
    stale = []
    for path in sorted((BACKEND / "app").rglob("*.py")):
        for lineno, text in _prompt_strings(path):
            if any(marker in text for marker in _EXPLAINS_THE_RETIREMENT):
                continue
            if re.search(r"brain_type\s*=|category\s*=\s*['\"]", text):
                stale.append(f"{path.relative_to(BACKEND)}:{lineno}")
    assert not stale, (
        "a prompt still teaches the pre-v3 memory_store signature "
        "(brain_type / category) — the tool takes `content` alone and will "
        "reject the call: " + ", ".join(stale)
    )


def test_the_onboarding_scripts_teach_the_v3_tool_surface():
    """Anti-vacuity for the test above: deleting the scripts would pass it."""
    from app.agent.voice_context import render_onboarding

    voice = render_onboarding()
    assert "memory_store(content=" in voice, (
        "the voice onboarding script no longer shows the one-argument form"
    )
    assert "finalize_onboarding(agent_name=" in voice, (
        "the agent's own name is SOUL, not memory — it must travel to "
        "finalize_onboarding, which is the only thing that persists it"
    )
    # Both text-chat scripts still run onboarding, and both still say how.
    for rel in ("app/agent/agent_runner.py", "app/api/ws_chat.py"):
        joined = " ".join(t for _, t in _prompt_strings(BACKEND / rel))
        assert "memory_store(content=" in joined, f"{rel} lost its store guidance"


def test_finalize_onboarding_takes_the_agents_identity_as_parameters():
    """It used to recover the agent's name by searching agent-brain rows for
    the substring "my name is" — a string search standing in for a contract,
    over a route v3 deleted. Now it is an argument, and a missing one is
    visible instead of silently yielding "Agent Soul"."""
    import inspect

    from app.api import ws_realtime

    params = inspect.signature(ws_realtime._finalize_onboarding).parameters
    assert {"agent_name", "personality"} <= set(params)

    # code_of() strips docstrings and comments — this function's own
    # docstring explains what was removed and names `brain_type` doing it.
    module = code_of(BACKEND / "app/api/ws_realtime.py")
    body = module[module.index("async def _finalize_onboarding"):]
    body = body[:body.index("\nasync def ") if "\nasync def " in body else len(body)]
    assert '"my name is" in content.lower()' not in body, (
        "the agent's name is still recovered by substring-searching rows"
    )
    assert "_MEMORIES_MAX_LIMIT" not in body, (
        "onboarding still asks for a row-list limit"
    )


def test_the_legacy_voice_builder_reads_no_deleted_route():
    """[c]: four `GET /api/memories` reads — two in the warm-context gather,
    two in onboarding finalize. The route is deleted, so each returned 404,
    `_vps_api` folded it to None, and the `_missing` list logged
    "agent_brain, user_brain" on every voice call forever — which is exactly
    the "looks like the user has nothing" failure that list exists to
    prevent."""
    src = code_of(BACKEND / "app/api/ws_realtime.py")
    assert '"/api/memories"' not in src, (
        "ws_realtime still reads the deleted row-list route"
    )
    # `brain_type` as a QUERY PARAMETER, not as a word. The new onboarding
    # guidance says "never pass brain_type or category", so a substring test
    # over the module would trip on the fix itself.
    tree = ast.parse((BACKEND / "app/api/ws_realtime.py").read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "params" and isinstance(kw.value, ast.Dict):
                keys = [
                    k.value for k in kw.value.keys
                    if isinstance(k, ast.Constant)
                ]
                if "brain_type" in keys:
                    offenders.append(f"line {node.lineno}")
    assert not offenders, (
        "a request still sends brain_type — that is the row API: " + ", ".join(offenders)
    )
    # The fallback must still say what it is — a degraded prompt announcing
    # itself is the whole point of the 2026-07-31 scar tissue.
    assert "NO MEMORY by construction" in (BACKEND / "app/api/ws_realtime.py").read_text()


# ── 12. The onboarding PROSE is checked against the live tool schema ──
#
# Nothing type-checks a prompt. That is why `voice_context.render_onboarding`
# kept instructing `memory_store(brain_type='user', category='identity', …)`
# through a whole rebuild that removed both parameters — the model simply got
# a rejection at run time.
#
# Why this matters more than a rejected call: it is the FIRST voice session a
# new user ever has, and the flow's entire purpose is to capture their name
# and their agent's name. The failure presents as the user finishing
# onboarding with nothing stored, and then the agent forgetting them
# immediately afterwards. Nobody traces that back to a tool schema.
#
# So the prose is parsed and its parameter names are checked against the
# SCHEMA the model is actually handed. Mutation-tested: putting `category=`
# back into the script fails this.

_TOOL_CALL_RE = re.compile(r"\b(memory_store|finalize_onboarding)\(([^)]*)\)")
_KWARG_RE = re.compile(r"(\w+)\s*=")


def _tool_calls_taught_by(script: str):
    """[(tool, {param names}), …] for every call the prose demonstrates."""
    out = []
    for tool, args in _TOOL_CALL_RE.findall(script):
        out.append((tool, set(_KWARG_RE.findall(args))))
    return out


def _live_schema(tool_name: str) -> set:
    """The parameter names the model is actually offered."""
    from app.agent.tool_definitions import get_agent_tools

    for t in get_agent_tools():
        if t["name"] == tool_name:
            return set(t["input_schema"]["properties"])
    # The onboarding-only tools are relay-local, declared in ws_realtime.
    tree = ast.parse((BACKEND / "app/api/ws_realtime.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
        if not {"name", "parameters"} <= keys:
            continue
        entry = {
            k.value: v for k, v in zip(node.keys, node.values)
            if isinstance(k, ast.Constant)
        }
        name_node = entry.get("name")
        if not (isinstance(name_node, ast.Constant) and name_node.value == tool_name):
            continue
        params = entry["parameters"]
        for pk, pv in zip(params.keys, params.values):
            if isinstance(pk, ast.Constant) and pk.value == "properties":
                return {
                    k.value for k in pv.keys if isinstance(k, ast.Constant)
                }
    raise AssertionError(f"no schema found for {tool_name!r}")


def test_the_onboarding_script_only_teaches_parameters_that_exist():
    from app.agent.voice_context import render_onboarding

    taught = _tool_calls_taught_by(render_onboarding())
    assert taught, "the onboarding script demonstrates no tool calls at all"

    seen_tools = set()
    for tool, params in taught:
        seen_tools.add(tool)
        live = _live_schema(tool)
        unknown = params - live
        assert not unknown, (
            f"the onboarding script tells the model to call {tool} with "
            f"{sorted(unknown)}, which the live schema does not accept "
            f"(it takes {sorted(live)}). A new user's first session would "
            "get a rejected tool call in the one flow whose job is to store "
            "their name."
        )
    # Both halves must actually be demonstrated — a script that taught
    # neither would pass every assertion above.
    assert seen_tools == {"memory_store", "finalize_onboarding"}, seen_tools


def test_the_agent_soul_half_does_not_go_through_the_users_memory():
    """The two halves are different corpora and must not be conflated.

    The USER's facts are memory (`memory_store` → the curator → you/profile).
    The AGENT's own name and personality are SOUL, which v3 does not touch —
    `AgentConfig.agent_name` is owned by `PUT /api/soul`, and there is no
    agent-side tool for it. So onboarding carries them as
    `finalize_onboarding` parameters. Routing them through memory_store would
    put the agent's identity in the user's biography.
    """
    from app.agent.voice_context import render_onboarding

    script = render_onboarding()
    for tool, params in _tool_calls_taught_by(script):
        if tool == "memory_store":
            assert not (params & {"brain_type", "category"}), params
    assert {"agent_name", "personality"} <= _live_schema("finalize_onboarding")
    # And the script says so in as many words, because a model reads prose.
    assert "not facts about them" in script or "your identity" in script
