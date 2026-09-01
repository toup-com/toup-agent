"""Template catalog (Round 28) — every template proven against the
REAL manifest-built registry, the flagship pinned, and the boot sync's
upsert semantics (admin `enabled` survives).

PLATFORM lane. The validation tests are pure; the sync tests use the
shared test DB via async_session_maker.
"""

import json
import re

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.agent.automations.narrator import BRIEF_GROUPS
from app.agent.automations.spec import SpecError, validate_spec
from app.agent.automations.spec_v2 import ValidatedSpecV2
from app.agent.automations.workflow import run_blockers
from app.db.database import async_session_maker
from app.db.models.platform_automation import (
    AutomationTemplate, TEMPLATE_CATEGORIES,
)
from app.services.automation_template_catalog import (
    CATALOG, sync_template_catalog, template_payload,
)
from app.services.connector_registry import ConnectorRegistry

# The two rail-forbidden tools — a template containing either is a
# product-invariant violation, not a style problem.
_SEND_TOOLS = ("gmail__send_message", "outlook__send_message")


@pytest.fixture(scope="module")
def registry():
    r = ConnectorRegistry()
    r.load_all(include_experimental=True)
    return {e["connector_id"]: e for e in r.automation_registry()}


def _validated(entry, registry):
    tvars = {v["name"] for v in entry.get("variables") or []}
    return validate_spec(entry["spec"], registry, template_mode=True,
                         template_vars=tvars)


def test_catalog_size_and_slug_uniqueness():
    slugs = [e["slug"] for e in CATALOG]
    assert len(slugs) == len(set(slugs))
    assert 25 <= len(CATALOG) <= 40


def test_every_template_names_a_real_category_and_connectors(registry):
    for e in CATALOG:
        assert e["category"] in TEMPLATE_CATEGORIES, e["slug"]
        for cid in e["connectors"]:
            assert cid in registry, f"{e['slug']}: unknown connector {cid}"


def test_every_template_spec_validates_against_the_real_registry(registry):
    failures = []
    for e in CATALOG:
        try:
            _validated(e, registry)
        except SpecError as exc:
            failures.append((e["slug"], exc.errors))
    assert not failures, failures


def test_no_template_can_send_mail():
    for e in CATALOG:
        blob = json.dumps(e["spec"])
        for tool in _SEND_TOOLS:
            assert tool not in blob, f"{e['slug']} references {tool}"


def test_every_write_target_is_the_grant_pin(registry):
    """A grant pins ONE target — a template pointing a write's target
    param at an event-derived value would arm-fail (or worse, fire and
    be refused per-call). The catalog must never teach that shape."""
    for e in CATALOG:
        v = _validated(e, registry)
        if isinstance(v, ValidatedSpecV2):
            writes = [(st.connector_id, st.tool, st.params_template)
                      for st in v.steps if st.mutates]
        else:
            writes = ([(v.action_connector_id, v.action_tool,
                        v.action_params_template)]
                      if v.action_mutates else [])
        for cid, tool, params in writes:
            target_param = (registry[cid].get("target_param_by_action")
                            or {}).get(tool)
            assert target_param, f"{e['slug']}: {tool} has no target param"
            assert params.get(target_param) == "{{grant.target.id}}", (
                f"{e['slug']}: {tool}.{target_param} must be the grant "
                f"pin, got {params.get(target_param)!r}"
            )


def test_declared_variables_have_names_labels_and_shape():
    for e in CATALOG:
        for var in e.get("variables") or []:
            assert var.get("name") and var.get("label"), (e["slug"], var)
            assert isinstance(var.get("required"), bool)


def test_flagship_morning_work_brief_pins(registry):
    """R42. The brief is seven steps and it interpolates ONE value.

    Until this round it was five connector reads stitched by a string
    template, so a read that failed published `(0)` above its own "Could
    not read …" sentence and a step that never ran published `()` — a
    missing `steps.<id>` renders as "". The post carries one literal and
    one placeholder now; the counting is gone with the arithmetic.
    """
    entry = next(e for e in CATALOG if e["slug"] == "morning-work-brief")
    assert entry["category"] == "work"
    assert set(entry["connectors"]) == {"calendar", "gmail", "slack", "jira"}
    # No declared variables at all: an unanswered one is a read that
    # fails and blames a healthy account every weekday (the GitHub
    # owner/repo defect), and the setup thread has nothing to ask.
    assert entry["variables"] == []
    assert entry["spec"]["variables"] == {}

    v = _validated(entry, registry)
    assert isinstance(v, ValidatedSpecV2)
    reads = [st for st in v.steps if st.kind == "tool" and not st.mutates]
    writes = [st for st in v.steps if st.mutates]
    assert len(v.steps) == 7
    assert len(reads) == 5 and len(v.agent_steps) == 1 and len(writes) == 1
    assert [st.id for st in v.steps] == [
        "cal", "mail", "waiting", "rooms", "board", "rank", "post",
    ]
    assert {st.connector_id for st in reads} == {
        "calendar", "gmail", "slack", "jira",
    }
    # One dead source must not kill the brief; every read still collects.
    assert all(st.on_error == "continue" for st in reads)
    assert all(st.collect for st in reads)
    # The agent step keeps its `fail` default: its answer IS the post, and
    # a swallowed failure binds "" and publishes a bare title.
    rank = v.agent_steps[0]
    assert rank.on_error == "fail" and rank.output_var == "brief"

    # Exactly one write, to Slack — mail is only ever READ here.
    assert len(writes) == 1 and writes[0].tool == "slack__send_message"
    text = writes[0].params_template["text"]
    assert text == "*Morning brief*\n\n{{var.brief}}"
    assert "{{steps." not in json.dumps(entry["spec"]), (
        "a step placeholder in the post is how a failed read became a zero"
    )

    # The five headings are the narrator's own §3.6 vocabulary, so the
    # Slack post and the thread's result card rank one run once. Compared
    # word-for-word: the prompt writes the separators as ASCII hyphens
    # (Slack `*bold*` copy), BRIEF_GROUPS uses · and —.
    def _words(label):
        return re.sub(r"[^A-Z ]+", " ", label).split()
    for _rank, label, _tone in BRIEF_GROUPS:
        heading = next(
            (ln for ln in rank.prompt.splitlines() if ln.startswith("*")
             and _words(ln) == _words(label)),
            None,
        )
        assert heading, f"{label!r} is not a heading in the ranking prompt"

    # The only thing the product must ask for is where to post.
    assert [b["code"] for b in run_blockers(v.raw)] == ["needs_destination"]


def test_boss_email_draft_pins(registry):
    """R28-C's proactive-assist flagship, as coordinated."""
    entry = next(e for e in CATALOG if e["slug"] == "boss-email-draft")
    assert entry["category"] == "email"
    v = _validated(entry, registry)
    assert isinstance(v, ValidatedSpecV2)
    assert v.mode == "confirm"
    assert v.sources[0].mode == "push"
    assert v.sources[0].connector_id == "gmail"
    writes = [st for st in v.steps if st.mutates]
    assert len(writes) == 1 and writes[0].tool == "gmail__create_draft"
    var_names = {x["name"]: x for x in entry["variables"]}
    assert var_names["boss_email"]["required"] is True
    assert var_names["draft_style"]["required"] is False


# ── Boot sync semantics ──────────────────────────────────────────────


@pytest_asyncio.fixture
async def synced_db():
    async with async_session_maker() as db:
        await sync_template_catalog(db)
        yield db


@pytest.mark.asyncio
async def test_sync_is_an_idempotent_upsert(synced_db):
    rows = (await synced_db.execute(
        select(AutomationTemplate)
    )).scalars().all()
    by_slug = {r.slug: r for r in rows}
    assert set(e["slug"] for e in CATALOG) <= set(by_slug)
    stats = await sync_template_catalog(synced_db)
    assert stats["inserted"] == 0 and stats["updated"] == 0


@pytest.mark.asyncio
async def test_sync_heals_drift_but_never_touches_enabled(synced_db):
    row = (await synced_db.execute(
        select(AutomationTemplate)
        .where(AutomationTemplate.slug == "morning-work-brief")
    )).scalar_one()
    row.name = "tampered"
    row.enabled = False       # the admin kill-switch
    await synced_db.commit()

    await sync_template_catalog(synced_db)

    row = (await synced_db.execute(
        select(AutomationTemplate)
        .where(AutomationTemplate.slug == "morning-work-brief")
    )).scalar_one()
    assert row.name == "Morning work brief"   # content healed
    assert row.enabled is False               # kill-switch survived
    # Restore for sibling tests.
    row.enabled = True
    await synced_db.commit()


@pytest.mark.asyncio
async def test_payload_carries_category_and_variables(synced_db):
    row = (await synced_db.execute(
        select(AutomationTemplate)
        .where(AutomationTemplate.slug == "boss-email-draft")
    )).scalar_one()
    payload = template_payload(row)
    assert payload["category"] == "email"
    assert any(v["name"] == "boss_email" for v in payload["variables"])
    assert payload["spec"]["version"] == 2


def test_the_set_up_button_path_validates_for_every_template(registry):
    """R36-1 — the ENDPOINT's exact pre-create mutations, not the bare
    spec. The lint above stayed green for weeks while 19 of 28
    templates 422'd on "Set up", because `from_template` stamped
    `variables` onto v1 specs (an unknown top-level field there) and
    never passed `template_vars` — so a required variable with no
    default was an `unknown_variable` error instead of a setup-thread
    question. This test IS that endpoint path; if it and the endpoint
    ever drift again, drift the test first."""
    failures = []
    for e in CATALOG:
        spec = dict(e.get("spec") or {})
        variables = dict(spec.get("variables") or {})
        declared: set = set()
        for v in e.get("variables") or []:
            name = v.get("name")
            if not name:
                continue
            declared.add(str(name))
            if not variables.get(name) and v.get("default"):
                variables[name] = v["default"]
        if spec.get("version") == 2:
            spec["variables"] = variables
        if spec.get("description") is None and e.get("description"):
            spec["description"] = e.get("description")
        try:
            validate_spec(spec, registry, template_mode=True,
                          template_vars=declared)
        except SpecError as exc:
            failures.append((e["slug"], exc.errors))
    assert not failures, failures


def test_digest_templates_carry_their_own_narration():
    """R36-7 — a template whose product is a digest names it, so the
    narrator stops dressing every result as the morning triage."""
    by_slug = {e["slug"]: e for e in CATALOG}
    hint = (by_slug["newsletter-roundup"]["spec"].get("narration") or {})
    assert hint.get("style") == "digest"
    assert hint.get("title") == "This week's newsletters"
    assert hint.get("goal")
    for slug in ("daily-repo-digest", "week-ahead-digest",
                 "class-email-digest", "daily-agenda",
                 "weekly-work-log", "daily-standup-notes"):
        h = by_slug[slug]["spec"].get("narration") or {}
        assert h.get("style") == "digest", slug
        assert 1 <= len(h.get("title") or "") <= 80, slug


# ── The already-installed flagship (alembic 099) ─────────────────────
#
# A catalog fix does not fix an automation somebody already has, and the
# founder's account is posting the old shape every weekday. These pin the
# migration's DECISION — pure, so no database is needed for it.


@pytest.fixture(scope="module")
def respec():
    import importlib.util
    import pathlib
    path = (pathlib.Path(__file__).resolve().parents[1] / "alembic" /
            "versions" / "20260831_0099_099_morning_brief_respec.py")
    spec = importlib.util.spec_from_file_location("m099", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_099_ships_the_catalog_plan_step_for_step(respec):
    """The migration's frozen copy IS the catalog's plan.

    Frozen means "does not follow the catalog from here on", not
    "already differs from it". The copy went out with `on_error:
    continue` on the POST step, where the catalog leaves it implicit —
    and implicit on a MUTATING step is `fail`, so a migrated brief whose
    Slack post failed would have been recorded as a run that carried on.
    """
    catalog = next(e for e in CATALOG if e["slug"] == respec._SLUG)
    assert respec._NEW_SPEC["steps"] == catalog["spec"]["steps"]


def test_099_re_specs_a_shipped_installation_and_keeps_what_is_theirs(
        respec, registry):
    """The destination, the pins, the cron and a renamed title survive;
    the plan and the description are replaced."""
    installed = json.loads(json.dumps(respec._NEW_SPEC))   # start from new…
    installed["steps"] = [                                  # …but the OLD plan
        {"id": sid, "connector_id": sid, "tool": tool, "params": {}}
        for sid, tool in respec._OLD_STEPS
    ]
    installed["steps"][-1]["params"] = {
        "channel": "{{grant.target.id}}", "text": respec._OLD_POST_TEXT,
    }
    installed["steps"][-1]["grant_id"] = "grant-123"
    installed["steps"][-1]["grant_target"] = {
        "kind": "channel", "id": "C0ALL", "label": "#all-toup",
    }
    installed["description"] = respec._OLD_DESCRIPTION
    installed["name"] = "Nariman's brief"
    installed["focus"] = {"slack": [{"kind": "channel", "id": "C0ALL",
                                     "label": "#all-toup",
                                     "note": "Ali first"}]}
    installed["trigger"]["sources"][0]["schedule"] = {"cron_local": "15 7 * * 1-5"}

    new = respec._target(installed)
    assert new is not None
    assert [s["id"] for s in new["steps"]] == [
        "cal", "mail", "waiting", "rooms", "board", "rank", "post",
    ]
    assert new["name"] == "Nariman's brief"
    assert new["focus"] == installed["focus"]
    assert new["trigger"] == installed["trigger"]
    assert new["steps"][-1]["grant_id"] == "grant-123"
    assert new["steps"][-1]["grant_target"]["id"] == "C0ALL"
    assert new["description"] != respec._OLD_DESCRIPTION
    # It must still be a legal spec with its destination already pinned,
    # or the migration would silently unarm the automation it fixed.
    v = validate_spec(new, registry, template_mode=True, template_vars=set())
    assert isinstance(v, ValidatedSpecV2) and len(v.steps) == 7
    assert run_blockers(new) == []
    # Running it twice must not run it twice.
    assert respec._target(new) is None


@pytest.mark.parametrize("mutate,why", [
    (lambda s: s["steps"][-1]["params"].update(text="mine"), "edited post"),
    (lambda s: s["steps"].pop(1), "edited plan"),
    (lambda s: s.setdefault("variables", {}).update(
        jira_jql="project = ENG ORDER BY rank"), "their own Jira filter"),
    (lambda s: s.update(version=1), "a v1 spec"),
])
def test_099_leaves_an_automation_the_user_changed_alone(respec, mutate, why):
    installed = json.loads(json.dumps(respec._NEW_SPEC))
    installed["steps"] = [
        {"id": sid, "connector_id": sid, "tool": tool, "params": {}}
        for sid, tool in respec._OLD_STEPS
    ]
    installed["steps"][-1]["params"] = {"text": respec._OLD_POST_TEXT}
    installed["description"] = respec._OLD_DESCRIPTION
    assert respec._target(installed) is not None, "the fixture must match"
    mutate(installed)
    assert respec._target(installed) is None, why
