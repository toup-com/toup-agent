"""Template catalog (Round 28) — every template proven against the
REAL manifest-built registry, the flagship pinned, and the boot sync's
upsert semantics (admin `enabled` survives).

PLATFORM lane. The validation tests are pure; the sync tests use the
shared test DB via async_session_maker.
"""

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.agent.automations.spec import SpecError, validate_spec
from app.agent.automations.spec_v2 import ValidatedSpecV2
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
    import json
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
    entry = next(e for e in CATALOG if e["slug"] == "morning-work-brief")
    assert entry["category"] == "work"
    assert set(entry["connectors"]) == {
        "jira", "github", "teams", "gmail", "outlook", "slack",
    }
    v = _validated(entry, registry)
    assert isinstance(v, ValidatedSpecV2)
    reads = [st for st in v.steps if not st.mutates]
    writes = [st for st in v.steps if st.mutates]
    # Five sources of news, every one skip-tolerant — a dead Jira must
    # not kill the whole brief.
    assert len(reads) == 5
    assert all(st.on_error == "skip" for st in reads)
    assert all(st.collect for st in reads)
    # Exactly one write, to Slack — mail is only ever READ here.
    assert len(writes) == 1 and writes[0].tool == "slack__send_message"
    read_conns = {st.connector_id for st in reads}
    assert read_conns == {"jira", "github", "teams", "gmail", "outlook"}


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
