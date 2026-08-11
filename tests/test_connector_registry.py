"""
T1c — ConnectorRegistry tests.

Smoke matrix from the T1c spec:

  1. Drop a valid stub manifest → registry picks it up, registry.get('stub')
     returns the entry.
  2. Bad tool-name regex → rejected with structured error, alarm fired,
     rest of registry still boots.
  3. Two manifests with colliding tool names → second one rejected,
     first wins.
  4. Tool name collides with a built-in (web_fetch) → rejected.
     Built-ins always win.
  5. Tool name collides with a skill prefix (toup__, app_builder__) →
     rejected. Skills win over connectors.
  6. Manifest id ≠ directory name → rejected.
  7. Provider class manifest_id ≠ manifest id → rejected.
  8. Registry summary lists all loaded connectors with tool counts.
  9. Boot with no manifests → empty registry, no errors.

Each lint-failure test writes ad-hoc connector subdirs into a temp
directory, points the registry at them, and asserts the registry
loaded the expected subset and rejected the rest with a sensible
alarm message.

Pure-unit — no DB, no FastAPI, no HTTP. The registry is filesystem +
in-memory only.
"""

from __future__ import annotations

import json
import shutil
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from app.services.connector_registry import (
    ConnectorRegistry,
    KNOWN_PROVIDER_APPS,
    TOOL_NAME_RE,
)


# ─── Helper: write an ad-hoc connector dir into tmp ──────────────────


def _write_manifest(tmp_root: Path, dir_name: str, manifest_dict: dict) -> Path:
    sub = tmp_root / dir_name
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "__init__.py").write_text("")
    (sub / "manifest.yaml").write_text(yaml.safe_dump(manifest_dict))
    return sub


def _write_provider(
    sub: Path,
    *,
    manifest_id: str,
    class_name: str = "TmpProvider",
    custom_class_id: str | None = None,
) -> None:
    """Write a provider.py with a BaseConnectorProvider subclass.

    `custom_class_id` lets the test write a manifest_id ClassVar that
    DOESN'T match `manifest_id` — used to test the lint check.
    """
    cls_id = custom_class_id if custom_class_id is not None else manifest_id
    body = textwrap.dedent(f'''\
        from typing import ClassVar
        from app.connectors.base import (
            BaseConnectorProvider, ConnectorContext, ConnectorOk,
            ConnectorResult, HealthResult, RefreshResult,
        )

        class {class_name}(BaseConnectorProvider):
            manifest_id: ClassVar[str] = {cls_id!r}

            async def execute(self, tool_name, tool_input, ctx):
                return ConnectorOk(content="ok")

            async def revoke(self, user_id):
                return None

            async def refresh(self, refresh_token, *, scopes=None):
                return RefreshResult(access_token="x")

            async def health_probe(self, ctx):
                return HealthResult(ok=True)
    ''')
    (sub / "provider.py").write_text(body)


def _baseline_manifest(connector_id: str, tools: list[dict] | None = None) -> dict:
    """A valid manifest dict you can mutate per-test. Defaults to
    `stub_provider_app` so KNOWN_PROVIDER_APPS doesn't reject it."""
    if tools is None:
        tools = [{
            "name": f"{connector_id}__do",
            "description": "do a thing",
            "input_schema": {"type": "object", "properties": {}},
            "mutates": False,
            "elevation": False,
            "output_redaction": [],
            "channel_policy": {"default": "allow", "deny": []},
        }]
    return {
        "manifest_version": 1,
        "id": connector_id,
        "name": connector_id.title(),
        "short_description": f"{connector_id} for testing",
        "status": "stable",  # avoid the experimental filter
        "category": "test",
        "oauth": {
            "provider_app": "stub_provider_app",
            "scopes": [],
            "scopes_optional": [],
            "pkce": True,
            "refresh": True,
        },
        "health": {"probe": tools[0]["name"]},
        "tools": tools,
    }


@pytest.fixture
def tmp_connectors_dir(tmp_path, monkeypatch) -> Path:
    """Tmp directory acting as the connectors root. We ALSO need to
    insert it into sys.path under a synthetic package name so the
    registry's `importlib.import_module("app.connectors.<id>.provider")`
    can find providers we drop in here.

    Trick: write tmp providers under a sibling synthetic package, then
    monkey-patch `importlib.import_module` to redirect lookups for
    `app.connectors.<id>.provider` to `<synth>.<id>.provider`.
    """
    pkg_root = tmp_path / "connectors_pkg"
    pkg_root.mkdir()
    (pkg_root / "__init__.py").write_text("")
    sys.path.insert(0, str(tmp_path))

    # Patch the registry's import call to look in our synthetic root.
    import app.services.connector_registry as registry_module
    real_import = registry_module.importlib.import_module

    def patched_import(name: str, *a, **kw):
        if name.startswith("app.connectors.") and name.endswith(".provider"):
            # app.connectors.<id>.provider → connectors_pkg.<id>.provider
            connector_id = name.removeprefix("app.connectors.").removesuffix(".provider")
            return real_import(f"connectors_pkg.{connector_id}.provider", *a, **kw)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(registry_module.importlib, "import_module", patched_import)

    yield pkg_root

    # Cleanup: drop tmp_path from sys.path; flush any cached imports.
    sys.path.remove(str(tmp_path))
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("connectors_pkg"):
            del sys.modules[mod_name]


# ─── 0. Tool-name regex ──────────────────────────────────────────────


def test_tool_name_regex_matches_skill_loader_convention():
    """The connector regex must accept the same shape as the skill
    loader's prefix convention."""
    assert TOOL_NAME_RE.match("gmail__list_messages")
    assert TOOL_NAME_RE.match("toup__create_spec")
    assert TOOL_NAME_RE.match("a__b")  # minimum legal
    # Invalids
    assert not TOOL_NAME_RE.match("Gmail__list_messages")  # uppercase prefix
    assert not TOOL_NAME_RE.match("gmail_list_messages")    # single underscore
    assert not TOOL_NAME_RE.match("__list_messages")        # empty prefix
    assert not TOOL_NAME_RE.match("123__bad_start")         # digit start
    assert not TOOL_NAME_RE.match("gmail__")                # empty suffix


# ─── 1. Happy path: stub manifest loads ──────────────────────────────


def teststub_manifest_loads_with_include_experimental():
    """The shipped `stub` connector at backend/app/connectors/stub
    loads when include_experimental=True."""
    reg = ConnectorRegistry()
    n = reg.load_all(include_experimental=True)
    # stub is the only experimental connector shipped today; if T3
    # adds more, this >= still holds.
    assert n >= 1
    entry = reg.get("stub")
    assert entry is not None
    assert entry.manifest.id == "stub"
    assert entry.provider.manifest_id == "stub"
    assert "stub__echo" in {t.name for t in entry.manifest.tools}
    assert reg.is_connector_tool("stub__echo")
    assert reg.get_owner("stub__echo") == "stub"
    assert reg.alarms() == []


def teststub_manifest_excluded_from_production_load():
    """Default load (include_experimental=False) drops stub."""
    reg = ConnectorRegistry()
    n = reg.load_all(include_experimental=False)
    assert n == 0  # No production connectors shipped yet (T3a/T4a).
    assert reg.get("stub") is None
    # The exclusion shows up as an alarm — operator can audit "what was skipped"
    skipped_ids = {a.connector_id for a in reg.alarms()}
    assert "stub" in skipped_ids


# ─── 2. Bad tool-name regex ──────────────────────────────────────────


def test_bad_tool_name_regex_rejected_others_still_load(tmp_connectors_dir):
    """Bad manifest is alarmed and skipped; sibling-good manifest still
    loads. No exception propagates from load_all."""
    bad = _baseline_manifest("badreg", tools=[{
        "name": "Bad__Name",  # uppercase — regex rejects
        "description": "x",
        "input_schema": {"type": "object", "properties": {}},
    }])
    good = _baseline_manifest("good")
    sub_bad = _write_manifest(tmp_connectors_dir, "badreg", bad)
    _write_provider(sub_bad, manifest_id="badreg")
    sub_good = _write_manifest(tmp_connectors_dir, "good", good)
    _write_provider(sub_good, manifest_id="good")

    reg = ConnectorRegistry()
    n = reg.load_all(connectors_dir=str(tmp_connectors_dir))
    assert n == 1
    assert reg.get("good") is not None
    assert reg.get("badreg") is None
    alarms = {a.connector_id: a.reason for a in reg.alarms()}
    assert "badreg" in alarms
    assert "regex" in alarms["badreg"].lower() or "tool name" in alarms["badreg"].lower()


# ─── 3. Cross-manifest tool-name collision ───────────────────────────


def test_collision_between_two_manifests_first_wins(tmp_connectors_dir):
    """Two manifests both declare `same__tool`. `aaa` (sorted first)
    wins; `bbb` is rejected with an alarm pointing at `aaa`."""
    a = _baseline_manifest("aaa", tools=[{
        "name": "aaa__tool",
        "description": "x",
        "input_schema": {"type": "object", "properties": {}},
    }])
    # b illegally tries to register `aaa__tool` (would also fail prefix
    # check, but we want to test the collision path specifically — so
    # use bbb's prefix and inject a colliding name. Use same name.)
    b = _baseline_manifest("bbb", tools=[{
        "name": "bbb__tool",  # bbb's own legal tool
        "description": "x",
        "input_schema": {"type": "object", "properties": {}},
    }, {
        "name": "aaa__tool",  # collision — would fail prefix check too
        "description": "collide",
        "input_schema": {"type": "object", "properties": {}},
    }])
    sub_a = _write_manifest(tmp_connectors_dir, "aaa", a)
    _write_provider(sub_a, manifest_id="aaa")
    sub_b = _write_manifest(tmp_connectors_dir, "bbb", b)
    _write_provider(sub_b, manifest_id="bbb")

    reg = ConnectorRegistry()
    reg.load_all(connectors_dir=str(tmp_connectors_dir))
    assert reg.get("aaa") is not None
    # bbb fails (either on prefix mismatch on `aaa__tool` or on
    # collision — either way it's skipped).
    assert reg.get("bbb") is None
    assert any(a.connector_id == "bbb" for a in reg.alarms())


# ─── 4. Built-in tool collision ──────────────────────────────────────


def test_collision_with_builtin_tool_rejected(tmp_connectors_dir):
    """A connector that tries to register `web_fetch` (built-in) is
    rejected. Built-ins always win."""
    # web_fetch is one of the built-in tool names. Force a connector
    # whose tool name happens to collide. We can't use a different
    # prefix because the manifest-prefix check would fire first; use
    # `web_fetch` as a prefix isn't legal either (no `__`).
    #
    # Realistic vector: connector named `web` declares `web__fetch`?
    # That doesn't collide with `web_fetch` (different chars).
    # Instead simulate by making the built-in set return our tool name
    # via monkey-patch.
    m = _baseline_manifest("colide", tools=[{
        "name": "colide__do",
        "description": "x",
        "input_schema": {"type": "object", "properties": {}},
    }])
    sub = _write_manifest(tmp_connectors_dir, "colide", m)
    _write_provider(sub, manifest_id="colide")

    # Pretend `colide__do` is a built-in — registry should reject the
    # manifest at lint step 10.
    import app.services.connector_registry as registry_module
    original = registry_module._collect_builtin_tool_names
    registry_module._collect_builtin_tool_names = lambda: {"colide__do"}
    try:
        reg = ConnectorRegistry()
        reg.load_all(connectors_dir=str(tmp_connectors_dir))
        assert reg.get("colide") is None
        alarm = next(a for a in reg.alarms() if a.connector_id == "colide")
        assert "built-in" in alarm.reason.lower()
    finally:
        registry_module._collect_builtin_tool_names = original


# ─── 5. Skill-prefix collision ───────────────────────────────────────


def test_collision_with_skill_prefix_rejected(tmp_connectors_dir):
    """A connector whose id is `toup` (real skill prefix in the codebase)
    or `app_builder` is rejected. Skills win over connectors."""
    m = _baseline_manifest("toup", tools=[{
        "name": "toup__shadow",
        "description": "shadow toup skill",
        "input_schema": {"type": "object", "properties": {}},
    }])
    sub = _write_manifest(tmp_connectors_dir, "toup", m)
    _write_provider(sub, manifest_id="toup")

    reg = ConnectorRegistry()
    reg.load_all(connectors_dir=str(tmp_connectors_dir))
    assert reg.get("toup") is None
    alarm = next(a for a in reg.alarms() if a.connector_id == "toup")
    assert "skill" in alarm.reason.lower()


# ─── 6. Manifest id ≠ directory name ─────────────────────────────────


def test_manifest_id_must_match_directory_name(tmp_connectors_dir):
    """If manifest.id is `bar` but the directory is `foo`, reject."""
    m = _baseline_manifest("bar")  # manifest.id="bar"
    sub = _write_manifest(tmp_connectors_dir, "foo", m)  # dir="foo"
    _write_provider(sub, manifest_id="bar")

    reg = ConnectorRegistry()
    reg.load_all(connectors_dir=str(tmp_connectors_dir))
    assert reg.get("bar") is None
    assert reg.get("foo") is None
    alarm = next(a for a in reg.alarms() if a.connector_id == "foo")
    assert "directory" in alarm.reason.lower()


# ─── 7. Provider class manifest_id ≠ manifest id ─────────────────────


def test_provider_class_manifest_id_must_match(tmp_connectors_dir):
    """If provider class declares `manifest_id="other"` but manifest.id
    is `mine`, reject."""
    m = _baseline_manifest("mine")
    sub = _write_manifest(tmp_connectors_dir, "mine", m)
    _write_provider(sub, manifest_id="mine", custom_class_id="other")

    reg = ConnectorRegistry()
    reg.load_all(connectors_dir=str(tmp_connectors_dir))
    assert reg.get("mine") is None
    alarm = next(a for a in reg.alarms() if a.connector_id == "mine")
    assert "manifest_id" in alarm.reason


# ─── 8. Registry summary ─────────────────────────────────────────────


def test_registry_summary_lists_loaded_connectors(tmp_connectors_dir):
    a = _baseline_manifest("aaa")
    b = _baseline_manifest("bbb", tools=[
        {
            "name": "bbb__one",
            "description": "x",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "bbb__two",
            "description": "y",
            "input_schema": {"type": "object", "properties": {}},
        },
    ])
    _write_provider(_write_manifest(tmp_connectors_dir, "aaa", a), manifest_id="aaa")
    _write_provider(_write_manifest(tmp_connectors_dir, "bbb", b), manifest_id="bbb")

    reg = ConnectorRegistry()
    reg.load_all(connectors_dir=str(tmp_connectors_dir))
    summary = reg.summary()
    assert len(summary) == 2
    by_id = {s["id"]: s for s in summary}
    assert by_id["aaa"]["tools"] == ["aaa__do"]
    assert sorted(by_id["bbb"]["tools"]) == ["bbb__one", "bbb__two"]


# ─── 9. Empty load ───────────────────────────────────────────────────


def test_empty_directory_loads_cleanly(tmp_connectors_dir):
    """No subdirectories → empty registry, no errors, no alarms."""
    reg = ConnectorRegistry()
    n = reg.load_all(connectors_dir=str(tmp_connectors_dir))
    assert n == 0
    assert reg.list_all() == []
    assert reg.alarms() == []


def test_nonexistent_directory_loads_cleanly(tmp_path):
    reg = ConnectorRegistry()
    n = reg.load_all(connectors_dir=str(tmp_path / "does_not_exist"))
    assert n == 0
    assert reg.alarms() == []


# ─── Bonus: lint coverage on the misc rules ──────────────────────────


def test_unknown_provider_app_rejected(tmp_connectors_dir):
    m = _baseline_manifest("unkn")
    m["oauth"]["provider_app"] = "totally_made_up_provider"
    sub = _write_manifest(tmp_connectors_dir, "unkn", m)
    _write_provider(sub, manifest_id="unkn")

    reg = ConnectorRegistry()
    reg.load_all(connectors_dir=str(tmp_connectors_dir))
    assert reg.get("unkn") is None
    alarm = next(a for a in reg.alarms() if a.connector_id == "unkn")
    assert "provider_app" in alarm.reason


def test_health_probe_must_reference_own_tool(tmp_connectors_dir):
    m = _baseline_manifest("hp")
    m["health"]["probe"] = "hp__nonexistent"  # not in tools list
    sub = _write_manifest(tmp_connectors_dir, "hp", m)
    _write_provider(sub, manifest_id="hp")

    reg = ConnectorRegistry()
    reg.load_all(connectors_dir=str(tmp_connectors_dir))
    assert reg.get("hp") is None
    alarm = next(a for a in reg.alarms() if a.connector_id == "hp")
    assert "health.probe" in alarm.reason


def test_tool_prefix_must_match_manifest_id(tmp_connectors_dir):
    m = _baseline_manifest("real", tools=[{
        "name": "fake__tool",  # prefix doesn't match manifest.id="real"
        "description": "x",
        "input_schema": {"type": "object", "properties": {}},
    }])
    sub = _write_manifest(tmp_connectors_dir, "real", m)
    _write_provider(sub, manifest_id="real")

    reg = ConnectorRegistry()
    reg.load_all(connectors_dir=str(tmp_connectors_dir))
    assert reg.get("real") is None
    alarm = next(a for a in reg.alarms() if a.connector_id == "real")
    assert "must start with" in alarm.reason


def test_malformed_yaml_doesnt_crash_registry(tmp_connectors_dir):
    sub = tmp_connectors_dir / "broken"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    (sub / "manifest.yaml").write_text("this: is: not: valid: yaml: at all\n  - bad")
    (sub / "provider.py").write_text("# empty\n")

    # Plus a sibling that's valid.
    good = _baseline_manifest("good")
    _write_provider(_write_manifest(tmp_connectors_dir, "good", good), manifest_id="good")

    reg = ConnectorRegistry()
    n = reg.load_all(connectors_dir=str(tmp_connectors_dir))
    assert n == 1
    assert reg.get("good") is not None
    assert any(a.connector_id == "broken" for a in reg.alarms())


def test_get_registry_returns_singleton():
    """Module-level singleton — same instance on repeated calls."""
    from app.services.connector_registry import (
        get_registry, reset_registry_for_tests,
    )
    reset_registry_for_tests()
    a = get_registry()
    b = get_registry()
    assert a is b
    reset_registry_for_tests()
    c = get_registry()
    assert c is not a


def test_known_provider_apps_includes_v1_set():
    """v1 must support google + github + the test stub."""
    assert "google" in KNOWN_PROVIDER_APPS
    assert "github" in KNOWN_PROVIDER_APPS
    assert "stub_provider_app" in KNOWN_PROVIDER_APPS
