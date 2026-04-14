"""Tests for the AppManifest schema and loader (Checkpoint 4a)."""

import json
import os
import tempfile

import pytest

from app.models.app_manifest import AppManifest, detect_unknown_fields
from app.services.app_manifest_loader import load_app_manifest


# ── Fixtures ─────────────────────────────────────────────────────

VALID_MANIFEST = {
    "domain": "Handsome Hype Compliments",
    "description": "Tools for viewing compliment history and navigation.",
    "actions": [
        {
            "name": "open_home",
            "description": "Navigate to the main screen.",
            "type": "navigate",
            "screen": "Home",
            "params": {},
        },
        {
            "name": "get_recent",
            "description": "Get recent compliments.",
            "type": "query",
            "sql": "SELECT id, text FROM compliments ORDER BY id DESC LIMIT :limit;",
            "params": {
                "limit": {
                    "type": "integer",
                    "description": "Max rows",
                    "default": 10,
                }
            },
        },
        {
            "name": "clear_history",
            "description": "Remove all compliments.",
            "type": "mutation",
            "sql": "DELETE FROM compliments;",
            "params": {},
        },
    ],
}


def _write_manifest(app_dir: str, data) -> str:
    """Write a manifest file to {app_dir}/lib/agentSkill.json."""
    lib_dir = os.path.join(app_dir, "lib")
    os.makedirs(lib_dir, exist_ok=True)
    path = os.path.join(lib_dir, "agentSkill.json")
    if isinstance(data, (dict, list)):
        with open(path, "w") as f:
            json.dump(data, f)
    elif isinstance(data, str):
        with open(path, "w") as f:
            f.write(data)
    return path


# ── Test 1: Happy path ──────────────────────────────────────────

def test_valid_manifest_parses():
    """A valid manifest with all three action types parses correctly."""
    manifest = AppManifest.model_validate(VALID_MANIFEST)
    assert manifest.domain == "Handsome Hype Compliments"
    assert len(manifest.actions) == 3
    assert manifest.actions[0].type == "navigate"
    assert manifest.actions[0].screen == "Home"
    assert manifest.actions[1].type == "query"
    assert manifest.actions[1].sql.startswith("SELECT")
    assert manifest.actions[2].type == "mutation"


def test_loader_happy_path():
    """Loader reads a valid manifest from disk and returns AppManifest."""
    with tempfile.TemporaryDirectory() as app_dir:
        _write_manifest(app_dir, VALID_MANIFEST)
        result = load_app_manifest(app_dir, app_id="test-app-123")
        assert result is not None
        assert result.domain == "Handsome Hype Compliments"
        assert len(result.actions) == 3


# ── Test 2: Missing manifest ────────────────────────────────────

def test_loader_missing_manifest():
    """Loader returns None for apps without agentSkill.json."""
    with tempfile.TemporaryDirectory() as app_dir:
        result = load_app_manifest(app_dir, app_id="no-manifest-app")
        assert result is None


# ── Test 3: Invalid JSON ────────────────────────────────────────

def test_loader_invalid_json():
    """Loader returns None for manifests with broken JSON syntax."""
    with tempfile.TemporaryDirectory() as app_dir:
        _write_manifest(app_dir, '{"oops this isnt valid json')
        result = load_app_manifest(app_dir, app_id="broken-json-app")
        assert result is None


# ── Test 4: Schema validation failures ──────────────────────────

def test_missing_required_field_domain():
    """Manifest missing 'domain' fails validation."""
    bad = {
        "description": "test",
        "actions": [],
    }
    with pytest.raises(Exception):
        AppManifest.model_validate(bad)


def test_missing_required_field_actions():
    """Manifest missing 'actions' fails validation."""
    bad = {
        "domain": "test",
        "description": "test",
    }
    with pytest.raises(Exception):
        AppManifest.model_validate(bad)


def test_wrong_type_actions_not_list():
    """Manifest with 'actions' as a string fails validation."""
    bad = {
        "domain": "test",
        "description": "test",
        "actions": "not a list",
    }
    with pytest.raises(Exception):
        AppManifest.model_validate(bad)


def test_navigate_action_with_sql_field():
    """Navigate action with a 'sql' field is accepted (extra='allow')
    but the discriminator routes it to NavigateAction which requires 'screen'."""
    bad = {
        "domain": "test",
        "description": "test",
        "actions": [
            {
                "name": "go_home",
                "description": "Navigate",
                "type": "navigate",
                # Missing required 'screen' field
                "sql": "SELECT 1;",
            }
        ],
    }
    with pytest.raises(Exception):
        AppManifest.model_validate(bad)


def test_query_action_missing_sql():
    """Query action missing 'sql' fails validation."""
    bad = {
        "domain": "test",
        "description": "test",
        "actions": [
            {
                "name": "get_stuff",
                "description": "Get stuff",
                "type": "query",
                # Missing 'sql' field
                "params": {},
            }
        ],
    }
    with pytest.raises(Exception):
        AppManifest.model_validate(bad)


# ── Test 5: SQL param mismatch ──────────────────────────────────

def test_sql_param_undeclared():
    """SQL uses :param but it's not declared in params dict."""
    bad = {
        "domain": "test",
        "description": "test",
        "actions": [
            {
                "name": "search",
                "description": "Search",
                "type": "query",
                "sql": "SELECT * FROM t WHERE name = :name AND age > :min_age;",
                "params": {
                    "name": {"type": "string", "description": "Name"},
                    # missing min_age
                },
            }
        ],
    }
    with pytest.raises(Exception) as exc_info:
        AppManifest.model_validate(bad)
    assert "min_age" in str(exc_info.value)


def test_sql_param_unused():
    """Param declared but not used in SQL string."""
    bad = {
        "domain": "test",
        "description": "test",
        "actions": [
            {
                "name": "list_all",
                "description": "List all",
                "type": "query",
                "sql": "SELECT * FROM t;",
                "params": {
                    "unused_param": {"type": "string", "description": "Not in SQL"},
                },
            }
        ],
    }
    with pytest.raises(Exception) as exc_info:
        AppManifest.model_validate(bad)
    assert "unused_param" in str(exc_info.value)


# ── Test 6: Unknown fields ──────────────────────────────────────

def test_unknown_fields_accepted():
    """Manifest with extra fields parses successfully (lax mode)."""
    extended = {
        **VALID_MANIFEST,
        "some_new_field": "hello",
        "version": 2,
    }
    manifest = AppManifest.model_validate(extended)
    assert manifest.domain == "Handsome Hype Compliments"


def test_unknown_fields_detected():
    """detect_unknown_fields correctly identifies extra fields."""
    extended = {
        **VALID_MANIFEST,
        "some_new_field": "hello",
    }
    unknown = detect_unknown_fields(extended)
    assert "some_new_field" in unknown


def test_unknown_action_fields_detected():
    """detect_unknown_fields finds extra fields inside actions."""
    extended = json.loads(json.dumps(VALID_MANIFEST))
    extended["actions"][0]["experimental_flag"] = True
    unknown = detect_unknown_fields(extended)
    assert any("experimental_flag" in u for u in unknown)


# ── Test 7: Loader with mixed apps ──────────────────────────────

def test_loader_mixed_apps():
    """Three app dirs: valid, missing, broken — all return correct results."""
    with tempfile.TemporaryDirectory() as root:
        # App 1: valid manifest
        app1 = os.path.join(root, "app1")
        os.makedirs(app1)
        _write_manifest(app1, VALID_MANIFEST)

        # App 2: no manifest
        app2 = os.path.join(root, "app2")
        os.makedirs(app2)

        # App 3: broken JSON
        app3 = os.path.join(root, "app3")
        os.makedirs(app3)
        _write_manifest(app3, "{broken json!!")

        r1 = load_app_manifest(app1, app_id="app1")
        r2 = load_app_manifest(app2, app_id="app2")
        r3 = load_app_manifest(app3, app_id="app3")

        assert r1 is not None
        assert r1.domain == "Handsome Hype Compliments"
        assert r2 is None
        assert r3 is None


# ── Test 8: from_file and from_app_dir classmethods ─────────────

def test_from_app_dir():
    """AppManifest.from_app_dir reads from the correct path."""
    with tempfile.TemporaryDirectory() as app_dir:
        _write_manifest(app_dir, VALID_MANIFEST)
        manifest = AppManifest.from_app_dir(app_dir)
        assert manifest.domain == "Handsome Hype Compliments"


def test_from_file_not_found():
    """AppManifest.from_file raises on missing file."""
    with pytest.raises(FileNotFoundError):
        AppManifest.from_file("/nonexistent/path/agentSkill.json")
