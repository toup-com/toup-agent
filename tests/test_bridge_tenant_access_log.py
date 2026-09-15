"""No tenant request is logged at the edge. Make that optional instead of fixed.

`/etc/caddy/Caddyfile` puts `log` inside `bridge.agents.toup.ai` and
`monitoring.agents.toup.ai` only, so the adapter compiled the tenant wildcard
into `srv0.logs.skip_hosts` — and `grep -c 'agents.toup.ai","uri'
/var/log/caddy/access.log` returns 0. A 404 "Unknown tenant" and a 502 "upstream
down" are therefore indistinguishable from outside the container, which is the
whole reason the 12 Sep incident had to be reconstructed from agent-side logs.

Two things this does NOT do, both of which were tried first:

* edit the Caddyfile — Caddy now runs `--resume` from its autosave, so that file
  is validated at boot and never loaded;
* add a per-route log handler — the exclusion is at the SERVER level, so a route
  handler cannot lift it.

Default OFF. The first install of this file must change nothing at the edge
until an operator sets `BRIDGE_TENANT_ACCESS_LOG=1`.

Run:
    cd backend && PYTHONPATH=. pytest tests/test_bridge_tenant_access_log.py
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _bridge_main_loader import load_bridge_main  # noqa: E402

HOST = "*.agents.toup.ai"


@pytest.fixture
def bridge(monkeypatch):
    mod = load_bridge_main()
    monkeypatch.delenv("BRIDGE_TENANT_ACCESS_LOG", raising=False)
    return mod


# ── the flag ───────────────────────────────────────────────────────


def test_default_off_is_a_no_op_that_does_not_even_read(bridge, monkeypatch):
    calls = []
    monkeypatch.setattr(bridge.httpx, "get", lambda *a, **k: calls.append(a))
    assert bridge._apply_tenant_access_log() == {"enabled": False}
    assert calls == []


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_the_flag_accepts_the_usual_spellings(bridge, monkeypatch, value):
    monkeypatch.setenv("BRIDGE_TENANT_ACCESS_LOG", value)
    assert bridge._tenant_access_log_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", " "])
def test_anything_else_is_off(bridge, monkeypatch, value):
    monkeypatch.setenv("BRIDGE_TENANT_ACCESS_LOG", value)
    assert bridge._tenant_access_log_enabled() is False


# ── the server-logs planner ────────────────────────────────────────


def test_the_tenant_wildcard_stops_being_skipped(bridge):
    """This is the exclusion. Everything else is downstream of removing it."""
    logs = {
        "logger_names": {"bridge.agents.toup.ai": ["log0"]},
        "skip_hosts": [HOST, "127.0.0.1"],
    }
    updated, changed = bridge._caddy_server_logs_plan(logs)
    assert changed is True
    assert updated["skip_hosts"] == ["127.0.0.1"]
    assert updated["logger_names"][HOST] == ["tenants"]
    # Another site's logger is not disturbed.
    assert updated["logger_names"]["bridge.agents.toup.ai"] == ["log0"]


def test_an_emptied_skip_list_is_removed_not_left_empty(bridge):
    updated, changed = bridge._caddy_server_logs_plan({"skip_hosts": [HOST]})
    assert changed is True
    assert "skip_hosts" not in updated


def test_the_logger_names_shape_follows_the_running_config(bridge):
    """Caddy's `logger_names` is map[string]string before 2.9 and
    map[string][]string from 2.9 on. Copy what is there rather than assume."""
    old_style, _ = bridge._caddy_server_logs_plan(
        {"logger_names": {"bridge.agents.toup.ai": "log0"}}
    )
    assert old_style["logger_names"][HOST] == "tenants"

    new_style, _ = bridge._caddy_server_logs_plan(
        {"logger_names": {"bridge.agents.toup.ai": ["log0"]}}
    )
    assert new_style["logger_names"][HOST] == ["tenants"]

    empty, _ = bridge._caddy_server_logs_plan({})
    assert empty["logger_names"][HOST] == ["tenants"]


def test_applying_twice_changes_nothing_the_second_time(bridge):
    first, changed = bridge._caddy_server_logs_plan({"skip_hosts": [HOST]})
    assert changed is True
    second, changed_again = bridge._caddy_server_logs_plan(first)
    assert changed_again is False
    assert second == first


def test_a_missing_logs_object_is_created_not_crashed_on(bridge):
    updated, changed = bridge._caddy_server_logs_plan(None)
    assert changed is True
    assert updated["logger_names"][HOST] == ["tenants"]


# ── the logging-sink planner ───────────────────────────────────────


def test_the_sink_is_json_at_the_named_file(bridge):
    updated, changed = bridge._caddy_logging_plan({})
    assert changed is True
    sink = updated["logs"]["tenants"]
    assert sink["encoder"] == {"format": "json"}
    assert sink["writer"]["output"] == "file"
    assert sink["writer"]["filename"] == "/var/log/caddy/tenants.log"
    assert sink["include"] == ["http.log.access.tenants"]


def test_the_default_logger_excludes_the_tenant_lines(bridge):
    """Without this every tenant request is written twice."""
    updated, _ = bridge._caddy_logging_plan(
        {"logs": {"default": {"writer": {"output": "file"}, "exclude": ["x"]}}}
    )
    assert updated["logs"]["default"]["exclude"] == ["x", "http.log.access.tenants"]
    # The default sink itself is untouched.
    assert updated["logs"]["default"]["writer"] == {"output": "file"}


def test_the_logging_plan_is_idempotent(bridge):
    first, changed = bridge._caddy_logging_plan({})
    assert changed is True
    _second, changed_again = bridge._caddy_logging_plan(first)
    assert changed_again is False


def test_the_log_file_is_configurable(bridge, monkeypatch):
    monkeypatch.setenv("BRIDGE_TENANT_ACCESS_LOG_PATH", "/tmp/t.log")
    fresh = load_bridge_main()
    assert fresh.TENANT_LOG_FILE == "/tmp/t.log"
    assert fresh._caddy_logging_plan({})[0]["logs"]["tenants"]["writer"][
        "filename"
    ] == "/tmp/t.log"


# ── the applier ────────────────────────────────────────────────────


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.headers = {"etag": '"e1"'}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_apply_writes_both_paths_once_and_then_stops(bridge, monkeypatch):
    monkeypatch.setenv("BRIDGE_TENANT_ACCESS_LOG", "1")
    state = {
        "/config/logging": {},
        "/config/apps/http/servers/srv0/logs": {"skip_hosts": [HOST]},
    }
    patched = []

    def _get(url, **k):
        path = url.split("2019", 1)[1]
        return _Resp(state[path])

    def _patch(url, headers=None, json=None, **k):
        path = url.split("2019", 1)[1]
        state[path] = json
        patched.append(path)
        return _Resp(json)

    monkeypatch.setattr(bridge.httpx, "get", _get)
    monkeypatch.setattr(bridge.httpx, "patch", _patch)

    out = bridge._apply_tenant_access_log()
    assert out["enabled"] is True
    assert out["logging_changed"] and out["server_logs_changed"]
    assert sorted(patched) == [
        "/config/apps/http/servers/srv0/logs", "/config/logging",
    ]

    patched.clear()
    out2 = bridge._apply_tenant_access_log()
    assert out2["logging_changed"] is False and out2["server_logs_changed"] is False
    assert patched == []


def test_a_caddy_failure_never_breaks_route_restoration(bridge, monkeypatch):
    monkeypatch.setenv("BRIDGE_TENANT_ACCESS_LOG", "1")

    def _boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(bridge.httpx, "get", _boom)
    out = bridge._apply_tenant_access_log()
    assert "OSError" in out["error"]

    # And the restore that calls it still returns a summary.
    monkeypatch.setattr(bridge, "AGENTS_DATA_DIR", pathlib.Path("/nonexistent-xyz"))
    summary = bridge._restore_caddy_routes_from_disk_v2()
    assert summary["access_log"]["enabled"] is True
    assert "error" in summary["access_log"]


def test_a_path_that_does_not_exist_yet_is_put_not_patched(bridge, monkeypatch):
    """Caddy's PATCH requires an existing value; a host with no `logging` block
    would 404 forever if we only ever PATCHed."""
    monkeypatch.setenv("BRIDGE_TENANT_ACCESS_LOG", "1")
    puts, patches = [], []
    monkeypatch.setattr(bridge.httpx, "get", lambda url, **k: _Resp(None, status=404))
    monkeypatch.setattr(bridge.httpx, "put",
                        lambda url, **k: puts.append(url) or _Resp({}))
    monkeypatch.setattr(bridge.httpx, "patch",
                        lambda url, **k: patches.append(url) or _Resp({}))
    bridge._apply_tenant_access_log()
    assert len(puts) == 2 and patches == []


def test_the_cas_precondition_is_sent(bridge, monkeypatch):
    monkeypatch.setenv("BRIDGE_TENANT_ACCESS_LOG", "1")
    seen = []
    monkeypatch.setattr(bridge.httpx, "get", lambda url, **k: _Resp({}))
    monkeypatch.setattr(
        bridge.httpx, "patch",
        lambda url, headers=None, **k: seen.append(headers) or _Resp({}),
    )
    bridge._apply_tenant_access_log()
    assert all(h.get("If-Match") == '"e1"' for h in seen)
