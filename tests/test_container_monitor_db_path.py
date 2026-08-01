"""The tenant DB path must alert — /agent/health 200 is not enough.

Regression guard for the 2026-08-01 outage: pgbouncer died, every tenant lost
its database, chat 500'd fleet-wide for six minutes, and nothing alerted. The
containers were up, Postgres was up, and /agent/health kept answering

    {"status": "healthy", "db_ok": false, ...}

The agent's own db_watchdog had already diagnosed it and put `db_ok: false` in
that payload. The monitor simply never read the field.

The load-bearing test is `test_the_exact_outage_payload_alerts`: it feeds the
real production body and requires an alert. It fails on the pre-fix code,
which only ever looked at `status`.

Direct invocation works:
    cd backend && ENVIRONMENT=development python tests/test_container_monitor_db_path.py
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("ENVIRONMENT", "development")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import container_monitor as cm  # noqa: E402


def _container(name: str, cid: str = None):
    return SimpleNamespace(id=cid or name, container_name=name, host_port=9000,
                           user_id=f"user-{name}", status="running")


def _reset():
    cm._db_down_counts.clear()
    cm._last_db_alert = None


def _probe_with(payload, status=200):
    """Drive the pure verdict function against a canned /agent/health body.

    Deliberately not exercising _probe_agent_health here: it opens a real DB
    session to resolve agent_url, which is why the decision worth testing was
    extracted into a pure function in the first place.
    """
    if status != 200:
        return False, None
    return cm.verdict_from_health_body(payload)


def _capture_alerts(coro):
    sent = []

    async def _fake_send(msg):
        sent.append(msg)

    with patch.object(cm, "_send_telegram_alert", _fake_send):
        asyncio.run(coro())
    return sent


# ── the probe reads the field at all ──────────────────────────────

def test_the_exact_outage_payload_is_parsed():
    healthy, db_ok = _probe_with({"status": "healthy", "db_ok": False})
    assert healthy is True, "the agent WAS answering — liveness must stay true"
    assert db_ok is False, (
        "db_ok=false was not read off the health body; this is the field that "
        "was ignored during the 2026-08-01 fleet-wide outage"
    )


def test_healthy_db_ok_true():
    assert _probe_with({"status": "healthy", "db_ok": True}) == (True, True)


def test_missing_db_ok_is_unknown_not_down():
    """Older images and pool-generic boots omit the field. Absent must never
    be read as down, or every one of them alerts forever."""
    healthy, db_ok = _probe_with({"status": "healthy"})
    assert healthy is True
    assert db_ok is None


def test_non_bool_db_ok_is_unknown():
    assert _probe_with({"status": "healthy", "db_ok": "yes"})[1] is None


def test_dead_container_reports_no_db_verdict():
    healthy, db_ok = _probe_with({"status": "healthy"}, status=500)
    assert healthy is False
    assert db_ok is None, "a container that isn't answering cannot report db_ok"


# ── the alert decision ────────────────────────────────────────────

def test_the_exact_outage_payload_alerts():
    """Two or more tenants reporting db_ok=false is the outage. Alert now."""
    _reset()
    downs = [_container("toup-agent-a"), _container("toup-agent-b")]
    for c in downs:
        cm._db_down_counts[c.id] = 1
    sent = _capture_alerts(lambda: cm._alert_on_db_path(downs))
    assert len(sent) == 1, "a fleet-wide DB outage produced no alert"
    assert "shared component" in sent[0]
    assert "Restarting containers will NOT help" in sent[0], (
        "the alert must steer the operator away from the restart reflex — "
        "restarting 55 containers does not fix pgbouncer"
    )


def test_one_alert_not_one_per_tenant():
    """55 tenants go down together; the operator gets ONE message."""
    _reset()
    downs = [_container(f"toup-agent-{i}") for i in range(55)]
    for c in downs:
        cm._db_down_counts[c.id] = 1
    sent = _capture_alerts(lambda: cm._alert_on_db_path(downs))
    assert len(sent) == 1, f"expected 1 aggregated alert, got {len(sent)}"
    assert "55 tenants" in sent[0]


def test_single_tenant_waits_one_cycle_then_alerts():
    _reset()
    c = _container("toup-agent-solo")
    cm._db_down_counts[c.id] = 1
    assert _capture_alerts(lambda: cm._alert_on_db_path([c])) == [], (
        "one tenant on its first failed cycle should not page anyone"
    )
    cm._db_down_counts[c.id] = 2
    sent = _capture_alerts(lambda: cm._alert_on_db_path([c]))
    assert len(sent) == 1
    assert "Single tenant" in sent[0], (
        "a single tenant must be diagnosed differently from a shared outage"
    )


def test_nothing_down_never_alerts():
    _reset()
    assert _capture_alerts(lambda: cm._alert_on_db_path([])) == []


def test_cooldown_suppresses_the_second_alert():
    _reset()
    downs = [_container("a"), _container("b")]
    for c in downs:
        cm._db_down_counts[c.id] = 1
    assert len(_capture_alerts(lambda: cm._alert_on_db_path(downs))) == 1
    assert _capture_alerts(lambda: cm._alert_on_db_path(downs)) == [], (
        "a 5-minute monitor loop must not re-page every cycle"
    )


def test_cooldown_expires():
    _reset()
    downs = [_container("a"), _container("b")]
    for c in downs:
        cm._db_down_counts[c.id] = 1
    _capture_alerts(lambda: cm._alert_on_db_path(downs))
    cm._last_db_alert = datetime.utcnow() - cm.DB_ALERT_COOLDOWN - timedelta(seconds=1)
    assert len(_capture_alerts(lambda: cm._alert_on_db_path(downs))) == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
