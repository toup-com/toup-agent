"""The host's connection ceiling, asserted where somebody will trip over it.

Tenant Postgres is the binding resource on the VPS and nothing measured it.
Read-only on production, 2026-09-07 15:34Z — mid-hour, no user traffic:

    92 tenant databases
    294 client backends, 292 of them `idle`
    per-database 2-5, mean 2.9
    max_connections 300, superuser_reserved_connections 3  ->  297 usable

Three connections of headroom, before the :00/:05 fleet-wide cron burst
(`_MMCron(minute=0)` day_archival and `_MMCron(minute=5)`
current_context_rollover, fired by every container in the same second) which
takes it to 297-299. The next signup creates one more database and ~3 more
backends. `04-postgres.sh` still carried the original sizing comment,
"headroom for 30 tenants x 8".

`FATAL: sorry, too many clients already` does not queue. It fails the connect,
and a tenant container that boots into it fails `alembic upgrade` (swallowed by
`Dockerfile.agent`) and misses `GENERIC_HEALTH_TIMEOUT_S=90`, so the slot is
reaped and respawned into the same wall.

These assertions read the installer scripts as text, because that is the only
form these settings exist in that anything can check — the live
`/etc/pgbouncer/pgbouncer.ini` is `0640 postgres` and `postgresql.conf` is
edited in place by `pg_set`.

Run:
    cd backend && PYTHONPATH=. pytest tests/test_host_capacity_config.py
"""
from __future__ import annotations

import pathlib
import re

import pytest

NEW_VPS = pathlib.Path(__file__).resolve().parents[2] / "new-vps"
PG = NEW_VPS / "04-postgres.sh"
OBS = NEW_VPS / "06-observability.sh"

# Live fleet, measured 2026-09-07. Keep these honest; every threshold below is
# derived from them rather than picked.
TENANT_DBS = 92
MEAN_BACKENDS_PER_DB = 2.9
PEAK_BACKENDS_PER_DB = 5


@pytest.fixture(scope="module")
def pg_src() -> str:
    return PG.read_text("utf-8")


@pytest.fixture(scope="module")
def obs_src() -> str:
    return OBS.read_text("utf-8")


def _pg_set(src: str, key: str) -> str:
    m = re.search(rf'^pg_set\s+{re.escape(key)}\s+"([^"]+)"', src, re.M)
    assert m, f"pg_set {key} not found in 04-postgres.sh"
    return m.group(1)


def _ini(src: str, key: str) -> str:
    m = re.search(rf"^{re.escape(key)}\s*=\s*(\S+)", src, re.M)
    assert m, f"{key} not found in the pgbouncer ini heredoc"
    return m.group(1)


# ── anti-vacuity ─────────────────────────────────────────────────


def test_the_installer_scripts_are_the_ones_we_think(pg_src, obs_src):
    assert "pgbouncer" in pg_src and "[databases]" in pg_src
    assert "prom/node-exporter" in obs_src and "alerts.yml" in obs_src
    assert _pg_set(pg_src, "shared_buffers") == "16GB", (
        "shared_buffers moved; the memory arithmetic in the max_connections "
        "comment has to move with it"
    )


# ── the ceiling ──────────────────────────────────────────────────


def test_max_connections_leaves_room_for_the_fleet_that_exists(pg_src):
    """The falsifier. At 300 this fails with the live numbers."""
    maxconn = int(_pg_set(pg_src, "max_connections"))
    reserved = 3  # superuser_reserved_connections, PG default
    usable = maxconn - reserved
    steady = TENANT_DBS * MEAN_BACKENDS_PER_DB
    assert usable > steady * 1.25, (
        f"max_connections={maxconn} leaves {usable} usable slots for a steady "
        f"state of {steady:.0f} ({TENANT_DBS} tenant DBs x {MEAN_BACKENDS_PER_DB} "
        "backends). That is under 25% headroom, and the :00 cron burst alone "
        "eats it — measured 294/297 at rest on 2026-09-07."
    )


def test_pgbouncer_caps_what_one_tenant_can_take(pg_src):
    """`max_db_connections` is what turns an outage into a queue.

    Without it pgbouncer may open `default_pool_size` server connections for
    every tenant DB — 92 x 20 = 1840 against a Postgres that accepts a few
    hundred — and the overflow surfaces as a FATAL on somebody ELSE's connect.
    """
    per_db = int(_ini(pg_src, "max_db_connections"))
    maxconn = int(_pg_set(pg_src, "max_connections"))
    assert per_db >= PEAK_BACKENDS_PER_DB, (
        f"max_db_connections={per_db} is at or below the measured per-tenant "
        f"high-water mark ({PEAK_BACKENDS_PER_DB}); it would bind in normal "
        "operation and make every tenant wait"
    )
    assert per_db * TENANT_DBS < maxconn - 3, (
        f"max_db_connections={per_db} x {TENANT_DBS} tenant DBs = "
        f"{per_db * TENANT_DBS} exceeds the {maxconn - 3} usable slots, so the "
        "cap does not actually bound the fleet"
    )


def test_idle_server_connections_are_reaped_within_the_fleets_touch_interval(pg_src):
    """600 s never fires: every container touches its DB at least every 30 s.

    Measured on the canary's own pool 2026-09-07: idle ages 13 s and 255 s. A
    600 s timeout reaps neither; a 180 s one reaps the second.
    """
    idle = int(_ini(pg_src, "server_idle_timeout"))
    assert 0 < idle <= 300, (
        f"server_idle_timeout={idle}: at 0 it is disabled and pools never "
        "shrink; above ~300 it never fires in this fleet"
    )
    lifetime = int(_ini(pg_src, "server_lifetime"))
    assert idle < lifetime, "an idle timeout above server_lifetime is dead config"


def test_pgbouncer_restarts_itself(pg_src):
    """`Restart=no` was the whole 2026-08-01 outage.

    Every tenant DB goes through pgbouncer. It died at 19:43Z that day and
    nothing in the system could end the outage; it died twice more on
    2026-09-06. The cron watchdog is the no-root half and caps at 3/hour.
    """
    assert "pgbouncer.service.d/restart.conf" in pg_src, (
        "no systemd drop-in for pgbouncer — the packaged unit is Restart=no"
    )
    m = re.search(r"restart\.conf > /dev/null <<'CONF'\n(.*?)\nCONF", pg_src, re.S)
    assert m, "the drop-in heredoc is not readable"
    dropin = m.group(1)
    assert re.search(r"^Restart=always$", dropin, re.M)
    # StartLimitIntervalSec moved to [Unit] in systemd 229 and is silently
    # ignored in [Service] — a drop-in can look applied and still rate-limit.
    unit, _, service = dropin.partition("[Service]")
    assert "StartLimit" not in service, (
        "StartLimit* in [Service] is ignored by systemd >= 229; it belongs in "
        "[Unit] or the restart cascade is still rate-limited"
    )
    assert "StartLimit" in unit
    assert 'systemctl show pgbouncer -p Restart' in pg_src, (
        "the installer never verifies the drop-in took effect"
    )


# ── the alerting that has to notice ──────────────────────────────


def test_there_is_an_alert_on_the_connection_ceiling(obs_src):
    """It was commented out with 'requires pg_exporter if/when added'."""
    assert re.search(r"^\s*- alert: PGConnectionsHigh$", obs_src, re.M), (
        "PGConnectionsHigh is still commented out or missing"
    )
    assert "toup_pg_backends / toup_pg_max_connections" in obs_src, (
        "the rule still references pg_exporter metrics nothing produces"
    )
    assert "/usr/local/sbin/toup-pg-connections" in obs_src, (
        "nothing writes toup_pg_backends, so the rule can never fire"
    )
    assert re.search(r"^\* \* \* \* \* root /usr/local/sbin/toup-pg-connections$",
                     obs_src, re.M), (
        "the collector must run every minute — the :00 burst is ~70 s wide and "
        "a coarser sample misses it entirely"
    )


def test_node_exporter_is_reachable_from_prometheus(obs_src):
    """`network_mode: bridge` made every host rule permanently inert.

    prometheus/cadvisor/grafana share the compose network; the default bridge
    has no embedded DNS, so `node_exporter:9100` never resolved. Verified on
    the host 2026-09-07: `getent hosts cadvisor` -> 172.20.0.6 from grafana,
    `getent hosts node_exporter` -> nothing.
    """
    m = re.search(r"^  node_exporter:\n(.*?)(?=^  \w+:$)", obs_src, re.S | re.M)
    assert m, "node_exporter service block not found in the compose heredoc"
    block = m.group(1)
    # Config lines only — the comment explaining this is allowed to say it.
    settings = [l for l in block.splitlines() if not l.lstrip().startswith("#")]
    assert not [l for l in settings if "network_mode:" in l], (
        "node_exporter is on the default docker bridge while prometheus is on "
        "the compose network, so the `node` scrape target cannot resolve and "
        "HostHighMemory / HostCPULoad / HostSwapInUse / the textfile metrics "
        "are all dead config"
    )
    assert "targets: ['node_exporter:9100']" in obs_src


def test_a_dead_instrument_is_itself_an_alert(obs_src):
    """Prometheus was down for four months and nothing said so.

    `up == 0` catches a scrape target; `node_textfile_mtime_seconds` catches a
    cron that stopped writing, which otherwise freezes a gauge at its last
    healthy value forever.
    """
    for alert in ("ScrapeTargetDown", "TextfileMetricsStale"):
        assert re.search(rf"^\s*- alert: {alert}$", obs_src, re.M), (
            f"{alert} is missing (a substring match would pass for a renamed "
            "alert, so this is anchored)"
        )
    assert re.search(r"^\s*expr: up == 0$", obs_src, re.M)
    assert "node_textfile_mtime_seconds" in obs_src


# ── the rendered artefacts must actually be valid ────────────────


def _extract(src: str, pattern: str) -> str:
    m = re.search(pattern, src, re.S | re.M)
    assert m, f"heredoc not found: {pattern}"
    return m.group(1) + "\n"


def test_the_collector_script_is_valid_shell(obs_src, tmp_path):
    """It is written into a heredoc, so nothing else would ever parse it."""
    import shutil
    import subprocess

    body = _extract(
        obs_src,
        r"^sudo tee /usr/local/sbin/toup-pg-connections > /dev/null <<'SHELL'\n(.*?)\nSHELL$",
    )
    assert "toup_pg_backends" in body and "toup_pg_max_connections" in body
    # A collector that leaves a STALE file is worse than one that writes none.
    assert 'exit 0' in body, "no bail-out path when Postgres is unreachable"
    bash = shutil.which("bash")
    if not bash:  # pragma: no cover
        pytest.skip("no bash")
    f = tmp_path / "collector.sh"
    f.write_text(body, "utf-8")
    r = subprocess.run([bash, "-n", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_alert_rules_parse(obs_src, tmp_path):
    """`promtool check rules` on the rendered alerts.yml.

    The YAML half always runs. The promtool half is opt-in
    (`TOUP_TEST_PROMTOOL=1`): it pulls a 200 MB image, which does not belong in
    the CI sweep that runs every file in tests/ in its own process. Run locally
    it reports `SUCCESS: 14 rules found`.

    A rules file Prometheus rejects means Prometheus does not start, which is
    one of the ways this stack has been dark before.
    """
    import os
    import subprocess

    body = _extract(
        obs_src,
        r"^sudo tee \"\$OBS_DIR/prometheus/alerts\.yml\" > /dev/null <<'ALERTS'\n(.*?)\nALERTS$",
    )
    f = tmp_path / "alerts.yml"
    f.write_text(body, "utf-8")

    # Cheap check that always runs: it must at least be YAML with groups.
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(body)
    names = [r["alert"] for g in doc["groups"] for r in g["rules"] if "alert" in r]
    assert "PGConnectionsHigh" in names and "ScrapeTargetDown" in names
    assert len(names) >= 12, names

    if os.environ.get("TOUP_TEST_PROMTOOL") != "1":
        pytest.skip("set TOUP_TEST_PROMTOOL=1 to run promtool; YAML shape checked")
    try:
        ok = subprocess.run(["docker", "version"], capture_output=True, timeout=15).returncode == 0
    except Exception:
        ok = False
    if not ok:
        pytest.skip("TOUP_TEST_PROMTOOL=1 but the docker daemon is not reachable")
    r = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "/bin/promtool",
         "-v", f"{f}:/tmp/alerts.yml:ro", "prom/prometheus:v2.54.1",
         "check", "rules", "/tmp/alerts.yml"],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SUCCESS" in r.stdout
