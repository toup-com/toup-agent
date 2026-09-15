"""Import `bridge/main.py` in an environment that has no VPS under it.

`main.py` is bridge-host code: it imports `docker`, `structlog` and `psycopg2`
(none is a backend dependency), calls `docker.from_env()` at module scope and
refuses to load without `BRIDGE_PG_ADMIN_URL` / `BRIDGE_DOCKER_IMAGE`. The other
bridge guards side-step all of that by parsing the file with `ast` — which
cannot execute a decision function, and every defect in this wave is a decision.

So: stub the three missing modules, give the two required env vars a value, and
load the real file by path. `docker.errors.NotFound` / `.APIError` are real
exception classes because `main.py` catches them; everything else on the fake
client is a recording no-op, so a test that accidentally reaches Docker gets a
harmless object rather than a hang.

`psycopg2` is the one of the three that is SOMETIMES installed: the Postgres
lane's `psycopg2-binary` sits in any venv built from that lane's list, so the
first version of this loader (which stubbed only `docker` and `structlog`) was
green on every developer machine and red in CI's hermetic sqlite lane, where
all four callers errored at fixture setup with `No module named 'psycopg2'`.
The stub is therefore swapped in for the duration of `exec_module` regardless
of what the process has, and the real module — if any — is put back after:
`main.py` binds the name at import, so the unit under test always sees the
stub, and nothing else in the process loses its driver.

Each caller gets its OWN module instance (`load_bridge_main()`), so monkeypatched
globals — `AGENTS_DATA_DIR`, `CADDY_ADMIN` — never leak between test files.
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import pathlib
import sys
import types

BRIDGE_MAIN = pathlib.Path(__file__).resolve().parents[2] / "bridge" / "main.py"

_counter = itertools.count()


class _FakeContainers:
    def __init__(self, owner):
        self._owner = owner

    def get(self, name):
        raise _docker_module().errors.NotFound(f"no such container: {name}")

    def list(self, *a, **k):
        return []

    def run(self, *a, **k):
        raise AssertionError("test reached docker.containers.run")


class _FakeClient:
    def __init__(self):
        self.containers = _FakeContainers(self)
        self.networks = types.SimpleNamespace(
            get=lambda name: (_ for _ in ()).throw(
                _docker_module().errors.NotFound(name)
            ),
            create=lambda *a, **k: types.SimpleNamespace(name="net"),
        )
        self.images = types.SimpleNamespace(pull=lambda *a, **k: None)


def _docker_module() -> types.ModuleType:
    mod = sys.modules.get("docker")
    if mod is not None:
        return mod
    mod = types.ModuleType("docker")
    errors = types.ModuleType("docker.errors")

    class DockerException(Exception):
        pass

    class NotFound(DockerException):
        pass

    class APIError(DockerException):
        pass

    errors.DockerException = DockerException
    errors.NotFound = NotFound
    errors.APIError = APIError
    mod.errors = errors
    mod.from_env = lambda **k: _FakeClient()
    sys.modules["docker"] = mod
    sys.modules["docker.errors"] = errors
    return mod


class _Logger:
    """structlog's kwargs-logger, reduced to a recorder the tests can read."""

    def __init__(self):
        self.records: list[tuple[str, str, dict]] = []

    def _rec(self, level):
        def _log(event="", *args, **kw):
            self.records.append((level, str(event), kw))

        return _log

    def __getattr__(self, name):
        return self._rec(name)


def _structlog_module() -> types.ModuleType:
    mod = sys.modules.get("structlog")
    if mod is not None:
        return mod
    mod = types.ModuleType("structlog")
    mod.get_logger = lambda *a, **k: _Logger()
    sys.modules["structlog"] = mod
    return mod


def _psycopg2_module() -> types.ModuleType:
    """A `psycopg2` whose only entry point refuses, loudly.

    `main.py` touches the driver in exactly one place (`psycopg2.connect` inside
    its admin-connection helper). No test in this suite may reach a database,
    so reaching it is a test defect and is reported as one — the same rule the
    fake Docker client applies to `containers.run`.
    """
    mod = types.ModuleType("psycopg2")

    class Error(Exception):
        pass

    class OperationalError(Error):
        pass

    def _connect(*a, **k):
        raise AssertionError("test reached psycopg2.connect")

    mod.Error = Error
    mod.OperationalError = OperationalError
    mod.connect = _connect
    return mod


def load_bridge_main() -> types.ModuleType:
    """A fresh `bridge/main.py`, loaded under stubs. Safe to call per-test."""
    _docker_module()
    _structlog_module()
    os.environ.setdefault("BRIDGE_PG_ADMIN_URL", "postgres://t:t@127.0.0.1:5432/postgres")
    os.environ.setdefault("BRIDGE_DOCKER_IMAGE", "ghcr.io/toup-com/toup-agent:test")

    name = f"bridge_main_uut_{next(_counter)}"
    spec = importlib.util.spec_from_file_location(name, BRIDGE_MAIN)
    mod = importlib.util.module_from_spec(spec)
    # `main.py` ends with `from pool_addon import attach_pool_routes` and
    # `import tenant_health`, both wrapped in try/except — pool_addon would pull
    # the whole pool stack in, so block it and let the except branch log.
    sys.modules[name] = mod
    blocked = "pool_addon" not in sys.modules
    if blocked:
        sys.modules["pool_addon"] = None  # type: ignore[assignment]
    # See the module docstring: the stub is swapped in for the load only, and
    # whatever the process had (a real driver, or nothing) is restored after.
    had_psycopg2 = "psycopg2" in sys.modules
    real_psycopg2 = sys.modules.get("psycopg2")
    sys.modules["psycopg2"] = _psycopg2_module()
    # Module scope creates /var/log/toup-bridge for the audit trail. Not
    # writable here and not what any of these tests are about.
    real_mkdir = pathlib.Path.mkdir

    def _tolerant_mkdir(self, *a, **k):
        try:
            return real_mkdir(self, *a, **k)
        except (PermissionError, OSError):
            return None

    pathlib.Path.mkdir = _tolerant_mkdir  # type: ignore[method-assign]
    try:
        spec.loader.exec_module(mod)
    finally:
        pathlib.Path.mkdir = real_mkdir  # type: ignore[method-assign]
        if blocked:
            sys.modules.pop("pool_addon", None)
        if had_psycopg2:
            sys.modules["psycopg2"] = real_psycopg2  # type: ignore[assignment]
        else:
            sys.modules.pop("psycopg2", None)
    return mod
