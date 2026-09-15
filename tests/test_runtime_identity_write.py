"""`write_runtime` must never rename a truncated file over runtime.json.

The temp path was a fixed `runtime.json.tmp`, shared by every writer.
`write_text` opens it with O_TRUNC, so there is a window in which that file is
EMPTY; another writer's `os.replace` landing inside that window renames an
empty file into place. `_load_from_disk` soft-fails to None, `is_bound()` goes
False, and `LobbyAndDrainMiddleware` 503s every route with `X-Lobby-Mode: 1`
on a container that still reports healthy.

Reachability, stated honestly: NOT reachable in the deployed configuration.
`admin_bind` is `async def`, it is `write_runtime`'s only backend caller, it
calls it synchronously, there is no `await` between the write and the rename,
and the container runs one uvicorn worker — so two concurrent binds cannot
interleave. Loki confirms: zero "Failed to read /etc/toup-agent" lines over
11-13 Sep. This is hardening for the day one of those premises stops holding,
which is why the first test drives real concurrency through a thread pool
(where an `asyncio.Lock` would NOT have helped and the unique temp does).
"""
from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.services import runtime_identity as ri


@pytest.fixture
def runtime_path(tmp_path, monkeypatch):
    p = tmp_path / "runtime.json"
    monkeypatch.setattr(ri, "RUNTIME_PATH", p)
    monkeypatch.setattr(ri, "_runtime", None, raising=False)
    monkeypatch.setattr(ri, "_runtime_loaded", False, raising=False)
    return p


def _payload(uid: str) -> dict:
    # ~600 bytes, the shape the bridge actually sends.
    return {
        "user_id": uid,
        "agent_api_key": "k" * 43,
        "user_name": "Mobin Moosapoor",
        "user_email": f"{uid}@privaterelay.appleid.com",
        "connect_token": "c" * 40,
        "llm_mode": "managed",
        "whatsapp_mode": "baileys",
    }


def test_concurrent_writers_never_leave_a_corrupt_runtime_json(runtime_path):
    """Hammer it from real threads. Every observed state of the file must be
    valid JSON naming one of the two tenants — never empty, never partial."""
    a, b = _payload("aaaaaaaa"), _payload("bbbbbbbb")
    seen: list[str] = []
    errors: list[BaseException] = []

    def write(p):
        for _ in range(60):
            try:
                ri.write_runtime(p)
            except BaseException as e:  # noqa: BLE001 - recorded, then asserted
                errors.append(e)
                return

    def read():
        for _ in range(400):
            if not runtime_path.exists():
                continue
            raw = runtime_path.read_text("utf-8")
            if raw == "":
                seen.append("EMPTY")
                continue
            try:
                seen.append(json.loads(raw)["user_id"])
            except (json.JSONDecodeError, KeyError):
                seen.append("CORRUPT:" + raw[:40])

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(lambda f: f(), [lambda: write(a), lambda: write(b), read]))

    assert not errors, f"write_runtime raised under concurrency: {errors[:3]}"
    bad = [s for s in seen if s not in ("aaaaaaaa", "bbbbbbbb")]
    assert not bad, f"runtime.json was observed corrupt/empty: {bad[:5]}"
    assert seen, "the reader never observed the file at all"

    # And the file that survived is one of the two payloads, intact.
    final = json.loads(runtime_path.read_text("utf-8"))
    assert final in (a, b)


def test_the_temp_path_is_unique_per_write(runtime_path):
    """The mechanism, asserted directly: no two writes may share a temp name.

    A fixed `runtime.json.tmp` passes every single-threaded test ever written,
    which is why this one names the invariant rather than the symptom.
    """
    names: list[str] = []
    real_mkstemp = ri.tempfile.mkstemp

    def spy(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        names.append(name)
        return fd, name

    ri.tempfile.mkstemp = spy
    try:
        for i in range(5):
            ri.write_runtime(_payload(f"user{i:04d}"))
    finally:
        ri.tempfile.mkstemp = real_mkstemp

    assert len(names) == 5, "write_runtime no longer goes through mkstemp"
    assert len(set(names)) == 5, f"temp path reused across writes: {names}"
    assert str(runtime_path) not in names


def test_no_temp_files_are_left_behind(runtime_path):
    for i in range(4):
        ri.write_runtime(_payload(f"user{i:04d}"))
    leftovers = [p.name for p in runtime_path.parent.iterdir() if p.name != "runtime.json"]
    assert leftovers == [], f"temp files left in the runtime dir: {leftovers}"


def test_a_failed_write_leaves_no_temp_and_does_not_clobber(runtime_path):
    """A doomed write must not take the previous identity down with it."""
    ri.write_runtime(_payload("good0000"))

    class Unserialisable:
        pass

    with pytest.raises(TypeError):
        ri.write_runtime({"user_id": "bad00000", "x": Unserialisable()})

    assert json.loads(runtime_path.read_text("utf-8"))["user_id"] == "good0000"
    leftovers = [p.name for p in runtime_path.parent.iterdir() if p.name != "runtime.json"]
    assert leftovers == [], f"a failed write left {leftovers}"


def test_write_runtime_still_rejects_a_payload_with_no_user_id(runtime_path):
    with pytest.raises(ValueError):
        ri.write_runtime({"agent_api_key": "k"})
    assert not runtime_path.exists()


def test_the_file_is_owner_readable_only(runtime_path):
    import stat

    ri.write_runtime(_payload("perm0000"))
    mode = stat.S_IMODE(runtime_path.stat().st_mode)
    assert mode == 0o600, f"runtime.json holds an api key at mode {oct(mode)}"


# ── The bind handler's identity section is serialised ────────────────────

@pytest.mark.asyncio
async def test_the_bind_lock_is_rebuilt_when_the_event_loop_changes():
    """An `asyncio.Lock` binds to the loop it is first awaited on. Caching one
    across loops would wedge every bind after a loop swap."""
    from app.api import admin_pool

    admin_pool._BIND_LOCK = None
    lock1 = admin_pool._bind_lock()
    assert admin_pool._bind_lock() is lock1, "a fresh lock per call serialises nothing"

    # A different loop must get a different lock. `asyncio.run` cannot nest
    # inside a running loop, so the second loop runs on its own thread.
    holder: dict = {}

    def in_its_own_loop():
        async def other():
            holder["lock"] = admin_pool._bind_lock()

        asyncio.run(other())

    with ThreadPoolExecutor(max_workers=1) as ex:
        ex.submit(in_its_own_loop).result()
    assert holder["lock"] is not lock1


@pytest.mark.asyncio
async def test_the_identity_section_of_bind_is_mutually_exclusive():
    """Two binds may not interleave settings-apply and runtime.json.

    Asserted by making the write itself yield: without the lock the second
    bind's apply_to_settings lands between the first bind's apply and its
    write, which is the "container bound to a mix" state.
    """
    from app.api import admin_pool

    admin_pool._BIND_LOCK = None
    order: list[str] = []

    def fake_apply(fields):
        order.append(f"apply:{fields['user_id']}")
        return len(fields)

    def fake_write(fields):
        order.append(f"write:{fields['user_id']}")

    async def one(uid: str):
        async with admin_pool._bind_lock():
            fake_apply({"user_id": uid})
            await asyncio.sleep(0)  # the yield a real handler could acquire
            fake_write({"user_id": uid})

    await asyncio.gather(one("aaa"), one("bbb"))

    assert order in (
        ["apply:aaa", "write:aaa", "apply:bbb", "write:bbb"],
        ["apply:bbb", "write:bbb", "apply:aaa", "write:aaa"],
    ), f"the two binds interleaved: {order}"


def test_bind_holds_the_lock_over_steps_1_to_2_but_not_over_mcp():
    """Scope probe. A lock spanning step 2d's network call would let one slow
    platform round-trip wedge every future bind on this container."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1].joinpath(
        "app/api/admin_pool.py"
    ).read_text()
    at = src.find("async with _bind_lock():")
    assert at > 0, "the bind lock is gone"
    tail = src[at:]

    apply_at = tail.find("runtime_identity.apply_to_settings")
    write_at = tail.find("runtime_identity.write_runtime")
    mcp_at = tail.find("ensure_mcp_initialized")
    assert 0 < apply_at < write_at, "steps 1-2 are not both inside the lock"

    # Everything inside the `async with` is indented past the handler body's
    # 4 spaces; the first line back at 4 spaces ends the block.
    block_end = len(tail)
    for line in tail.splitlines(keepends=True)[1:]:
        idx = tail.find(line, 0)
        if line.strip() and not line.startswith("        "):
            block_end = tail.find(line)
            break
    assert write_at < block_end, "write_runtime fell out of the lock"
    assert mcp_at > block_end, "the lock spans the MCP bootstrap's network call"
