"""The transcript lane rides out a tenant outage as a LANE (P3, 2026-10-02).

Production, founder tenant 871bac24, 02:29:47-02:34:42 UTC: the tenant's event
loop stalled (every write timed out at 15 s), then the container was restarted
in place and the edge answered 502 while it booted. The relay's lane gave every
row its own retry budget and spent them one row at a time — three 15 s
ReadTimeouts per row while the call was live, a 12-attempt 502 ladder per row
after hang-up — and its finisher stopped at a fixed 120 s from hang-up, 32 s
before the tenant came back: 13 LOST verdicts for one call.

Every fake here sits on the SHARED agent HTTP client, under the real
`_save_voice_messages` and `_vps_api`, and answers the lane's probe as well as
its writes. Clocks are the production ones at 1/100 scale (the suite's
fast-clock pattern): the fake holds a stalled request for the caller's real
timeout times `SCALE`, and the lane's own windows are scaled to match, so a
two-minute outage is about a second of test.
"""

import asyncio
import logging
import time
import types
from datetime import datetime, timedelta, timezone

import httpx
import pytest

import test_live_harness as H


SCALE = 0.01
T0 = datetime(2026, 10, 3, 2, 29, 47, tzinfo=timezone.utc)


class _Resp:
    def __init__(self, status, body=b'{"id": "m1"}'):
        self.status_code = status
        self.content = body

    def json(self):
        import json

        return json.loads(self.content)


class _Tenant:
    """One tenant behind its edge, on the shared agent HTTP client.

    `windows` are (start, end, mode) in wall-clock seconds from creation:
    "edge" answers 502 to everything (the edge for a container that is
    restarting); "stall" holds every request for the caller's FULL timeout
    (scaled) and then raises ReadTimeout (an event loop that stopped
    answering). `down_after_writes` takes the tenant away for good after that
    many landed writes. `slow_contents` are rows whose write always stalls
    while everything else, the probe included, answers. `write_script(body)`
    may return a status for one write. Writes upsert by client_msg_id and keep
    `occurred_at` unless the body sends it (sessions.py).
    """

    def __init__(self, *, latency=0.0):
        self.t0 = time.monotonic()
        self.latency = latency
        self.windows: list[tuple[float, float, str]] = []
        self.down_after_writes = None
        self.slow_contents: set[str] = set()
        self.write_script = None
        self.requests: list[dict] = []
        self.rows: dict[str, dict] = {}
        self.landed = 0
        self.last_landed_at = None

    def now(self):
        return time.monotonic() - self.t0

    def mode(self):
        if self.down_after_writes is not None and self.landed >= self.down_after_writes:
            return "edge"
        t = self.now()
        for start, end, mode in self.windows:
            if start <= t < end:
                return mode
        return None

    async def _away(self, mode, timeout):
        if mode == "edge":
            return _Resp(502, b"<html>502 Bad Gateway</html>")
        await asyncio.sleep(float(timeout) * SCALE)
        raise httpx.ReadTimeout("")

    async def post(self, url, **kw):
        body = dict(kw.get("json") or {})
        timeout = kw.get("timeout")
        entry = {
            "method": "POST", "t": round(self.now(), 3), "timeout": timeout,
            "content": body.get("content"), "revision": body.get("revision"),
            "spoken": (body.get("voice") or {}).get("spoken"),
        }
        self.requests.append(entry)
        mode = self.mode() or ("stall" if body.get("content") in self.slow_contents else None)
        if mode:
            entry["outcome"] = mode
            return await self._away(mode, timeout)
        status = self.write_script(body) if self.write_script else None
        await asyncio.sleep(self.latency)
        if status is not None:
            entry["outcome"] = status
            return _Resp(status)
        row = self.rows.setdefault(body.get("client_msg_id"), {})
        row.update({k: v for k, v in body.items() if k != "occurred_at"})
        if "occurred_at" in body:
            row["occurred_at"] = body["occurred_at"]
        self.landed += 1
        self.last_landed_at = time.monotonic()
        entry["outcome"] = 201
        return _Resp(201)

    async def get(self, url, **kw):
        timeout = kw.get("timeout")
        entry = {"method": "GET", "t": round(self.now(), 3), "timeout": timeout,
                 "url": url, "params": dict(kw.get("params") or {}),
                 "headers": dict(kw.get("headers") or {})}
        self.requests.append(entry)
        mode = self.mode()
        if mode:
            entry["outcome"] = mode
            return await self._away(mode, timeout)
        entry["outcome"] = 200
        return _Resp(200, b'{"id": "db-session"}')

    def writes(self, content=None):
        return [
            r for r in self.requests
            if r["method"] == "POST" and (content is None or r["content"] == content)
        ]

    def probes(self):
        return [r for r in self.requests if r["method"] == "GET"]


def _install(monkeypatch, **kw):
    from app.api import ws_realtime as rt
    import app.services.agent_http as ah

    tenant = _Tenant(**kw)

    async def vps(_uid):
        return ("http://agent.test", "k")

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(ah, "get_agent_http_client", lambda: tenant)
    return tenant


def _lane(monkeypatch):
    """A transcript lane on the production clocks at 1/100 scale: backoff
    0.5…8 s, the 30 s per-row window and the 600 s lane budget. The away
    timeout stays 5 s on purpose: the fake turns timeouts into time."""

    import app.services.live_voice_protocol as live

    monkeypatch.setattr(
        live.PersistenceQueue, "BACKOFF_S", tuple(b * SCALE for b in (0.5, 1.0, 2.0, 4.0, 8.0)),
    )
    queue = live.PersistenceQueue("user-1", "db-session")
    queue.retry_window_s = 30.0 * SCALE
    queue.lane_deadline_s = 600.0 * SCALE
    # getattr: the helper must also drive a queue that predates the lane
    # state, so these tests fail there on behaviour rather than on setup.
    assert getattr(queue, "away_timeout_s", 5.0) == 5.0
    return queue


def _row(i):
    import app.services.live_voice_protocol as live

    key = f"live-utt:live_u2_EUjmDp3oNPqa8sGGsyFvQ6Ibn1F4cfwB:{i}"
    return live.PersistenceRecord(kind="message", key=key, payload={
        "user_text": f"sentence {i}", "assistant_text": "", "user_ref": key,
        "user_revision": 1, "user_occurred_at": T0 + timedelta(seconds=i),
    })


def _names(counters):
    return [name for name, _fields in counters]


# ── logs name the row ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_lost_and_retry_lines_name_the_rows_key_tail(monkeypatch, caplog):
    """`key[:48]` printed only the shared `live-output:<psid>` prefix: every
    row of the incident call logged the same key. The tail — the epoch, the
    ordinal, the delegation id — is what tells the rows apart."""

    import app.services.live_voice_protocol as live

    caplog.set_level(logging.INFO)
    tenant = _install(monkeypatch)
    tenant.write_script = lambda body: 500
    monkeypatch.setattr(live.PersistenceQueue, "BACKOFF_S", (0.0,))
    key = "live-output:live_u2_EUjmDp3oNPqa8sGGsyFvQ6Ibn1F4cfwB:7"
    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    queue.submit(live.PersistenceRecord(kind="message", key=key, payload={
        "user_text": "", "assistant_text": "program.", "assistant_ref": key,
    }))
    assert await queue.drain(2.0)
    await queue.stop()

    tail = live._key_tail(key)
    assert tail == key[-24:] and tail.endswith(":7")
    lines = [r.getMessage() for r in caplog.records
             if "persistence LOST" in r.getMessage() or "persistence attempt" in r.getMessage()]
    assert len(lines) == 3, lines                 # two retries, one verdict
    # Integration with PR-1 (same sites, same helper): the existing `key=` head
    # stays where it was (additive) and every line ENDS with the row's tail —
    # the part that tells this row from the call's other rows.
    assert all(line.endswith(f"key_tail={tail}") for line in lines), lines
    assert all(f"key={key[:48]} " in line for line in lines), lines
    assert not [line for line in lines if "program." in line], "ids only, never words"


# ── the lane, not the row, owns an outage ─────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["live", "after_hangup"])
async def test_a_backlog_survives_an_outage_longer_than_one_rows_ladder(monkeypatch, phase):
    """Ten rows; the edge answers 502 for longer than TWO rows' ladders — the
    30 s window per row while the call is live, the 12-attempt ladder per row
    the finisher used to burn after hang-up. All ten land and none is reported
    lost: only the head row was retried while the tenant was away, without
    being charged, and the rows behind it spent nothing waiting."""

    import app.services.live_voice_protocol as live

    counters = H.capture_counters(monkeypatch)
    tenant = _install(monkeypatch)
    tenant.windows.append((0.0, 1.5, "edge"))
    queue = _lane(monkeypatch)
    queue.start()
    for i in range(10):
        queue.submit(_row(i))
    if phase == "live":
        assert await queue.drain(6.0)
        await queue.stop()
    else:
        await live._finish_detached(queue, [], 0.01, 300.0 * SCALE)

    assert queue.written == 10 and queue.lost == 0, tenant.requests[-5:]
    assert len(tenant.rows) == 10
    names = _names(counters)
    assert "live_persist_lost" not in names and "voice_transcript_lost" not in names
    assert names.count("live_persist_tenant_away") == 1
    assert names.count("live_persist_tenant_back") == 1
    down = [r for r in tenant.writes() if r["outcome"] == "edge"]
    assert len(down) > 3 and {r["content"] for r in down} == {"sentence 0"}, (
        "only the head row may be retried while the tenant is away"
    )


@pytest.mark.asyncio
async def test_a_slow_read_timeout_row_lands(monkeypatch):
    """The stall itself (02:29:47-02:31:51): every request is held for its
    whole timeout and then times out. The old lane spent 15 s per attempt and
    reported the row LOST at attempts=3 (47 s > its 30 s window) while the
    call queued behind it. Now the probe confirms the tenant is away, the row
    is retried with the short timeout without being charged, and it lands
    when the tenant answers again."""

    counters = H.capture_counters(monkeypatch)
    tenant = _install(monkeypatch)
    tenant.windows.append((0.0, 1.0, "stall"))
    queue = _lane(monkeypatch)
    queue.start()
    queue.submit(_row(0))
    assert await queue.drain(5.0)
    await queue.stop()

    assert queue.written == 1 and queue.lost == 0, tenant.requests
    assert "live_persist_lost" not in _names(counters)
    writes = tenant.writes()
    assert len(writes) > 3, "more attempts than the row's own ladder allows"
    # The first attempt had no reason to expect an outage; once the probe
    # said the tenant was away, the next one used the short timeout.
    assert writes[0]["timeout"] == 15.0 and writes[1]["timeout"] == 5.0, writes
    assert tenant.probes() and {p["timeout"] for p in tenant.probes()} == {5.0}


@pytest.mark.asyncio
async def test_a_merely_slow_tenant_still_lands_its_rows_while_away(monkeypatch):
    """The short timeout must not lock out a tenant that is slow rather than
    gone (booting under load, SLOW_SQL 9.5 s after the restart): it took the
    connection and did not answer in 5 s, so every other attempt gets the
    full timeout — and a 9.5 s write lands on one of those. (A guard on the
    new short timeout: the old lane, at 15 s per attempt, passes too.)"""

    tenant = _install(monkeypatch)
    tenant.windows.append((0.0, 0.3, "stall"))
    queue = _lane(monkeypatch)
    real_post = tenant.post

    async def slow_post(url, **kw):
        # After the stall: alive, but every write takes 9.5 s (scaled).
        if tenant.mode() is None:
            if float(kw.get("timeout") or 15.0) < 9.5:
                tenant.requests.append({"method": "POST", "t": round(tenant.now(), 3),
                                        "timeout": kw.get("timeout"), "content": None,
                                        "outcome": "slow"})
                await asyncio.sleep(float(kw["timeout"]) * SCALE)
                raise httpx.ReadTimeout("")
            await asyncio.sleep(9.5 * SCALE)
        return await real_post(url, **kw)

    async def slow_get(url, **kw):
        if tenant.mode() is None:              # the probe is slow too
            await asyncio.sleep(float(kw["timeout"]) * SCALE)
            raise httpx.ReadTimeout("")
        return await _Tenant.get(tenant, url, **kw)

    tenant.post = slow_post
    tenant.get = slow_get
    queue.start()
    queue.submit(_row(0))
    assert await queue.drain(5.0)
    await queue.stop()
    assert queue.written == 1 and queue.lost == 0, tenant.requests
    assert tenant.landed == 1


@pytest.mark.asyncio
async def test_a_slow_row_cannot_hold_the_lane_while_the_tenant_answers(monkeypatch):
    """The probe is what keeps one slow row from being mistaken for an outage:
    its write always times out while the tenant answers everything else. It
    keeps the ladder it always had — three full-timeout attempts, one LOST
    verdict — the rows behind it land, and the lane never calls the tenant
    away. (A guard on the new state: this passes on the old lane too.)"""

    counters = H.capture_counters(monkeypatch)
    tenant = _install(monkeypatch)
    tenant.slow_contents = {"sentence 0"}
    queue = _lane(monkeypatch)
    queue.start()
    for i in range(4):
        queue.submit(_row(i))
    assert await queue.drain(5.0)
    await queue.stop()

    assert queue.lost == 1 and queue.written == 3
    slow = tenant.writes("sentence 0")
    assert len(slow) == 3 and {r["timeout"] for r in slow} == {15.0}, slow
    assert "live_persist_tenant_away" not in _names(counters)


@pytest.mark.asyncio
async def test_an_outage_past_the_lane_deadline_is_still_lost_once_per_row(monkeypatch, caplog):
    """The lane's budget is finite: past it the LOST verdict is still reached,
    once per row, and the rows behind the head fail fast instead of each
    spending a ladder. When the tenant answers again the lane is back."""

    counters = H.capture_counters(monkeypatch)
    caplog.set_level(logging.INFO)
    tenant = _install(monkeypatch)
    tenant.windows.append((0.0, 1.2, "edge"))
    queue = _lane(monkeypatch)
    queue.lane_deadline_s = 0.4
    queue.start()
    for i in range(3):
        queue.submit(_row(i))
    started = time.monotonic()
    assert await queue.drain(3.0)
    elapsed = time.monotonic() - started

    assert queue.lost == 3 and queue.written == 0
    reasons = [f.get("reason") for name, f in counters if name == "live_persist_lost"]
    assert reasons == ["exhausted"] * 3
    assert len(tenant.writes("sentence 0")) > 3, "the head was retried, uncharged, to the deadline"
    assert len(tenant.writes("sentence 1")) == 1 and len(tenant.writes("sentence 2")) == 1
    assert 0.35 <= elapsed < 0.8, elapsed
    lost_lines = [r.getMessage() for r in caplog.records
                  if r.levelno >= logging.ERROR and "persistence LOST" in r.getMessage()]
    assert len(lost_lines) == 3 and "lane_deadline_s=" in lost_lines[0], lost_lines

    await asyncio.sleep(max(0.0, 1.2 - tenant.now()) + 0.02)
    queue.submit(_row(3))
    assert await queue.drain(2.0)
    await queue.stop()
    assert queue.written == 1 and queue.away_since is None
    assert "live_persist_tenant_back" in _names(counters)


# ── the probe ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(("answer", "expected", "verdict"), [
    (200, {"status": 200}, "up"),
    (404, {"status": 404}, "up"),
    (500, {"status": 500}, "up"),
    (502, {"status": 502}, "away"),
    (503, {"status": 503}, "away"),
    (httpx.ReadTimeout(""), {"error": "ReadTimeout", "transport": True}, "away"),
    (httpx.ConnectError("refused"), {"error": "ConnectError", "transport": True}, "away"),
    (AttributeError("no get"), {"error": "AttributeError"}, "unknown"),
])
async def test_the_probe_reads_no_body_and_says_what_happened(
    monkeypatch, answer, expected, verdict,
):
    import app.services.live_voice_protocol as live
    from app.api import ws_realtime as rt
    import app.services.agent_http as ah

    seen: list[tuple] = []

    class _NoBody(_Resp):
        def json(self):                            # pragma: no cover - must not run
            raise AssertionError("the probe must never read the body")

    class _Client:
        async def get(self, url, **kw):
            seen.append((url, kw))
            if isinstance(answer, BaseException):
                raise answer
            return _NoBody(answer)

    async def vps(_uid):
        return ("http://agent.test", "agent-key")

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(ah, "get_agent_http_client", lambda: _Client())
    outcome = await rt._probe_voice_session("user-1", "sess-1", timeout=5.0)
    assert outcome == expected
    assert live._probe_verdict(outcome) == verdict
    ((url, kw),) = seen
    assert url == "http://agent.test/api/sessions/sess-1"
    assert kw["params"] == {"include_messages": "false"}
    assert kw["headers"] == {"X-Agent-Key": "agent-key"} and kw["timeout"] == 5.0


@pytest.mark.asyncio
async def test_a_probe_with_no_tenant_address_is_no_evidence(monkeypatch):
    import app.services.live_voice_protocol as live
    from app.api import ws_realtime as rt

    async def none(_uid):
        return None

    monkeypatch.setattr(rt, "_get_vps_info", none)
    outcome = await rt._probe_voice_session("user-1", "sess-1", timeout=5.0)
    assert outcome == {"error": "vps_unavailable"}
    assert live._probe_verdict(outcome) == "unknown"


def test_the_lane_settings_are_real():
    """Declared settings: a defaulted lookup of a name `Settings` never
    declares is silently the literal forever."""

    import app.services.live_voice_protocol as live
    from app.config import Settings, settings

    fields = Settings.model_fields
    assert fields["voice_live_persist_lane_deadline_s"].default == 600.0
    assert fields["voice_live_persist_away_timeout_s"].default == 5.0
    assert fields["voice_live_persist_retry_window_s"].default == 30.0
    queue = live.PersistenceQueue("user-1", "db-session")
    assert queue.lane_deadline_s == float(settings.voice_live_persist_lane_deadline_s)
    assert queue.away_timeout_s == float(settings.voice_live_persist_away_timeout_s)


# ── a flag-only revision never holds new rows ─────────────────────────

def _delegated(ref):
    return {
        "user_text": "", "assistant_text": "I couldn't find specific information.",
        "model": "m", "assistant_ref": ref, "assistant_occurred_at": T0,
        "assistant_voice": {"source": "delegated", "delegation_id": "d1"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["queued", "in_flight"])
async def test_flag_only_revision_does_not_block_new_rows(monkeypatch, when):
    """02:29:47: the delivery verdict (spoken=false) of an answer that had
    ALREADY landed went to the head of the lane, failed for 47 s, and held the
    call's next rows behind it. A revision that only flips a flag on a landed
    row now waits behind rows that carry content — queued behind them, or
    stepping aside when it fails in flight — and still lands, on its row, in
    its place."""

    import app.services.live_voice_protocol as live
    from app.api import ws_realtime as rt

    tenant = _install(monkeypatch, latency=0.02)
    flag_writes = {"n": 0}

    def script(body):
        if (body.get("voice") or {}).get("spoken") is False:
            flag_writes["n"] += 1
            if flag_writes["n"] <= 2:
                return 500                     # the row's own problem: charged
        return None

    tenant.write_script = script
    monkeypatch.setattr(live.PersistenceQueue, "BACKOFF_S", (0.01,))
    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    ref = "live-delegation:live_u2_EUjmDp3oNPqa8sGGsyFvQ6Ibn1F4cfwB:item_EUjmcVJ"
    first = _delegated(ref)
    queue.submit(live.PersistenceRecord(kind="message", key=ref, payload=dict(first)))
    assert await queue.drain(2.0)                 # the answer has landed

    session = types.SimpleNamespace(persistence=queue)
    delivery = types.SimpleNamespace(claim={"ref": ref, "revision": 1, "payload": dict(first)})
    await live._LiveSession.write_spoken_flag(session, delivery, False)
    if when == "in_flight":
        await asyncio.sleep(0.005)                # the flag revision is on the wire
    queue.submit(_row(1))
    queue.submit(_row(2))
    assert await queue.drain(3.0)
    await queue.stop()

    order = ["flag" if r["spoken"] is False else r["content"] for r in tenant.writes()]
    answer = first["assistant_text"]
    if when == "queued":
        assert order[:3] == [answer, "sentence 1", "sentence 2"], order
    else:
        assert order[:4] == [answer, "flag", "sentence 1", "sentence 2"], order
    assert order.count("flag") == 3, order        # still its three attempts
    assert queue.lost == 0 and queue.written == 4
    row = tenant.rows[rt.voice_client_msg_id("db-session", ref)]
    assert row["voice"]["spoken"] is False and row["revision"] == 2
    assert row["occurred_at"] == T0.isoformat(), "the flag revision never moves its row"


@pytest.mark.asyncio
async def test_a_flag_revision_of_a_row_that_never_landed_keeps_its_place(monkeypatch):
    """The other half of the rule: a flag revision whose row has NOT landed may
    have to create it, so it is never passed over — it goes in order, carrying
    the first write's stamp (F34)."""

    import app.services.live_voice_protocol as live
    from app.api import ws_realtime as rt

    tenant = _install(monkeypatch, latency=0.01)
    monkeypatch.setattr(live.PersistenceQueue, "BACKOFF_S", (0.0,))
    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    ref = "live-delegation:p:d9"
    session = types.SimpleNamespace(persistence=queue)
    queue.submit(_row(0))
    await asyncio.sleep(0.003)                    # the lane is busy from here
    first = _delegated(ref)
    queue.submit(live.PersistenceRecord(kind="message", key=ref, payload=dict(first)))
    delivery = types.SimpleNamespace(claim={"ref": ref, "revision": 1, "payload": dict(first)})
    await live._LiveSession.write_spoken_flag(session, delivery, False)
    queue.submit(_row(1))
    assert await queue.drain(2.0)
    await queue.stop()

    order = ["flag" if r["spoken"] is False else r["content"] for r in tenant.writes()]
    # The waiting first write and its flag revision coalesced into one write
    # (P2) that keeps its queue position — ahead of the row submitted after it.
    assert order == ["sentence 0", "flag", "sentence 1"], order
    row = tenant.rows[rt.voice_client_msg_id("db-session", ref)]
    assert row["occurred_at"] == T0.isoformat() and row["voice"]["spoken"] is False


# ── the finisher's budget counts from the lane's last progress ────────

@pytest.mark.asyncio
async def test_finisher_budget_counts_from_last_success(monkeypatch):
    """The budget is the time the TENANT has to come back, so it counts from
    the lane's last landed write, not from hang-up: a tenant already silent
    for most of it at hang-up does not get all of it again, and
    `teardown_budget` still counts what is left, once per row."""

    import app.services.live_voice_protocol as live

    counters = H.capture_counters(monkeypatch)
    tenant = _install(monkeypatch, latency=0.01)
    tenant.down_after_writes = 1                  # the first row lands, then away
    queue = _lane(monkeypatch)
    queue.start()
    for i in range(4):
        queue.submit(_row(i))
    while tenant.last_landed_at is None:
        await asyncio.sleep(0.005)
    # Hang-up 0.35 s after the last write that landed; the budget is 0.5 s.
    await asyncio.sleep(max(0.0, tenant.last_landed_at + 0.35 - time.monotonic()))
    started = time.monotonic()
    await live._finish_detached(queue, [], 0.01, 0.5)
    elapsed = time.monotonic() - started

    assert elapsed < 0.35, f"the finisher waited {elapsed:.2f}s after hang-up"
    assert queue.written == 1 and queue.lost == 3
    reasons = [f.get("reason") for name, f in counters if name == "live_persist_lost"]
    assert reasons == ["teardown_budget"] * 3, reasons


@pytest.mark.asyncio
async def test_finisher_budget_moves_with_every_landed_write(monkeypatch):
    """…and a tenant that keeps taking rows keeps the finisher going: a slow
    tenant drains a backlog far longer than the budget, because every landed
    write restarts the count. A fixed budget from hang-up cut it off mid-way
    and counted the rest lost."""

    import app.services.live_voice_protocol as live

    tenant = _install(monkeypatch, latency=0.05)
    queue = _lane(monkeypatch)
    queue.start()
    for i in range(12):
        queue.submit(_row(i))
    await live._finish_detached(queue, [], 0.01, 0.2)
    assert queue.written == 12 and queue.lost == 0, len(tenant.rows)


def test_the_finisher_budget_is_sized_for_a_restart():
    """…and a budget that covers the call that needed one: 02:29:47 → 02:34:42
    is 295 s, and 120 s gave up 32 s before the tenant came back."""

    from app.config import Settings

    default = Settings.model_fields["voice_live_persist_background_drain_s"].default
    assert default == 420.0 and default >= 295


# ── tunnel hints: timing only, never a verdict ────────────────────────

@pytest.mark.asyncio
async def test_a_tunnel_reconnect_retries_the_head_row_now(monkeypatch):
    """The tenant's tunnel coming back on this replica cuts a tenant-away
    backoff short, so the head row is retried at once instead of up to 8 s
    later. Rows never travel the tunnel, so it is a wake-up, nothing more."""

    import app.services.live_voice_protocol as live

    tenant = _install(monkeypatch)
    tenant.windows.append((0.0, 0.2, "edge"))
    queue = _lane(monkeypatch)
    monkeypatch.setattr(live.PersistenceQueue, "BACKOFF_S", (5.0,))   # a long wait
    queue.start()
    queue.submit(_row(0))
    while queue.away_since is None:
        await asyncio.sleep(0.005)
    await asyncio.sleep(max(0.0, 0.25 - tenant.now()))
    woke = time.monotonic()
    live.note_tenant_link("user-1", up=True)
    assert await queue.drain(1.0), "the reconnect did not wake the lane"
    assert time.monotonic() - woke < 0.5
    await queue.stop()
    assert queue.written == 1
    live.note_tenant_link("user-1", up=True)       # a stopped lane is gone: no-op


@pytest.mark.asyncio
async def test_a_tunnel_drop_makes_only_the_next_attempt_short(monkeypatch):
    import app.services.live_voice_protocol as live

    tenant = _install(monkeypatch)
    queue = _lane(monkeypatch)
    queue.start()
    live.note_tenant_link("user-1", up=False)
    live.note_tenant_link("someone-else", up=False)
    queue.submit(_row(0))
    assert await queue.drain(1.0)
    queue.submit(_row(1))
    assert await queue.drain(1.0)
    await queue.stop()
    assert [w["timeout"] for w in tenant.writes()] == [5.0, 15.0]
    assert queue.away_since is None


class _EndpointWebSocket:
    def __init__(self):
        self.scope = {"subprotocols": []}
        self.query_params = {"token": "opaque"}
        self.sent: list[dict] = []
        self.hang_up = asyncio.Event()

    async def accept(self, **kwargs):
        return None

    async def send_json(self, value):
        self.sent.append(value)

    async def close(self, code=1000):
        return None

    async def receive_text(self):
        from fastapi import WebSocketDisconnect

        await self.hang_up.wait()
        raise WebSocketDisconnect


@pytest.mark.asyncio
async def test_the_tunnel_tells_the_lanes_when_it_comes_and_goes(monkeypatch):
    import app.services.live_voice_protocol as live
    from app.api import ws_agent_tunnel as tunnel

    hints: list[tuple] = []
    monkeypatch.setattr(live, "note_tenant_link", lambda uid, *, up: hints.append((uid, up)))

    async def authenticate(_token):
        return "tenant-user"

    async def config_ok(_tunnel):
        return True

    monkeypatch.setattr(tunnel, "_authenticate_tunnel", authenticate)
    monkeypatch.setattr(tunnel, "_push_latest_config", config_ok)
    monkeypatch.setattr(tunnel, "_tunnels", {})
    monkeypatch.setattr(tunnel, "_pending_calls", {})
    ws = _EndpointWebSocket()
    task = asyncio.create_task(tunnel.agent_tunnel_ws(ws, token="opaque"))
    for _ in range(100):
        if tunnel.is_agent_connected("tenant-user"):
            break
        await asyncio.sleep(0.005)
    assert hints == [("tenant-user", True)]

    # A forward that times out says the tenant may be stalled.
    assert await tunnel.send_http_forward(
        "tenant-user", "GET", "/api/apps/", timeout=0.01,
    ) is None
    assert hints[-1] == ("tenant-user", False)

    ws.hang_up.set()
    await asyncio.wait_for(task, timeout=1.0)
    assert hints[-1] == ("tenant-user", False) and len(hints) == 3
    assert tunnel.get_tunnel("tenant-user") is None
