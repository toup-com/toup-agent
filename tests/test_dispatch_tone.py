"""The dispatch tone — one catalogue, carried end to end, silent nowhere.

An operator picks the sound a dispatch arrives with. Every way of getting
that wrong is INAUDIBLE on the device — iOS plays the system default for a
sound it cannot find, cannot decode, or is not allowed to honour, which is
byte-identical to "the operator chose the default". So the properties below
are the only place any of it is observable.

What each test defends:

  1. ONE list. The picker's options, the create route's validator and the
     APNs resolver all read `dispatch_tones`, so the panel cannot offer a
     tone the server rejects or the app does not bundle.
  2. An unknown tone is a 422 on the WRITE path (the operator is told) and
     collapses to the default on every READ path (an accepted dispatch
     still delivers if the catalogue is later trimmed).
  3. The choice survives the whole pipeline: create → row → fan-out
     snapshot → notification `data_json`.
  4. It rides under `data.tone`, NEVER `data.sound`. `data.sound` is the
     ALARM-CLASS marker in live_activity_service — an announcement stamped
     with it would re-ring three times, twenty seconds apart.
  5. `apns_push` refuses to put a named sound on an ActivityKit payload.
     iOS has never honoured one there and the result is SILENCE, not a
     fallback (verified on device 2026-07-20/21). This is the test that
     stops a future "let's wire the picker up properly" from silencing
     every dispatch card on every phone.
  6. Exactly one alembic head. 086 and 087 exist on sibling branches of
     this arc; two revisions sharing a parent makes `alembic upgrade head`
     fail at ENVIRONMENT LOAD, i.e. no migration runs at all.

Run: cd backend && RUN_MODE=platform python3 -m pytest tests/test_dispatch_tone.py -q -p no:cacheprovider
"""

from __future__ import annotations

import types
import uuid
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.api.admin.dispatch import router as dispatch_router
from app.api.auth import get_current_user
from app.config import settings
from app.db import async_session_maker, get_db
from app.db.models import (
    AdminDispatch,
    LiveActivity,
    LiveActivityDevice,
    NotificationQueue,
    NQ_QUEUED,
    User,
)
from app.services import apns_push, dispatch_tones

BACKEND = Path(__file__).resolve().parent.parent


# ── Harness (same shape as test_admin_dispatch.py) ────────────────

def _principal(uid: str, role: str = "user", email: str = "u@x.com"):
    return types.SimpleNamespace(id=uid, role=role, email=email, name="U")


@pytest_asyncio.fixture
async def dispatch_client():
    app = FastAPI()
    app.include_router(dispatch_router, prefix=settings.api_prefix)

    async def _override_db():
        async with async_session_maker() as db:
            yield db

    app.dependency_overrides[get_db] = _override_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c, app


async def _mk_user(*, role: str = "user") -> str:
    from app.services.auth_service import get_password_hash

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"tone-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("x" * 12),
            name="Tone Test",
            role=role,
            timezone="America/Toronto",
            created_at=datetime.utcnow(),
            notification_preferences={
                "quiet_hours": {"enabled": False, "start": "22:00", "end": "08:00"},
            },
        ))
        await db.commit()
    return user_id


def _compose(**overrides) -> dict:
    body = {
        "mode": "once",
        "audience": "user",
        "title": "Scheduled maintenance",
        "body": "Toup will be briefly unavailable at 02:00 UTC.",
    }
    body.update(overrides)
    return body


# ── 1. one list ───────────────────────────────────────────────────


def test_the_catalogue_is_the_only_list():
    """The route serves exactly what the validator accepts.

    Two hand-maintained copies is how a panel comes to offer a tone the
    server 422s (or worse, one the phone does not bundle, which is silent).
    """
    served = {t["id"] for t in dispatch_tones.catalogue()}
    assert served == set(dispatch_tones.TONE_IDS)
    for tone_id in served:
        assert dispatch_tones.is_valid(tone_id)


@pytest.mark.asyncio
async def test_the_route_serves_the_shape_the_panel_is_built_against(dispatch_client):
    """`DispatchTone` in frontend/src/api.ts is a CLAIM about this response
    that TypeScript cannot check. Asserted here, against the handler, because
    the panel renders `label`, `description` and `preview` directly — a
    renamed key arrives as `undefined` at runtime with tsc perfectly happy.
    """
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    res = await client.get("/api/admin/dispatch/tones")
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["default"] == "default"
    assert [t["id"] for t in payload["tones"]][0] == "default", "default renders first"
    for t in payload["tones"]:
        assert set(t) == {
            "id", "label", "description", "file", "android_channel", "preview",
        }, t
        assert t["label"] and t["description"], t
        for event in t["preview"]:
            start_ms, freq, dur_ms, timbre = event
            assert start_ms >= 0 and freq > 0 and dur_ms > 0
            assert timbre in ("bell", "blip", "thud"), event


def test_every_tone_is_a_bundled_file_or_the_system_default():
    """The DO-NOT of this feature: never offer a sound the app cannot play.

    iOS cannot be told to play an arbitrary system ringtone from a push, so
    the only legal values are `default` and a file the app bundles. The file
    names are pinned here rather than merely spot-checked, because the app
    is a different repo — this list IS the cross-repo contract, and a rename
    on either side has to break something.
    """
    by_id = {t["id"]: t for t in dispatch_tones.catalogue()}
    assert by_id["default"]["file"] is None, "the system default is not a file"
    assert {i: t["file"] for i, t in by_id.items() if t["file"]} == {
        "ping": "toup_ping.caf",
        "chime": "toup_chime.caf",
        "pulse": "toup_pulse.caf",
        "rise": "toup_rise.caf",
        "knock": "toup_knock.caf",
        "alarm": "toup_alarm.caf",
    }
    # Bundled into the app AND the widget target by
    # toup-platform-app/plugins/with-alarm-sound.js, whose SOUND_FILES list
    # is held to assets/sounds/ by scripts/check-dispatch-tones.js there.
    for tone in by_id.values():
        if tone["file"]:
            assert tone["file"].endswith((".caf", ".aiff", ".wav")), tone


def test_android_binds_one_channel_per_tone():
    """Android fixes a channel's sound at CREATION and it cannot be changed
    afterwards, so a tone is a channel. The ids are declared here and
    created by nobody — the app installs no expo-notifications — but they
    are derived from the same list rather than invented at the call site."""
    channels = [t["android_channel"] for t in dispatch_tones.catalogue()]
    assert len(channels) == len(set(channels)), "one channel per tone, or a tone is unreachable"


# ── 2. unknown ids: loud on write, quiet on read ──────────────────


@pytest.mark.asyncio
async def test_an_unknown_tone_is_rejected_at_the_door(dispatch_client):
    """422, not a silent fall back to the default.

    Everything else in this feature fails inaudibly; the one moment we can
    still tell the operator their choice did not take is the request."""
    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    recipient = await _mk_user()
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    res = await client.post("/api/admin/dispatch", json=_compose(
        target_user_id=recipient, tone="foghorn",
    ))
    assert res.status_code == 422, res.text
    assert "tone" in res.text


def test_a_retired_tone_still_delivers_as_the_default():
    """The read path is deliberately forgiving where the write path is not:
    a dispatch that was accepted must not become undeliverable because the
    catalogue was later trimmed."""
    assert dispatch_tones.normalize("foghorn") == dispatch_tones.TONE_DEFAULT
    assert dispatch_tones.normalize(None) == dispatch_tones.TONE_DEFAULT
    assert dispatch_tones.normalize("knock") == "knock"
    assert dispatch_tones.apns_sound("foghorn") == "default"
    assert dispatch_tones.apns_sound("knock") == "toup_knock.caf"


# ── 3 + 4. the choice survives the pipeline, under the right key ──


@pytest.mark.asyncio
async def test_the_chosen_tone_reaches_the_notification_row(dispatch_client, monkeypatch):
    """create → admin_dispatches.tone → snapshot → data_json['tone'].

    And under `tone`, NEVER `sound`: `live_activity_service._is_alarm_row`
    reads `data.sound`, so an announcement stamped with it becomes
    ALARM-CLASS and re-rings _ALARM_MAX_RINGS times, twenty seconds apart —
    an operator's one message arriving as a three-ring reminder alarm.
    """
    from app.services import live_activity_service as las
    from app.services.admin_dispatch_worker import run_dispatch_fanout

    client, app = dispatch_client
    admin = await _mk_user(role="admin")
    recipient = await _mk_user()
    app.dependency_overrides[get_current_user] = lambda: _principal(admin, role="admin")

    async def _no_spawn(dispatch_id: str):
        return None

    monkeypatch.setattr("app.api.admin.dispatch.spawn_dispatch_fanout", _no_spawn)

    res = await client.post("/api/admin/dispatch", json=_compose(
        target_user_id=recipient, tone="knock",
    ))
    assert res.status_code == 201, res.text
    dispatch_id = res.json()["dispatch"]["id"]
    assert res.json()["dispatch"]["tone"] == "knock"

    async with async_session_maker() as db:
        stored = await db.get(AdminDispatch, dispatch_id)
        assert stored.tone == "knock", "the operator's choice is persisted"

    await run_dispatch_fanout(dispatch_id)

    async with async_session_maker() as db:
        nq = (await db.execute(
            select(NotificationQueue).where(NotificationQueue.user_id == recipient)
        )).scalars().one()

    # Asserted BEFORE the value, so moving the tone onto `sound` fails on the
    # reason rather than on a KeyError.
    assert "sound" not in nq.data_json, (
        "data.sound is the ALARM-CLASS marker — this row would ring three times"
    )
    assert las._is_alarm_row(nq) is False, "an announcement is not an alarm"
    assert nq.data_json.get("tone") == "knock"


# ── 5. an ActivityKit alert may never carry a named sound ─────────


def test_no_activitykit_payload_ever_carries_a_named_sound():
    """iOS has never honoured one — the card arrives in SILENCE, not with
    the default (verified on the founder's device 2026-07-20/21 with the
    file present in both the app and the widget-extension bundles).

    So the tone is resolved and then dropped, and this is the test that has
    to fail if anyone "fixes" the picker by putting the name on the wire.
    Every builder in apns_push is checked, because the alarm-class lane
    alerts on update and end too.
    """
    named = dispatch_tones.apns_sound("knock")
    assert named == "toup_knock.caf", "precondition: a real file name went in"

    start = apns_push.build_start_payload(
        mission_id="admin:x", title="T", alert_title="T", alert_body="B",
        alert_sound=named,
    )
    update = apns_push.build_update_payload(
        title="T", alert_title="T", alert_body="B", alert_sound=named,
    )
    end = apns_push.build_end_payload(
        title="T", alert_title="T", alert_body="B", alert_sound=named,
    )
    for label, payload in (("start", start), ("update", update), ("end", end)):
        sound = payload["aps"]["alert"]["sound"]
        assert sound == apns_push.ACTIVITY_KIT_SOUND == "default", (
            f"{label} payload carries {sound!r} — a named sound on an "
            f"ActivityKit alert is silence on device"
        )

    # …and a QUIET start stays quiet. `_alert` short-circuits on a missing
    # alert_title, so the synthesised alert below it has no `sound` key at
    # all — iOS 26 rejects an alertless start outright, which is why a
    # "silent" producer still ships an alert. Passing a tone alongside must
    # not resurrect a sound here: a countdown arm and the self-healing
    # restart path both take this branch, and either one banging is a
    # notification the user did not earn.
    quiet = apns_push.build_start_payload(
        mission_id="reminder:x", title="T", subtitle="S", alert_sound=named,
    )
    assert "sound" not in quiet["aps"]["alert"], quiet["aps"]["alert"]


@pytest.mark.asyncio
async def test_the_announcement_card_still_rings_with_a_tone_chosen(monkeypatch):
    """The end-to-end version of the rule above, through the real lane.

    A dispatch with a tone must degrade to the SYSTEM tone, never to no
    sound at all — the failure mode this guards is a picker that makes
    every lock-screen card silent.
    """
    from app.services import live_activity_service as las

    sent: list = []

    async def fake_send(token, payload, *, environment="development", priority=10):
        sent.append({"token": token, "payload": payload})
        return 200, ""

    monkeypatch.setattr(las.apns_push, "send_live_activity", fake_send)
    monkeypatch.setattr(settings, "apns_key_b64", "eA==")
    monkeypatch.setattr(settings, "apns_key_id", "KEY123")
    monkeypatch.setattr(settings, "apns_team_id", "TEAM123")

    user_id = await _mk_user()
    dispatch_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(LiveActivityDevice(
            id=str(uuid.uuid4()), user_id=user_id,
            push_to_start_token=uuid.uuid4().hex + uuid.uuid4().hex,
            apns_environment="development", created_at=datetime.utcnow(),
        ))
        await db.commit()

    row_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(NotificationQueue(
            id=row_id, user_id=user_id, source="platform",
            event_kind="announcement", title="Scheduled maintenance",
            body="Back at 02:30 UTC.", priority="default",
            data_json={
                "kind": "announcement",
                "mission_id": f"admin:{dispatch_id}",
                "dispatch_id": dispatch_id,
                "mode": "once",
                "deep_link": f"toup://notices?mission=admin:{dispatch_id}",
                "cap_exempt": True,
                "urgent": False,
                "tone": "alarm",
            },
            idempotency_key=f"admin-dispatch:{dispatch_id}:{user_id}",
            status=NQ_QUEUED, created_at=datetime.utcnow(),
        ))
        await db.commit()

    async with async_session_maker() as db:
        row = await db.get(NotificationQueue, row_id)
        out = await las.handle_notification_row(db, row, datetime.utcnow())
        await db.commit()

    assert out["delivered"] is True, out
    alert = sent[0]["payload"]["aps"]["alert"]
    assert alert["sound"] == "default", (
        f"card would arrive silent: {alert['sound']!r}"
    )

    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(
                LiveActivity.mission_id == f"admin:{dispatch_id}")
        )).scalars().one()
        assert la.mission_id == f"admin:{dispatch_id}"


# ── 6. exactly one alembic head ───────────────────────────────────


def test_exactly_one_head():
    """One head, whatever the numbering ends up being.

    This branch sits at the end of the Admin Dispatch arc's migration chain.
    It was authored as 088-off-085 while its siblings were in flight, which
    collided head-on: the product_events branch had ALSO taken 088. A duplicate
    revision id merges cleanly in git — different filenames, no conflict — and
    then alembic raises at ENVIRONMENT LOAD, so `upgrade head` runs NOTHING, on
    every boot, in both images. This repo has hit that before with two parallel
    PRs numbered 080.

    THE NUMBER IS READ, NOT WRITTEN DOWN. It has already moved twice — 088 on
    collision with product_events, then again when #648 landed 086 ahead of the
    whole arc — and both times a hardcoded literal here would have failed for a
    reason that has nothing to do with what this test is for. An earlier version
    of this docstring said exactly that while the assertion below still pinned
    the string, which is how the second move was caught: by this test failing
    for precisely the wrong reason it warned about.

    What must never change is that there is exactly one head, and that THIS
    file's revision is reachable from it.
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    script = ScriptDirectory.from_config(cfg)

    try:
        heads = list(script.get_heads())
    except KeyError as e:  # pragma: no cover - only on an un-rebased branch
        pytest.fail(
            f"down_revision {e} is not present in this tree. That is the "
            "expected state while the parent revision is still on an unmerged "
            "sibling branch — rebase onto main once it lands. It is NOT safe "
            "to 'fix' this by re-pointing at an older revision: two revisions "
            "sharing a parent is two heads, and alembic then fails at "
            "environment load and runs no migrations at all."
        )

    assert len(heads) == 1, f"expected exactly one alembic head, got {heads}"

    # ...and this revision is on the path to it, so a renumber that orphans
    # the file is caught here too.
    # Read this migration's own id out of the file rather than restating it.
    import re
    src = next(
        p for p in (BACKEND / "alembic" / "versions").glob("*_admin_dispatch_tone.py")
    ).read_text()
    mine = re.search(r'^revision = "([^"]+)"', src, re.M).group(1)

    reachable = {r.revision for r in script.walk_revisions(base="base", head=heads[0])}
    assert mine in reachable, f"{mine} is not reachable from head {heads[0]}"
