"""Disabling the durable voice supervisor has to actually disable it.

Two independent halves:

* the CONFIG half — which signal wins when the image bakes the flag ON;
* the RUNTIME half — that the stop path really stops the polling loop.

The config half is a falsifier by construction: every assertion below that
carries a ``# FALSIFIER`` marker fails on the pre-incident validator
(``if run_mode == "agent" and voice_tasks_force_on: enabled = True``), which
discarded an operator's explicit off and left the feature with no reliable
kill switch on the 2026-09-06 fleet.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from app.config import Settings


class _EnvPatch:
    """Set/unset real process env vars — the channel `docker run -e` uses.

    Init kwargs are NOT interchangeable here: the whole point of the fix is
    that a value present in ``os.environ`` outranks the image default while a
    value that only exists in a cloned ``.env`` file does not.
    """

    def __init__(self, **values: str | None) -> None:
        self.values = values
        self.saved: dict[str, str | None] = {}

    def __enter__(self):
        for key, value in self.values.items():
            self.saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return self

    def __exit__(self, *_exc):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


# ── config precedence ────────────────────────────────────────────────


@pytest.mark.parametrize("off", ["0", "false", "False", "off", "no"])
def test_explicit_container_env_off_beats_the_image_default(off: str) -> None:
    # FALSIFIER: the pre-incident validator returned True for every one of
    # these, so `docker run -e VOICE_TASKS_ENABLED=0` changed nothing.
    with _EnvPatch(VOICE_TASKS_ENABLED=off, VOICE_TASKS_FORCE_ON=None):
        settings = Settings(_env_file=None, run_mode="agent")
    assert settings.voice_tasks_enabled is False


@pytest.mark.parametrize("on", ["1", "true", "True", "on", "yes"])
def test_explicit_container_env_on_is_still_honoured(on: str) -> None:
    with _EnvPatch(VOICE_TASKS_ENABLED=on, VOICE_TASKS_FORCE_ON=None):
        settings = Settings(_env_file=None, run_mode="agent")
    assert settings.voice_tasks_enabled is True


def test_force_on_false_disables_even_against_the_baked_true() -> None:
    # FALSIFIER: Dockerfile.agent bakes VOICE_TASKS_ENABLED=true, so under the
    # pre-incident validator dropping FORCE_ON simply left the baked value
    # standing and the documented emergency rollback was a no-op.
    with _EnvPatch(VOICE_TASKS_ENABLED="true", VOICE_TASKS_FORCE_ON="false"):
        settings = Settings(_env_file=None, run_mode="agent")
    assert settings.voice_tasks_enabled is False


def test_force_on_false_disables_when_the_flag_is_absent_entirely() -> None:
    # Regression guard, not a falsifier: this one already held before the
    # change (the old `if` was simply False and the field default is False).
    # Pinned so a future "restore the default when not forcing" edit is loud.
    with _EnvPatch(VOICE_TASKS_ENABLED=None, VOICE_TASKS_FORCE_ON="0"):
        settings = Settings(_env_file=None, run_mode="agent")
    assert settings.voice_tasks_enabled is False


def test_absent_container_env_still_gets_the_image_default() -> None:
    """The regression guard on the fix above.

    This is the case 7edaed3a was written for and it must keep working: no
    VOICE_TASKS_ENABLED reaches the container, so the image default wins.
    """
    with _EnvPatch(VOICE_TASKS_ENABLED=None, VOICE_TASKS_FORCE_ON=None):
        settings = Settings(_env_file=None, run_mode="agent")
    assert settings.voice_tasks_enabled is True


def test_a_stale_dotenv_file_does_not_count_as_explicit(tmp_path) -> None:
    """A blue-green upgrade clones the old container's `.env`.

    pydantic reads that file, but a dotenv value never lands in os.environ —
    so it must fall through to the image default rather than silently keeping
    a whole upgraded fleet dark. This is the exact scenario 7edaed3a fixed,
    and the new precedence must not reopen it.
    """
    stale = tmp_path / ".env"
    stale.write_text("VOICE_TASKS_ENABLED=false\n")
    with _EnvPatch(VOICE_TASKS_ENABLED=None, VOICE_TASKS_FORCE_ON=None):
        settings = Settings(_env_file=str(stale), run_mode="agent")
    assert settings.voice_tasks_enabled is True


@pytest.mark.parametrize("value,expected", [("1", True), ("0", False)])
def test_platform_run_mode_is_untouched(value: str, expected: bool) -> None:
    with _EnvPatch(VOICE_TASKS_ENABLED=value, VOICE_TASKS_FORCE_ON="true"):
        settings = Settings(_env_file=None, run_mode="platform")
    assert settings.voice_tasks_enabled is expected


def test_an_unparseable_flag_fails_loudly_rather_than_defaulting_on() -> None:
    """Presence alone must not be read as "on".

    A blank or malformed VOICE_TASKS_ENABLED is a pydantic ValidationError
    while Settings is being built, i.e. the agent refuses to boot rather than
    quietly falling through to the image default. Recorded because the new
    precedence tests PRESENCE, and a silent coercion here would turn a typo
    into an un-disableable supervisor.
    """
    import pydantic

    for bad in ("", " false ", "maybe"):
        with _EnvPatch(VOICE_TASKS_ENABLED=bad, VOICE_TASKS_FORCE_ON=None):
            with pytest.raises(pydantic.ValidationError):
                Settings(_env_file=None, run_mode="agent")


# ── runtime stop path ────────────────────────────────────────────────


class _CountingMaker:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.acquisitions = 0

    def __call__(self):
        self.acquisitions += 1
        return self.delegate()


async def test_close_voice_task_services_stops_the_polling_loop() -> None:
    """Disabling has to reach the loop, not just the HTTP gate.

    ``app/api/voice_tasks.py`` refuses requests on ``voice_tasks_enabled``,
    but the supervisor reads that flag ONLY in agent_main at boot — nothing in
    ``_supervise`` consults it again. So the registry teardown is the only
    thing that can stop an already-running loop, and it has to actually stop
    it: cancel the task, clear the registry, and issue no further sessions.
    """
    from app.agent import voice_tasks as vt
    from app.db import async_session_maker

    runner = _NeverRunner()
    counting = _CountingMaker(async_session_maker)
    service = vt.VoiceTaskService(
        runner, session_maker=counting,
        poll_seconds=0.05, idle_poll_seconds=0.05, lease_seconds=90.0,
        claim_allowed=lambda: True,
    )
    vt._services[id(runner)] = service
    try:
        await service.start()
        await asyncio.sleep(0.35)
        polled = counting.acquisitions
        assert polled >= 3, f"supervisor was not polling to begin with ({polled})"

        await vt.close_voice_task_services()

        assert service._closing is True
        assert service._supervisor is None
        assert vt._services == {}
        settled = counting.acquisitions
        await asyncio.sleep(0.4)  # >= 8 poll intervals
        assert counting.acquisitions == settled, (
            "supervisor kept opening database sessions after close: "
            f"{counting.acquisitions - settled} more"
        )
    finally:
        vt._services.pop(id(runner), None)
        await service.close()


class _NeverRunner:
    async def run(self, **_kwargs):  # pragma: no cover - a claim is a failure
        raise AssertionError("idle supervisor claimed unexpected work")


def test_shutdown_close_is_not_gated_on_the_flag_it_is_undoing() -> None:
    """A source probe, because agent_main's lifespan is not unit-testable here.

    The close call used to sit under ``if settings.voice_tasks_enabled:``.
    That is the one condition guaranteed to be FALSE in the case the close
    exists for — an operator turned the flag off while a supervisor was
    already running — so the loop kept polling through shutdown and its
    in-flight tasks were never marked unknown. Asserts ORDER, not just
    presence: the call must not appear inside such a branch.
    """
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "agent_main.py"
    lines = source.read_text().splitlines()
    hits = [i for i, line in enumerate(lines)
            if "close_voice_task_services()" in line and "await" in line]
    assert len(hits) == 1, f"expected one shutdown close call, found {len(hits)}"
    # Walk back over the enclosing block; no voice_tasks_enabled gate may
    # appear as CODE between the top of the shutdown section and the call.
    # Comments are stripped first — the block deliberately explains the gate
    # it no longer has, and that prose must not trip its own guard.
    window = [line.split("#", 1)[0] for line in lines[max(0, hits[0] - 15):hits[0]]]
    offenders = [line for line in window if "voice_tasks_enabled" in line]
    assert not offenders, (
        "shutdown close is gated on voice_tasks_enabled again: " + repr(offenders)
    )
