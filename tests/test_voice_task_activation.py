"""The voice-task activation gates must converge after an agent upgrade."""

from app.config import Settings


def test_agent_image_default_overrides_stale_disabled_env():
    settings = Settings(
        _env_file=None,
        run_mode="agent",
        voice_tasks_enabled=False,
        voice_tasks_force_on=True,
    )
    assert settings.voice_tasks_enabled is True


def test_platform_gate_still_respects_its_explicit_value():
    settings = Settings(
        _env_file=None,
        run_mode="platform",
        voice_tasks_enabled=False,
        voice_tasks_force_on=True,
    )
    assert settings.voice_tasks_enabled is False
