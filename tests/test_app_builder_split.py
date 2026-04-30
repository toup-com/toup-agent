"""
Tests for the auto-builder Planner/Builder split (commit F of the LLM
model-selection refactor).

Asserts that:
  - The `role` kwarg on `AppBuilderSkill._call_llm` resolves the correct
    model (planner vs builder) when no explicit `model` is passed.
  - Default behaviour (both columns NULL, both env settings unset) keeps
    Planner and Builder on the same model — preserves today's single-model
    behaviour.
  - Per-tenant overrides on `agent_configs.app_builder_planner_model` /
    `_builder_model` win over `settings.agent_model`.

These tests exercise the resolver through the public functions in
`app.services.model_resolver`. They don't make real LLM calls — the
verification is that the right model id flows through to the call site.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest_asyncio

from app.services import model_resolver as mr


# Override the suite-wide DB autouse fixture; this file is pure unit-level.
@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    yield


def _settings_with(**overrides):
    """Mirror of test_model_resolver._settings_with — patches the resolver's
    settings reference so each test is hermetic."""
    return patch.object(mr, "settings", SimpleNamespace(**overrides))


class TestSplitNullInheritance(unittest.TestCase):
    """Default state: both columns NULL, both env settings unset.
    Planner and Builder both use agent_model. Preserves pre-refactor
    behaviour exactly until an operator deliberately splits."""

    def test_planner_inherits_agent_model_when_all_null(self):
        with _settings_with(
            agent_model="claude-opus-4-7",
            app_builder_planner_model=None,
            app_builder_builder_model=None,
        ):
            self.assertEqual(mr.app_builder_planner_model(), "claude-opus-4-7")

    def test_builder_inherits_agent_model_when_all_null(self):
        with _settings_with(
            agent_model="claude-opus-4-7",
            app_builder_planner_model=None,
            app_builder_builder_model=None,
        ):
            self.assertEqual(mr.app_builder_builder_model(), "claude-opus-4-7")

    def test_planner_and_builder_match_when_unsplit(self):
        with _settings_with(
            agent_model="some-default",
            app_builder_planner_model=None,
            app_builder_builder_model=None,
        ):
            self.assertEqual(
                mr.app_builder_planner_model(),
                mr.app_builder_builder_model(),
            )


class TestSplitEnvOverride(unittest.TestCase):
    """`settings.app_builder_planner_model` / `_builder_model` set via
    env var overrides agent_model for the relevant role only."""

    def test_planner_env_override_wins(self):
        with _settings_with(
            agent_model="agent-default",
            app_builder_planner_model="opus-strong",
            app_builder_builder_model=None,
        ):
            self.assertEqual(mr.app_builder_planner_model(), "opus-strong")
            # Builder still inherits from agent_model
            self.assertEqual(mr.app_builder_builder_model(), "agent-default")

    def test_builder_env_override_wins(self):
        with _settings_with(
            agent_model="agent-default",
            app_builder_planner_model=None,
            app_builder_builder_model="sonnet-cheap",
        ):
            self.assertEqual(mr.app_builder_builder_model(), "sonnet-cheap")
            # Planner still inherits from agent_model
            self.assertEqual(mr.app_builder_planner_model(), "agent-default")

    def test_independent_env_splits(self):
        with _settings_with(
            agent_model="agent-default",
            app_builder_planner_model="planner-strong",
            app_builder_builder_model="builder-cheap",
        ):
            self.assertEqual(mr.app_builder_planner_model(), "planner-strong")
            self.assertEqual(mr.app_builder_builder_model(), "builder-cheap")


class TestPerTenantSplitOverride(unittest.TestCase):
    """`agent_configs.app_builder_planner_model` / `_builder_model` columns
    take precedence over both env and agent_model. This is the runtime
    knob a Settings UI would expose."""

    def test_per_tenant_planner_column_wins_over_env(self):
        cfg = SimpleNamespace(
            agent_model="tenant-default",
            app_builder_planner_model="tenant-planner",
            app_builder_builder_model=None,
        )
        with _settings_with(
            agent_model="env-default",
            app_builder_planner_model="env-planner",
            app_builder_builder_model=None,
        ):
            self.assertEqual(mr.app_builder_planner_model(cfg), "tenant-planner")

    def test_per_tenant_builder_column_wins_over_env(self):
        cfg = SimpleNamespace(
            agent_model="tenant-default",
            app_builder_planner_model=None,
            app_builder_builder_model="tenant-builder",
        )
        with _settings_with(
            agent_model="env-default",
            app_builder_planner_model=None,
            app_builder_builder_model="env-builder",
        ):
            self.assertEqual(mr.app_builder_builder_model(cfg), "tenant-builder")

    def test_per_tenant_null_falls_through_to_env(self):
        cfg = SimpleNamespace(
            agent_model="tenant-default",
            app_builder_planner_model=None,
            app_builder_builder_model=None,
        )
        with _settings_with(
            agent_model="env-default",
            app_builder_planner_model="env-planner",
            app_builder_builder_model="env-builder",
        ):
            # Tenant column null → env wins for the per-role split.
            self.assertEqual(mr.app_builder_planner_model(cfg), "env-planner")
            self.assertEqual(mr.app_builder_builder_model(cfg), "env-builder")


class TestSplitTransitionalSafety(unittest.TestCase):
    """Resolver works even when an `agent_config` row predates migration 029
    and so doesn't have the new columns yet. SimpleNamespace without the
    columns mimics that state."""

    def test_split_resolves_when_cfg_has_no_split_columns(self):
        cfg = SimpleNamespace(agent_model="legacy-model")
        with _settings_with(
            agent_model=None,
            app_builder_planner_model=None,
            app_builder_builder_model=None,
        ):
            self.assertEqual(mr.app_builder_planner_model(cfg), "legacy-model")
            self.assertEqual(mr.app_builder_builder_model(cfg), "legacy-model")


if __name__ == "__main__":
    unittest.main()
