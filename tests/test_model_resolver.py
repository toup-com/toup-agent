"""
Tests for app.services.model_resolver — the single source of truth for
LLM model defaults.

Covers:
- Resolution chain (per-tenant override → settings env → canonical default).
- Auto-builder Planner/Builder split + null inheritance.
- Provider classification (`is_claude_model`, `is_openai_model`).
- OAuth token detection (`is_oauth_anthropic_token`).
- Registry helpers (`context_window_for`, `pricing_for`) wrap canonical
  data, return sensible defaults on miss.

Tests construct lightweight stand-in `agent_config` objects (plain
`SimpleNamespace`) instead of full SQLAlchemy `AgentConfig` rows. The
resolver only ever reads attributes via `getattr`, so attribute-bearing
duck types are sufficient.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest_asyncio

from app.services import model_resolver as mr


# Override the suite-wide `_reset_database` autouse fixture from
# conftest.py — these are pure unit tests and don't touch the DB. The
# override returns immediately so a test environment without a reachable
# Postgres can still run this file.
@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    yield

from app.services.model_resolver import (
    default_model,
    default_fallback_model,
    app_builder_planner_model,
    app_builder_builder_model,
    default_anthropic_model,
    default_openai_model,
    context_window_for,
    pricing_for,
    is_claude_model,
    is_openai_model,
    is_oauth_anthropic_token,
)


def _settings_with(**overrides):
    """Patch `app.services.model_resolver.settings` so the resolver
    sees specific field values for the duration of a test."""
    return patch.object(mr, "settings", SimpleNamespace(**overrides))


class TestDefaultModel(unittest.TestCase):
    def test_falls_through_to_canonical_when_nothing_set(self):
        with _settings_with(agent_model=None):
            self.assertEqual(default_model(), mr._CANONICAL_AGENT_MODEL)

    def test_settings_env_wins_when_no_per_tenant_override(self):
        with _settings_with(agent_model="claude-opus-4-99"):
            self.assertEqual(default_model(), "claude-opus-4-99")

    def test_per_tenant_override_wins_over_settings(self):
        cfg = SimpleNamespace(agent_model="custom-tenant-model")
        with _settings_with(agent_model="canonical-default"):
            self.assertEqual(default_model(cfg), "custom-tenant-model")

    def test_per_tenant_null_falls_through_to_settings(self):
        cfg = SimpleNamespace(agent_model=None)
        with _settings_with(agent_model="from-env"):
            self.assertEqual(default_model(cfg), "from-env")

    def test_per_tenant_empty_string_treated_as_null(self):
        cfg = SimpleNamespace(agent_model="")
        with _settings_with(agent_model="from-env"):
            self.assertEqual(default_model(cfg), "from-env")

    def test_per_tenant_whitespace_treated_as_null(self):
        cfg = SimpleNamespace(agent_model="   ")
        with _settings_with(agent_model="from-env"):
            self.assertEqual(default_model(cfg), "from-env")


class TestFallbackModel(unittest.TestCase):
    def test_canonical_fallback_when_settings_unset(self):
        with _settings_with(agent_fallback_model=None):
            self.assertEqual(default_fallback_model(), mr._CANONICAL_FALLBACK_MODEL)

    def test_settings_override_wins(self):
        with _settings_with(agent_fallback_model="custom-fallback"):
            self.assertEqual(default_fallback_model(), "custom-fallback")


class TestAppBuilderSplit(unittest.TestCase):
    def test_planner_inherits_from_agent_model_when_unset(self):
        cfg = SimpleNamespace(
            agent_model="X",
            app_builder_planner_model=None,
            app_builder_builder_model=None,
        )
        with _settings_with(
            agent_model=None,
            app_builder_planner_model=None,
            app_builder_builder_model=None,
        ):
            self.assertEqual(app_builder_planner_model(cfg), "X")

    def test_builder_inherits_from_agent_model_when_unset(self):
        cfg = SimpleNamespace(
            agent_model="Y",
            app_builder_planner_model=None,
            app_builder_builder_model=None,
        )
        with _settings_with(
            agent_model=None,
            app_builder_planner_model=None,
            app_builder_builder_model=None,
        ):
            self.assertEqual(app_builder_builder_model(cfg), "Y")

    def test_planner_per_tenant_override_wins(self):
        cfg = SimpleNamespace(
            agent_model="X",
            app_builder_planner_model="opus-strong",
            app_builder_builder_model="sonnet-cheap",
        )
        with _settings_with(
            agent_model=None,
            app_builder_planner_model=None,
            app_builder_builder_model=None,
        ):
            self.assertEqual(app_builder_planner_model(cfg), "opus-strong")
            self.assertEqual(app_builder_builder_model(cfg), "sonnet-cheap")

    def test_planner_settings_override_when_no_per_tenant(self):
        cfg = SimpleNamespace(
            agent_model="X",
            app_builder_planner_model=None,
            app_builder_builder_model=None,
        )
        with _settings_with(
            agent_model=None,
            app_builder_planner_model="env-planner",
            app_builder_builder_model="env-builder",
        ):
            self.assertEqual(app_builder_planner_model(cfg), "env-planner")
            self.assertEqual(app_builder_builder_model(cfg), "env-builder")

    def test_split_works_with_no_agent_config(self):
        with _settings_with(
            agent_model="from-env",
            app_builder_planner_model=None,
            app_builder_builder_model=None,
        ):
            self.assertEqual(app_builder_planner_model(), "from-env")
            self.assertEqual(app_builder_builder_model(), "from-env")

    def test_split_falls_through_to_canonical_when_everything_null(self):
        with _settings_with(
            agent_model=None,
            app_builder_planner_model=None,
            app_builder_builder_model=None,
        ):
            self.assertEqual(app_builder_planner_model(), mr._CANONICAL_AGENT_MODEL)
            self.assertEqual(app_builder_builder_model(), mr._CANONICAL_AGENT_MODEL)


class TestProviderSpecificDefaults(unittest.TestCase):
    def test_anthropic_default_returns_primary_when_already_claude(self):
        with _settings_with(agent_model="claude-opus-4-7", anthropic_model=None):
            self.assertEqual(default_anthropic_model(), "claude-opus-4-7")

    def test_anthropic_default_falls_through_when_primary_is_openai(self):
        with _settings_with(agent_model="gpt-5.4", anthropic_model=None):
            self.assertEqual(default_anthropic_model(), mr._CANONICAL_ANTHROPIC_MODEL)

    def test_anthropic_default_honours_settings_override(self):
        with _settings_with(agent_model="gpt-5.4", anthropic_model="claude-opus-9"):
            self.assertEqual(default_anthropic_model(), "claude-opus-9")

    def test_openai_default_returns_primary_when_already_openai(self):
        with _settings_with(agent_model="gpt-5.4"):
            self.assertEqual(default_openai_model(), "gpt-5.4")

    def test_openai_default_falls_through_when_primary_is_claude(self):
        with _settings_with(agent_model="claude-opus-4-7"):
            self.assertEqual(default_openai_model(), mr._CANONICAL_OPENAI_MODEL)


class TestProviderClassification(unittest.TestCase):
    def test_is_claude_model_matches_prefix(self):
        self.assertTrue(is_claude_model("claude-opus-4-7"))
        self.assertTrue(is_claude_model("claude-sonnet-4-6"))
        self.assertTrue(is_claude_model("claude-haiku-4-5-20251001"))
        self.assertTrue(is_claude_model("claude-3-5-haiku-20241022"))

    def test_is_claude_model_case_insensitive(self):
        self.assertTrue(is_claude_model("Claude-Opus-4-7"))

    def test_is_claude_model_rejects_non_claude(self):
        self.assertFalse(is_claude_model("gpt-5.4"))
        self.assertFalse(is_claude_model("o1-mini"))
        self.assertFalse(is_claude_model("o3"))
        self.assertFalse(is_claude_model("gemini-pro"))

    def test_is_claude_model_handles_empty_and_none(self):
        self.assertFalse(is_claude_model(None))
        self.assertFalse(is_claude_model(""))

    def test_is_openai_model_matches_gpt_family(self):
        self.assertTrue(is_openai_model("gpt-5.4"))
        self.assertTrue(is_openai_model("gpt-5"))
        self.assertTrue(is_openai_model("gpt-4o"))
        self.assertTrue(is_openai_model("gpt-4o-mini"))
        self.assertTrue(is_openai_model("gpt-4-turbo"))

    def test_is_openai_model_matches_reasoning_family(self):
        self.assertTrue(is_openai_model("o1-mini"))
        self.assertTrue(is_openai_model("o3-mini"))
        self.assertTrue(is_openai_model("o4-mini"))
        self.assertTrue(is_openai_model("o1"))
        self.assertTrue(is_openai_model("o3"))

    def test_is_openai_model_rejects_non_openai(self):
        self.assertFalse(is_openai_model("claude-opus-4-7"))
        self.assertFalse(is_openai_model("gemini-pro"))

    def test_is_openai_model_handles_empty_and_none(self):
        self.assertFalse(is_openai_model(None))
        self.assertFalse(is_openai_model(""))


class TestOAuthDetection(unittest.TestCase):
    def test_detects_claude_code_oauth_token(self):
        self.assertTrue(is_oauth_anthropic_token("sk-ant-oat01-eACw6b1XYZ"))

    def test_rejects_anthropic_api_key(self):
        self.assertFalse(is_oauth_anthropic_token("sk-ant-api03-USERKEY"))

    def test_rejects_openai_keys(self):
        self.assertFalse(is_oauth_anthropic_token("sk-proj-DOXYZ"))
        self.assertFalse(is_oauth_anthropic_token("sk-1234567890abcdef"))

    def test_rejects_empty_and_none(self):
        self.assertFalse(is_oauth_anthropic_token(None))
        self.assertFalse(is_oauth_anthropic_token(""))


class TestRegistryHelpers(unittest.TestCase):
    def test_context_window_for_known_model(self):
        # Reads the canonical context_manager registry, not a duplicate.
        from app.agent.context_manager import MODEL_CONTEXT_WINDOWS
        for known_model, expected in MODEL_CONTEXT_WINDOWS.items():
            self.assertEqual(context_window_for(known_model), expected)

    def test_context_window_for_unknown_model_returns_default(self):
        # Unknown model returns the canonical default + logs a warning.
        from app.agent.context_manager import DEFAULT_CONTEXT_WINDOW
        self.assertEqual(
            context_window_for("nonexistent-model-xyz"),
            DEFAULT_CONTEXT_WINDOW,
        )

    def test_context_window_for_empty_returns_safe_default(self):
        self.assertEqual(context_window_for(""), mr._safe_default_window())
        self.assertEqual(context_window_for(None), mr._safe_default_window())

    def test_pricing_for_known_model(self):
        # Returns a (input, output) tuple in float per-1k.
        result = pricing_for("gpt-5.4")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)

    def test_pricing_for_unknown_returns_none(self):
        self.assertIsNone(pricing_for("nonexistent-model-xyz"))

    def test_pricing_for_empty_returns_none(self):
        self.assertIsNone(pricing_for(""))
        self.assertIsNone(pricing_for(None))


class TestResolverIsTolerantOfMissingColumns(unittest.TestCase):
    """The resolver must work even if `agent_config` doesn't yet have the
    new columns (transitional deploy window between commit A and commit E
    where the SQLAlchemy model gains the new fields)."""

    def test_split_resolves_when_cfg_has_no_split_columns(self):
        # AgentConfig-shaped object without the new auto-builder columns.
        cfg = SimpleNamespace(agent_model="legacy")
        with _settings_with(
            agent_model=None,
            app_builder_planner_model=None,
            app_builder_builder_model=None,
        ):
            self.assertEqual(app_builder_planner_model(cfg), "legacy")
            self.assertEqual(app_builder_builder_model(cfg), "legacy")

    def test_resolver_when_settings_has_no_split_fields(self):
        # Settings object without the new auto-builder env fields.
        with _settings_with(agent_model="from-env"):  # no split fields at all
            self.assertEqual(app_builder_planner_model(), "from-env")
            self.assertEqual(app_builder_builder_model(), "from-env")


if __name__ == "__main__":
    unittest.main()
