"""The app-building guard refused every React file ever written.

`_is_app_building_write` blocks a write and tells the model to call
`app_builder__build_app` instead. It fires when a path looks like app
scaffolding AND the content scores >= 2 "app indicators". Two of those
indicators were:

    "expo"          -- a SUBSTRING of the word `export`
    "import React"  -- the second most common line in any React file

So `src/components/Button.tsx` containing nothing but an import and an
export scored exactly 2 and was refused. Every `**/components/*.tsx`,
every `**/screens/*Screen.tsx`, every `App.tsx`, `package.json` and
`tsconfig.json` in the user's own repo, in a workspace whose whole
purpose is writing code.

Two properties are load-bearing and both are tested here:

  1. ORDINARY REACT IS NOT APP SCAFFOLDING. The signals must distinguish
     "a React file" from "an Expo project being scaffolded", or the
     guard is a blanket ban.

  2. THE GUARD MUST NOT OUTLIVE THE TOOL IT REDIRECTS TO. Its entire
     output is "call `app_builder__build_app`". A tenant whose
     `AGENT_TOOL_FAMILIES` withholds that family has no such tool, so
     the model is blocked, redirected to something absent, and refused
     again by the entitlement layer — two refusals and no route. Latent
     while the family was always on; live the moment withholding it
     became the way to stay under OpenAI's 128-tool cap (#536).

A false positive here is far more expensive than a false negative: this
is a redirect, not a security boundary. Nothing is unsafe about writing
a .tsx file.
"""

from __future__ import annotations

import pytest

from app.agent import tool_entitlements as te_mod
from app.agent.tool_executor import (
    _is_app_building_exec,
    _is_app_building_write,
    _pipeline_guard_active,
)
from app.config import settings


@pytest.fixture(autouse=True)
def _restore_entitlements():
    yield
    settings.agent_tool_families = "*"
    te_mod.reset_cache_for_tests()


def _entitle(value: str) -> None:
    settings.agent_tool_families = value
    te_mod.reset_cache_for_tests()


# ── 1. Ordinary React must pass ──────────────────────────────────────

ORDINARY = [
    (
        "src/components/Button.tsx",
        'import React from "react";\n\nexport function Button() {\n'
        '  return <button>go</button>;\n}\n',
    ),
    (
        "web/src/components/Chart.tsx",
        'import React, { useState } from "react";\n'
        "export default function Chart() { return null; }\n",
    ),
    (
        "frontend/src/screens/LoginScreen.tsx",
        'import React from "react";\nexport const LoginScreen = () => null;\n',
    ),
    (
        "backend/tsconfig.json",
        '{"compilerOptions": {"strict": true}, "exclude": ["node_modules"]}',
    ),
    (
        "package.json",
        '{"name": "api", "dependencies": {"fastify": "^4.0.0"}}',
    ),
]


@pytest.mark.parametrize("path,content", ORDINARY, ids=[p for p, _ in ORDINARY])
def test_ordinary_react_and_config_files_are_not_blocked(path, content):
    assert _is_app_building_write(path, content) is False


def test_the_exact_file_that_proved_the_bug():
    """`expo` matched inside `export`; `import React` supplied the second
    indicator. Nothing else was needed."""
    content = 'import React from "react";\nexport function Button() {}\n'
    assert "expo" in content          # the substring is genuinely there
    assert "import React" in content  # and so is the other indicator
    assert _is_app_building_write("src/components/Button.tsx", content) is False


def test_export_alone_never_counts_as_expo():
    """Directly pins the substring. A file that is nothing but exports
    must score zero, not one."""
    content = "export const a = 1;\nexport const b = 2;\nexport default a;\n"
    assert _is_app_building_write("src/components/Only.tsx", content) is False


# ── 1b. Each half of the fix, pinned INDEPENDENTLY ───────────────────
#
# Mutation testing caught this. The bug needed BOTH `expo`-matches-`export`
# AND `import React` to reach a score of 2, so reverting either one alone
# still scores 1 and the tests above stay green. Two independently
# sufficient fixes means no single mutant reproduces the defect — and a
# later edit could drop one, relying silently on the other.
#
# Each test below pairs the suspect signal with EXACTLY ONE genuine
# indicator, so the score is 1 if the fix holds and 2 if it does not.


def test_export_does_not_count_as_an_expo_indicator():
    """Kills a revert of `\\bexpo\\b` back to `expo`.

    One real indicator (`StyleSheet.create`) plus the word `export`. If
    `expo` matched as a substring this would score 2 and be blocked.
    """
    content = "import { StyleSheet } from 'x';\nStyleSheet.create({});\nexport default 1;\n"
    assert "expo" in content, "the substring must be present for this to prove anything"
    assert _is_app_building_write("src/components/A.tsx", content) is False


def test_import_react_does_not_count_as_an_indicator():
    """Kills re-adding `import React` to the signal list.

    One real indicator (`react-navigation`) plus `import React`. If the
    latter counted, this would score 2 and be blocked.
    """
    content = 'import React from "react";\nimport { x } from "react-navigation";\n'
    assert _is_app_building_write("src/components/B.tsx", content) is False


def test_one_indicator_is_not_enough():
    """Kills lowering the threshold from 2 to 1."""
    content = "const s = StyleSheet.create({ a: {} });\n"
    assert _is_app_building_write("App.tsx", content) is False


# ── 2. Real Expo scaffolding must still be caught ────────────────────
#
# ANTI-VACUITY. Without these the tests above pass on a guard that has
# been deleted outright.

REAL_SCAFFOLD = [
    (
        "App.tsx",
        'import { NavigationContainer } from "@react-navigation/native";\n'
        'import { createNativeStackNavigator } from "@react-navigation/native-stack";\n'
        "const Stack = createNativeStackNavigator();\n",
    ),
    (
        "package.json",
        '{"dependencies": {"expo": "~52.0.0", "react-native": "0.76.9"}}',
    ),
    (
        "src/screens/HomeScreen.tsx",
        'import { View, StyleSheet } from "react-native";\n'
        "const s = StyleSheet.create({ c: { flex: 1 } });\n",
    ),
]


@pytest.mark.parametrize("path,content", REAL_SCAFFOLD, ids=[p for p, _ in REAL_SCAFFOLD])
def test_genuine_expo_scaffolding_is_still_blocked(path, content):
    assert _is_app_building_write(path, content) is True


def test_a_path_outside_the_pattern_list_is_never_blocked():
    """The content signals only apply to paths that look like app files."""
    content = '{"dependencies": {"expo": "~52.0.0", "react-native": "0.76.9"}}'
    assert _is_app_building_write("docs/notes/example.md", content) is False


@pytest.mark.parametrize("cmd", [
    "npx create-expo-app my-app",
    "expo init my-app",
    "yarn create expo my-app",
    "npx create-next-app web",
])
def test_scaffolding_commands_are_still_caught(cmd):
    assert _is_app_building_exec(cmd) is True


@pytest.mark.parametrize("cmd", [
    "npm run build",
    "npx tsc --noEmit",
    "git commit -m 'export the report'",
    "npm install",
])
def test_ordinary_commands_are_not_caught(cmd):
    assert _is_app_building_exec(cmd) is False


# ── 3. The guard must not outlive the tool it redirects to ───────────


def test_guard_is_inactive_when_the_app_builder_is_withheld():
    _entitle("doc_generation,toup")
    assert _pipeline_guard_active() is False


def test_guard_is_active_by_default():
    """ANTI-VACUITY: `AGENT_TOOL_FAMILIES` defaults to `*`, so every tenant
    on the default keeps today's behaviour exactly."""
    _entitle("*")
    assert _pipeline_guard_active() is True


def test_guard_is_active_when_the_family_is_explicitly_entitled():
    _entitle("app_builder")
    assert _pipeline_guard_active() is True


def test_both_call_sites_actually_consult_the_entitlement():
    """Structural, because the behavioural tests above cannot see the wiring.

    Mutation-proven: dropping `_pipeline_guard_active() and` from either
    call site left every other test in this file GREEN while restoring the
    exact defect — a blocked write redirecting to a tool the tenant does
    not have. Same class as the OAuth quirk call sites in
    `test_oauth_provider_dialects.py`: a predicate that is correct in
    isolation and simply never consulted.

    Asserted per call site rather than by counting, so moving one guard
    without the other is caught too.
    """
    import inspect

    from app.agent import tool_executor as tx

    src = inspect.getsource(tx)
    for predicate in ("_is_app_building_write(", "_is_app_building_exec("):
        calls = [
            line.strip() for line in src.splitlines()
            if predicate in line and "def " not in line and not line.strip().startswith("#")
        ]
        assert calls, f"no call site found for {predicate} — did it get renamed?"
        for line in calls:
            assert "_pipeline_guard_active()" in line, (
                f"call site `{line}` does not consult _pipeline_guard_active(). "
                f"With the app_builder family withheld this blocks the write and "
                f"redirects to a tool that is not in the model's tool list."
            )


def test_guard_survives_an_entitlement_layer_that_raises(monkeypatch):
    """It must fail OPEN — a broken entitlement read must not silently
    switch off a guard, and must never break `exec`."""
    import app.agent.tool_executor as tx

    def boom(_):
        raise RuntimeError("entitlements unavailable")

    monkeypatch.setattr(
        "app.agent.tool_entitlements.family_enabled", boom, raising=True,
    )
    assert tx._pipeline_guard_active() is True
