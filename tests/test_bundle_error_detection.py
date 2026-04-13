"""
Tests for bundle error detection in the Auto Builder pipeline.

The bundle validator scans Metro bundle responses for compilation errors.
Key design: only the first 150 lines of large bundles (>50KB) are scanned,
because Metro puts error overlays at the top.  The rest of the bundle is
compiled JS that naturally contains error-related keywords (TypeError,
SyntaxError, etc.) as class names and variable names — scanning it would
produce endless false positives.

Run: python tests/test_bundle_error_detection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.skills.builtins.app_builder.skill import AppBuilderSkill


# ── Fixtures ──────────────────────────────────────────────────────────

# Real Metro/Expo error lines that MUST be detected (appear in preamble).
REAL_METRO_ERRORS = [
    "error: Unable to resolve module 'react-native-svg' from '/app/components/Chart.tsx'",
    "Error: Cannot find module '@expo/vector-icons'",
    "SyntaxError: /app/screens/Home.tsx: Unexpected token (42:10)",
    "TransformError: /app/App.tsx: Missing semicolon",
    "TypeError: Cannot read properties of undefined (reading 'map')",
    "ReferenceError: navigation is not defined",
    "  Unable to resolve module './nonexistent' from '/app/lib/utils.ts'",
    "Module not found: Can't resolve './missing.tsx' in '/app/components'",
    "EADDRINUSE: Port 4001 is already in use",
    "error: Metro has encountered an error: ENOENT /app/assets/missing.png",
    "Error: /app/screens/Settings.tsx: Unexpected token, expected \",\"",
    "  Invariant Violation: Text strings must be rendered within a <Text> component",
    "Fatal error: Metro bundler process exited unexpectedly",
    "CompileError: WebAssembly.compile(): expected magic word",
    "ParseError: /app/utils/parser.ts: Unterminated string constant (15:22)",
]

# Lines from the ErrorBoundary template — safe because 'error:' only matches
# at start-of-line (BUNDLE_ERROR_LINE_PREFIXES), and these are mid-line.
ERROR_BOUNDARY_LINES = [
    "  state: State = { hasError: false, error: null };",
    "    return { hasError: true, error };",
    "    onPress={() => this.setState({ hasError: false, error: null })}",
    "  if (this.state.hasError) {",
    "interface State { hasError: boolean; error: Error | null; }",
]

# Metro runtime boilerplate — these contain keywords but appear deep in
# the compiled bundle (line 500+), so _validate_bundle never scans them.
# _extract_errors_from_text WILL flag them if passed directly, which is
# correct — the fix is in _validate_bundle (only scan first 150 lines),
# not in the keyword list.
METRO_RUNTIME_DEEP_BUNDLE_LINES = [
    '  throw new Error("Cannot find module \'" + moduleIdHint + "\'");',
    '  let currentCompileErrorMessage = null;',
    '  if (__DEV__) { throw new TypeError("Expected a component class..."); }',
    '  console.error("Cannot find module", moduleName);',
]


# ── Tests ─────────────────────────────────────────────────────────────

def test_real_metro_errors_detected():
    """Every real Metro/Expo error line must be detected."""
    for line in REAL_METRO_ERRORS:
        bundle = f"// bundle preamble\n{line}\n// bundle continues"
        errors = AppBuilderSkill._extract_errors_from_text(bundle)
        assert len(errors) > 0, (
            f"Missed real Metro error: {line!r}"
        )


def test_error_boundary_not_flagged():
    """ErrorBoundary template lines must not trigger (error: is mid-line)."""
    bundle = "\n".join(ERROR_BOUNDARY_LINES)
    errors = AppBuilderSkill._extract_errors_from_text(bundle)
    assert errors == [], (
        f"ErrorBoundary false positive — got {len(errors)} error(s): "
        f"{[e['error'] for e in errors]}"
    )


def test_preamble_only_scanning_skips_deep_bundle():
    """Simulate _validate_bundle's preamble-only scan for large bundles.

    A large bundle (>50KB) should only have its first 150 lines scanned.
    Metro runtime boilerplate at line 500+ should never be seen.
    """
    # Build a fake bundle: 150 clean preamble lines + deep runtime at line 500+
    preamble = ["// line " + str(i) for i in range(200)]
    deep_runtime = METRO_RUNTIME_DEEP_BUNDLE_LINES
    padding = ["// padding " + str(i) for i in range(300)]
    full_bundle = "\n".join(preamble + padding + deep_runtime)

    # Simulate the _validate_bundle logic: large bundle → only first 150 lines
    lines = full_bundle.split("\n", 150)
    scan_text = "\n".join(lines[:150])
    errors = AppBuilderSkill._extract_errors_from_text(scan_text)
    assert errors == [], (
        f"Deep bundle false positive leaked through preamble scan — got: "
        f"{[e['error'] for e in errors]}"
    )


def test_small_error_response_fully_scanned():
    """A small response (<50KB) should be fully scanned — it's an error page."""
    error_page = (
        "error: Unable to resolve module 'react-native-svg' "
        "from '/app/components/Chart.tsx'"
    )
    errors = AppBuilderSkill._extract_errors_from_text(error_page)
    assert len(errors) > 0, "Small error response should be caught"


def test_error_in_preamble_detected():
    """Real errors in the first 150 lines of a large bundle are caught."""
    preamble_error = "SyntaxError: /app/App.tsx: Unexpected token (10:5)"
    lines = [preamble_error] + ["// module code " + str(i) for i in range(200)]
    scan_text = "\n".join(lines[:150])
    errors = AppBuilderSkill._extract_errors_from_text(scan_text)
    assert len(errors) > 0, "Error in preamble should be detected"


def test_has_error_false_regression():
    """Regression: { hasError: false, error: null } must not be flagged."""
    bundle = (
        "var ErrorBoundary = function() {\n"
        "  this.state = { hasError: false, error: null };\n"
        "};\n"
    )
    errors = AppBuilderSkill._extract_errors_from_text(bundle)
    assert errors == [], (
        f"hasError:false regression — got: {[e['error'] for e in errors]}"
    )


# ── Runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_real_metro_errors_detected,
        test_error_boundary_not_flagged,
        test_preamble_only_scanning_skips_deep_bundle,
        test_small_error_response_fully_scanned,
        test_error_in_preamble_detected,
        test_has_error_false_regression,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
