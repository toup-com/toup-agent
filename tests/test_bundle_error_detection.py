"""
Tests for bundle error detection in the Auto Builder pipeline.

Regression: the broad keyword 'error:' in BUNDLE_ERROR_KEYWORDS caused false
positives when scanning compiled JS bundles — the generated ErrorBoundary
component contains `{ hasError: false, error: null }`, and the substring
`error:` matched, causing every build to fail with:
    "Bundle compilation failed: hasError: false,"

Run: python tests/test_bundle_error_detection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.skills.builtins.app_builder.skill import AppBuilderSkill


# ── Fixtures ──────────────────────────────────────────────────────────

# Lines that appear in compiled dev bundles from the pipeline's own
# ErrorBoundary template — these MUST NOT be flagged as errors.
ERROR_BOUNDARY_LINES = [
    "  state: State = { hasError: false, error: null };",
    "    return { hasError: true, error };",
    "    console.error('[ErrorBoundary]', error, errorInfo);",
    "    onPress={() => this.setState({ hasError: false, error: null })}",
    "  if (this.state.hasError) {",
    "    {this.state.error?.message || 'An unexpected error occurred'}",
    "interface State { hasError: boolean; error: Error | null; }",
]

# Typical compiled JS patterns that should NOT trigger false positives.
SAFE_JS_LINES = [
    "  var _error = require('./error');",
    "  exports.error = function(msg) { return new Error(msg); };",
    "  } catch (error) { console.log(error); }",
    "  const errorHandler = (error) => setError(error);",
    "  Promise.reject(new Error('timeout'));",
    "  this.setState({ loading: false, error: null, data: result });",
    "  if (response.error) { throw new Error(response.error.message); }",
]

# Real Metro/Expo error lines that MUST be detected.
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


# ── Tests ─────────────────────────────────────────────────────────────

def test_error_boundary_not_flagged():
    """ErrorBoundary template lines must produce zero errors."""
    bundle = "\n".join(ERROR_BOUNDARY_LINES)
    errors = AppBuilderSkill._extract_errors_from_text(bundle)
    assert errors == [], (
        f"ErrorBoundary false positive — got {len(errors)} error(s): "
        f"{[e['error'] for e in errors]}"
    )


def test_safe_js_not_flagged():
    """Common JS patterns containing the word 'error' must not trigger."""
    bundle = "\n".join(SAFE_JS_LINES)
    errors = AppBuilderSkill._extract_errors_from_text(bundle)
    assert errors == [], (
        f"False positive on safe JS — got {len(errors)} error(s): "
        f"{[e['error'] for e in errors]}"
    )


def test_real_metro_errors_detected():
    """Every real Metro/Expo error line must be detected."""
    for line in REAL_METRO_ERRORS:
        bundle = f"// bundle preamble\n{line}\n// bundle continues"
        errors = AppBuilderSkill._extract_errors_from_text(bundle)
        assert len(errors) > 0, (
            f"Missed real Metro error: {line!r}"
        )


def test_mixed_bundle_only_catches_real_errors():
    """A bundle mixing safe code and real errors should only flag the errors."""
    safe_section = "\n".join(ERROR_BOUNDARY_LINES + SAFE_JS_LINES)
    error_section = "\n".join(REAL_METRO_ERRORS[:3])
    bundle = f"{safe_section}\n{error_section}"
    errors = AppBuilderSkill._extract_errors_from_text(bundle)
    # Should only find errors from the real error section
    error_texts = [e["error"] for e in errors]
    for safe_line in ERROR_BOUNDARY_LINES:
        assert safe_line.strip() not in error_texts, (
            f"ErrorBoundary line leaked through: {safe_line}"
        )
    assert len(errors) >= 1, "Should detect at least one real error"


def test_has_error_false_regression():
    """Exact regression: the string that caused job 34bdd032 to fail."""
    bundle = (
        "var ErrorBoundary = function() {\n"
        "  this.state = { hasError: false, error: null };\n"
        "};\n"
        "ErrorBoundary.prototype.setState = function(s) {\n"
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
        test_error_boundary_not_flagged,
        test_safe_js_not_flagged,
        test_real_metro_errors_detected,
        test_mixed_bundle_only_catches_real_errors,
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
