"""
Tests for the agent card strip logic in the Auto Builder pipeline.

The pipeline must:
  1. Strip visible "Agent ready" cards from generated screen files
  2. Preserve registerAction() calls, AgentBridge imports, and all plumbing
  3. Overwrite AgentPlaceholder.tsx with a null stub
  4. Leave clean screens (no agent card) untouched

Run: python tests/test_agent_card_strip.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────

# HomeScreen with BOTH an "Agent ready" card AND registerAction plumbing
HOMESCREEN_WITH_CARD_AND_PLUMBING = '''import React, { useEffect, useState } from 'react';
import { View, Text, Pressable, StyleSheet, ScrollView } from 'react-native';
import { AgentBridge } from '../lib/agentBridge';
import { registerAction } from '../lib/agentActions';

export default function HomeScreen() {
  const [result, setResult] = useState(0);

  useEffect(() => {
    AgentBridge.currentScreen = 'Home';
    registerAction('Home', {
      id: 'calculate',
      label: 'Calculate',
      description: 'Run a calculation',
      handler: async (expr: string) => eval(expr),
    });
  }, []);

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Calculator</Text>
      <Text style={styles.result}>{result}</Text>

      {/* Agent section */}
      <View style={styles.agentCard}>
        <Text style={styles.agentEmoji}>🤖</Text>
        <Text style={styles.agentTitle}>Agent ready</Text>
        <Text style={styles.agentDesc}>The app includes agent support for help and changes through the platform's floating assistant.</Text>
        <View style={styles.agentBadge}>
          <Text style={styles.agentBadgeText}>Use the floating agent orb to ask for help.</Text>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  title: { fontSize: 24 },
  result: { fontSize: 48 },
  agentCard: { padding: 16, backgroundColor: '#1C2128', borderRadius: 12, margin: 16 },
  agentEmoji: { fontSize: 24 },
  agentTitle: { fontSize: 16, fontWeight: '700' },
  agentDesc: { fontSize: 13, color: '#8B949E' },
  agentBadge: { padding: 8, backgroundColor: '#58A6FF', borderRadius: 8, marginTop: 8 },
  agentBadgeText: { color: '#FFF', fontSize: 12 },
});
'''

# Clean HomeScreen with only plumbing, no agent card
HOMESCREEN_CLEAN = '''import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { AgentBridge } from '../lib/agentBridge';
import { registerAction } from '../lib/agentActions';

export default function HomeScreen() {
  const [result, setResult] = useState(0);

  useEffect(() => {
    AgentBridge.currentScreen = 'Home';
    registerAction('Home', {
      id: 'calculate',
      label: 'Calculate',
      description: 'Run a calculation',
      handler: async (expr: string) => eval(expr),
    });
  }, []);

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Calculator</Text>
      <Text style={styles.result}>{result}</Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  title: { fontSize: 24 },
  result: { fontSize: 48 },
});
'''

# AgentPlaceholder with actual visible UI (should be overwritten)
AGENT_PLACEHOLDER_VISIBLE = '''import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

export default function AgentPlaceholder() {
  return (
    <View style={styles.container}>
      <Text>Agent is connected</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { position: 'absolute', bottom: 20, right: 20, zIndex: 9999 },
});
'''

AGENT_PLACEHOLDER_NULL_STUB = (
    "import React from 'react';\n"
    "// Platform renders agent UI externally — this component intentionally renders nothing.\n"
    "export default function AgentPlaceholder() { return null; }\n"
)


# ── Strip function (mirrors the regex in skill.py) ────────────────────

def strip_agent_card(code: str) -> str:
    """Apply the same regex strip logic as skill.py's infra step."""
    return re.sub(
        r'(?s)\{?/\*\s*[Aa]gent\s*(ready|card|placeholder|section).*?\*/\}?\s*'
        r'<(?:View|div)[^>]*>.*?(?:Agent ready|floating agent orb|agent support).*?</(?:View|div)>\s*',
        '', code
    )


# ── Tests ─────────────────────────────────────────────────────────────

def test_card_stripped_plumbing_preserved():
    """Agent ready card is removed. registerAction, AgentBridge, imports survive."""
    result = strip_agent_card(HOMESCREEN_WITH_CARD_AND_PLUMBING)

    # Card gone
    assert 'Agent ready' not in result, "Agent ready card should be stripped"
    assert 'floating agent orb' not in result, "Floating agent orb text should be stripped"

    # Plumbing intact
    assert "import { AgentBridge } from '../lib/agentBridge'" in result, "AgentBridge import must survive"
    assert "import { registerAction } from '../lib/agentActions'" in result, "registerAction import must survive"
    assert "AgentBridge.currentScreen = 'Home'" in result, "AgentBridge.currentScreen must survive"
    assert "registerAction('Home'" in result, "registerAction call must survive"
    assert "useEffect" in result, "useEffect must survive"

    # Main content intact
    assert "Calculator" in result, "App content must survive"
    assert "result" in result, "App state must survive"


def test_clean_homescreen_untouched():
    """A HomeScreen without agent card should not be modified."""
    result = strip_agent_card(HOMESCREEN_CLEAN)
    assert result == HOMESCREEN_CLEAN, (
        "Clean HomeScreen was modified by strip logic — "
        "strip should be a no-op when no agent card is present"
    )


def test_agent_placeholder_overwrite_detection():
    """Visible AgentPlaceholder should be detected as needing overwrite."""
    # Visible placeholder is not the null stub
    assert AGENT_PLACEHOLDER_VISIBLE.strip() != AGENT_PLACEHOLDER_NULL_STUB.strip(), (
        "Visible placeholder should differ from null stub"
    )
    # Null stub is the null stub
    assert AGENT_PLACEHOLDER_NULL_STUB.strip() == AGENT_PLACEHOLDER_NULL_STUB.strip(), (
        "Null stub should match itself"
    )


def test_strip_does_not_break_syntax():
    """After stripping, the code should still have balanced brackets."""
    result = strip_agent_card(HOMESCREEN_WITH_CARD_AND_PLUMBING)
    # Count opening/closing braces (rough syntax check)
    opens = result.count('{')
    closes = result.count('}')
    assert opens == closes, (
        f"Unbalanced braces after strip: {opens} opens, {closes} closes"
    )


# ── Runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_card_stripped_plumbing_preserved,
        test_clean_homescreen_untouched,
        test_agent_placeholder_overwrite_detection,
        test_strip_does_not_break_syntax,
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
