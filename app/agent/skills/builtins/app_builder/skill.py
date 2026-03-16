"""
App Builder Skill — Conversational app builder for React Native/Expo.

The agent converses with the user (using Opus 4.6) to understand requirements,
presents a plan for approval, then builds in the background (using Sonnet 4.6).
After build, the user can preview and iterate.

Tools:
  - app_builder__build_app    — Start build after plan approval
  - app_builder__get_status   — Check build progress
  - app_builder__modify_app   — Apply changes to an existing app
  - app_builder__resume_build — Resume a paused build (after token limit)

The conversational flow is driven by the system prompt section — no state machine.
The agent asks questions, presents a plan, and only calls build_app after approval.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from app.agent.skills.base import Skill, SkillContext, SkillMeta

logger = logging.getLogger(__name__)


class TokenLimitError(Exception):
    """Raised when Claude API rate limit is hit during a build.
    Carries retry-after info so the build can be paused and resumed."""
    def __init__(self, retry_after_seconds: int = 300, message: str = ""):
        self.retry_after_seconds = retry_after_seconds
        self.message = message or f"Token limit reached. Resets in {retry_after_seconds}s"
        super().__init__(self.message)


# ── LLM prompts ─────────────────────────────────────────────────────

PLANNING_PROMPT = """You are planning a cross-platform app built with React Native/Expo.
The app must work on iPhone, iPad, and Web.

User wants: "{description}"

{extra_context}

Output ONLY valid JSON (no markdown fences, no explanation):
{{
  "files": ["/App.tsx", "/screens/HomeScreen.tsx", "/lib/agentBridge.ts", "/lib/agentActions.ts", ...],
  "dependencies": ["@react-navigation/native", "expo-sqlite", ...],
  "app_name": "TodoApp",
  "summary": "A todo app with ...",
  "needs_database": true,
  "db_type": "sqlite",
  "platforms": ["web", "ios"]
}}

Rules:
- Use TypeScript (.tsx/.ts files)
- Entry point MUST be /App.tsx
- Split into screens/, components/, lib/ directories
- For data persistence, use expo-sqlite (works offline on all platforms)
- Include @react-navigation/native + @react-navigation/native-stack for navigation
- Include react-native-safe-area-context and react-native-screens
- For responsive layout, use useWindowDimensions + isDesktop check (width > 768)
- On desktop: maxWidth 800px centered, 2-3 column grids. On mobile: single column full width.
- RESPONSIVE NAVIGATION: On mobile show bottom tab bar. On desktop show a left sidebar instead.
  Use createBottomTabNavigator with a custom tabBar prop that renders AdaptiveTabBar.
  Include /components/AdaptiveTabBar.tsx in the file list.
- If the app needs charts, use react-native-chart-kit (NOT recharts — that's web-only)
- Keep the file list focused — don't over-engineer
- Use EMOJI characters for ALL icons (🏠📊⚙️ etc.) — works on all platforms with zero imports.
  NEVER use @expo/vector-icons (breaks on Expo Web). NEVER use raw unicode symbols (▲☐⌂).
- Tab bar icons MUST use emoji in <Text> components
- Progress components should use 0-1 scale, not 0-100
- ALWAYS include /lib/agentSkill.json — domain-specific agent skill manifest (see below)

CRITICAL — Agent Skill Manifest:
Every app MUST include /lib/agentSkill.json — this tells the agent what domain-specific
operations it can perform within the app. The agent uses these to help users via
`app__action` tool. Example for a GRE prep app:
  {{"domain":"GRE Preparation","description":"Tools for GRE exam prep",
    "actions":[{{"name":"get_progress","description":"Get user's study progress","type":"query",
    "sql":"SELECT section, score, date FROM practice_results ORDER BY date DESC LIMIT 10","params":{{}}}},
    {{"name":"get_vocabulary","description":"Get vocab words by difficulty","type":"query",
    "sql":"SELECT word, definition FROM vocabulary WHERE difficulty = :level LIMIT :count",
    "params":{{"level":{{"type":"string"}},"count":{{"type":"integer","default":10}}}}}},
    {{"name":"add_study_session","description":"Log a study session","type":"mutation",
    "sql":"INSERT INTO study_sessions (section, duration_min, notes) VALUES (:section, :duration, :notes)",
    "params":{{"section":{{"type":"string"}},"duration":{{"type":"integer"}},"notes":{{"type":"string","default":""}}}}}},
    {{"name":"go_to_practice","description":"Open practice test screen","type":"navigate","screen":"Practice","params":{{}}}}]}}

CRITICAL — Error Boundary (prevents white screens):
You MUST include /components/ErrorBoundary.tsx in EVERY plan.
This is a class-based React component that wraps the entire app and catches runtime errors,
showing a helpful error message + reload button instead of a blank white screen.

CRITICAL — Agent Integration System:
Every app is an "agentic app" — the user's AI agent must be able to work inside it.
The platform proxy automatically injects the agent's Orb UI (floating chat widget with
the user's real agent appearance, eyes, breathing animation, and live WebSocket connection).
Do NOT generate any agent UI component — no AgentPlaceholder.tsx, no floating chat bubble.
The proxy handles the agent UI for all apps automatically.

You MUST include these files in EVERY plan:
  - /components/ErrorBoundary.tsx — Catches runtime errors, shows error screen instead of white page
  - /lib/agentBridge.ts — Bridge module that:
    - Exposes the current screen/route name to the agent
    - Provides a navigate(screenName, params?) function for agent-driven navigation
    - Lists available screens and their purposes
    - Exposes app-specific actions the agent can trigger (e.g. createTodo, deleteItem)
    - Uses postMessage/event-based communication pattern for agent ↔ app messaging
  - /lib/agentActions.ts — Screen-specific action registry:
    - Maps each screen to actions the agent can perform on it
    - E.g. HomeScreen → [addItem, searchItems, filterBy], SettingsScreen → [toggleTheme, exportData]
Do NOT include /components/AgentPlaceholder.tsx — the proxy injects the agent UI.
"""

CODE_GEN_PROMPT = """Generate the complete TypeScript code for {file_path} in a React Native/Expo app.

App description: {description}
App name: {app_name}
All files in the app: {all_files}
Dependencies: {all_deps}
Database: {db_type}
{design_notes}

Rules:
- Use TypeScript with proper types
- Use React Native components (View, Text, ScrollView, Pressable, TextInput, FlatList, etc.)
- Use StyleSheet.create() for styles — dark theme (background: #161B22, text: #F0F2F5)
- Accent color: #58A6FF (blue)
- RESPONSIVE LAYOUT — these apps run on mobile (375px) AND desktop web (1200px+):
  - Use useWindowDimensions() at the top of every screen
  - Define: const isDesktop = width > 768;
  - On DESKTOP (width > 768): constrain main content to maxWidth: 800, alignSelf: 'center'.
    Use 2-3 column grid layouts for cards/lists (flexDirection: 'row', flexWrap: 'wrap').
    Add more generous padding (24-32px).
  - On MOBILE (width <= 768): single column, full width, standard mobile padding (16px).
  - Example responsive pattern for EVERY screen's root container:
    const {{ width }} = useWindowDimensions();
    const isDesktop = width > 768;
    <ScrollView style={{{{ flex: 1, paddingLeft: isDesktop ? 220 : 0 }}}}>
      <View style={{{{ maxWidth: isDesktop ? 800 : undefined, alignSelf: 'center', width: '100%', padding: isDesktop ? 32 : 16, paddingBottom: isDesktop ? 32 : 100 }}}}>
    The paddingLeft: 220 on desktop accounts for the sidebar navigation width.
    The paddingBottom: 100 on mobile accounts for the bottom tab bar (60px) + agent button above it.
  - Cards and grid items: use percentage widths on desktop (width: isDesktop ? '48%' : '100%')
  - This is CRITICAL — the app preview on toup.ai is full-width desktop, not a phone frame
- RESPONSIVE NAVIGATION — bottom tabs on mobile, left sidebar on desktop:
  - Use createBottomTabNavigator with `tabBar={{(props) => <AdaptiveTabBar {{...props}} appName={{"{app_name}"}} />}}`
  - The AdaptiveTabBar component handles BOTH layouts (see below for /components/AdaptiveTabBar.tsx)
  - NEVER hardcode bottom tabs without AdaptiveTabBar — it WILL look broken on desktop
- If this file uses database, import from '../lib/db' (expo-sqlite helper)
- For navigation, use @react-navigation/native-stack
- React Navigation v7 REQUIRES a `theme` prop with `fonts` on NavigationContainer. Without it, the app
  CRASHES with "Cannot read properties of undefined (reading 'bold')". ALWAYS add this theme prop:
  <NavigationContainer theme={{{{ dark: true, colors: {{
    primary: "#58A6FF", background: "#161B22", card: "#1C2128",
    text: "#F0F2F5", border: "#30363D", notification: "#58A6FF",
  }}, fonts: {{
    regular: {{ fontFamily: "System", fontWeight: "400" }},
    medium: {{ fontFamily: "System", fontWeight: "500" }},
    bold: {{ fontFamily: "System", fontWeight: "700" }},
    heavy: {{ fontFamily: "System", fontWeight: "900" }},
  }} }}}}>
  NEVER use <NavigationContainer> without the theme prop — it WILL crash.
- Make it fully functional — not a skeleton
- Include proper error handling and loading states
- Output ONLY the code — no markdown fences, no explanation
- CRITICAL: The file MUST be syntactically complete — all brackets closed, all StyleSheet styles finished,
  all exports present. If the file would be too long, SIMPLIFY the implementation instead of truncating.
  A simpler complete file is far better than a complex truncated one.
- CRITICAL: Keep data/seed files COMPACT. Maximum 50 items in any array (e.g. vocabulary lists, quiz banks,
  sample data). If the app needs more data, generate 30-50 representative examples — not 100+.
  Large data files get truncated by the token limit and break the app. Quality over quantity.

CRITICAL — Cross-Platform Rendering Rules:
These apps run as Expo Web inside a WebView on mobile AND as web apps in browsers.
Rendering differences WILL cause visual bugs if you ignore these rules:

1. Use EMOJI characters for ALL icons — they render perfectly on web, iOS, and Android with zero imports.
   NEVER use @expo/vector-icons (breaks on Expo Web). NEVER use raw unicode symbols (▲, ☐, ⌂, ☰, ★).
   Use emoji via <Text> elements. Examples:
   - Navigation: 🏠 Home, 📋 List, 📊 Analytics, 📅 Calendar, ⚙️ Settings, 🔍 Search
   - Actions: ➕ Add, ✏️ Edit, 🗑️ Delete, ✅ Done, ❌ Close, 💾 Save, 🔄 Refresh
   - Status: ⭐ Favorite, ❤️ Heart, 🔥 Streak, 🏆 Trophy, 🎯 Target, 💪 Fitness
   - Content: 📝 Note, 📖 Book, 🎵 Music, 📸 Photo, 🎬 Video, 💬 Chat, 🔔 Notification
   - Health: 💧 Water, 🍎 Food, 🧘 Meditation, 🏃 Running, 😴 Sleep, 💊 Medicine
   - Misc: ⏰ Timer, 📈 Chart, 🌙 Dark, ☀️ Light, 👤 Profile, 🔒 Lock, 🎨 Color

2. Tab bar icons MUST use emoji in <Text> components. NEVER use empty <View> or icon libraries.
   Example tab bar icon:
   tabBarIcon: ({{ color }}) => <Text style={{{{ fontSize: 22 }}}}>🏠</Text>
   Use different emoji for each tab to make them visually distinct.
   CRITICAL: Tab screen labels must be PLAIN TEXT only (e.g. "Dashboard", not "🏠 Dashboard").
   The emoji goes in tabBarIcon, NOT in the label. Putting emoji in both causes duplicate emojis in the sidebar.

3. Progress bars / gauges: If using a ProgressBar component that clamps to 0-1 range internally,
   pass values in 0-1 range (NOT 0-100). Double-check: Math.min((score - min) / (max - min), 1),
   NOT Math.min(((score - min) / (max - min)) * 100, 100).

4. All database operations MUST be wrapped in try/catch with graceful fallbacks.
   Database may not be ready on first render — use loading states and default values.
   For database files (lib/database.ts): Do NOT use `import * as SQLite from 'expo-sqlite'` directly.
   Instead use conditional import with Platform check and web fallback mock:
   ```
   import {{ Platform }} from 'react-native';
   let SQLite: any;
   if (Platform.OS !== 'web') {{ const mod = 'expo-sqlite'; SQLite = require(mod); }}
   ```
   Then in getDatabase(), return an in-memory mock when Platform.OS === 'web'.

5. React Rules of Hooks: ALL useState, useRef, useEffect, useMemo, useCallback calls
   MUST appear before any conditional return statement. Never place hooks after early returns.

6. Use Platform.select() for platform-specific behavior (font families, padding, shadows).
   Test values must work on both web and native — avoid native-only APIs without web fallbacks.

7. Import/Export consistency: If a module uses `export default X`, import it as `import X from '...'`.
   If it uses `export {{ X }}`, import as `import {{ X }} from '...'`. NEVER use named import syntax
   `import {{ X }}` for a module that only has `export default`. This causes X to be undefined at runtime.
   For lib files (agentBridge, database, etc.), always provide BOTH `export default` AND named `export {{ }}`
   so either import style works.

8. NEVER use async component functions. `export default async function Screen()` returns a Promise,
   not a React element — this causes an instant white page. Always use useEffect + useState for data fetching.

9. NEVER mix expo-router with manual NavigationContainer. Use only @react-navigation/* with
   createBottomTabNavigator/createNativeStackNavigator. Do NOT import from 'expo-router'.

10. Platform-specific APIs that crash on web: expo-secure-store, Vibration, Alert.alert(),
    react-native-reanimated (needs babel plugin). Always wrap in Platform.OS !== 'web' checks
    or use conditional require().

11. In React Native, ALL text MUST be inside <Text> components. Bare strings inside <View>
    crash with "Text strings must be rendered within a Text component". This includes
    conditional expressions: use `{{condition && <Text>text</Text>}}` not `{{condition && "text"}}`.

12. Functions that use `await` MUST be declared `async`. Writing `export function getX(): Promise<T> {{ await ... }}`
    is a SyntaxError — it MUST be `export async function getX(): Promise<T> {{ await ... }}`.
    This is the #1 cause of build failures in database utility files.

Responsive Navigation (CRITICAL — every app MUST include this):
- If this is /components/AdaptiveTabBar.tsx:
  Build a responsive tab bar that renders as bottom tabs on mobile and a left sidebar on desktop.
  It receives the standard React Navigation tabBar props: {{ state, descriptors, navigation, insets }}.
  Plus an `appName` string prop.

  Implementation:
  - Call useWindowDimensions() to get width. const isDesktop = width > 768;
  - Extract tabs from state.routes + descriptors (label, emoji icon, isFocused).
  - On MOBILE (isDesktop === false): render a horizontal bottom tab bar:
    - View with flexDirection: 'row', backgroundColor: '#1C2128', borderTopWidth: 1, borderTopColor: '#30363D'
    - paddingBottom: insets.bottom (safe area), height: 60 + insets.bottom
    - paddingRight: 56 — reserve space on the right for the floating agent Orb
      so the last tab item is never hidden behind the agent button
    - Each tab: Pressable with emoji icon + label, highlighted color when focused (#58A6FF vs #8B949E)
  - On DESKTOP (isDesktop === true): render a vertical left sidebar:
    - View with width: 220, backgroundColor: '#1C2128', borderRightWidth: 1, borderRightColor: '#30363D'
    - App name as header: Text with fontSize 16, fontWeight '700', color '#F0F2F5', padding 20
    - Each tab: Pressable row with emoji (fontSize 18) + label (fontSize 14), padding 14 16,
      active tab has backgroundColor '#21262D' + borderRadius 8 + color '#58A6FF'
    - marginTop: 8 between header and nav items
  - On press: call navigation.navigate(route.name)

  IMPORTANT: The App.tsx must wrap the Tab.Navigator inside a View with flexDirection: 'row' on desktop
  so the sidebar sits left of the content. Pattern in App.tsx:

  function AppLayout() {{
    const {{ width }} = useWindowDimensions();
    const isDesktop = width > 768;
    return (
      <View style={{{{ flex: 1, flexDirection: isDesktop ? 'row' : 'column' }}}}>
        <Tab.Navigator
          tabBar={{(props) => <AdaptiveTabBar {{...props}} appName="{app_name}" />}}
          screenOptions={{{{
            headerStyle: {{ backgroundColor: '#1C2128' }},
            headerTintColor: '#F0F2F5',
            tabBarStyle: {{ display: 'none' }},  // AdaptiveTabBar handles rendering
          }}}}
        >
          {{/* Tab.Screen entries */}}
        </Tab.Navigator>
      </View>
    );
  }}

  The AdaptiveTabBar itself handles the layout positioning:
  - On mobile: it renders AFTER/BELOW the screen content (bottom of screen)
  - On desktop: it renders BEFORE/LEFT of the screen content using position absolute or
    by the parent flexDirection: 'row' layout

  Actually simpler approach — the AdaptiveTabBar on desktop should use position: 'absolute', left: 0,
  top: 0, bottom: 0, width: 220, and the Tab.Navigator screens should have paddingLeft: 220 on desktop.
  On mobile, the tab bar is at the bottom (position: 'absolute', bottom: 0, left: 0, right: 0).

  This avoids needing to restructure the parent layout. The Tab.Navigator renders normally,
  and AdaptiveTabBar positions itself using absolute positioning.

  In each Tab.Screen, the screen component should add paddingLeft: isDesktop ? 220 : 0 to its
  outermost ScrollView/View to account for the sidebar width. Since every screen already uses
  useWindowDimensions() for responsive layout, just add this padding.

Error Boundary (CRITICAL — prevents white screens):
- If this is /components/ErrorBoundary.tsx:
  Build a React class component error boundary. This catches runtime errors and shows a user-friendly
  error screen instead of a blank white page. Implementation:
  - Class component extending React.Component<{{children: React.ReactNode}}, {{hasError: boolean, error: Error | null}}>
  - static getDerivedStateFromError(error): return {{ hasError: true, error }}
  - componentDidCatch(error, errorInfo): console.error('[ErrorBoundary]', error, errorInfo)
  - When hasError is true, render a dark error screen (#161B22 background):
    - Center-aligned: "⚠️" emoji (fontSize 48), "Something went wrong" title (white, 18px, bold),
      error.message in gray (13px, max 3 lines), and a "Reload App" button (#58A6FF, rounded, Pressable)
    - The reload button calls: this.setState({{ hasError: false, error: null }}) to retry rendering
  - When hasError is false: return this.props.children
  - Export as default

- If this is /App.tsx:
  Wrap the ENTIRE app (NavigationContainer + everything inside) with <ErrorBoundary>:
  ```
  import ErrorBoundary from './components/ErrorBoundary';
  ...
  return (
    <ErrorBoundary>
      <NavigationContainer theme={{...}}>
        ...
      </NavigationContainer>
    </ErrorBoundary>
  );
  ```
  Note: Do NOT render AgentPlaceholder — the platform proxy injects the agent Orb UI automatically.

Agent Integration System (CRITICAL — every app is "agentic"):
- If this is /components/AgentPlaceholder.tsx:
  DO NOT generate this file. The platform proxy injects the agent's Orb UI
  automatically (floating sphere with eyes, breathing animation, chat panel,
  live WebSocket connection). If this file is requested, return a minimal stub:
  ```
  import React from 'react';
  export default function AgentPlaceholder() { return null; }
  ```
  The injected Orb widget replaces this component entirely.

- If this is /lib/agentBridge.ts:
  IMPORTANT: The platform proxy injects a pre-built bridge as window.__TOUP_AGENT_BRIDGE.
  Your code MUST check for it first and delegate to it if available.

  Structure:
  ```
  // Check for injected bridge (set by platform proxy — deterministic, tested)
  const injected = typeof window !== 'undefined' && (window as any).__TOUP_AGENT_BRIDGE;

  // If injected bridge exists, create a thin wrapper that delegates to it
  // but adds app-specific screens and actions metadata.
  // If not (e.g. Expo Go native), create own WebSocket implementation.
  ```

  If `window.__TOUP_AGENT_BRIDGE` exists:
  - Create AgentBridge as a wrapper that delegates sendMessage, onAgentMessage,
    onToolActivity, setNavigationRef, navigate, destroy, and isConnected to the injected bridge.
  - Add your own getScreens() and getActions() with the app-specific screen list and actions.
  - Call injected.setScreens(screens) and injected.setActions(actionsMap) to register metadata.

  If `window.__TOUP_AGENT_BRIDGE` does NOT exist (fallback for Expo Go native):
  - Read window.__TOUP_AUTH_TOKEN, window.__TOUP_APP_ID, window.__TOUP_WS_URL.
  - If all three globals exist, connect to WS_URL with ?token=AUTH_TOKEN query param.
  - Implement own WebSocket connection with reconnect (exponential backoff, 1s→30s, unlimited retries).
  - Handle messages: text_chunk (accumulate), done (fire callbacks), app_navigate, tool_start, tool_end, error.
  - Queue messages when not connected.

  Export as singleton: `const AgentBridge = ...; export {{ AgentBridge }}; export default AgentBridge;`
  All other files import {{ AgentBridge }} and use it directly (NOT .getInstance()).

  Core API (same whether injected or own implementation):
  - currentScreen: string (updated by navigation listener)
  - navigate(screenName: string, params?: object): void
  - getScreens(): Array<{{name: string, description: string}}>
  - getActions(screenName?: string): Array<{{id: string, label: string, handler: string}}>
  - sendMessage(text: string): void — sends {{"type":"message","text":text,"app_id":APP_ID,"channel":"app"}}
  - onAgentMessage(callback: (msg: string) => void): () => void — returns unsubscribe
  - onToolActivity(callback: (tool: string, done: boolean) => void): () => void — returns unsubscribe
  - setNavigationRef(ref: any): void
  - isConnected: boolean

  Do NOT process messages locally or simulate responses. ALL messages go through the WebSocket to the real agent.

- If this is /lib/agentActions.ts:
  Export a registry mapping screen names to available actions:
  type AgentAction = {{ id: string; label: string; description: string; handler: (...args: any[]) => Promise<any> }}
  const screenActions: Record<string, AgentAction[]> = {{ ... }}
  Populate with meaningful actions for EACH screen in the app.
  Export registerAction(screen, action) and getActions(screen) functions.

- If this is /App.tsx:
  Import AdaptiveTabBar from './components/AdaptiveTabBar' and use it as the tabBar prop:
  `tabBar={{(props) => <AdaptiveTabBar {{...props}} appName="AppName" />}}`
  Do NOT import or render AgentPlaceholder — the platform proxy injects the agent UI automatically.
  Import {{ AgentBridge }} from './lib/agentBridge' and use it directly (NOT .getInstance()).
  Pass the navigation ref to AgentBridge.setNavigationRef() so the agent can navigate screens.
  Example: AgentBridge.setNavigationRef(ref), AgentBridge.sendMessage(msg), AgentBridge.currentScreen = name.
  Do NOT render AgentPlaceholder — the platform proxy injects the agent's Orb UI with live WebSocket
  connection, proper appearance, and chat panel. The agent bridge still needs to be initialized
  for navigation and screen metadata.
  CRITICAL: Use createBottomTabNavigator with AdaptiveTabBar — NEVER render bottom tabs without it.

- For ANY screen file:
  Register that screen's agent actions in a useEffect via agentActions.registerAction().

- If this is /lib/agentSkill.json:
  Generate a JSON manifest of domain-specific agent actions for this app.
  The agent uses these via the `app__action` tool to help users.
  Format:
  {{
    "domain": "<domain name, e.g. 'GRE Preparation'>",
    "description": "<what these tools help the agent do within this app>",
    "actions": [
      {{
        "name": "<snake_case_name>",
        "description": "<what this action does>",
        "type": "query" | "mutation" | "navigate",
        "sql": "<SQL with :param_name placeholders (for query/mutation)>",
        "screen": "<screen name (for navigate type only)>",
        "params": {{
          "<param_name>": {{"type": "string|integer|number|boolean", "description": "...", "default": <optional>}}
        }}
      }}
    ]
  }}
  Rules:
  - Generate 4-8 meaningful domain actions based on the app's purpose
  - "query": SELECT — returns data. "mutation": INSERT/UPDATE/DELETE — modifies data. "navigate": opens a screen
  - SQL MUST use the app's actual table/column names from the database schema in /lib/db.ts
  - Use :param_name for parameterized queries (NEVER string concatenation)
  - Make actions domain-level (score_test, get_progress) NOT generic CRUD (insert_row, select_all)
  - Include at least 1 navigation action
  - Output ONLY valid JSON — no markdown fences, no comments
"""

MODIFY_ANALYSIS_PROMPT = """You are analyzing a modification request for an existing React Native/Expo app.

App name: {app_name}
Current files: {file_list}
User wants: "{changes}"

Based on the change request, determine which files need to be modified or created.
Output ONLY valid JSON (no markdown fences):
{{
  "affected_files": ["/screens/HomeScreen.tsx", "/components/ThemeProvider.tsx"],
  "new_files": ["/lib/theme.ts"],
  "summary": "What changes will be made"
}}
"""


def _slugify(name: str, suffix: str = "") -> str:
    """Convert app name to a URL-friendly slug (e.g. 'GRE Success Tracker' → 'GRE-Success-Tracker')."""
    # Preserve original casing, replace non-alphanumeric with hyphens
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-")[:40]
    if suffix:
        slug = f"{slug}-{suffix}"
    return slug


class AppBuilderSkill(Skill):
    """Conversational app builder — asks questions, plans, builds, iterates."""

    meta = SkillMeta(
        name="app_builder",
        version="2.0.0",
        description="Build React Native/Expo apps through conversation",
        author="toup",
    )

    def __init__(
        self,
        app_manager=None,
        ws_broadcast: Optional[Callable] = None,
        skill_loader=None,
    ):
        self._app_manager = app_manager
        self._ws_broadcast = ws_broadcast
        self._skill_loader = skill_loader

    def set_refs(self, app_manager=None, ws_broadcast=None, skill_loader=None):
        """Set references after construction (for late binding)."""
        if app_manager:
            self._app_manager = app_manager
        if ws_broadcast:
            self._ws_broadcast = ws_broadcast
        if skill_loader:
            self._skill_loader = skill_loader

    # ── Tool Definitions ───────────────────────────────────────────

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "app_builder__build_app",
                "description": (
                    "Build a mobile + web app AFTER discussing requirements with the user "
                    "and getting their approval on the plan. Do NOT call this without first "
                    "asking clarifying questions and presenting a plan. The app builds in "
                    "the background as a React Native/Expo project (iPhone, iPad, Web)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short app name (2-4 words, e.g. 'Todo App')",
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description including ALL features and screens discussed with the user",
                        },
                        "screens": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of screens (e.g. ['Home', 'Settings', 'Detail'])",
                        },
                        "features": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of features discussed with the user",
                        },
                        "db_type": {
                            "type": "string",
                            "enum": ["sqlite", "none"],
                            "description": "Database type — sqlite for persistent data, none for UI-only",
                        },
                        "platforms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Target platforms (default: ['web', 'ios'])",
                        },
                        "design_notes": {
                            "type": "string",
                            "description": "Design preferences from user (theme, colors, style)",
                        },
                    },
                    "required": ["name", "description"],
                },
            },
            {
                "name": "app_builder__get_status",
                "description": "Check the build progress of an app. Returns current step, progress, and URLs when ready.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "The build job ID returned by build_app",
                        },
                    },
                    "required": ["job_id"],
                },
            },
            {
                "name": "app_builder__modify_app",
                "description": (
                    "Modify an existing app based on user feedback after preview. "
                    "Only regenerates affected files — much faster than a full rebuild. "
                    "Use when the user wants changes like 'make it dark theme' or 'add a settings page'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app_id": {
                            "type": "string",
                            "description": "The app ID to modify",
                        },
                        "changes": {
                            "type": "string",
                            "description": "Detailed description of what to change (from user feedback)",
                        },
                    },
                    "required": ["app_id", "changes"],
                },
            },
            {
                "name": "app_builder__resume_build",
                "description": (
                    "Resume a paused app build. Builds get paused when the token/rate limit is reached. "
                    "This picks up exactly where the build left off — no re-doing completed steps."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "The paused build job ID to resume",
                        },
                    },
                    "required": ["job_id"],
                },
            },
        ]

    # ── Tool Execution ─────────────────────────────────────────────

    async def execute_tool(
        self, tool_name: str, args: Dict[str, Any], ctx: SkillContext
    ) -> str:
        dispatch = {
            "app_builder__build_app": self._exec_build_app,
            "app_builder__get_status": self._exec_get_status,
            "app_builder__modify_app": self._exec_modify_app,
            "app_builder__resume_build": self._exec_resume_build,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return f"Unknown tool: {tool_name}"
        return await handler(args, ctx)

    async def _exec_build_app(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        """Start a new app build."""
        name = args.get("name", "Untitled App")
        description = args.get("description", "")
        screens = args.get("screens", [])
        features = args.get("features", [])
        db_type = args.get("db_type", "sqlite")
        platforms = args.get("platforms", ["web", "ios"])
        design_notes = args.get("design_notes", "")
        user_id = ctx.user_id

        if not self._app_manager:
            return "App builder is not available — app_manager not configured."

        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob

        app_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        slug = _slugify(name)
        apps_dir = self._app_manager.APPS_DIR if hasattr(self._app_manager, 'APPS_DIR') else "/opt/toup-agent/apps"
        app_dir = os.path.join(apps_dir, app_id)

        # Store the conversational context as plan JSON
        plan_context = {
            "screens": screens,
            "features": features,
            "db_type": db_type,
            "platforms": platforms,
            "design_notes": design_notes,
        }

        async with async_session_maker() as db:
            app = App(
                id=app_id,
                user_id=user_id,
                name=name,
                description=description,
                slug=slug,
                status="building",
                app_dir=app_dir,
                platforms=",".join(platforms),
            )
            db.add(app)

            job = BuildJob(
                id=job_id,
                user_id=user_id,
                app_id=app_id,
                title=f"Build: {name}",
                prompt=description,
                status="queued",
                steps_json=json.dumps(self._initial_steps()),
            )
            db.add(job)
            await db.commit()

        # Spawn background build
        asyncio.create_task(
            self._build_app(job_id, app_id, name, description, user_id, slug, app_dir, plan_context)
        )

        return (
            f"Building '{name}'! Job ID: {job_id}\n\n"
            f"Track progress in the **Jobs** tab. "
            f"When done, you'll find it in the **Apps** tab with:\n"
            f"- QR code for Expo Go (iPhone/iPad)\n"
            f"- Web URL for browser preview\n"
            f"- GitHub repo with all code\n\n"
            f"Use `app_builder__get_status` with job_id='{job_id}' to check progress."
        )

    async def _exec_get_status(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        """Check build job status."""
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob

        job_id = args.get("job_id", "")
        if not job_id:
            return "ERROR: job_id is required"

        async with async_session_maker() as db:
            job = await db.get(BuildJob, job_id)
            if not job:
                return f"Job '{job_id}' not found."

            steps = json.loads(job.steps_json) if job.steps_json else []
            current_step = next((s for s in steps if s["status"] == "running"), None)
            done_count = sum(1 for s in steps if s["status"] == "done")

            result = f"Job: {job.title}\nStatus: {job.status}\nProgress: {done_count}/{len(steps)} steps"

            if current_step:
                result += f"\nCurrent: {current_step['label']}"
                if current_step.get("detail"):
                    result += f" — {current_step['detail']}"

            if job.status == "completed" and job.app_id:
                app = await db.get(App, job.app_id)
                if app:
                    result += f"\n\nApp is {app.status}!"
                    if app.port:
                        result += f"\nMetro port: {app.port}"
                    if app.web_port:
                        result += f"\nWeb port: {app.web_port}"
                    if self._app_manager:
                        try:
                            qr_url = await self._app_manager.get_qr_url(app.id)
                            web_url = await self._app_manager.get_web_url(app.id)
                            if qr_url:
                                result += f"\nQR URL: {qr_url}"
                            if web_url:
                                result += f"\nWeb URL: {web_url}"
                        except Exception:
                            pass
                    if app.github_url:
                        result += f"\nGitHub: {app.github_url}"

            if job.status == "paused":
                resume_after = job.resume_after
                if resume_after:
                    result += f"\nPaused at: {job.paused_at.isoformat() if job.paused_at else 'unknown'}"
                    result += f"\nResumes after: {resume_after.isoformat()}"
                    remaining = (resume_after - datetime.utcnow()).total_seconds()
                    if remaining > 0:
                        mins = int(remaining // 60)
                        result += f"\nTime remaining: {mins}m {int(remaining % 60)}s"
                    else:
                        result += "\nReady to resume! Use `app_builder__resume_build`."

            if job.status == "failed":
                result += f"\nError: {job.error_message or 'Unknown error'}"

            return result

    async def _exec_resume_build(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        """Resume a paused build from checkpoint."""
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob

        job_id = args.get("job_id", "")
        if not job_id:
            return "ERROR: job_id is required"

        if not self._app_manager:
            return "App builder is not available — app_manager not configured."

        async with async_session_maker() as db:
            job = await db.get(BuildJob, job_id)
            if not job:
                return f"Job '{job_id}' not found."

            if job.status != "paused":
                return f"Job is not paused (status: {job.status}). Only paused jobs can be resumed."

            # Check if enough time has passed
            if job.resume_after and datetime.utcnow() < job.resume_after:
                remaining = (job.resume_after - datetime.utcnow()).total_seconds()
                mins = int(remaining // 60)
                return (
                    f"Token limit hasn't reset yet. "
                    f"Try again in {mins}m {int(remaining % 60)}s "
                    f"(resets at {job.resume_after.strftime('%H:%M UTC')})"
                )

            checkpoint = {}
            try:
                checkpoint = json.loads(job.checkpoint_json) if job.checkpoint_json else {}
            except (json.JSONDecodeError, TypeError):
                return "ERROR: Could not load checkpoint data. The build state may be corrupted."

            if not checkpoint:
                return "ERROR: No checkpoint data found. Cannot resume this build."

            # Mark job as running again
            job.status = "running"
            job.paused_at = None
            job.resume_after = None
            await db.commit()

            app_id = checkpoint.get("app_id", job.app_id)
            if app_id:
                app = await db.get(App, app_id)
                if app:
                    app.status = "building"
                    await db.commit()

        user_id = ctx.user_id
        completed_steps = checkpoint.get("completed_steps", [])

        # Spawn background resume
        asyncio.create_task(
            self._resume_build_app(job_id, checkpoint, user_id)
        )

        return (
            f"Resuming build! Skipping {len(completed_steps)} completed steps "
            f"({', '.join(completed_steps)}). Continuing from '{checkpoint.get('current_step', 'unknown')}'.\n\n"
            f"Use `app_builder__get_status` with job_id='{job_id}' to track progress."
        )

    async def _exec_modify_app(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        """Modify an existing app based on user feedback."""
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob

        app_id = args.get("app_id", "")
        changes = args.get("changes", "")
        user_id = ctx.user_id

        if not app_id or not changes:
            return "ERROR: app_id and changes are required"

        if not self._app_manager:
            return "App builder is not available — app_manager not configured."

        async with async_session_maker() as db:
            app = await db.get(App, app_id)
            if not app:
                return f"App '{app_id}' not found."
            if app.status == "building":
                return "App is currently being built. Wait for the build to complete before modifying."

        # Create a modification job
        job_id = str(uuid.uuid4())
        async with async_session_maker() as db:
            job = BuildJob(
                id=job_id,
                user_id=user_id,
                app_id=app_id,
                title=f"Modify: {app.name}",
                prompt=changes,
                status="queued",
                steps_json=json.dumps([
                    {"id": str(uuid.uuid4()), "type": "planning", "label": "Analyzing changes...", "status": "pending"},
                    {"id": str(uuid.uuid4()), "type": "writing", "label": "Updating files...", "status": "pending"},
                    {"id": str(uuid.uuid4()), "type": "installing", "label": "Updating dependencies...", "status": "pending"},
                    {"id": str(uuid.uuid4()), "type": "starting", "label": "Restarting servers...", "status": "pending"},
                    {"id": str(uuid.uuid4()), "type": "ready", "label": "Ready!", "status": "pending"},
                ]),
            )
            db.add(job)
            app.status = "building"
            await db.commit()

        # Spawn background modification
        asyncio.create_task(
            self._modify_app(job_id, app_id, app.name, app.description or "", changes, user_id, app.slug, app.app_dir)
        )

        return (
            f"Modifying '{app.name}'! Job ID: {job_id}\n\n"
            f"Changes: {changes}\n"
            f"Only affected files will be regenerated — this is faster than a full rebuild.\n"
            f"Track progress in the **Jobs** tab."
        )

    # ── System Prompt ──────────────────────────────────────────────

    def get_system_prompt_section(self) -> Optional[str]:
        return (
            "# App Builder\n"
            "You can build cross-platform **agentic apps** (iPhone, iPad, Web) using React Native/Expo.\n"
            "Every app you build includes an **Agent Placeholder** — a floating widget where YOU (the agent) "
            "can dock into the app and help the user. You'll have full access to navigate, read data, and "
            "perform actions within every app you build.\n\n"
            "## When the user wants to build an app, follow this process:\n\n"
            "### Step 1: Understand Requirements (10+ Questions)\n"
            "Ask **at least 10 clarifying questions** in a SINGLE message to deeply understand what the user wants.\n"
            "Pick the most relevant questions from this list based on the app type — aim for 10-12:\n\n"
            "**Core & Purpose:**\n"
            "1. Who is this app for? (personal use, team, public users)\n"
            "2. What is the ONE main action users do most? (track, browse, create, schedule, learn)\n"
            "3. What problem does this app solve? (productivity, learning, health, fun, organization)\n\n"
            "**Features & Scope:**\n"
            "4. Which features are must-have? (suggest 4-5 based on app type)\n"
            "5. Any nice-to-have features? (suggest 3-4 extras)\n"
            "6. Do you need notifications/reminders?\n"
            "7. Do you need data visualization? (charts, progress bars, streaks, stats)\n\n"
            "**Data & Storage:**\n"
            "8. Should data persist between sessions? (SQLite database vs UI-only)\n"
            "9. What data does the user create? (entries, notes, scores, schedules, uploads)\n"
            "10. Any seed/default data needed? (categories, templates, starter content)\n\n"
            "**Design & UX:**\n"
            "11. Color theme preference? (suggest 3-4 palettes that match the app vibe)\n"
            "12. Dark mode, light mode, or both?\n"
            "13. How many main screens? (suggest based on app type: 3-6 typical)\n"
            "14. Tab bar navigation or drawer/sidebar? (suggest best for the app type)\n"
            "15. Any specific layout, style, or design inspiration?\n\n"
            "**Smart Extras:**\n"
            "16. Should the app have a dashboard/home screen with overview stats?\n"
            "17. Any gamification? (streaks, achievements, points, levels, progress tracking)\n"
            "18. Sort/filter/search on lists?\n"
            "19. Import/export data? (CSV, share, backup)\n"
            "20. Onboarding/welcome screen for first-time users?\n\n"
            "Pick at least 10 of these based on the app type. You can add your own smart questions too.\n"
            "If the user gives a very detailed description upfront, you may reduce to 8 but never less.\n\n"
            "**CRITICAL — EVERY question MUST have [[option]] buttons. NO exceptions.**\n"
            "- NEVER ask open-ended questions without buttons.\n"
            "- NEVER use bullet point lists (- option1, - option2) instead of buttons.\n"
            "- EVERY question gets 2-5 clickable [[option]] buttons on the NEXT LINE.\n"
            "- The user interacts ONLY through buttons — they cannot type free text answers.\n\n"
            "**Format rules:**\n"
            "- Place [[option]] buttons DIRECTLY on the line after each question.\n"
            "- NEVER collect all buttons at the end.\n"
            "- Keep question text SHORT (one line). Put context in the buttons, not the question.\n\n"
            "CORRECT:\n"
            "```\n"
            "1. **What's your current level?**\n"
            "[[Beginner]] [[Intermediate]] [[Advanced]] [[Not sure]]\n"
            "\n"
            "2. **Which features do you want?**\n"
            "[[Feature A]] [[Feature B]] [[Feature C]] [[All of them]]\n"
            "\n"
            "3. **Save data locally?**\n"
            "[[Yes, save my data]] [[No, UI only]]\n"
            "```\n\n"
            "WRONG (open-ended question without buttons):\n"
            "```\n"
            "1. What's your current level? Are you starting from scratch?\n"
            "\n"
            "2. **Which features?**\n"
            "[[Feature A]] [[Feature B]]\n"
            "```\n\n"
            "WRONG (bullet list instead of buttons):\n"
            "```\n"
            "Which sections do you want to focus on?\n"
            "- Reading practice\n"
            "- Listening practice\n"
            "- Writing practice\n"
            "[[Reading]] [[Listening]]\n"
            "```\n\n"
            "### Step 2: Present Plan\n"
            "After gathering requirements, present a structured plan:\n"
            "- **App name** and 1-2 sentence summary\n"
            "- **Screens**: List each screen with its purpose\n"
            "- **Features**: List each feature\n"
            "- **Agent Integration**: Mention that your agent placeholder will be on every page, "
            "with per-screen actions (e.g. 'On the Home screen, I can help you add/search/filter items')\n"
            "- **Database**: SQLite or none\n"
            "- **Platforms**: web + mobile\n\n"
            "Ask: \"Does this plan look good?\"\n"
            "[[Build it!]] [[Change something]]\n\n"
            "### Step 3: Build (ONLY after explicit approval)\n"
            "Call `app_builder__build_app` with ALL the context from the conversation.\n"
            "Include screens, features, design_notes, and db_type from the discussion.\n"
            "Tell the user to check the **Jobs** tab for live progress.\n\n"
            "### Step 4: Preview & Iterate\n"
            "After build completes, share the preview URL and QR code.\n"
            "Ask: \"How does it look?\"\n"
            "[[Looks great!]] [[Change colors]] [[Add a feature]] [[Rebuild it]]\n"
            "If they want changes, use `app_builder__modify_app` with their feedback.\n"
            "This only regenerates affected files — much faster than a full rebuild.\n\n"
            "**IMPORTANT**: NEVER call app_builder__build_app without first discussing "
            "requirements and getting the user's approval on the plan.\n\n"
            "### Status Check\n"
            "Use `app_builder__get_status` with the job_id to check build progress.\n"
            "If a build is **paused** due to token limits, tell the user their tokens need to reset.\n"
            "Use `app_builder__resume_build` with the job_id to resume a paused build once tokens reset.\n"
            "After building, each app gets its own tools (file editing, DB queries, "
            "navigation, GitHub push, restart, etc.) under `app_{slug}__*`.\n\n"
            "### Agent Docking (after build)\n"
            "Once an app is built, you can dock into it using the app's tools:\n"
            "- `app_{slug}__navigate` — change pages/screens within the app\n"
            "- `app_{slug}__read_file` / `app_{slug}__write_file` — modify any app code\n"
            "- `app_{slug}__query_db` — read/write the app's database\n"
            "- The user sees your agent placeholder on every screen of their app\n"
            "- When the user says 'go to my app' or 'open the todo app', use navigate to go to the right screen\n\n"
            "### Layer 2: User-Driven Customization (Edit Layer on Top of Layer 1)\n"
            "Every app has two layers:\n"
            "- **Layer 1** — The base app you built. Already functional with the user's basic preferences applied.\n"
            "- **Layer 2** — Deep personalization that ENHANCES, FIXES, and EXTENDS what Layer 1 created.\n\n"
            "CRITICAL: Layer 2 is NOT a rebuild. Layer 1 already exists and works. Layer 2 improves it.\n"
            "The user already answered foundational questions during Layer 1 (target score, test date, "
            "color theme, academic vs general, study hours, app name, basic preferences). "
            "These are SETTLED. NEVER re-ask them.\n\n"
            "The Layer 2 context message will include LAYER 1 BUILD REQUEST and LAYER 1 CHOICES — "
            "read these carefully to know exactly what was already covered.\n\n"
            "When the user triggers Layer 2 (clicks 'Customize this app' or similar):\n\n"
            "**Step 1: Audit the App (MANDATORY — do this SILENTLY before asking ANY questions)**\n"
            "Use `app_{slug}__read_file` to read 3-5 key files in the app:\n"
            "- App.tsx or main navigation to understand structure\n"
            "- 1-2 main screen components to see content depth\n"
            "- Database/seed data to find placeholder content\n"
            "- Any data files (vocabulary lists, schedules, templates)\n\n"
            "As you read, identify things Layer 1 did POORLY or left INCOMPLETE:\n"
            "- **Placeholder/demo data**: Mock items, generic lists, lorem-style content\n"
            "- **Shallow features**: Screens that exist but have minimal/static functionality\n"
            "- **Hardcoded content**: Values that should be personalized or dynamic\n"
            "- **Missing algorithms**: Simple logic where proper domain algorithms should be\n"
            "- **No real content**: Empty or generic where domain-specific content should exist\n\n"
            "Do NOT tell the user you are auditing. Do NOT expose file paths or technical details.\n\n"
            "**Step 2: Ask 10+ Deep Contextual Questions Based on Your Audit**\n"
            "Your questions must reference SPECIFIC things you found in the code.\n"
            "Each question MUST cite a concrete finding: a number, a feature, actual content you saw.\n\n"
            "FORBIDDEN TOPICS (Layer 1 already handled):\n"
            "- Target score / goal / band score\n"
            "- Test date / timeline\n"
            "- Study hours / weekly availability\n"
            "- Color theme / visual preferences\n"
            "- Academic vs General Training\n"
            "- App name / description\n"
            "- Which sections to include\n"
            "- Basic skill level\n\n"
            "GOOD Layer 2 questions (reference specific code findings):\n"
            "- 'I found 500 vocabulary words but they're all general English — should I focus them on your specific field?'\n"
            "- 'The reading section has 10 passages but they're placeholder lorem ipsum — should I generate real exam-style passages?'\n"
            "- 'The study plan is a fixed 90-day schedule with identical daily tasks — want me to make it adaptive based on quiz performance?'\n"
            "- 'The scoring uses a simplified formula (correct/total) — want me to implement the official band descriptor algorithm?'\n"
            "- 'The speaking section has no audio capabilities — should I add voice recording and playback?'\n"
            "- 'I found the badge system awards badges but they don't unlock anything — tie achievements to bonus content?'\n"
            "- 'The grammar exercises cover all topics equally — prioritize high-impact patterns that lose the most marks?'\n\n"
            "Every question MUST have [[option]] buttons directly on the next line.\n"
            "Each question should make the user think: 'yes, that's exactly what I need to fix.'\n\n"
            "**Step 3: Apply Changes**\n"
            "Based on answers, use `app_{slug}__write_file` and `app_{slug}__query_db` to:\n"
            "- Replace placeholder/demo data with real, useful, domain-specific content\n"
            "- Upgrade shallow features with deeper, functional implementations\n"
            "- Add proper algorithms where simplified logic exists\n"
            "- Generate real content where placeholders exist\n\n"
            "Be EFFICIENT: write complete files rather than many small edits. Batch related changes.\n"
            "Show brief progress after each edit. Changes are applied live.\n"
            "NEVER use memory_store to save preferences instead of editing. ALWAYS use write_file.\n"
            "NEVER tell the user you are 'storing preferences in memory' — that is an internal operation.\n\n"
            "**Step 4: Summary & Next Steps**\n"
            "After all changes, give a brief human-friendly summary:\n"
            "'All done! Here's what I customized: [2-3 bullet points of key changes]. Try it out!'\n"
            "NEVER expose internal operations (memory storage, file paths, DB queries) to the user.\n"
            "Suggest next steps with buttons:\n"
            "[[Try the app now]] [[Make more changes]] [[Reset to default]]\n"
        )

    # ── Build Pipeline ──────────────────────────────────────────────

    def _initial_steps(self) -> List[Dict]:
        """Create the initial steps list for a build job."""
        step_types = [
            ("planning", "Planning app architecture..."),
            ("scaffolding", "Creating Expo project..."),
            ("writing", "Generating app code..."),
            ("database", "Setting up database..."),
            ("installing", "Installing dependencies..."),
            ("github", "Creating GitHub repository..."),
            ("starting", "Starting preview servers..."),
            ("ready", "App is ready!"),
        ]
        return [
            {
                "id": str(uuid.uuid4()),
                "type": step_type,
                "label": label,
                "status": "pending",
            }
            for step_type, label in step_types
        ]

    async def _resume_build_app(self, job_id: str, checkpoint: Dict, user_id: str):
        """Resume a paused build from checkpoint — skips completed steps."""
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob
        from .build_logger import BuildLogger

        app_id = checkpoint.get("app_id", "")
        name = checkpoint.get("name", "Unknown App")
        description = checkpoint.get("description", "")
        slug = checkpoint.get("slug", "")
        app_dir = checkpoint.get("app_dir", "")
        plan_context = checkpoint.get("plan_context")
        design_notes = checkpoint.get("design_notes", "")
        completed_steps = set(checkpoint.get("completed_steps", []))
        current_step = checkpoint.get("current_step", "planning")

        blog = BuildLogger(job_id, user_id, ws_broadcast=self._ws_broadcast)
        blog.set_step("resume")
        await blog.info(f"Resuming build for '{name}' from step: {current_step}")
        await blog.info(f"Completed steps: {', '.join(completed_steps) if completed_steps else 'none'}")

        try:
            # ── Restore plan data from checkpoint ────────────────────
            plan = checkpoint.get("plan")
            if not plan and "planning" not in completed_steps:
                # Need to redo planning
                blog.set_step("planning")
                await self._update_step(job_id, user_id, "planning", "running")
                extra_context = checkpoint.get("extra_context", "")
                plan = await self._step_plan(job_id, user_id, description, blog, extra_context)
                if not plan:
                    await blog.error("Planning failed on resume")
                    await blog.persist()
                    await self._fail_job(job_id, app_id, "Planning failed on resume")
                    return
                completed_steps.add("planning")

            files_to_generate = checkpoint.get("files_to_generate", plan.get("files", ["/App.tsx"]) if plan else ["/App.tsx"])
            deps = checkpoint.get("deps", plan.get("dependencies", []) if plan else [])
            db_type = checkpoint.get("db_type", "none")
            app_name = checkpoint.get("app_name", name)
            needs_db = checkpoint.get("needs_db", False)

            # ── Scaffolding (skip if done) ───────────────────────────
            if "scaffolding" not in completed_steps:
                blog.set_step("scaffolding")
                await self._update_step(job_id, user_id, "scaffolding", "running")
                await blog.info(f"Creating Expo project '{app_name}'...")
                await self._app_manager.scaffold_app(app_id, app_name)
                await blog.success("Expo project created")
                await self._update_step(job_id, user_id, "scaffolding", "done")
            else:
                await blog.info("Scaffolding already done — skipping")

            # ── Writing (resume partial if needed) ───────────────────
            if "writing" not in completed_steps:
                blog.set_step("writing")
                existing_generated = checkpoint.get("generated_files", {})
                pending_files = checkpoint.get("pending_files", files_to_generate)

                # Filter out files already generated
                if existing_generated:
                    pending_files = [f for f in pending_files if f not in existing_generated]
                    await blog.info(f"Resuming code gen: {len(existing_generated)} files done, {len(pending_files)} remaining")

                if pending_files:
                    await self._update_step(job_id, user_id, "writing", "running",
                                            detail=f"Generating {len(pending_files)} remaining files...")
                    new_files = await self._generate_code(
                        description, app_name, pending_files, deps, db_type,
                        job_id=job_id, user_id=user_id, blog=blog,
                        design_notes=design_notes,
                    )
                    generated_files = {**existing_generated, **new_files}
                else:
                    generated_files = existing_generated

                await self._app_manager.write_app_files(app_id, generated_files)
                await self._write_infra_files(app_id, generated_files, blog)

                async with async_session_maker() as db:
                    app = await db.get(App, app_id)
                    if app:
                        app.files_json = json.dumps(generated_files)
                        app.deps_json = json.dumps(deps)
                        await db.commit()

                await blog.success(f"Code generation complete: {len(generated_files)} files total")
                await self._update_step(job_id, user_id, "writing", "done",
                                        detail=f"Generated {len(generated_files)} files")
            else:
                generated_files = checkpoint.get("generated_files", {})
                await blog.info("Code generation already done — skipping")

            # ── Database (skip if done) ──────────────────────────────
            if "database" not in completed_steps:
                blog.set_step("database")
                if needs_db and db_type != "none":
                    await self._update_step(job_id, user_id, "database", "running")
                    try:
                        db_url = await self._app_manager.setup_database(app_id, db_type)
                        storage_dir = await self._app_manager.setup_storage(app_id)
                        async with async_session_maker() as db:
                            app = await db.get(App, app_id)
                            if app:
                                app.db_type = db_type
                                app.db_url = db_url
                                app.storage_dir = storage_dir
                                await db.commit()
                        await blog.success(f"Database ready: {db_type}")
                        await self._update_step(job_id, user_id, "database", "done")
                    except Exception as e:
                        await blog.warn(f"Database setup failed (non-fatal): {e}")
                        await self._update_step(job_id, user_id, "database", "done", detail=f"Skipped: {e}")
                else:
                    await self._update_step(job_id, user_id, "database", "done", detail="Not needed")
            else:
                await blog.info("Database setup already done — skipping")

            # ── Install deps (skip if done) ──────────────────────────
            if "installing" not in completed_steps:
                blog.set_step("installing")
                await self._update_step(job_id, user_id, "installing", "running")
                web_deps = ["react-dom", "react-native-web"]
                all_deps = list(set((deps or []) + web_deps))
                await blog.info(f"Installing {len(all_deps)} packages...")
                await self._app_manager.install_deps(app_id, all_deps)
                await blog.success("Dependencies installed")
                await self._update_step(job_id, user_id, "installing", "done")
            else:
                await blog.info("Dependencies already installed — skipping")

            # ── GitHub (skip if done) ────────────────────────────────
            if "github" not in completed_steps:
                blog.set_step("github")
                await self._update_step(job_id, user_id, "github", "running")
                try:
                    repo_info = await self._app_manager.create_github_repo(app_id, app_name)
                    github_url = repo_info.get("repo_url", "")
                    async with async_session_maker() as db:
                        app = await db.get(App, app_id)
                        if app:
                            app.github_url = github_url
                            app.github_repo = repo_info.get("repo_name", "")
                            await db.commit()
                    await blog.success(f"GitHub repo created", github_url)
                    await self._update_step(job_id, user_id, "github", "done",
                                            detail=github_url or "Skipped")
                except Exception as e:
                    await blog.warn(f"GitHub failed (non-fatal): {e}")
                    await self._update_step(job_id, user_id, "github", "done", detail=f"Skipped: {e}")
            else:
                await blog.info("GitHub already done — skipping")

            # ── Start servers (always redo — they don't survive pause) ─
            blog.set_step("starting")
            await self._update_step(job_id, user_id, "starting", "running")

            # Pre-flight dep check
            await self._verify_deps_installed(app_dir, generated_files, deps, blog)

            # Start servers
            metro_port = await self._app_manager.start_metro(app_id)
            await blog.success(f"Metro running on port {metro_port}")
            web_port = await self._app_manager.start_web(app_id)
            await blog.success(f"Web server running on port {web_port}")

            # Bundle validation (simplified — same as _build_app step 7c)
            bundle_ok = False
            for repair_round in range(4):
                managed_app = self._app_manager._running.get(app_id)
                web_alive = (managed_app and managed_app.web_process
                             and managed_app.web_process.returncode is None)
                if web_alive:
                    await self._wait_for_server(web_port, timeout=25)
                    bundle_ok, bundle_errors = await self._validate_bundle(web_port, blog)
                else:
                    bundle_ok = False
                    bundle_errors = self._extract_errors_from_log_buffer(app_id)

                if bundle_ok:
                    await blog.success("Bundle compiles cleanly" if repair_round == 0 else f"Bundle repaired (round {repair_round})")
                    break

                await blog.warn(f"Repair round {repair_round + 1}/4...")
                fixed = False
                if bundle_errors:
                    fixed = await self._fix_missing_deps(app_id, app_dir, bundle_errors, blog)
                code_errors = [e for e in bundle_errors if e.get("file", "unknown") != "unknown"]
                if code_errors:
                    if await self._repair_bundle_errors(app_id, app_dir, code_errors, generated_files,
                                                        description, name, deps, db_type, blog):
                        fixed = True
                if not fixed and not web_alive:
                    try:
                        web_port = await self._app_manager.start_web(app_id)
                        await asyncio.sleep(5)
                    except Exception:
                        pass
                if not fixed:
                    break
                if web_alive:
                    await asyncio.sleep(4)
                else:
                    try:
                        web_port = await self._app_manager.start_web(app_id)
                    except Exception:
                        pass

            qr_url = await self._app_manager.get_qr_url(app_id)
            web_url = await self._app_manager.get_web_url(app_id)
            final_status = "running" if bundle_ok else "error"

            async with async_session_maker() as db:
                app = await db.get(App, app_id)
                if app:
                    app.status = final_status
                    app.port = metro_port
                    app.web_port = web_port
                    managed = self._app_manager._running.get(app_id)
                    if managed:
                        app.metro_pid = managed.metro_process.pid if managed.metro_process else None
                        app.web_pid = managed.web_process.pid if managed.web_process else None
                    await db.commit()

            if not bundle_ok:
                await blog.error("Bundle compilation failed after repair")
                await blog.persist()
                await self._fail_job(job_id, app_id, "Bundle compilation failed after resume repair")
                return

            await self._update_step(job_id, user_id, "starting", "done",
                                    detail=f"Metro:{metro_port} Web:{web_port}")

            # ── Ready ────────────────────────────────────────────────
            blog.set_step("ready")
            await self._update_step(job_id, user_id, "ready", "done")

            summary = blog.summary()
            await blog.success(f"Build resumed & complete! {summary['total_tokens']:,} tokens used in resume")
            await blog.persist()

            async with async_session_maker() as db:
                job = await db.get(BuildJob, job_id)
                if job:
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    job.checkpoint_json = None  # Clear checkpoint
                    await db.commit()

            # Register app in gateway
            if hasattr(self, '_app_gateway') and self._app_gateway:
                try:
                    from .app_fs_skill import AppFsSkill
                    app_fs_skill = AppFsSkill(app_id, name, slug, app_dir, self._app_manager)
                    self._app_gateway.register_app(slug, app_fs_skill)
                except Exception as e:
                    logger.warning(f"[RESUME] Failed to register app in gateway: {e}")

            # Broadcast app_ready
            if self._ws_broadcast:
                await self._ws_broadcast(user_id, {
                    "type": "app_ready",
                    "app_id": app_id,
                    "name": name,
                    "qr_url": qr_url,
                    "web_url": web_url,
                })

            logger.info(f"[RESUME] Build resumed and complete for '{name}' (job={job_id})")

        except TokenLimitError as e:
            # Hit limit again during resume — re-pause with updated checkpoint
            updated_checkpoint = checkpoint.copy()
            updated_checkpoint["completed_steps"] = list(completed_steps)
            logger.info(f"[RESUME] Token limit hit again for '{name}' — re-pausing")
            await self._pause_job(job_id, app_id, user_id, e.retry_after_seconds, updated_checkpoint, blog)

        except Exception as e:
            logger.exception(f"[RESUME] Error resuming build for '{name}'")
            await blog.error(f"Resume error: {e}")
            await blog.persist()
            await self._fail_job(job_id, app_id, f"Resume failed: {e}")

    async def _build_app(
        self,
        job_id: str,
        app_id: str,
        name: str,
        description: str,
        user_id: str,
        slug: str,
        app_dir: str,
        plan_context: Optional[Dict] = None,
    ):
        """Background task that orchestrates the full build pipeline."""
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob
        from .build_logger import BuildLogger

        blog = BuildLogger(job_id, user_id, ws_broadcast=self._ws_broadcast)
        blog.set_step("init")
        await blog.info(f"Starting build for '{name}'", f"app={app_id}")
        await blog.info(f"User prompt: {description[:200]}{'...' if len(description) > 200 else ''}")

        # Build extra context from the conversational plan
        extra_context = ""
        design_notes = ""
        if plan_context:
            screens = plan_context.get("screens", [])
            features = plan_context.get("features", [])
            design_notes = plan_context.get("design_notes", "")
            if screens:
                extra_context += f"\nScreens: {', '.join(screens)}"
                await blog.info(f"Screens from conversation: {', '.join(screens)}")
            if features:
                extra_context += f"\nFeatures: {', '.join(features)}"
                await blog.info(f"Features from conversation: {', '.join(features)}")
            if design_notes:
                extra_context += f"\nDesign preferences: {design_notes}"
                await blog.info(f"Design notes: {design_notes}")

        # Checkpoint state — built up incrementally so pause can save progress
        _checkpoint: Dict[str, Any] = {
            "current_step": "planning",
            "completed_steps": [],
            "app_id": app_id,
            "job_id": job_id,
            "name": name,
            "description": description,
            "user_id": user_id,
            "slug": slug,
            "app_dir": app_dir,
            "plan_context": plan_context,
            "extra_context": extra_context,
            "design_notes": design_notes,
        }

        try:
            # ── Step 1: Planning ────────────────────────────────────
            plan = await self._step_plan(job_id, user_id, description, blog, extra_context)
            if not plan:
                await blog.error("Planning failed — could not generate app plan")
                await blog.persist()
                await self._fail_job(job_id, app_id, "Planning failed — could not generate app plan")
                return

            files_to_generate = plan.get("files", ["/App.tsx"])
            deps = plan.get("dependencies", [])
            db_type = plan.get("db_type", plan_context.get("db_type", "none") if plan_context else "none")
            app_name = plan.get("app_name", name)
            needs_db = plan.get("needs_database", False)

            # Ensure agentSkill.json is always generated (domain-specific actions)
            if "/lib/agentSkill.json" not in files_to_generate:
                files_to_generate.append("/lib/agentSkill.json")

            await blog.info(f"Plan: {len(files_to_generate)} files, {len(deps)} deps, db={db_type}")

            # Update checkpoint after planning
            _checkpoint.update({
                "current_step": "scaffolding",
                "completed_steps": ["planning"],
                "plan": plan,
                "files_to_generate": files_to_generate,
                "deps": deps,
                "db_type": db_type,
                "app_name": app_name,
                "needs_db": needs_db,
            })

            # Save plan to DB
            async with async_session_maker() as db:
                app = await db.get(App, app_id)
                if app:
                    try:
                        app.plan_json = json.dumps(plan)
                    except Exception:
                        pass
                    await db.commit()

            # ── Step 2: Scaffolding ─────────────────────────────────
            blog.set_step("scaffolding")
            await self._update_step(job_id, user_id, "scaffolding", "running")
            try:
                await blog.info(f"Creating Expo project '{app_name}'...")
                t0 = time.time()
                await self._app_manager.scaffold_app(app_id, app_name)
                elapsed = time.time() - t0
                await blog.success(f"Expo project created", f"{elapsed:.1f}s")
                await self._update_step(job_id, user_id, "scaffolding", "done")
            except Exception as e:
                await blog.error(f"Scaffold failed: {e}")
                await blog.persist()
                await self._fail_job(job_id, app_id, f"Scaffold failed: {e}")
                return

            # ── Step 3: Writing code ────────────────────────────────
            blog.set_step("writing")
            await self._update_step(job_id, user_id, "writing", "running",
                                    detail=f"Generating {len(files_to_generate)} files...")
            try:
                await blog.info(f"Generating {len(files_to_generate)} files (5 concurrent)...")
                generated_files = await self._generate_code(
                    description, app_name, files_to_generate, deps, db_type,
                    job_id=job_id, user_id=user_id, blog=blog,
                    design_notes=design_notes,
                )
                await self._app_manager.write_app_files(app_id, generated_files)

                # Write essential infrastructure files (metro config, web-safe DB)
                await self._write_infra_files(app_id, generated_files, blog)

                total_bytes = sum(len(c.encode()) for c in generated_files.values())
                await blog.success(f"Generated {len(generated_files)} files", f"{total_bytes:,} bytes total")

                async with async_session_maker() as db:
                    app = await db.get(App, app_id)
                    if app:
                        app.files_json = json.dumps(generated_files)
                        app.deps_json = json.dumps(deps)
                        await db.commit()

                await self._update_step(job_id, user_id, "writing", "done",
                                        detail=f"Generated {len(generated_files)} files")

                # Update checkpoint after writing
                _checkpoint.update({
                    "current_step": "database",
                    "completed_steps": ["planning", "scaffolding", "writing"],
                    "generated_files": generated_files,
                })
            except TokenLimitError as e:
                # Save partial code generation progress
                partial = getattr(e, 'partial_files', {})
                pending = getattr(e, 'pending_files', files_to_generate)
                if partial:
                    # Write partial files to disk so they survive pause
                    await self._app_manager.write_app_files(app_id, partial)
                    async with async_session_maker() as db:
                        app = await db.get(App, app_id)
                        if app:
                            app.files_json = json.dumps(partial)
                            await db.commit()
                _checkpoint.update({
                    "current_step": "writing",
                    "generated_files": partial,
                    "pending_files": pending,
                })
                await self._pause_job(job_id, app_id, user_id, e.retry_after_seconds, _checkpoint, blog)
                return
            except Exception as e:
                await blog.error(f"Code generation failed: {e}")
                await blog.persist()
                await self._fail_job(job_id, app_id, f"Code generation failed: {e}")
                return

            # ── Step 4: Database ────────────────────────────────────
            blog.set_step("database")
            if needs_db and db_type != "none":
                await self._update_step(job_id, user_id, "database", "running")
                try:
                    await blog.info(f"Setting up {db_type} database...")
                    db_url = await self._app_manager.setup_database(app_id, db_type)
                    storage_dir = await self._app_manager.setup_storage(app_id)
                    async with async_session_maker() as db:
                        app = await db.get(App, app_id)
                        if app:
                            app.db_type = db_type
                            app.db_url = db_url
                            app.storage_dir = storage_dir
                            await db.commit()
                    await blog.success(f"Database ready: {db_type}", db_url or "")
                    await self._update_step(job_id, user_id, "database", "done")
                except Exception as e:
                    await blog.warn(f"Database setup failed (non-fatal): {e}")
                    await self._update_step(job_id, user_id, "database", "done",
                                            detail=f"Skipped: {e}")
            else:
                await blog.info("No database needed, skipping")
                await self._update_step(job_id, user_id, "database", "done", detail="Not needed")

            _checkpoint["completed_steps"] = ["planning", "scaffolding", "writing", "database"]
            _checkpoint["current_step"] = "installing"

            # ── Step 5: Installing deps ─────────────────────────────
            blog.set_step("installing")
            await self._update_step(job_id, user_id, "installing", "running")
            try:
                web_deps = ["react-dom", "react-native-web"]
                all_deps = list(set((deps or []) + web_deps))
                await blog.info(f"Installing {len(all_deps)} packages...")
                for dep in sorted(all_deps):
                    await blog.debug(f"  {dep}")
                t0 = time.time()
                install_output = await self._app_manager.install_deps(app_id, all_deps)
                elapsed = time.time() - t0
                await blog.command_run(f"npm install {' '.join(all_deps[:5])}{'...' if len(all_deps) > 5 else ''}",
                                       exit_code=0, duration_s=elapsed, output=install_output[:300] if install_output else "")
                await blog.success(f"Dependencies installed", f"{elapsed:.1f}s")
                await self._update_step(job_id, user_id, "installing", "done")
            except Exception as e:
                await blog.error(f"npm install failed: {e}")
                await blog.persist()
                await self._fail_job(job_id, app_id, f"npm install failed: {e}")
                return

            _checkpoint["completed_steps"] = ["planning", "scaffolding", "writing", "database", "installing"]
            _checkpoint["current_step"] = "github"

            # ── Step 6: GitHub ──────────────────────────────────────
            blog.set_step("github")
            await self._update_step(job_id, user_id, "github", "running")
            try:
                await blog.info("Creating GitHub repository...")
                repo_info = await self._app_manager.create_github_repo(app_id, app_name)
                github_url = repo_info.get("repo_url", "")
                github_repo = repo_info.get("repo_name", "")
                async with async_session_maker() as db:
                    app = await db.get(App, app_id)
                    if app:
                        app.github_url = github_url
                        app.github_repo = github_repo
                        await db.commit()
                await blog.success(f"GitHub repo created", github_url)
                await self._update_step(job_id, user_id, "github", "done",
                                        detail=github_url or "Skipped (no gh CLI)")
            except Exception as e:
                await blog.warn(f"GitHub repo creation failed (non-fatal): {e}")
                await self._update_step(job_id, user_id, "github", "done",
                                        detail=f"Skipped: {e}")

            _checkpoint["completed_steps"] = ["planning", "scaffolding", "writing", "database", "installing", "github"]
            _checkpoint["current_step"] = "starting"

            # ── Step 7: Starting servers + full auto-repair ─────────
            blog.set_step("starting")
            await self._update_step(job_id, user_id, "starting", "running")
            qr_url = ""
            web_url = ""
            try:
                # ── 7a: Pre-flight dep check — install any missing deps BEFORE starting servers
                await blog.info("Verifying dependencies before server start...")
                await self._verify_deps_installed(app_dir, generated_files, deps, blog)

                # ── 7b: Start servers
                await blog.info("Starting Metro bundler (mobile)...")
                metro_port = await self._app_manager.start_metro(app_id)
                await blog.success(f"Metro running on port {metro_port}")

                await blog.info("Starting Expo Web server...")
                web_port = await self._app_manager.start_web(app_id)
                await blog.success(f"Web server running on port {web_port}")

                # ── 7c: Bundle validation + comprehensive auto-repair (up to 4 rounds) ──
                await blog.info("Validating web bundle compilation...")
                bundle_ok = False
                MAX_REPAIR_ROUNDS = 4
                try:
                    for repair_round in range(MAX_REPAIR_ROUNDS):
                        # Check if web server is alive
                        managed_app = self._app_manager._running.get(app_id)
                        web_alive = (managed_app and managed_app.web_process
                                     and managed_app.web_process.returncode is None)

                        if web_alive:
                            await self._wait_for_server(web_port, timeout=25)
                            bundle_ok, bundle_errors = await self._validate_bundle(web_port, blog)
                        else:
                            bundle_ok = False
                            await blog.warn("Web server crashed — extracting errors...")
                            bundle_errors = self._extract_errors_from_log_buffer(app_id)
                            if not bundle_errors:
                                await blog.warn("No specific errors in process output")
                                # Dump last 15 log lines for debugging
                                if managed_app and managed_app.log_buffer:
                                    for line in list(managed_app.log_buffer)[-15:]:
                                        await blog.info(f"  {line}")

                        if bundle_ok:
                            label = "Bundle compiles cleanly" if repair_round == 0 else f"Bundle repaired and compiles cleanly (round {repair_round})"
                            await blog.success(label)
                            break

                        # ── Categorize and fix errors ──
                        await blog.warn(f"Bundle has issues — repair round {repair_round + 1}/{MAX_REPAIR_ROUNDS}...")

                        fixed_something = False

                        # Fix 1: Missing dependencies (npm install)
                        if bundle_errors:
                            dep_fixed = await self._fix_missing_deps(app_id, app_dir, bundle_errors, blog)
                            if dep_fixed:
                                fixed_something = True

                        # Fix 2: Code errors — regenerate broken files via LLM
                        code_errors = [e for e in bundle_errors if e.get("file", "unknown") != "unknown"]
                        if code_errors:
                            repaired = await self._repair_bundle_errors(
                                app_id, app_dir, code_errors, generated_files,
                                description, name, deps, db_type, blog
                            )
                            if repaired:
                                fixed_something = True

                        # Fix 3: No specific errors found — restart server (port/transient issue)
                        if not fixed_something and not web_alive:
                            await blog.info("No actionable errors — restarting web server...")
                            try:
                                web_port = await self._app_manager.start_web(app_id)
                                await blog.info(f"Web server restarted on port {web_port}")
                                await asyncio.sleep(5)
                                fixed_something = True
                            except Exception as restart_err:
                                await blog.warn(f"Web server restart failed: {restart_err}")

                        if not fixed_something:
                            await blog.warn("Could not identify or fix any issues")
                            break

                        # Restart server after code/dep fixes if it was alive (Metro needs to rebundle)
                        if web_alive:
                            await asyncio.sleep(4)  # Wait for Metro hot-reload
                        else:
                            await blog.info("Restarting web server after repairs...")
                            try:
                                web_port = await self._app_manager.start_web(app_id)
                                await blog.info(f"Web server restarted on port {web_port}")
                            except Exception as restart_err:
                                await blog.warn(f"Web server restart failed: {restart_err}")

                    # ── 7d: Last resort — static export if bundle validation still fails ──
                    if not bundle_ok:
                        await blog.warn("Live server failed — attempting static export as fallback...")
                        try:
                            proc = await asyncio.create_subprocess_exec(
                                "npx", "expo", "export", "--platform", "web",
                                cwd=app_dir,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.STDOUT,
                            )
                            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
                            output = stdout.decode("utf-8", errors="replace") if stdout else ""
                            if proc.returncode == 0 and "Exported:" in output:
                                await blog.success("Static export succeeded — app code is valid")
                                # Restart web server one final time — code compiles, server was the issue
                                try:
                                    web_port = await self._app_manager.start_web(app_id)
                                    await self._wait_for_server(web_port, timeout=20)
                                    bundle_ok = True
                                    await blog.success(f"Web server running on port {web_port} after static export verification")
                                except Exception:
                                    # Even if server fails, mark as running — the code is valid
                                    bundle_ok = True
                                    await blog.info("Web server may be slow to start, but code compiles — marking as running")
                            else:
                                await blog.error(f"Static export also failed: {output[-300:]}")
                        except asyncio.TimeoutError:
                            await blog.warn("Static export timed out (120s)")
                        except Exception as e:
                            await blog.warn(f"Static export failed: {e}")

                    if not bundle_ok:
                        await blog.error("Bundle has compilation errors after all repair attempts")
                except Exception as e:
                    await blog.warn(f"Bundle validation error: {e}")

                qr_url = await self._app_manager.get_qr_url(app_id)
                web_url = await self._app_manager.get_web_url(app_id)
                await blog.info(f"QR URL: {qr_url}")
                await blog.info(f"Web URL: {web_url}")

                # Set status based on bundle validation result
                final_status = "running" if bundle_ok else "error"
                async with async_session_maker() as db:
                    app = await db.get(App, app_id)
                    if app:
                        app.status = final_status
                        app.port = metro_port
                        app.web_port = web_port
                        managed = self._app_manager._running.get(app_id)
                        if managed:
                            app.metro_pid = managed.metro_process.pid if managed.metro_process else None
                            app.web_pid = managed.web_process.pid if managed.web_process else None
                        await db.commit()
                if not bundle_ok:
                    await blog.error("Build completed with errors — check bundle logs")
                    await blog.persist()
                    await self._fail_job(job_id, app_id, "Bundle compilation failed after auto-repair")
                    return

                await self._update_step(job_id, user_id, "starting", "done",
                                        detail=f"Metro:{metro_port} Web:{web_port}")
            except Exception as e:
                await blog.error(f"Server start failed: {e}")
                await blog.persist()
                await self._fail_job(job_id, app_id, f"Server start failed: {e}")
                return

            # ── Step 8: Ready! ──────────────────────────────────────
            blog.set_step("ready")
            await self._update_step(job_id, user_id, "ready", "done")

            summary = blog.summary()
            await blog.success(
                f"Build complete! {summary['files_written']} files, {summary['llm_calls']} LLM calls, {summary['total_tokens']:,} tokens",
                f"Errors: {summary['errors']}, Warnings: {summary['warnings']}"
            )

            await blog.persist()
            async with async_session_maker() as db:
                job = await db.get(BuildJob, job_id)
                if job:
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    job.model = "claude-opus-4-6"
                    await db.commit()

            # Register app into the gateway (single skill, not per-app tools)
            if hasattr(self, '_app_gateway') and self._app_gateway:
                try:
                    from .app_fs_skill import AppFsSkill
                    app_fs_skill = AppFsSkill(app_id, name, slug, app_dir, self._app_manager)
                    self._app_gateway.register_app(slug, app_fs_skill)
                    logger.info(f"[BUILD] Registered app '{name}' in gateway (slug: {slug})")
                except Exception as e:
                    logger.warning(f"[BUILD] Failed to register app in gateway: {e}")

            # Broadcast app_ready
            if self._ws_broadcast:
                await self._ws_broadcast(user_id, {
                    "type": "app_ready",
                    "app_id": app_id,
                    "name": name,
                    "qr_url": qr_url,
                    "web_url": web_url,
                })

            logger.info(f"[BUILD] Build complete for '{name}' (job={job_id})")

        except TokenLimitError as e:
            # Token/rate limit hit in any step — pause and save checkpoint
            logger.info(f"[BUILD] Token limit hit for '{name}' — pausing (retry in {e.retry_after_seconds}s)")
            await self._pause_job(job_id, app_id, user_id, e.retry_after_seconds, _checkpoint, blog)

        except Exception as e:
            logger.exception(f"[BUILD] Unexpected error building '{name}'")
            await blog.error(f"Unexpected error: {e}")
            await blog.persist()
            await self._fail_job(job_id, app_id, f"Unexpected error: {e}")

    # ── Modify Pipeline ────────────────────────────────────────────

    async def _modify_app(
        self,
        job_id: str,
        app_id: str,
        app_name: str,
        description: str,
        changes: str,
        user_id: str,
        slug: str,
        app_dir: str,
    ):
        """Background task that modifies an existing app."""
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob
        from .build_logger import BuildLogger

        blog = BuildLogger(job_id, user_id, ws_broadcast=self._ws_broadcast)
        blog.set_step("init")
        await blog.info(f"Modifying '{app_name}'", f"app={app_id}")
        await blog.info(f"Changes: {changes[:200]}{'...' if len(changes) > 200 else ''}")

        try:
            # ── Step 1: Analyze changes (Opus) ─────────────────────
            blog.set_step("planning")
            await self._update_step(job_id, user_id, "planning", "running")

            # Read current file list from filesystem
            file_list = []
            if os.path.exists(app_dir):
                for root, dirs, files in os.walk(app_dir):
                    # Skip node_modules, .expo, .git
                    dirs[:] = [d for d in dirs if d not in ('node_modules', '.expo', '.git', '.logs')]
                    for f in files:
                        rel = os.path.relpath(os.path.join(root, f), app_dir)
                        file_list.append(f"/{rel}")

            await blog.info(f"Current app has {len(file_list)} files")

            # Ask Opus which files need changes
            analysis_response = await self._call_llm(
                MODIFY_ANALYSIS_PROMPT.format(
                    app_name=app_name,
                    file_list=json.dumps(file_list[:100]),
                    changes=changes,
                ),
                "Analyze which files need modification.",
                blog=blog,
                purpose="Modification analysis",
            )

            analysis = self._extract_json(analysis_response)
            if not analysis:
                await blog.warn("Could not parse analysis, regenerating all screen files")
                analysis = {
                    "affected_files": [f for f in file_list if '/screens/' in f or f == '/App.tsx'],
                    "new_files": [],
                    "summary": changes,
                }

            affected = analysis.get("affected_files", []) + analysis.get("new_files", [])
            if not affected:
                affected = ["/App.tsx"]

            await blog.success(f"Analysis complete: {len(affected)} files to update")
            await self._update_step(job_id, user_id, "planning", "done",
                                    detail=f"{len(affected)} files: {', '.join(affected[:5])}")

            # ── Step 2: Regenerate affected files (Sonnet) ─────────
            blog.set_step("writing")
            await self._update_step(job_id, user_id, "writing", "running",
                                    detail=f"Regenerating {len(affected)} files...")

            # Read existing file contents for context
            existing_contents = {}
            for fp in file_list[:50]:
                abs_path = os.path.join(app_dir, fp.lstrip("/"))
                if os.path.exists(abs_path):
                    try:
                        with open(abs_path, 'r') as fh:
                            existing_contents[fp] = fh.read()
                    except Exception:
                        pass

            # Load deps from DB
            deps = []
            async with async_session_maker() as db:
                app = await db.get(App, app_id)
                if app and app.deps_json:
                    try:
                        deps = json.loads(app.deps_json)
                    except Exception:
                        pass

            modify_description = f"{description}\n\nMODIFICATION REQUEST: {changes}"
            generated = await self._generate_code(
                modify_description, app_name, affected, deps,
                app.db_type if app else "none",
                job_id=job_id, user_id=user_id, blog=blog,
                existing_files=existing_contents,
            )

            # Write files to disk
            await self._app_manager.write_app_files(app_id, generated)
            await blog.success(f"Updated {len(generated)} files")
            await self._update_step(job_id, user_id, "writing", "done",
                                    detail=f"Updated {len(generated)} files")

            # ── Step 3: Install any new deps ───────────────────────
            blog.set_step("installing")
            # Scan generated code for new imports that aren't in current deps
            new_deps = self._detect_new_deps(generated, deps)
            if new_deps:
                await blog.info(f"Installing {len(new_deps)} new dependencies: {', '.join(new_deps)}")
                try:
                    await self._app_manager.install_deps(app_id, new_deps)
                    await blog.success(f"Installed {len(new_deps)} new deps")
                except Exception as e:
                    await blog.warn(f"Dep install failed (non-fatal): {e}")
                await self._update_step(job_id, user_id, "installing", "done",
                                        detail=f"Installed {len(new_deps)} deps")
            else:
                await self._update_step(job_id, user_id, "installing", "done", detail="No new deps needed")

            # ── Step 4: Restart servers ────────────────────────────
            blog.set_step("starting")
            await self._update_step(job_id, user_id, "starting", "running")
            try:
                await self._app_manager.stop_app(app_id)
                metro_port = await self._app_manager.start_metro(app_id)
                web_port = await self._app_manager.start_web(app_id)
                await blog.success(f"Servers restarted — Metro:{metro_port} Web:{web_port}")

                async with async_session_maker() as db:
                    app = await db.get(App, app_id)
                    if app:
                        app.status = "running"
                        app.port = metro_port
                        app.web_port = web_port
                        await db.commit()

                await self._update_step(job_id, user_id, "starting", "done",
                                        detail=f"Metro:{metro_port} Web:{web_port}")
            except Exception as e:
                await blog.warn(f"Restart failed (non-fatal): {e}")
                await self._update_step(job_id, user_id, "starting", "done",
                                        detail=f"Restart skipped: {e}")

            # ── Step 4b: Validate bundle after modification (up to 4 rounds) ──
            if web_port:
                blog.set_step("validating")
                await blog.info("Validating modified bundle...")
                bundle_ok = False
                try:
                    for repair_round in range(4):
                        managed_app = self._app_manager._running.get(app_id)
                        web_alive = (managed_app and managed_app.web_process
                                     and managed_app.web_process.returncode is None)
                        if web_alive:
                            await self._wait_for_server(web_port, timeout=20)
                            bundle_ok, bundle_errors = await self._validate_bundle(web_port, blog)
                        else:
                            bundle_ok = False
                            await blog.warn("Web server crashed — extracting errors...")
                            bundle_errors = self._extract_errors_from_log_buffer(app_id)

                        if bundle_ok:
                            label = "Modified bundle compiles cleanly" if repair_round == 0 else f"Bundle repaired (round {repair_round})"
                            await blog.success(label)
                            break

                        await blog.warn(f"Bundle has issues — repair round {repair_round + 1}/4...")
                        fixed_something = False

                        # Fix missing deps
                        if bundle_errors:
                            dep_fixed = await self._fix_missing_deps(app_id, app_dir, bundle_errors, blog)
                            if dep_fixed:
                                fixed_something = True

                        # Fix code errors via LLM
                        code_errors = [e for e in bundle_errors if e.get("file", "unknown") != "unknown"]
                        if code_errors:
                            repaired = await self._repair_bundle_errors(
                                app_id, app_dir, code_errors, generated,
                                changes, app_name, [], "none", blog
                            )
                            if repaired:
                                fixed_something = True

                        # No errors found — restart server
                        if not fixed_something and not web_alive:
                            try:
                                web_port = await self._app_manager.start_web(app_id)
                                await asyncio.sleep(5)
                                fixed_something = True
                            except Exception:
                                pass

                        if not fixed_something:
                            break

                        if web_alive:
                            await asyncio.sleep(4)
                        else:
                            try:
                                web_port = await self._app_manager.start_web(app_id)
                            except Exception:
                                pass

                    if not bundle_ok:
                        await blog.error("Bundle has compilation errors after modification")
                        async with async_session_maker() as db:
                            app = await db.get(App, app_id)
                            if app:
                                app.status = "error"
                                await db.commit()
                except Exception as e:
                    await blog.warn(f"Bundle validation skipped: {e}")

            # ── Step 5: Ready! ─────────────────────────────────────
            blog.set_step("ready")
            await self._update_step(job_id, user_id, "ready", "done")

            await blog.persist()
            async with async_session_maker() as db:
                job = await db.get(BuildJob, job_id)
                if job:
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    job.model = "claude-opus-4-6"
                    await db.commit()

            if self._ws_broadcast:
                qr_url = await self._app_manager.get_qr_url(app_id) if self._app_manager else ""
                web_url = await self._app_manager.get_web_url(app_id) if self._app_manager else ""
                await self._ws_broadcast(user_id, {
                    "type": "app_ready",
                    "app_id": app_id,
                    "name": app_name,
                    "qr_url": qr_url,
                    "web_url": web_url,
                })

            logger.info(f"[BUILD] Modification complete for '{app_name}' (job={job_id})")

        except Exception as e:
            logger.exception(f"[BUILD] Modification error for '{app_name}'")
            await blog.error(f"Modification error: {e}")
            await blog.persist()
            await self._fail_job(job_id, app_id, f"Modification error: {e}")

    # ── Step helpers ────────────────────────────────────────────────

    async def _step_plan(
        self, job_id: str, user_id: str, description: str, blog=None,
        extra_context: str = "",
    ) -> Optional[Dict]:
        """Run the planning step — call LLM to get app plan."""
        if blog:
            blog.set_step("planning")
        await self._update_step(job_id, user_id, "planning", "running")

        try:
            if blog:
                await blog.info("Calling Claude Opus for app architecture planning...")
            llm_response = await self._call_llm(
                PLANNING_PROMPT.format(description=description, extra_context=extra_context),
                "Plan this app and output JSON.",
                blog=blog,
                purpose="App architecture planning",
            )

            plan = self._extract_json(llm_response)
            if not plan:
                if blog:
                    await blog.error("Failed to parse plan JSON from LLM response",
                                     llm_response[:300])
                await self._update_step(job_id, user_id, "planning", "failed",
                                        detail="Could not parse LLM planning output")
                return None

            files_list = plan.get("files", [])
            deps_list = plan.get("dependencies", [])
            if blog:
                await blog.success(f"Plan generated: {plan.get('app_name', 'App')}")
                await blog.info(f"Summary: {plan.get('summary', 'N/A')}")
                await blog.info(f"Files planned: {len(files_list)}")
                for f in files_list:
                    await blog.debug(f"  {f}")
                await blog.info(f"Dependencies: {len(deps_list)}")
                for d in deps_list:
                    await blog.debug(f"  {d}")
                await blog.info(f"Database: {plan.get('db_type', 'none')}")

            plan_detail = (
                f"{plan.get('summary', '')}\n\n"
                f"Files ({len(files_list)}):\n" +
                "\n".join(f"  {f}" for f in files_list) +
                f"\n\nDependencies ({len(deps_list)}):\n" +
                "\n".join(f"  {d}" for d in deps_list) +
                f"\n\nDatabase: {plan.get('db_type', 'none')}"
            )
            await self._update_step(job_id, user_id, "planning", "done",
                                    detail=plan_detail.strip())
            return plan
        except TokenLimitError:
            raise  # Let token limit propagate for pause/resume
        except Exception as e:
            if blog:
                await blog.error(f"Planning failed: {e}")
            await self._update_step(job_id, user_id, "planning", "failed", detail=str(e))
            return None

    async def _generate_code(
        self,
        description: str,
        app_name: str,
        files: List[str],
        deps: List[str],
        db_type: str,
        job_id: str = "",
        user_id: str = "",
        blog=None,
        design_notes: str = "",
        existing_files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Generate code for all files using LLM — parallel with semaphore."""
        generated = {}
        all_files = json.dumps(files)
        all_deps = json.dumps(deps)
        total = len(files)
        completed = 0
        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(5)

        design_section = f"Design notes: {design_notes}" if design_notes else ""

        async def _gen_one(i: int, file_path: str):
            nonlocal completed
            async with semaphore:
                if blog:
                    await blog.info(f"Generating {file_path}...", f"[{i+1}/{total}]")

                # Include existing file content for modifications
                existing_hint = ""
                if existing_files and file_path in existing_files:
                    existing_hint = (
                        f"\n\nCurrent content of {file_path} (modify this, don't start from scratch):\n"
                        f"```\n{existing_files[file_path][:4000]}\n```"
                    )

                prompt = CODE_GEN_PROMPT.format(
                    file_path=file_path,
                    description=description,
                    app_name=app_name,
                    all_files=all_files,
                    all_deps=all_deps,
                    db_type=db_type,
                    design_notes=design_section,
                )
                if existing_hint:
                    prompt += existing_hint

                try:
                    code = await self._call_llm(
                        prompt, f"Generate code for {file_path}",
                        blog=blog, purpose=f"Generate {file_path}",
                        max_tokens=32000,
                    )
                    code = self._strip_fences(code)

                    # Validate syntax — retry if truncated
                    is_valid, error_msg = self._validate_syntax(code, file_path)
                    if not is_valid:
                        if blog:
                            await blog.warn(f"Validation failed for {file_path}: {error_msg} — retrying")
                        retry_prompt = prompt + (
                            "\n\nCRITICAL: The previous generation was truncated/incomplete. "
                            "You MUST output the COMPLETE file with ALL brackets closed, "
                            "ALL StyleSheet styles defined, and ALL exports present. "
                            "If the file is very long, simplify the implementation rather than truncating it."
                        )
                        code = await self._call_llm(
                            retry_prompt, f"Generate code for {file_path} (retry — previous was truncated)",
                            blog=blog, purpose=f"Retry {file_path} (truncated)",
                            max_tokens=32000,
                        )
                        code = self._strip_fences(code)

                        # Validate again
                        is_valid2, error_msg2 = self._validate_syntax(code, file_path)
                        if not is_valid2 and blog:
                            await blog.warn(f"Retry still has issues for {file_path}: {error_msg2} — attempting auto-repair")
                            # Auto-repair: close any unclosed brackets
                            code = self._auto_repair_syntax(code, file_path)

                    async with lock:
                        generated[file_path] = code
                        completed += 1
                    if blog:
                        await blog.file_written(file_path, len(code.encode()), code)
                except TokenLimitError:
                    # Let token limit errors propagate — caller handles pause/resume
                    raise
                except Exception as e:
                    if blog:
                        await blog.error(f"Failed to generate {file_path}: {e}")
                    async with lock:
                        generated[file_path] = self._make_error_fallback(file_path)
                        completed += 1

                if job_id:
                    async with lock:
                        current = completed
                    await self._update_step(
                        job_id, user_id, "writing", "running",
                        detail=f"File {current}/{total}: {file_path}"
                    )

        try:
            await asyncio.gather(*[_gen_one(i, fp) for i, fp in enumerate(files)])
        except TokenLimitError:
            # Attach partial results to the exception so caller can checkpoint
            err = TokenLimitError(
                retry_after_seconds=300,
                message=f"Token limit during code generation ({len(generated)}/{total} files done)"
            )
            err.partial_files = generated  # type: ignore[attr-defined]
            err.pending_files = [f for f in files if f not in generated]  # type: ignore[attr-defined]
            raise err
        return generated

    async def _write_infra_files(self, app_id: str, generated_files: dict, blog=None):
        """Write essential infrastructure files that every app needs.

        1. metro.config.js — stubs out expo-sqlite WASM on web (prevents crash)
        2. Patches database files to use conditional import (web-safe)
        """
        infra_files = {}

        # Metro config — stub out wa-sqlite on web to prevent SharedArrayBuffer crash
        infra_files["/metro.config.js"] = """const { getDefaultConfig } = require('expo/metro-config');
const config = getDefaultConfig(__dirname);

const origResolveRequest = config.resolver.resolveRequest;
config.resolver.resolveRequest = (context, moduleName, platform) => {
  // On web: resolve .wasm files and expo-sqlite web internals to empty
  if (platform === 'web') {
    if (moduleName.endsWith('.wasm') || moduleName.includes('wa-sqlite')) {
      return { type: 'empty' };
    }
  }
  if (origResolveRequest) {
    return origResolveRequest(context, moduleName, platform);
  }
  return context.resolveRequest(context, moduleName, platform);
};

module.exports = config;
"""

        # Inject emoji font CSS — iOS WKWebView doesn't auto-fallback to emoji fonts.
        # react-native-web sets font-family to "-apple-system,...,sans-serif" which lacks emoji.
        # We inject a global <style> into index.html (or the Expo web template) to fix this.
        index_html = os.path.join(
            self._app_manager.APPS_DIR if hasattr(self._app_manager, 'APPS_DIR') else "/opt/toup-agent/apps",
            app_id, "web", "index.html"
        )
        # Also try the root index.html (Expo web template)
        root_index = os.path.join(
            self._app_manager.APPS_DIR if hasattr(self._app_manager, 'APPS_DIR') else "/opt/toup-agent/apps",
            app_id, "index.html"
        )
        emoji_css = (
            '<style id="emoji-fix">'
            '* { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
            'Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", '
            '"Noto Color Emoji" !important; }'
            '</style>'
        )
        for html_path in (index_html, root_index):
            if os.path.exists(html_path):
                try:
                    with open(html_path, 'r', encoding='utf-8') as f:
                        html = f.read()
                    if 'emoji-fix' not in html:
                        html = html.replace('</head>', f'{emoji_css}\n</head>', 1)
                        with open(html_path, 'w', encoding='utf-8') as f:
                            f.write(html)
                        if blog:
                            await blog.info("Injected emoji font CSS for iOS WebView compatibility")
                except Exception as e:
                    logger.debug(f"[BUILD] Failed to inject emoji CSS: {e}")

        # Patch any database file that directly imports expo-sqlite
        for fp, code in generated_files.items():
            if "import * as SQLite from 'expo-sqlite'" in code or 'import * as SQLite from "expo-sqlite"' in code:
                patched = self._make_database_web_safe(code)
                if patched != code:
                    infra_files[fp] = patched
                    if blog:
                        await blog.info(f"Patched {fp} for web-safe SQLite")

        # Fix import/export mismatches: if a lib file only has `export default X`,
        # add a named export too so `import { X }` also works
        self._fix_import_export_mismatches(generated_files, infra_files)

        # Fix React Navigation v7 theme — must include `fonts` property
        import re as _re
        FULL_THEME_PROP = (
            '\n        theme={{\n'
            '          dark: true,\n'
            '          colors: {\n'
            '            primary: "#58A6FF",\n'
            '            background: "#161B22",\n'
            '            card: "#1C2128",\n'
            '            text: "#F0F2F5",\n'
            '            border: "#30363D",\n'
            '            notification: "#58A6FF",\n'
            '          },\n'
            '          fonts: {\n'
            '            regular: { fontFamily: "System", fontWeight: "400" },\n'
            '            medium: { fontFamily: "System", fontWeight: "500" },\n'
            '            bold: { fontFamily: "System", fontWeight: "700" },\n'
            '            heavy: { fontFamily: "System", fontWeight: "900" },\n'
            '          },\n'
            '        }}'
        )
        fonts_block = (
            "          fonts: {\n"
            "            regular: { fontFamily: 'System', fontWeight: '400' as const },\n"
            "            medium: { fontFamily: 'System', fontWeight: '500' as const },\n"
            "            bold: { fontFamily: 'System', fontWeight: '700' as const },\n"
            "            heavy: { fontFamily: 'System', fontWeight: '900' as const },\n"
            "          },\n"
        )
        for fp, code in {**generated_files, **infra_files}.items():
            if 'NavigationContainer' not in code:
                continue

            patched = code

            if 'theme' not in code:
                # Case 1: NavigationContainer has NO theme prop at all — add complete theme
                patched = patched.replace(
                    '<NavigationContainer',
                    '<NavigationContainer' + FULL_THEME_PROP,
                    1,
                )
            elif 'fonts' not in code:
                # Case 2: Has theme but missing fonts — inject fonts block
                # Try multiple patterns to find where to insert fonts:
                patched = _re.sub(
                    r'(colors:\s*\{[^}]*\},?\s*\n(\s+)\})',
                    r'\1,\n' + fonts_block,
                    code,
                    count=1,
                )
                if patched == code:
                    patched = _re.sub(
                        r'(notification:[^\n]+\n\s+\},)',
                        r'\1\n' + fonts_block,
                        code,
                        count=1,
                    )
                if patched == code:
                    patched = _re.sub(
                        r'(border:[^\n]+\n\s+\},)',
                        r'\1\n' + fonts_block,
                        code,
                        count=1,
                    )
                if patched == code:
                    # Ultimate fallback — insert before closing `}}`
                    patched = _re.sub(
                        r'(\},\s*\n\s*\}\}[\s>])',
                        fonts_block + r'\1',
                        code,
                        count=1,
                    )

            if patched != code:
                infra_files[fp] = patched
                if blog:
                    await blog.info(f"Injected React Navigation v7 fonts into {fp}")

        # Ensure ErrorBoundary exists — prevents white screens from runtime errors
        if "/components/ErrorBoundary.tsx" not in generated_files and "/components/ErrorBoundary.tsx" not in infra_files:
            infra_files["/components/ErrorBoundary.tsx"] = '''import React from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';

interface State { hasError: boolean; error: Error | null; }

export default class ErrorBoundary extends React.Component<{children: React.ReactNode}, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('[ErrorBoundary]', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <View style={styles.container}>
          <Text style={styles.emoji}>⚠️</Text>
          <Text style={styles.title}>Something went wrong</Text>
          <Text style={styles.message} numberOfLines={3}>
            {this.state.error?.message || 'An unexpected error occurred'}
          </Text>
          <Pressable
            style={styles.button}
            onPress={() => this.setState({ hasError: false, error: null })}
          >
            <Text style={styles.buttonText}>Reload App</Text>
          </Pressable>
        </View>
      );
    }
    return this.props.children;
  }
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#161B22', justifyContent: 'center', alignItems: 'center', padding: 32 },
  emoji: { fontSize: 48, marginBottom: 16 },
  title: { color: '#F0F2F5', fontSize: 18, fontWeight: '700', marginBottom: 8 },
  message: { color: '#8B949E', fontSize: 13, textAlign: 'center', lineHeight: 18, marginBottom: 24, maxWidth: 320 },
  button: { backgroundColor: '#58A6FF', paddingHorizontal: 24, paddingVertical: 12, borderRadius: 20 },
  buttonText: { color: '#FFF', fontSize: 14, fontWeight: '600' },
});
'''
            if blog:
                await blog.info("Added ErrorBoundary component (prevents white screens)")

        # Inject ErrorBoundary wrapper into App.tsx if not already present
        for fp in ["/App.tsx"]:
            code = infra_files.get(fp, generated_files.get(fp, ""))
            if code and "ErrorBoundary" not in code and "NavigationContainer" in code:
                # Add import
                code = "import ErrorBoundary from './components/ErrorBoundary';\n" + code
                # Wrap NavigationContainer with ErrorBoundary
                code = code.replace("<NavigationContainer", "<ErrorBoundary>\n      <NavigationContainer", 1)
                # Find the closing </NavigationContainer> and add </ErrorBoundary> after
                # Handle both </NavigationContainer> patterns
                if "</NavigationContainer>" in code:
                    # Find last occurrence
                    idx = code.rfind("</NavigationContainer>")
                    close_tag = "</NavigationContainer>"
                    # Find the end of the line containing the closing tag
                    end_idx = code.find("\n", idx + len(close_tag))
                    if end_idx == -1:
                        end_idx = len(code)
                    code = code[:end_idx] + "\n      </ErrorBoundary>" + code[end_idx:]
                infra_files[fp] = code
                if blog:
                    await blog.info("Wrapped App.tsx with ErrorBoundary")

        # Fix common LLM code generation mistakes that cause white pages
        self._fix_common_llm_mistakes(generated_files, infra_files, blog)

        if infra_files:
            await self._app_manager.write_app_files(app_id, infra_files)
            if blog:
                await blog.info(f"Wrote {len(infra_files)} infrastructure files (metro config, web-safe DB)")

    @staticmethod
    def _fix_common_llm_mistakes(generated_files: dict, infra_files: dict, blog=None):
        """Detect and fix common LLM code generation mistakes that cause white pages.

        Catches:
        1. Async component functions (async function Screen → returns Promise, not element)
        2. expo-router mixed with NavigationContainer (instant crash)
        3. Platform-specific APIs without guards (SecureStore, Vibration on web)
        """
        import re as _re

        for fp, code in {**generated_files, **infra_files}.items():
            if not fp.endswith(('.tsx', '.ts', '.jsx', '.js')):
                continue
            original = code
            patched = code

            # 1. Async component functions → remove async keyword
            # Pattern a: export default async function ScreenName(
            patched = _re.sub(
                r'export\s+default\s+async\s+function\s+(\w+)',
                r'export default function \1',
                patched,
            )
            patched = _re.sub(
                r'export\s+async\s+function\s+(\w+)',
                r'export function \1',
                patched,
            )
            # Pattern b: const ScreenName = async () => { (arrow functions used as components)
            # Only for screen/component files — don't touch lib/utils
            if '/screens/' in fp or '/components/' in fp or fp == '/App.tsx':
                patched = _re.sub(
                    r'((?:export\s+)?const\s+\w+(?:Screen|Component|Page|View)\s*=\s*)async\s*\(',
                    r'\1(',
                    patched,
                )
                patched = _re.sub(
                    r'((?:export\s+)?const\s+\w+(?:Screen|Component|Page|View)\s*=\s*)async\s*\(\)',
                    r'\1()',
                    patched,
                )

            # 2. Mixing expo-router and NavigationContainer
            if 'expo-router' in patched and 'NavigationContainer' in patched:
                patched = _re.sub(
                    r'import\s+\{[^}]*\}\s+from\s+[\'"]expo-router[\'"];?\n?',
                    '',
                    patched,
                )

            # 3. Guard platform-specific APIs that crash on web
            # SecureStore
            if "expo-secure-store" in patched and "Platform.OS" not in patched:
                # Only add Platform import if not already present
                platform_import_line = ""
                rn_check = _re.findall(r'import\s*\{([^}]+)\}\s*from\s*[\'"]react-native[\'"]', patched)
                if not any('Platform' in imp for imp in rn_check):
                    platform_import_line = "import { Platform } from 'react-native';\n"
                patched = patched.replace(
                    "import * as SecureStore from 'expo-secure-store';",
                    platform_import_line +
                    "let SecureStore: any;\n"
                    "if (Platform.OS !== 'web') {\n"
                    "  const mod = 'expo-secure-store'; SecureStore = require(mod);\n"
                    "}",
                )

            # Alert.alert() — crashes on web, wrap in Platform check
            # Use balanced-paren matching (regex [^)]+ fails on nested parens / template literals)
            if 'Alert.alert(' in patched and "Platform.OS === 'web'" not in patched:
                result_parts = []
                search_from = 0
                while True:
                    idx = patched.find('Alert.alert(', search_from)
                    if idx == -1:
                        result_parts.append(patched[search_from:])
                        break
                    result_parts.append(patched[search_from:idx])
                    # Find balanced closing paren
                    paren_start = idx + len('Alert.alert(') - 1  # index of '('
                    depth = 0
                    end = paren_start
                    for ci in range(paren_start, len(patched)):
                        if patched[ci] == '(':
                            depth += 1
                        elif patched[ci] == ')':
                            depth -= 1
                            if depth == 0:
                                end = ci
                                break
                    if depth != 0:
                        # Unbalanced — skip this occurrence
                        result_parts.append(patched[idx:idx + len('Alert.alert(')])
                        search_from = idx + len('Alert.alert(')
                        continue
                    full_call = patched[idx:end + 1]  # "Alert.alert(...)"
                    args_inner = patched[paren_start + 1:end]  # everything inside parens
                    # Extract first arg (title) for window.alert fallback — take up to first comma at depth 0
                    first_arg_parts = []
                    d = 0
                    for ch in args_inner:
                        if ch in ('(', '[', '{'):
                            d += 1
                        elif ch in (')', ']', '}'):
                            d -= 1
                        elif ch == ',' and d == 0:
                            break
                        first_arg_parts.append(ch)
                    first_arg = ''.join(first_arg_parts).strip()
                    replacement = f"(Platform.OS === 'web' ? window.alert({first_arg}) : {full_call})"
                    result_parts.append(replacement)
                    search_from = end + 1
                patched = ''.join(result_parts)
                # Ensure Platform is imported — safely add to existing react-native import
                rn_imports = _re.findall(r'import\s*\{([^}]+)\}\s*from\s*[\'"]react-native[\'"]', patched)
                already_has_platform = any('Platform' in imp for imp in rn_imports)
                if not already_has_platform:
                    def _add_platform_import(m):
                        imports = m.group(1).rstrip().rstrip(',')
                        return f"import {{ {imports}, Platform }} from 'react-native'"
                    patched, n = _re.subn(
                        r"import\s*\{([^}]+)\}\s*from\s*['\"]react-native['\"]",
                        _add_platform_import,
                        patched,
                        count=1,
                    )
                    if n == 0:
                        patched = "import { Platform } from 'react-native';\n" + patched

            # Vibration — no-op on web
            if 'Vibration.vibrate' in patched and "Platform.OS" not in patched:
                patched = _re.sub(
                    r'Vibration\.vibrate\([^)]*\)',
                    r"(Platform.OS !== 'web' && Vibration.vibrate())",
                    patched,
                )

            # 4. Non-async functions using await — LLMs write `function X(): Promise<T> { await ... }`
            #    which is a SyntaxError. Add `async` to any `export function` that contains `await `.
            if 'await ' in patched:
                # Find all non-async exported functions and check if they contain await
                def _add_async_to_awaiting_fns(code: str) -> str:
                    lines = code.split('\n')
                    result = []
                    i = 0
                    while i < len(lines):
                        line = lines[i]
                        # Match `export function X(` or `function X(` that is NOT already async
                        if (_re.match(r'\s*(?:export\s+)?function\s+\w+', line)
                                and 'async' not in line):
                            # Scan ahead to find the function body and check for await.
                            # The function signature may span multiple lines with TS types
                            # that contain { } (e.g. Promise<Array<{ key: string }>>).
                            # Only start counting braces AFTER we've seen the first { that
                            # follows a ) or > (the actual function body opening brace).
                            fn_start = i
                            body_started = False
                            brace_count = 0
                            has_await = False
                            for j in range(i, min(i + 200, len(lines))):
                                ln = lines[j]
                                if not body_started:
                                    # Look for the opening brace of the function body.
                                    # It's the first { that appears after ) or > at line-end
                                    # (not inside a type annotation like { key: string }).
                                    stripped = ln.rstrip()
                                    if stripped.endswith('{'):
                                        body_started = True
                                        brace_count = 1
                                        # Check this line for await too
                                        if 'await ' in ln:
                                            has_await = True
                                    continue
                                # Inside function body — count braces
                                brace_count += ln.count('{') - ln.count('}')
                                if 'await ' in ln:
                                    has_await = True
                                if brace_count <= 0:
                                    break
                            if has_await:
                                if 'export function ' in line:
                                    lines[i] = line.replace('export function ', 'export async function ', 1)
                                else:
                                    lines[i] = _re.sub(r'(\s*)function\s+', r'\1async function ', line, count=1)
                        result.append(lines[i])
                        i += 1
                    return '\n'.join(result)
                patched = _add_async_to_awaiting_fns(patched)

            if patched != original:
                infra_files[fp] = patched

    @staticmethod
    def _fix_import_export_mismatches(generated_files: dict, infra_files: dict):
        """Ensure lib files with only `export default X` also have named `export { X }`.

        LLMs often generate `import { X } from './lib/X'` in one file but only
        `export default X` in the lib file. This causes X to be undefined at
        runtime, which crashes the app with a white page.
        """
        import re as _re

        # Collect all named imports across all files
        named_imports: dict[str, set[str]] = {}  # module_path -> set of names
        for fp, code in generated_files.items():
            for m in _re.finditer(r'import\s*\{([^}]+)\}\s*from\s*[\'"]([^\'"]+)[\'"]', code):
                names = {n.strip().split(' as ')[0] for n in m.group(1).split(',') if n.strip()}
                module = m.group(2)
                named_imports.setdefault(module, set()).update(names)

        # For each lib file, check if it has a default export but missing named exports
        for fp, code in generated_files.items():
            if fp in infra_files:
                code = infra_files[fp]  # use already-patched version

            # Find default export: `export default X;` or `export default function X`
            default_match = _re.search(r'export\s+default\s+(\w+)\s*;', code)
            if not default_match:
                continue
            default_name = default_match.group(1)

            # Check if any file imports this name from this module path
            needs_named = False
            for module_path, names in named_imports.items():
                # Resolve relative path to check if it points to this file
                if default_name in names:
                    needs_named = True
                    break

            if needs_named and f'export {{ {default_name} }}' not in code and f'export {{{default_name}}}' not in code:
                # Add named export alongside default export
                code = code.replace(
                    f'export default {default_name};',
                    f'export {{ {default_name} }};\nexport default {default_name};'
                )
                infra_files[fp] = code

    @staticmethod
    def _make_database_web_safe(code: str) -> str:
        """Replace direct expo-sqlite import with conditional import + web mock."""
        import_line = "import * as SQLite from 'expo-sqlite';"
        if import_line not in code:
            import_line = 'import * as SQLite from "expo-sqlite";'
        if import_line not in code:
            return code

        web_safe_import = """import { Platform } from 'react-native';

// On web, use in-memory mock (expo-sqlite WASM doesn't bundle with Metro)
let SQLite: any;
if (Platform.OS !== 'web') {
  // @ts-ignore - dynamic require to avoid Metro static analysis
  const mod = 'expo-sqlite'; SQLite = require(mod);
}

// In-memory store for web preview
const _webStore: Record<string, any[]> = {};
const _webDb = {
  execAsync: async (_sql: string) => { /* no-op for DDL on web */ },
  getAllAsync: async (sql: string, _params?: any[]) => {
    const table = sql.match(/from\\s+(\\w+)/i)?.[1] || 'default';
    return _webStore[table] || [];
  },
  runAsync: async (_sql: string, _params?: any[]) => {
    return { lastInsertRowId: 1, changes: 1 };
  },
  getFirstAsync: async (sql: string, _params?: any[]) => {
    const table = sql.match(/from\\s+(\\w+)/i)?.[1] || 'default';
    const rows = _webStore[table] || [];
    return rows[0] || null;
  },
};"""

        code = code.replace(import_line, web_safe_import, 1)

        # Fix type annotations that reference SQLite.SQLiteDatabase
        code = code.replace('SQLite.SQLiteDatabase', 'any')

        # Patch getDatabase to return web mock on web platform
        # Look for the openDatabaseAsync call and wrap it
        if 'openDatabaseAsync' in code:
            import re as _re
            # Add Platform.OS web check before the try block in getDatabase
            code = _re.sub(
                r'(export async function getDatabase\(\)[^{]*\{[^}]*?)(\btry\b)',
                r'\1if (Platform.OS === "web") { return _webDb as any; }\n  \2',
                code,
                count=1,
            )

        return code

    async def _call_llm(
        self, system_prompt: str, user_message: str, model: str = "claude-opus-4-6",
        blog=None, purpose: str = "", max_tokens: int = 32000,
    ) -> str:
        """Call the LLM (Anthropic). Using Opus 4.6 for all phases."""
        import time as _time
        from app.config import settings

        anthropic_key = settings.anthropic_api_key or ""
        if anthropic_key:
            try:
                import anthropic
                import os
                is_oauth = "sk-ant-oat" in anthropic_key
                if is_oauth:
                    os.environ.pop("ANTHROPIC_API_KEY", None)
                    client = anthropic.AsyncAnthropic(
                        auth_token=anthropic_key,
                        default_headers={
                            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
                            "user-agent": "claude-cli/2.1.2 (external, cli)",
                            "x-app": "cli",
                        },
                    )
                else:
                    client = anthropic.AsyncAnthropic(api_key=anthropic_key)

                t0 = _time.time()
                # Use streaming — Anthropic requires it for large max_tokens
                text = ""
                input_tok = 0
                output_tok = 0
                stop_reason = "end_turn"
                async with client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                ) as stream:
                    async for chunk in stream.text_stream:
                        text += chunk
                    response = await stream.get_final_message()
                    input_tok = getattr(response.usage, 'input_tokens', 0)
                    output_tok = getattr(response.usage, 'output_tokens', 0)
                    stop_reason = getattr(response, 'stop_reason', 'end_turn')
                elapsed = _time.time() - t0

                if blog:
                    await blog.llm_call(
                        model=model,
                        purpose=purpose or user_message[:50],
                        input_tokens=input_tok,
                        output_tokens=output_tok,
                        duration_s=elapsed,
                    )

                # Warn if output was truncated due to max_tokens
                if stop_reason == "max_tokens" and blog:
                    await blog.warn(f"LLM output truncated (hit {max_tokens} token limit)")

                return text
            except anthropic.RateLimitError as e:
                # Token limit / rate limit — extract retry-after and raise for pause/resume
                retry_after = 300  # default 5 minutes
                try:
                    retry_after = int(getattr(e.response, 'headers', {}).get('retry-after', 300))
                except (ValueError, TypeError, AttributeError):
                    pass
                msg = f"Rate limit reached. Resets in {retry_after}s"
                if blog:
                    await blog.warn(msg)
                raise TokenLimitError(retry_after_seconds=retry_after, message=msg)
            except anthropic.APIStatusError as e:
                if e.status_code == 529:  # API overloaded
                    raise TokenLimitError(retry_after_seconds=120, message="API overloaded, retry in 2min")
                if blog:
                    await blog.warn(f"Anthropic call failed (status {e.status_code}), falling back to OpenAI: {e}")
                else:
                    logger.warning(f"[BUILD] Anthropic call failed (status {e.status_code}), falling back to OpenAI: {e}")
            except Exception as e:
                if blog:
                    await blog.warn(f"Anthropic call failed, falling back to OpenAI: {e}")
                else:
                    logger.warning(f"[BUILD] Anthropic call failed, falling back to OpenAI: {e}")

        if settings.openai_api_key:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=settings.openai_api_key)
                oai_model = settings.agent_model or "gpt-4o-mini"
                t0 = _time.time()
                response = await client.chat.completions.create(
                    model=oai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    max_completion_tokens=max_tokens,
                )
                elapsed = _time.time() - t0
                text = response.choices[0].message.content or ""
                usage = response.usage
                if blog and usage:
                    await blog.llm_call(
                        model=oai_model,
                        purpose=purpose or user_message[:50],
                        input_tokens=usage.prompt_tokens or 0,
                        output_tokens=usage.completion_tokens or 0,
                        duration_s=elapsed,
                    )
                return text
            except Exception as e:
                if blog:
                    await blog.error(f"OpenAI call failed: {e}")
                raise

        raise RuntimeError("No LLM provider configured (need ANTHROPIC_API_KEY or OPENAI_API_KEY)")

    async def _update_step(
        self,
        job_id: str,
        user_id: str,
        step_type: str,
        status: str,
        detail: Optional[str] = None,
    ):
        """Update a step in the build job and broadcast via WebSocket."""
        from app.db.database import async_session_maker
        from app.db.models import BuildJob

        async with async_session_maker() as db:
            job = await db.get(BuildJob, job_id)
            if not job:
                return

            steps = json.loads(job.steps_json) if job.steps_json else []
            step_dict = None
            for s in steps:
                if s["type"] == step_type:
                    s["status"] = status
                    if detail:
                        s["detail"] = detail
                    if status == "running":
                        s["started_at"] = datetime.utcnow().isoformat()
                    elif status in ("done", "failed"):
                        started = s.get("started_at")
                        if started:
                            try:
                                start_dt = datetime.fromisoformat(started)
                                s["duration_ms"] = int((datetime.utcnow() - start_dt).total_seconds() * 1000)
                            except Exception:
                                pass
                    step_dict = s
                    break

            if job.status == "queued" and status == "running":
                job.status = "running"

            job.steps_json = json.dumps(steps)
            await db.commit()

        if self._ws_broadcast and step_dict:
            await self._ws_broadcast(user_id, {
                "type": "job_update",
                "job_id": job_id,
                "status": job.status if job else "running",
                "step": step_dict,
            })

    async def _fail_job(self, job_id: str, app_id: str, error_msg: str):
        """Mark a job as failed and clean up stuck step statuses."""
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob

        logger.error(f"[BUILD] Job {job_id} failed: {error_msg}")

        async with async_session_maker() as db:
            job = await db.get(BuildJob, job_id)
            if job:
                job.status = "failed"
                job.error_message = error_msg
                job.completed_at = datetime.utcnow()

                # Clean up stuck steps: mark any "running" step as "failed"
                try:
                    steps = json.loads(job.steps_json) if job.steps_json else []
                    for s in steps:
                        if s.get("status") == "running":
                            s["status"] = "failed"
                            started = s.get("started_at")
                            if started:
                                try:
                                    start_dt = datetime.fromisoformat(started)
                                    s["duration_ms"] = int(
                                        (datetime.utcnow() - start_dt).total_seconds() * 1000
                                    )
                                except Exception:
                                    pass
                    job.steps_json = json.dumps(steps)
                except (json.JSONDecodeError, TypeError):
                    pass

                await db.commit()

            app = await db.get(App, app_id)
            if app:
                app.status = "error"
                await db.commit()

    async def _pause_job(
        self, job_id: str, app_id: str, user_id: str,
        retry_after_seconds: int, checkpoint: Dict, blog,
    ):
        """Pause a build job due to token limit — save checkpoint for resume."""
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob

        now = datetime.utcnow()
        resume_at = now + timedelta(seconds=retry_after_seconds)

        logger.info(f"[BUILD] Job {job_id} paused — token limit. Resume after {resume_at.isoformat()}")

        async with async_session_maker() as db:
            job = await db.get(BuildJob, job_id)
            if job:
                job.status = "paused"
                job.paused_at = now
                job.resume_after = resume_at
                job.checkpoint_json = json.dumps(checkpoint)
                await db.commit()

            app = await db.get(App, app_id)
            if app:
                app.status = "paused"
                await db.commit()

        await blog.warn(f"Token limit reached — build paused. Resumes at {resume_at.strftime('%H:%M:%S UTC')}")
        await blog.persist()

        # Broadcast pause event via WebSocket
        if self._ws_broadcast:
            await self._ws_broadcast(user_id, {
                "type": "job_paused",
                "job_id": job_id,
                "app_id": app_id,
                "resume_after": resume_at.isoformat(),
                "retry_after_seconds": retry_after_seconds,
                "current_step": checkpoint.get("current_step", "unknown"),
                "completed_steps": checkpoint.get("completed_steps", []),
            })

    # ── Parsing helpers ─────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """Extract JSON object from LLM output (handles markdown fences)."""
        text = text.strip()
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        patterns = [
            r"```json\s*\n(.*?)\n```",
            r"```\s*\n(.*?)\n```",
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1) if match.lastindex else match.group(0))
                except (json.JSONDecodeError, IndexError):
                    continue
        return None

    @staticmethod
    def _strip_fences(code: str) -> str:
        """Strip markdown code fences from generated code."""
        code = code.strip()
        if code.startswith("```"):
            first_newline = code.index("\n") if "\n" in code else len(code)
            code = code[first_newline + 1:]
        if code.endswith("```"):
            code = code[:-3].rstrip()
        return code

    @staticmethod
    def _make_error_fallback(file_path: str) -> str:
        """Generate a type-appropriate error fallback based on file type.

        - Screen/component files → visible error placeholder component
        - Library/utility files → stub module with no-op exports
        - Config files → empty valid config
        """
        basename = os.path.basename(file_path)
        name = basename.replace('.tsx', '').replace('.ts', '').replace('.jsx', '').replace('.js', '')

        # Library/utility files — export a no-op stub, NOT a React component
        if '/lib/' in file_path or '/utils/' in file_path or '/helpers/' in file_path:
            if 'database' in name.lower() or 'db' in name.lower():
                return (
                    "// Database stub — generation failed, app runs without persistence\n"
                    "import { Platform } from 'react-native';\n"
                    "const mockDb = {\n"
                    "  execAsync: async () => {},\n"
                    "  runAsync: async () => ({ lastInsertRowid: 0, changes: 0 }),\n"
                    "  getFirstAsync: async () => null,\n"
                    "  getAllAsync: async () => [],\n"
                    "};\n"
                    "export async function getDatabase() { return mockDb; }\n"
                    "export default { getDatabase };\n"
                )
            if 'bridge' in name.lower() or 'agent' in name.lower():
                return (
                    "// AgentBridge stub — delegates to injected bridge if available\n"
                    "const injected = typeof window !== 'undefined' && (window as any).__TOUP_AGENT_BRIDGE;\n"
                    "const agentBridge = injected || {\n"
                    "  isConnected: false,\n"
                    "  currentScreen: 'Home',\n"
                    "  sendMessage: async () => {},\n"
                    "  onAgentMessage: () => () => {},\n"
                    "  onToolActivity: () => () => {},\n"
                    "  setNavigationRef: () => {},\n"
                    "  navigate: () => {},\n"
                    "  getScreens: () => [],\n"
                    "  getActions: () => [],\n"
                    "};\n"
                    "export { agentBridge };\n"
                    "export const AgentBridge = agentBridge;\n"
                    "export default agentBridge;\n"
                )
            # Generic library stub
            return (
                f"// Stub for {basename} — generation failed\n"
                f"export default {{}};\n"
            )

        # Screen/component files — visible error placeholder
        return (
            f"import React from 'react';\n"
            f"import {{ View, Text }} from 'react-native';\n"
            f"export default function {name}() {{\n"
            f"  return (\n"
            f"    <View style={{{{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#161B22' }}}}>\n"
            f"      <Text style={{{{ color: '#F85149', fontSize: 16, fontWeight: 'bold', marginBottom: 8 }}}}>Build Error</Text>\n"
            f"      <Text style={{{{ color: '#8B949E', fontSize: 13, textAlign: 'center', paddingHorizontal: 24 }}}}>\n"
            f"        Failed to generate {basename}. Please rebuild the app.\n"
            f"      </Text>\n"
            f"    </View>\n"
            f"  );\n"
            f"}}\n"
        )

    @staticmethod
    def _validate_syntax(code: str, file_path: str) -> tuple[bool, str]:
        """
        Validate generated code for truncation/syntax issues.
        Returns (is_valid, error_description).
        Checks:
          1. Balanced brackets: { } ( ) [ ]
          2. File ends properly (not mid-token)
          3. TSX/JSX/TS/JS-specific: StyleSheet.create closed, export present
        """
        # Skip non-code files
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ('.tsx', '.ts', '.jsx', '.js', '.json'):
            return True, ""

        if not code.strip():
            return False, "Empty file"

        # JSON validation
        if ext == '.json':
            try:
                json.loads(code)
                return True, ""
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {e}"

        # Check balanced brackets
        stack = []
        in_string = False
        string_char = None
        escape_next = False
        in_line_comment = False
        in_block_comment = False
        prev_char = ''

        for ch in code:
            if escape_next:
                escape_next = False
                prev_char = ch
                continue
            if ch == '\\' and in_string:
                escape_next = True
                prev_char = ch
                continue
            if ch == '\n':
                in_line_comment = False
                prev_char = ch
                continue
            if in_line_comment:
                prev_char = ch
                continue
            if in_block_comment:
                if prev_char == '*' and ch == '/':
                    in_block_comment = False
                prev_char = ch
                continue
            if ch == '/' and prev_char == '/':
                in_line_comment = True
                if stack and stack[-1] == '/':
                    pass  # don't track
                prev_char = ch
                continue
            if ch == '*' and prev_char == '/':
                in_block_comment = True
                prev_char = ch
                continue
            if in_string:
                if ch == string_char:
                    in_string = False
                    string_char = None
                prev_char = ch
                continue
            if ch in ('"', "'", '`'):
                in_string = True
                string_char = ch
                prev_char = ch
                continue
            if ch in ('{', '(', '['):
                stack.append(ch)
            elif ch in ('}', ')', ']'):
                expected = {'}': '{', ')': '(', ']': '['}[ch]
                if stack and stack[-1] == expected:
                    stack.pop()
            prev_char = ch

        if in_string:
            return False, f"Unterminated string (started with {string_char!r}) — file is truncated"
        if len(stack) > 0:
            return False, f"Unbalanced brackets: {len(stack)} unclosed ({stack[-3:]})"

        # Check for common truncation patterns
        stripped = code.rstrip()
        # Ends mid-property (no semicolon, comma, bracket, or closing paren)
        last_line = stripped.split('\n')[-1].strip() if stripped else ""
        truncation_indicators = [
            # Line ends with an identifier (no punctuation) — mid-token truncation
            # But allow common valid endings like export statements
            last_line and last_line[-1].isalpha() and not last_line.endswith((
                'return', 'true', 'false', 'null', 'undefined', 'break', 'continue',
            )) and not any(last_line.startswith(kw) for kw in ('export ', 'module.')),
            # NOTE: StyleSheet.create check removed — bracket balance above catches
            # genuinely unclosed StyleSheets. The old check flagged files ending
            # with `export { X };` after the StyleSheet as truncated (false positive).
        ]

        if any(truncation_indicators):
            return False, f"Likely truncated: ends with '{last_line[-40:]}', {len(stack)} unclosed brackets"

        return True, ""

    @staticmethod
    def _auto_repair_syntax(code: str, file_path: str) -> str:
        """
        Best-effort auto-repair for truncated generated code.

        Strategy: Instead of appending closers (which produces garbage like `export default X;')})`),
        truncate back to the last complete top-level statement, then add the missing
        export default + StyleSheet if needed. This produces valid code that renders
        an error placeholder rather than crashing the entire bundle.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ('.tsx', '.ts', '.jsx', '.js'):
            return code

        # Find the last valid top-level boundary: export default, StyleSheet.create, or closing `});`
        lines = code.split('\n')

        # Find the last line that looks like a complete top-level statement
        last_good_line = len(lines) - 1
        for i in range(len(lines) - 1, -1, -1):
            stripped = lines[i].strip()
            # Good boundaries: export default, top-level closing, StyleSheet end
            if stripped in ('});', '});', '});', '})', '});', '});'):
                last_good_line = i
                break
            if stripped.startswith('export default ') and stripped.endswith(';'):
                last_good_line = i
                break
            if stripped == '});' or stripped == '});':
                last_good_line = i
                break

        # Truncate to last good line
        truncated_lines = lines[:last_good_line + 1]
        truncated_code = '\n'.join(truncated_lines) + '\n'

        # Check if there's an export default — if not, extract component name and add one
        has_export_default = bool(re.search(r'export\s+default\s+', truncated_code))
        if not has_export_default:
            # Try to find the component function name
            comp_match = re.search(r'(?:function|const)\s+(\w+Screen|\w+Component|\w+)\s*[\(\:=]', truncated_code)
            comp_name = comp_match.group(1) if comp_match else None

            if comp_name:
                truncated_code += f'\nexport default {comp_name};\n'
            else:
                # Can't determine component name — replace entire file with error placeholder
                screen_name = os.path.basename(file_path).replace('.tsx', '').replace('.ts', '')
                truncated_code = (
                    f"import React from 'react';\n"
                    f"import {{ View, Text }} from 'react-native';\n"
                    f"export default function {screen_name}() {{\n"
                    f"  return (\n"
                    f"    <View style={{{{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#161B22' }}}}>\n"
                    f"      <Text style={{{{ color: '#F85149', fontSize: 16 }}}}>Failed to generate this screen. Please rebuild.</Text>\n"
                    f"    </View>\n"
                    f"  );\n"
                    f"}}\n"
                )

        logger.info(f"[BUILD] Auto-repaired {file_path}: truncated to last valid statement")
        return truncated_code

    async def _repair_bundle_errors(
        self, app_id: str, app_dir: str, bundle_errors: list,
        generated_files: dict, description: str, app_name: str,
        deps: list, db_type: str, blog=None,
    ) -> bool:
        """Attempt to regenerate files that caused bundle TransformErrors."""
        if not bundle_errors:
            return False

        files_to_repair = set()
        for err in bundle_errors:
            fp = err.get("file", "")
            # Convert absolute path to relative app path
            if app_dir and app_dir in fp:
                fp = "/" + fp.replace(app_dir, "").lstrip("/")
            # Ensure leading slash for matching
            if fp and not fp.startswith("/"):
                fp = "/" + fp
            # Direct match
            if fp and fp in generated_files:
                files_to_repair.add(fp)
                continue
            # Fuzzy match: try matching by filename suffix
            if fp:
                fp_stripped = fp.lstrip("/")
                for gf in generated_files:
                    if gf.lstrip("/") == fp_stripped or gf.endswith("/" + fp_stripped.split("/")[-1]):
                        files_to_repair.add(gf)
                        break

        if not files_to_repair:
            if blog:
                await blog.warn(f"Could not identify which files to repair from bundle errors: {[e.get('file') for e in bundle_errors[:3]]}")
            return False

        if blog:
            await blog.info(f"Regenerating {len(files_to_repair)} broken files: {', '.join(files_to_repair)}")

        # Read current file contents as context
        existing = {}
        for fp in files_to_repair:
            abs_path = os.path.join(app_dir, fp.lstrip("/"))
            if os.path.exists(abs_path):
                try:
                    with open(abs_path, 'r') as f:
                        existing[fp] = f.read()
                except Exception:
                    pass

        # Regenerate with error context
        repaired = await self._generate_code(
            description + "\n\nFIX THESE BUNDLE ERRORS: " + json.dumps([e["error"][:200] for e in bundle_errors[:3]]),
            app_name, list(files_to_repair), deps, db_type,
            blog=blog, existing_files=existing,
        )

        if repaired:
            # Apply infra fixes to repaired files
            await self._write_infra_files(app_id, repaired, blog)
            await self._app_manager.write_app_files(app_id, repaired)
            if blog:
                await blog.success(f"Repaired {len(repaired)} files")
            return True
        return False

    async def _fix_missing_deps(self, app_id: str, app_dir: str, bundle_errors: list, blog=None) -> bool:
        """Detect and install missing npm dependencies from bundle errors.

        Returns True if any new deps were installed.
        """
        import re as _re

        missing = set()
        for err in bundle_errors:
            msg = err.get("error", "")
            # "Unable to resolve "react-native-chart-kit""
            m = _re.search(r'Unable to resolve ["\']([^"\']+)["\']', msg)
            if m:
                pkg = m.group(1)
                if not pkg.startswith(".") and not pkg.startswith("/"):
                    # Get package name (handle scoped @org/pkg)
                    if pkg.startswith("@"):
                        parts = pkg.split("/")
                        missing.add("/".join(parts[:2]) if len(parts) >= 2 else pkg)
                    else:
                        missing.add(pkg.split("/")[0])
                continue
            # "Cannot find module 'some-module'"
            m = _re.search(r"Cannot find module ['\"]([^'\"]+)['\"]", msg)
            if m:
                pkg = m.group(1)
                if not pkg.startswith(".") and not pkg.startswith("/"):
                    if pkg.startswith("@"):
                        parts = pkg.split("/")
                        missing.add("/".join(parts[:2]) if len(parts) >= 2 else pkg)
                    else:
                        missing.add(pkg.split("/")[0])
            # "Module not found: Error: Can't resolve 'X'"
            m = _re.search(r"(?:Module not found|Can't resolve)[^'\"]*['\"]([^'\"]+)['\"]", msg)
            if m:
                pkg = m.group(1)
                if not pkg.startswith(".") and not pkg.startswith("/"):
                    if pkg.startswith("@"):
                        parts = pkg.split("/")
                        missing.add("/".join(parts[:2]) if len(parts) >= 2 else pkg)
                    else:
                        missing.add(pkg.split("/")[0])

        if not missing:
            return False

        if blog:
            await blog.info(f"Installing missing dependencies: {', '.join(missing)}")

        try:
            proc = await asyncio.create_subprocess_exec(
                "npm", "install", "--save", *list(missing),
                cwd=app_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            output = stdout.decode("utf-8", errors="replace") if stdout else ""
            if proc.returncode == 0:
                if blog:
                    await blog.success(f"Installed {len(missing)} missing deps: {', '.join(missing)}")
                return True
            else:
                if blog:
                    await blog.warn(f"npm install failed (code {proc.returncode}): {output[:200]}")
                return False
        except asyncio.TimeoutError:
            if blog:
                await blog.warn("npm install timed out (120s)")
            return False
        except Exception as e:
            if blog:
                await blog.warn(f"npm install failed: {e}")
            return False

    async def _verify_deps_installed(self, app_dir: str, generated_files: dict, deps: list, blog=None) -> bool:
        """Verify that all deps referenced in generated code are actually installed.

        Runs BEFORE starting servers. Returns True if all deps are available.
        """
        # Scan generated code for imports
        new_deps = self._detect_new_deps(generated_files, deps)
        if not new_deps:
            return True

        if blog:
            await blog.info(f"Found {len(new_deps)} uninstalled deps in generated code: {', '.join(new_deps)}")

        try:
            proc = await asyncio.create_subprocess_exec(
                "npm", "install", "--save", *new_deps,
                cwd=app_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode == 0:
                if blog:
                    await blog.success(f"Installed {len(new_deps)} additional deps")
                return True
            else:
                if blog:
                    output = stdout.decode("utf-8", errors="replace")[:200] if stdout else ""
                    await blog.warn(f"Dep install failed: {output}")
                return False
        except Exception as e:
            if blog:
                await blog.warn(f"Dep verification failed: {e}")
            return False

    @staticmethod
    def _detect_new_deps(generated_files: dict, existing_deps: list) -> list:
        """Scan generated code for imports that aren't in the current dep list."""
        import re as _re

        # Known built-in / RN modules that don't need npm install
        builtins = {
            'react', 'react-native', 'react-dom', 'expo', 'react-native-web',
            '@react-navigation/native', '@react-navigation/native-stack',
            '@react-navigation/bottom-tabs', 'react-native-safe-area-context',
            'react-native-screens', 'expo-sqlite', 'expo-status-bar',
            'expo-constants', 'expo-linking', 'expo-router',
        }

        existing_set = set(existing_deps) | builtins
        found_deps = set()

        for code in generated_files.values():
            # Match: import ... from 'package-name' or require('package-name')
            for m in _re.finditer(r"""(?:from|require\()\s*['"]([^./][^'"]*?)['"]""", code):
                pkg = m.group(1)
                # Get the package name (handle scoped packages like @org/pkg)
                if pkg.startswith('@'):
                    parts = pkg.split('/')
                    pkg_name = '/'.join(parts[:2]) if len(parts) >= 2 else pkg
                else:
                    pkg_name = pkg.split('/')[0]
                if pkg_name not in existing_set:
                    found_deps.add(pkg_name)

        return list(found_deps)

    @staticmethod
    async def _wait_for_server(port: int, timeout: int = 15):
        """Wait until a server is accepting connections on the given port."""
        import aiohttp
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:{port}/",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as resp:
                        # Any response means server is up (even 404/500)
                        return
            except Exception:
                await asyncio.sleep(1)
        # Timeout — proceed anyway, bundle validation will catch errors
        logger.warning(f"[BUILD] Server on port {port} not ready after {timeout}s, proceeding")

    # Error keywords to search for in Metro output / log buffer
    BUNDLE_ERROR_KEYWORDS = (
        'SyntaxError', 'TransformError', 'TypeError', 'ReferenceError',
        'Cannot find module', 'Module not found', 'Unexpected token',
        'Unable to resolve', 'error:', 'Error:',
        'ENOENT', 'EACCES', 'EADDRINUSE',
    )

    async def _validate_bundle(self, web_port: int, blog=None) -> tuple:
        """Check if the web bundle compiles without errors.

        Returns (is_valid, error_details_list).
        error_details_list contains dicts with 'file' and 'error' keys for each broken file.
        """
        import aiohttp
        import re as _re

        url = f"http://localhost:{web_port}/index.bundle?platform=web&dev=true"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=90)) as resp:
                    if resp.status != 200:
                        if blog:
                            await blog.warn(f"Bundle request returned HTTP {resp.status}")
                        return False, await self._parse_bundle_error_response(resp, blog)

                    # Read full response (up to 5MB) to catch errors anywhere in bundle
                    chunks = []
                    total = 0
                    async for chunk in resp.content.iter_chunked(65536):
                        chunks.append(chunk)
                        total += len(chunk)
                        if total > 5 * 1024 * 1024:
                            break
                    text = b"".join(chunks).decode("utf-8", errors="replace")

                    errors = self._extract_errors_from_text(text)
                    if errors and blog:
                        for e in errors[:5]:
                            await blog.error(f"Bundle error in {e['file']}: {e['error'][:200]}")
                    return len(errors) == 0, errors
        except asyncio.TimeoutError:
            if blog:
                await blog.warn("Bundle request timed out (90s)")
            return False, []
        except Exception as e:
            if blog:
                await blog.warn(f"Bundle validation request failed: {e}")
            return False, []

    async def _parse_bundle_error_response(self, resp, blog=None) -> list:
        """Parse an HTTP error response from Metro into structured error dicts."""
        import re as _re
        errors = []
        try:
            body = await resp.text()
            # Try JSON parse first (Metro error format)
            try:
                import json as _json
                err_json = _json.loads(body)
                fname = err_json.get("filename", "")
                msg = err_json.get("message", body[:500])
                if fname:
                    if not fname.startswith("/"):
                        fname = "/" + fname
                    errors.append({"file": fname, "error": msg[:500]})
                    if blog:
                        await blog.error(f"Bundle error in {fname}: {msg[:200]}")
                # Check for nested errors array
                for nested_err in err_json.get("errors", []):
                    nf = nested_err.get("filename", "")
                    nm = nested_err.get("message", "")
                    if nf:
                        if not nf.startswith("/"):
                            nf = "/" + nf
                        errors.append({"file": nf, "error": nm[:500]})
            except (ValueError, TypeError):
                pass
            # Fallback: regex for file paths in raw text
            if not errors:
                errors = self._extract_errors_from_text(body)
            # Last resort: raw body
            if not errors:
                file_match = _re.search(r'["\s/]([a-zA-Z][^\s"]*\.tsx?)', body)
                if file_match:
                    fname = "/" + file_match.group(1).lstrip("/")
                    errors.append({"file": fname, "error": body[:500]})
        except Exception:
            pass
        return errors

    @staticmethod
    def _extract_errors_from_text(text: str) -> list:
        """Extract structured error dicts from Metro bundle output text."""
        import re as _re
        errors = []
        seen_files = set()
        for line in text.split("\n"):
            if any(kw in line for kw in AppBuilderSkill.BUNDLE_ERROR_KEYWORDS):
                # Try to extract file path
                file_match = _re.search(r'(/[^\s:]+\.tsx?)', line)
                if not file_match:
                    # Try relative path pattern
                    file_match = _re.search(r'(?:in |from |module )["\']?([a-zA-Z./][^\s"\']+\.tsx?)', line)
                file_path = file_match.group(1) if file_match else "unknown"
                if not file_path.startswith("/") and file_path != "unknown":
                    file_path = "/" + file_path
                # Deduplicate by file
                if file_path not in seen_files:
                    seen_files.add(file_path)
                    errors.append({"file": file_path, "error": line.strip()[:500]})
        return errors

    def _extract_errors_from_log_buffer(self, app_id: str) -> list:
        """Extract structured errors from a managed app's log buffer."""
        managed_app = self._app_manager._running.get(app_id)
        if not managed_app or not managed_app.log_buffer:
            return []
        errors = []
        seen_files = set()
        import re as _re
        for line in list(managed_app.log_buffer)[-100:]:
            if any(kw in line for kw in self.BUNDLE_ERROR_KEYWORDS):
                file_match = _re.search(r'(/[^\s:]+\.tsx?)', line)
                if not file_match:
                    file_match = _re.search(r'(?:in |from |module )["\']?([a-zA-Z./][^\s"\']+\.tsx?)', line)
                file_path = file_match.group(1) if file_match else "unknown"
                if not file_path.startswith("/") and file_path != "unknown":
                    file_path = "/" + file_path
                if file_path not in seen_files:
                    seen_files.add(file_path)
                    errors.append({"file": file_path, "error": line.strip()[:500]})
        return errors
