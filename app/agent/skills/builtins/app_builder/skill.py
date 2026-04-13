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
- RESPONSIVE NAVIGATION — choose based on screen count:
  - If the app has ONLY 1 screen (e.g. a calculator, timer, converter, single-purpose tool):
    Do NOT include tab navigation or AdaptiveTabBar. Use a simple NativeStackNavigator with
    one screen, or render the screen component directly in App.tsx. A calculator does not need
    a sidebar or tab bar — it needs a keypad and a display, full stop.
  - If the app has 2+ screens: Use createBottomTabNavigator with a custom tabBar prop that
    renders AdaptiveTabBar. Include /components/AdaptiveTabBar.tsx in the file list.
    On mobile: bottom tab bar. On desktop: left sidebar.
- If the app needs charts, use react-native-chart-kit (NOT recharts — that's web-only)
- Keep the file list focused — don't over-engineer
- PREFER THE SIMPLEST CORRECT IMPLEMENTATION. Do not build complex engines when simple logic suffices.
  Examples:
  - "Simple calculator" → use an accumulator pattern (current value, pending operator, apply on =).
    Do NOT build a full expression parser, BODMAS engine, tokenizer, or shunting-yard algorithm.
  - "Timer" → use setInterval + state. Do NOT build a scheduler framework.
  - "Todo list" → flat array in state. Do NOT build a relational data model with categories/tags/priorities.
  Complexity should come from the user's requirements, not from the builder's assumptions.
  When in doubt, build the minimal version that works correctly.
- NEVER generate advertisements, ad banners, ad placeholders, "Ad banner (offline placeholder)",
  or any monetization-related components. These are personal user apps, not published products.
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

CRITICAL — Agent Integration: REGISTER vs RENDER distinction.
Every app is an "agentic app." The user's AI agent works inside it. There are two sides:

  REGISTRATION (REQUIRED — your app is broken without these):
  - /lib/agentBridge.ts — Bridge that exposes screens, actions, and navigation to the agent.
    Wire AgentBridge.setNavigationRef(ref) and AgentBridge.currentScreen = screenName in App.tsx.
  - /lib/agentActions.ts — Screen-specific action registry. Every screen calls
    registerAction() in a useEffect to tell the agent what it can do on that screen.
  - /lib/agentSkill.json — Domain action manifest (SQL queries, navigation, parameters).
  These files are REQUIRED. Without them the agent cannot interact with your app.

  RENDERING (FORBIDDEN — the platform handles all agent UI):
  - Do NOT render <AgentPlaceholder />, "Agent ready" cards, chat panels, floating orbs,
    "Ask Agent" buttons, "Agent Test Panel", or ANY visible agent UI component anywhere
    in your JSX. The platform renders the agent's Orb and chat panel OUTSIDE your app
    in a right-side split-view panel. Your app must not duplicate it.
  - Do NOT include /components/AgentPlaceholder.tsx in your file plan.
  - Do NOT add any JSX block with "Agent ready", "agent support", "floating agent orb",
    or similar text. If you include such a block, it will be stripped automatically.

You MUST include these files in EVERY plan:
  - /components/ErrorBoundary.tsx — Catches runtime errors, shows error screen instead of white page
  - /lib/agentBridge.ts — Bridge module (REGISTRATION, not UI)
  - /lib/agentActions.ts — Action registry (REGISTRATION, not UI)
Do NOT include /components/AgentPlaceholder.tsx — the platform renders all agent UI.
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
- MOBILE-FIRST RESPONSIVE LAYOUT — these apps run on MOBILE phones (375px) AND desktop web (1200px+):
  - Use useWindowDimensions() at the top of every screen
  - Define: const isDesktop = width > 768;
  - MOBILE IS THE PRIMARY TARGET (width <= 768):
    - Single column layout, full width
    - Padding: 16px sides, paddingBottom: 100 (for tab bar + agent orb)
    - All content MUST fit within 375px width — NEVER overflow horizontally
    - Cards: width '100%', no fixed pixel widths wider than 340px
    - Text: use flexShrink: 1 and numberOfLines to prevent overflow
    - Touch targets: minimum 44px height for tappable elements
    - Font sizes: titles 20-24px, body 14-16px, captions 11-12px
    - ScrollView for ALL screens (content may exceed screen height)
    - Wrap every screen in SafeAreaView or add paddingTop for status bar
  - On DESKTOP (width > 768): constrain main content to maxWidth: 800, alignSelf: 'center'.
    Use 2-3 column grid layouts for cards/lists (flexDirection: 'row', flexWrap: 'wrap').
    Add more generous padding (24-32px).
  - Example responsive pattern for EVERY screen's root container:
    const {{ width }} = useWindowDimensions();
    const isDesktop = width > 768;
    <ScrollView style={{{{ flex: 1, paddingLeft: isDesktop ? 220 : 0 }}}}>
      <View style={{{{ maxWidth: isDesktop ? 800 : undefined, alignSelf: 'center', width: '100%', padding: isDesktop ? 32 : 16, paddingBottom: isDesktop ? 32 : 100 }}}}>
    The paddingLeft: 220 on desktop accounts for the sidebar navigation width.
    The paddingBottom: 100 on mobile accounts for the bottom tab bar (60px) + agent button above it.
  - Cards and grid items: use percentage widths on desktop (width: isDesktop ? '48%' : '100%')
  - CRITICAL: Test all layouts mentally at 375px width. Nothing should overflow or be cut off.
- RESPONSIVE NAVIGATION — choose based on screen count:
  - If the app has ONLY 1 screen (e.g. calculator, timer, converter), do NOT use tab navigation.
    Render the screen directly or use a simple NativeStackNavigator. No AdaptiveTabBar needed.
  - If the app has 2+ screens: Use createBottomTabNavigator with `tabBar={{(props) => <AdaptiveTabBar {{...props}} appName={{"{app_name}"}} />}}`
    The AdaptiveTabBar component handles BOTH layouts (see below for /components/AdaptiveTabBar.tsx)
- NEVER generate advertisements, ad banners, ad placeholders, or monetization components.
- NEVER generate "Ask Agent" buttons, agent chat panels, or agent test UIs.
  The platform injects the agent Orb — apps must NOT duplicate it.
- PREFER THE SIMPLEST CORRECT IMPLEMENTATION. For a "simple calculator", use an accumulator pattern
  (current value + pending operator + apply on =), NOT a full expression parser/tokenizer/BODMAS engine.
  Build only what the user asked for — no extra complexity.
- CRITICAL — Cross-file API contracts (the #1 cause of silent white-page crashes):
  - If a lib module's default export is an OBJECT with methods (e.g., `export default {{ formatNumber, calcPress }}`),
    the consuming file MUST call it as `Module.methodName()`, NEVER as `Module()` — objects are not callable.
  - If a lib function returns a structured result (e.g., `{{ok: true, value: number}}`), the consumer MUST
    destructure it properly. Do NOT check `typeof result !== 'number'` when the function returns an object.
  - BEFORE writing any import, mentally verify: does the target module export this exact name? Is the export
    a function, object, or class? Does the consuming code use it in a way that matches its type?
  - When in doubt, use the NAMED export with the EXACT function name, not the default export object.
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

7. Import/Export consistency — THIS IS THE #1 CAUSE OF RUNTIME CRASHES:
   - Every named import MUST match an actual named export in the target module.
     If you import `{{ calcPress }}` from '../lib/calcEngine', that module MUST export `calcPress` by EXACTLY
     that name. A function named `pressKey` is NOT the same as `calcPress`.
   - If a module uses `export default X`, import it as `import X from '...'`.
   - For lib files, always provide BOTH `export default` AND named `export {{ }}` so either import style works.
   - CRITICAL: Use IDENTICAL function/type names across files. If the lib exports `pressKey`, import `pressKey`.
     Do NOT invent different names in the importing file — the bundler cannot resolve them.

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
  export default function AgentPlaceholder() {{ return null; }}
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
  Do NOT render any "Agent ready" card, agent placeholder, or agent UI in the screen's JSX.
  Registration is code (useEffect + imports). Rendering is JSX. Only registration belongs here.

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
            {
                "name": "app_builder__research_category",
                "description": (
                    "Research an app category via web search after the user picks a direction from "
                    "the 3 options in Layer 0. Returns research-informed questions with [[option]] buttons. "
                    "Call this AFTER the user selects a direction, BEFORE asking technical questions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "original_idea": {
                            "type": "string",
                            "description": "The user's original app idea (e.g. 'I want a fitness app')",
                        },
                        "chosen_direction": {
                            "type": "string",
                            "description": "The direction the user chose from the 3 options (e.g. 'A workout tracker with exercise logging and progress charts')",
                        },
                    },
                    "required": ["original_idea", "chosen_direction"],
                },
            },
            {
                "name": "app_builder__gather_requirements",
                "description": (
                    "Generate technical/implementation questions after the user answers research questions. "
                    "Returns 5-7 technical questions with [[option]] buttons covering UI, data, screens, etc. "
                    "Call this AFTER the user answers the research questions from app_builder__research_category."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "original_idea": {
                            "type": "string",
                            "description": "The user's original app idea",
                        },
                        "chosen_direction": {
                            "type": "string",
                            "description": "The direction the user chose from Layer 0",
                        },
                        "research_context": {
                            "type": "string",
                            "description": "The research questions that were asked (from app_builder__research_category)",
                        },
                        "research_answers": {
                            "type": "string",
                            "description": "The user's answers to the research questions",
                        },
                    },
                    "required": ["original_idea", "chosen_direction", "research_context", "research_answers"],
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
            "app_builder__research_category": self._exec_research_category,
            "app_builder__gather_requirements": self._exec_gather_requirements,
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
            logger.error("[BUILD] App Builder tool called but _app_manager is None — check startup logs for '[INIT]' errors")
            return (
                "App Builder isn't available right now — the app manager failed to initialize on startup. "
                "Check the agent logs for errors, or restart the agent from the Dashboard."
            )

        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob

        app_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        base_slug = _slugify(name)
        from app.agent.app_manager import APPS_DIR as _apps_dir
        # app_dir will be set after slug deduplication below

        # Store the conversational context as plan JSON
        plan_context = {
            "screens": screens,
            "features": features,
            "db_type": db_type,
            "platforms": platforms,
            "design_notes": design_notes,
        }

        async with async_session_maker() as db:
            # Deduplicate slug — append numeric suffix if collision.
            # Uses DB unique constraint as source of truth (handles concurrent races).
            from sqlalchemy import select as sa_select
            from sqlalchemy.exc import IntegrityError as _IntegrityError
            slug = base_slug
            for attempt in range(1, 100):
                existing = await db.execute(
                    sa_select(App.id).where(App.slug == slug).limit(1)
                )
                if not existing.scalar():
                    break
                slug = f"{base_slug}-{attempt}"

            app_dir = os.path.join(_apps_dir, slug)

            # Retry loop: if concurrent build races for the same slug,
            # the DB unique constraint catches it and we try the next suffix.
            _slug_committed = False
            for _slug_retry in range(5):
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
                    job_type="auto_builder",
                    title=f"Build: {name}",
                    prompt=description,
                    status="queued",
                    steps_json=json.dumps(self._initial_steps()),
                )
                db.add(job)
                try:
                    await db.commit()
                    _slug_committed = True
                    break
                except _IntegrityError:
                    await db.rollback()
                    slug = f"{base_slug}-{_slug_retry + 2}"
                    app_dir = os.path.join(_apps_dir, slug)
                    app_id = str(uuid.uuid4())
                    job_id = str(uuid.uuid4())
                    logger.warning("[BUILD] Slug collision on '%s', retrying with '%s'", base_slug, slug)

            if not _slug_committed:
                return f"Failed to create app — slug collision after 5 retries for '{base_slug}'"

        # Broadcast initial job_update immediately (before background task starts)
        if self._ws_broadcast:
            print(f"[BUILD] Sending initial job_update: job={job_id[:8]} name={name} user={user_id[:8]}", flush=True)
            await self._ws_broadcast(user_id, {
                "type": "job_update",
                "job_id": job_id,
                "name": name,
                "status": "queued",
                "step": "Starting build...",
                "total_steps": 8,
                "completed_steps": 0,
            })

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
            logger.error("[BUILD] App Builder tool called but _app_manager is None — check startup logs for '[INIT]' errors")
            return (
                "App Builder isn't available right now — the app manager failed to initialize on startup. "
                "Check the agent logs for errors, or restart the agent from the Dashboard."
            )

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
            logger.error("[BUILD] App Builder tool called but _app_manager is None — check startup logs for '[INIT]' errors")
            return (
                "App Builder isn't available right now — the app manager failed to initialize on startup. "
                "Check the agent logs for errors, or restart the agent from the Dashboard."
            )

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

    # ── Sub-Agent Handlers ────────────────────────────────────────

    async def _exec_research_category(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        """Run the Research Sub-Agent: web search + LLM to generate research-informed questions."""
        import time as _t
        from .research_agent import run_research

        original_idea = args.get("original_idea", "")
        chosen_direction = args.get("chosen_direction", "")
        print(f"[SKILL] _exec_research_category called: idea={original_idea[:50]}, direction={chosen_direction[:50]}", flush=True)

        if not original_idea or not chosen_direction:
            return "ERROR: original_idea and chosen_direction are required"

        _t0 = _t.time()
        try:
            questions = await run_research(
                original_idea=original_idea,
                chosen_direction=chosen_direction,
                call_llm=self._call_llm,
            )
            print(f"[SKILL] Research completed in {_t.time()-_t0:.1f}s — returning {len(questions)} chars", flush=True)
            return (
                "RESEARCH COMPLETE. Present these research-informed questions to the user exactly as formatted below. "
                "Do NOT add extra questions or rephrase — just present them in a friendly way with a brief intro like "
                "'I researched apps in this space — here are some questions based on what I found:'\n\n"
                f"{questions}"
            )
        except Exception as exc:
            print(f"[SKILL] Research FAILED in {_t.time()-_t0:.1f}s: {type(exc).__name__}: {exc}", flush=True)
            logger.exception("Research agent failed")
            return (
                f"Research agent encountered an error: {exc}\n"
                "Fall back to asking 5 general research-style questions with [[option]] buttons about "
                "the app category — cover: key features, target audience, content approach, "
                "engagement model, and differentiators."
            )

    async def _exec_gather_requirements(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        """Run the Building Sub-Agent: generate technical/implementation questions."""
        from .building_agent import run_building_questions

        original_idea = args.get("original_idea", "")
        chosen_direction = args.get("chosen_direction", "")
        research_context = args.get("research_context", "")
        research_answers = args.get("research_answers", "")

        if not original_idea or not chosen_direction:
            return "ERROR: original_idea and chosen_direction are required"

        try:
            questions = await run_building_questions(
                original_idea=original_idea,
                chosen_direction=chosen_direction,
                research_context=research_context,
                research_answers=research_answers,
                call_llm=self._call_llm,
            )
            return (
                "REQUIREMENTS GATHERED. Present these technical questions to the user exactly as formatted below. "
                "Do NOT add extra questions or rephrase — just present them with a brief intro like "
                "'Now let me ask about the technical details:'\n\n"
                f"{questions}"
            )
        except Exception as exc:
            logger.exception("Building agent failed")
            return (
                f"Building agent encountered an error: {exc}\n"
                "Fall back to asking 5 technical questions with [[option]] buttons about: "
                "UI style/theme, key screens, data storage, navigation style, and platform priority."
            )

    # ── System Prompt ──────────────────────────────────────────────

    def get_system_prompt_section(self) -> Optional[str]:
        return (
            "# App Builder\n"
            "## CRITICAL RULE — READ THIS FIRST\n"
            "You MUST use the `app_builder__build_app` tool for ALL app creation. NO EXCEPTIONS.\n"
            "This applies to ANY type of app — mobile, web, full-stack, Next.js, React, dashboards, "
            "tools, SaaS apps, data visualization, ANYTHING the user wants built as an application.\n"
            "Do NOT use exec, write_file, or edit_file to create apps, projects, or code files.\n"
            "Do NOT run npx, expo init, npm init, create-next-app, or any scaffolding command.\n"
            "Do NOT manually create package.json, App.tsx, screens, components, or config files.\n"
            "Do NOT use `toup__create_spec` or `toup__scaffold` for app creation — those are for "
            "planning/reviewing existing software projects, NOT building new apps.\n"
            "The pipeline guard WILL BLOCK these attempts and return an error.\n\n"
            "ANY request that involves building, creating, or making an app — regardless of how "
            "the user phrases it (\"make me\", \"create\", \"build\", \"I need\", \"can you make\", "
            "\"I want\", etc.) and regardless of the tech stack they mention (Next.js, React, "
            "PostgreSQL, D3, etc.) — MUST go through the App Builder pipeline below.\n\n"
            "The App Builder creates apps using React Native/Expo (runs on iPhone, iPad, and Web).\n"
            "Every app you build includes an **Agent Placeholder** — a floating widget where YOU (the agent) "
            "can dock into the app and help the user. You'll have full access to navigate, read data, and "
            "perform actions within every app you build.\n\n"

            "## When the user wants to build an app, follow this EXACT multi-layer process:\n\n"

            "### Layer 0: Help the User Understand What They Want (3 Direction Cards)\n"
            "Most users don't know exactly what they want. When the user gives a vague build prompt "
            "(e.g. 'I want a fitness app', 'build me a todo app', 'make something for cooking'), "
            "do NOT jump to questions. Instead:\n\n"
            "1. Interpret their vague idea\n"
            "2. Present **exactly 3 distinct directions** the app could take\n"
            "3. Each direction should be meaningfully different — not just variations of the same thing\n\n"
            "Format the 3 directions like this:\n"
            "```\n"
            "I can take this in a few different directions! Pick the one that resonates:\n\n"
            "**1. [Short Title]** — [1-2 sentence description of what this app would be]\n"
            "[[Short Title]]\n\n"
            "**2. [Short Title]** — [1-2 sentence description]\n"
            "[[Short Title]]\n\n"
            "**3. [Short Title]** — [1-2 sentence description]\n"
            "[[Short Title]]\n"
            "```\n\n"
            "RULES for Layer 0:\n"
            "- ALWAYS present exactly 3 options, no more, no less\n"
            "- CRITICAL: Each option MUST have a clickable [[button]] on the line RIGHT AFTER the description. "
            "The button text MUST be the option's short title (e.g. [[Workout Tracker]], [[Personal Trainer]]). "
            "NEVER use generic text like [[Pick this one]] — each button MUST be UNIQUE so the system knows which was clicked. "
            "NEVER omit the [[...]] lines — they are NOT optional.\n"
            "- Make the 3 directions genuinely different (e.g. for 'fitness app': workout tracker vs personal trainer vs social fitness challenge)\n"
            "- Keep descriptions short — 1-2 sentences max\n"
            "- Do NOT ask any questions yet — just present the 3 cards\n"
            "- If the user's description is already very specific and detailed (3+ sentences with clear features), "
            "you may skip Layer 0 and go directly to Layer 1A\n\n"

            "### Layer 1A: Research Agent (5+ Research-Informed Questions)\n"
            "**IMMEDIATELY after the user picks a direction from Layer 0, you MUST call "
            "`app_builder__research_category`.** Do NOT ask any additional questions, do NOT ask "
            "about platform preference, do NOT ask about tech stack, do NOT ask anything else — "
            "just call the tool right away with:\n"
            "- `original_idea`: the user's original prompt\n"
            "- `chosen_direction`: the direction they picked (title + description)\n\n"
            "This tool researches the app category via web search and returns 5-7 research-informed questions "
            "with [[option]] buttons. Present these questions to the user EXACTLY as returned.\n"
            "Add a brief friendly intro like: 'I did some research on apps like this — here are some "
            "questions based on what I found:'\n\n"
            "Wait for the user to answer ALL research questions before proceeding.\n"
            "**When the user replies with answers, match each answer to whichever question's "
            "[[option]] buttons it most closely matches — match by CONTENT SIMILARITY, not by line order.** "
            "Matching is case-insensitive and punctuation-tolerant. "
            "The user may answer in any order, skip questions, combine answers in one sentence, "
            "or use slightly different wording than the button labels (e.g., 'free no ads' matches '[[Free, no ads]]'). "
            "If the user provides fewer answers than questions, identify which specific questions were skipped "
            "by elimination after matching — do NOT assume the last questions were skipped. "
            "Only re-ask genuinely unanswered questions. If you're unsure which question an answer matches, "
            "state your best interpretation and ask for confirmation.\n\n"

            "### Layer 1B: Building Agent (5+ Technical Questions)\n"
            "**IMMEDIATELY after the user answers ALL research questions, call "
            "`app_builder__gather_requirements`.** Do NOT ask additional questions — call the tool with:\n"
            "- `original_idea`: the user's original prompt\n"
            "- `chosen_direction`: the direction from Layer 0\n"
            "- `research_context`: the research questions that were asked\n"
            "- `research_answers`: the user's answers to the research questions\n\n"
            "This tool returns 5-7 technical/implementation questions with [[option]] buttons. "
            "Present these to the user EXACTLY as returned.\n"
            "Add a brief intro like: 'Great choices! Now some technical details:'\n\n"
            "Wait for the user to answer ALL technical questions before proceeding.\n\n"

            "### Step 2: Present Plan\n"
            "After ALL questions from both Layer 1A and 1B are answered, present a structured plan:\n"
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

            "**IMPORTANT**: NEVER call app_builder__build_app without first going through "
            "Layer 0 → Layer 1A → Layer 1B → Plan approval. Never skip layers.\n\n"

            "**CRITICAL — ALL questions MUST have [[option]] buttons. NO exceptions.**\n"
            "- NEVER ask open-ended questions without buttons.\n"
            "- NEVER use bullet point lists (- option1, - option2) instead of buttons.\n"
            "- EVERY question gets 2-5 clickable [[option]] buttons on the NEXT LINE.\n"
            "- The user interacts ONLY through buttons — they should complete the entire pre-build "
            "flow by clicking, not typing.\n\n"

            "**Format rules for ALL questions (Layer 0, 1A, 1B):**\n"
            "- Place [[option]] buttons DIRECTLY on the line after each question.\n"
            "- NEVER collect all buttons at the end.\n"
            "- Keep question text SHORT (one line). Put context in the buttons, not the question.\n\n"

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
        await blog.info(f"app_dir={app_dir} slug={slug}")
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
                await self._app_manager.scaffold_app(app_id, app_name, slug=slug)
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

                await self._app_manager.write_app_files(app_id, generated_files, app_dir=app_dir)
                await self._write_infra_files(app_id, generated_files, blog, app_dir=app_dir, job_id=job_id)

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
                        db_url = await self._app_manager.setup_database(app_id, db_type, app_dir=app_dir)
                        storage_dir = await self._app_manager.setup_storage(app_id, app_dir=app_dir)
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
                await self._app_manager.install_deps(app_id, all_deps, app_dir=app_dir)
                await blog.success("Dependencies installed")
                await self._update_step(job_id, user_id, "installing", "done")
            else:
                await blog.info("Dependencies already installed — skipping")

            # ── GitHub (skip if done) ────────────────────────────────
            if "github" not in completed_steps:
                blog.set_step("github")
                await self._update_step(job_id, user_id, "github", "running")
                try:
                    repo_info = await self._app_manager.create_github_repo(app_id, app_name, app_dir=app_dir)
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
            metro_port = await self._app_manager.start_metro(app_id, app_dir=app_dir)
            await blog.success(f"Metro running on port {metro_port}")
            web_port = await self._app_manager.start_web(app_id, app_dir=app_dir)
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
                        web_port = await self._app_manager.start_web(app_id, app_dir=app_dir)
                        await asyncio.sleep(5)
                    except Exception:
                        pass
                if not fixed:
                    break
                if web_alive:
                    await asyncio.sleep(4)
                else:
                    try:
                        web_port = await self._app_manager.start_web(app_id, app_dir=app_dir)
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

            # Broadcast job completion so Jobs panel updates without polling
            if self._ws_broadcast:
                await self._ws_broadcast(user_id, {
                    "type": "job_update",
                    "job_id": job_id,
                    "app_id": app_id,
                    "name": name,
                    "status": "completed",
                })

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
                    "job_id": job_id,
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
        from app.config import settings
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob
        from .build_logger import BuildLogger

        blog = BuildLogger(job_id, user_id, ws_broadcast=self._ws_broadcast)
        blog.set_model(settings.agent_model or "claude-opus-4-6")
        blog.set_step("init")
        await blog.info(f"Starting build for '{name}'", f"app={app_id}")
        await blog.info(f"app_dir={app_dir} slug={slug}")
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
                await self._app_manager.scaffold_app(app_id, app_name, slug=slug)
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
                await self._app_manager.write_app_files(app_id, generated_files, app_dir=app_dir)

                # Write essential infrastructure files (metro config, web-safe DB)
                await self._write_infra_files(app_id, generated_files, blog, app_dir=app_dir, job_id=job_id)

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
                    await self._app_manager.write_app_files(app_id, partial, app_dir=app_dir)
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
                import traceback as _tb
                logger.error(f"[BUILD] Code generation exception: type={type(e).__name__}, repr={repr(e)}")
                logger.error(f"[BUILD] Traceback:\n{''.join(_tb.format_exception(type(e), e, e.__traceback__))}")
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
                    db_url = await self._app_manager.setup_database(app_id, db_type, app_dir=app_dir)
                    storage_dir = await self._app_manager.setup_storage(app_id, app_dir=app_dir)
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
                install_output = await self._app_manager.install_deps(app_id, all_deps, app_dir=app_dir)
                elapsed = time.time() - t0
                await blog.command_run(f"npm install {' '.join(all_deps[:5])}{'...' if len(all_deps) > 5 else ''}",
                                       exit_code=0, duration_s=elapsed, output=install_output[:300] if install_output else "")
                await blog.success(f"Dependencies installed", f"{elapsed:.1f}s")
                await self._update_step(job_id, user_id, "installing", "done")
            except Exception as e:
                error_text = str(e)
                classification = self._app_manager._classify_install_error(error_text) if self._app_manager else "unknown"
                await blog.error(f"[VALIDATOR] Install failed — classification={classification} job={job_id[:8]}")
                await blog.info(f"[VALIDATOR] Error detail: {error_text[:300]}")

                _install_repaired = False

                if classification == "bad_dep":
                    # Auto-repair: LLM fixes package.json (up to 2 attempts)
                    _repair_error_context = error_text
                    MAX_INSTALL_REPAIR_ATTEMPTS = 2

                    for repair_attempt in range(1, MAX_INSTALL_REPAIR_ATTEMPTS + 1):
                        await blog.info(
                            f"[REPAIR] Detected bad dependency — asking LLM to fix package.json "
                            f"(attempt {repair_attempt}/{MAX_INSTALL_REPAIR_ATTEMPTS}) job={job_id[:8]}"
                        )
                        try:
                            pkg_path = os.path.join(app_dir, "package.json")
                            with open(pkg_path, "r") as f:
                                pkg_content = f.read()
                            fix_prompt = (
                                f"npm install failed with this error:\n{_repair_error_context[:800]}\n\n"
                                f"Current package.json:\n{pkg_content}\n\n"
                                "Fix the package.json to resolve the error. Remove bad packages, "
                                "fix version numbers, or replace with working alternatives. "
                                "Return ONLY the corrected JSON, nothing else."
                            )
                            fixed_json = await self._call_llm(
                                system_prompt="You fix npm package.json errors. Return only valid JSON.",
                                user_message=fix_prompt,
                                model="", purpose="repair_package_json",
                                blog=blog,
                            )
                            import re as _re
                            json_match = _re.search(r'\{[\s\S]*\}', fixed_json)
                            if not json_match:
                                raise RuntimeError("LLM did not return valid JSON")
                            json.loads(json_match.group())  # validate
                            with open(pkg_path, "w") as f:
                                f.write(json_match.group())
                            await blog.info(f"[REPAIR] package.json rewritten, retrying install...")
                            install_output = await self._app_manager.install_deps(app_id, all_deps, app_dir=app_dir)
                            await blog.success(f"[REPAIR] Dependencies installed after repair (attempt {repair_attempt})")
                            await self._update_step(job_id, user_id, "installing", "done")
                            _install_repaired = True
                            break
                        except Exception as repair_err:
                            _repair_error_context = str(repair_err)
                            new_cls = self._app_manager._classify_install_error(_repair_error_context) if self._app_manager else "unknown"
                            await blog.warn(
                                f"[REPAIR] Attempt {repair_attempt} failed (new classification={new_cls}): "
                                f"{_repair_error_context[:200]}"
                            )
                            if new_cls != "bad_dep":
                                await blog.error(f"[REPAIR] Error type changed to {new_cls}, stopping repair")
                                break

                elif classification in ("disk",):
                    await blog.error(f"[VALIDATOR] Non-recoverable install failure ({classification}) — failing job")
                    await blog.persist()
                    await self._fail_job(job_id, app_id, f"npm install failed ({classification}): {error_text[:300]}")
                    return

                elif classification in ("transient", "stale", "unknown"):
                    # Orchestrator-level retry with escalating timeouts and cache clear
                    INSTALL_RETRY_TIMEOUTS = [450, 600]
                    for retry_idx, retry_timeout in enumerate(INSTALL_RETRY_TIMEOUTS, 1):
                        await blog.info(
                            f"[RETRY] Install retry {retry_idx}/{len(INSTALL_RETRY_TIMEOUTS)} "
                            f"with {retry_timeout}s timeout (was {classification})..."
                        )
                        try:
                            # Clear npm cache before retry to avoid corrupted state
                            import shutil as _shutil
                            npm_cache = os.path.join("/tmp", "npm-cache-shared")
                            _shutil.rmtree(npm_cache, ignore_errors=True)
                            # Clear node_modules lock artifacts
                            lock_path = os.path.join(app_dir, "package-lock.json")
                            if os.path.exists(lock_path):
                                os.remove(lock_path)

                            # Patch timeout for this attempt
                            from app.agent import app_manager as _am
                            orig_timeout = _am.NPM_INSTALL_TIMEOUT
                            _am.NPM_INSTALL_TIMEOUT = retry_timeout
                            _am.NPM_CI_TIMEOUT = retry_timeout
                            try:
                                install_output = await self._app_manager.install_deps(
                                    app_id, all_deps, app_dir=app_dir
                                )
                            finally:
                                _am.NPM_INSTALL_TIMEOUT = orig_timeout
                                _am.NPM_CI_TIMEOUT = 180

                            await blog.success(
                                f"[RETRY] Dependencies installed on retry {retry_idx} ({retry_timeout}s budget)"
                            )
                            await self._update_step(job_id, user_id, "installing", "done")
                            _install_repaired = True
                            break
                        except Exception as retry_err:
                            retry_text = str(retry_err)
                            retry_cls = self._app_manager._classify_install_error(retry_text) if self._app_manager else "unknown"
                            await blog.warn(
                                f"[RETRY] Attempt {retry_idx} failed ({retry_cls}): {retry_text[:200]}"
                            )
                            if retry_cls == "disk":
                                await blog.error("[RETRY] Disk/memory error — stopping retries")
                                break

                if not _install_repaired:
                    await blog.error(f"[VALIDATOR] Install failed after all recovery attempts ({classification}) job={job_id[:8]}")
                    await blog.persist()
                    await self._fail_job(job_id, app_id, f"npm install failed ({classification}): {error_text[:300]}")
                    return

            _checkpoint["completed_steps"] = ["planning", "scaffolding", "writing", "database", "installing"]
            _checkpoint["current_step"] = "github"

            # ── Step 6: GitHub ──────────────────────────────────────
            blog.set_step("github")
            await self._update_step(job_id, user_id, "github", "running")
            try:
                await blog.info("Creating GitHub repository...")
                repo_info = await self._app_manager.create_github_repo(app_id, app_name, app_dir=app_dir)
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
            logger.info(f"[BUILD] step=starting app_id={app_id} app_dir={app_dir}")
            qr_url = ""
            web_url = ""
            try:
                # ── 7a: Pre-flight dep check — install any missing deps BEFORE starting servers
                await blog.info("Verifying dependencies before server start...")
                await self._verify_deps_installed(app_dir, generated_files, deps, blog)

                # ── 7b: Start servers
                await blog.info("Starting Metro bundler (mobile)...")
                metro_port = await self._app_manager.start_metro(app_id, app_dir=app_dir)
                await blog.success(f"Metro running on port {metro_port}")

                await blog.info("Starting Expo Web server...")
                web_port = await self._app_manager.start_web(app_id, app_dir=app_dir)
                await blog.success(f"Web server running on port {web_port}")

                # ── 7b+: Pre-flight dependency reconciliation ──
                # Scan all generated files for imports and ensure they're in package.json
                await blog.info("Reconciling dependencies...")
                try:
                    import re as _re
                    all_imports = set()
                    for fpath, code in generated_files.items():
                        if isinstance(code, str):
                            # Match: import ... from 'package' or require('package')
                            for m in _re.finditer(r'''(?:from\s+['"]|require\s*\(\s*['"])(@?[a-zA-Z][a-zA-Z0-9._/-]*)''', code):
                                pkg = m.group(1)
                                # Skip relative imports and react/react-native (always available)
                                if pkg.startswith('.') or pkg.startswith('/'):
                                    continue
                                # Get top-level package name (e.g. @react-navigation/native → @react-navigation/native)
                                if pkg.startswith('@'):
                                    parts = pkg.split('/')
                                    pkg = '/'.join(parts[:2]) if len(parts) >= 2 else pkg
                                else:
                                    pkg = pkg.split('/')[0]
                                # Skip built-in React/RN packages
                                if pkg in ('react', 'react-native', 'react-dom', 'expo', 'expo-modules-core'):
                                    continue
                                all_imports.add(pkg)

                    if all_imports:
                        # Read current package.json to find what's already installed
                        pkg_json_path = os.path.join(app_dir, "package.json")
                        existing_deps = set()
                        if os.path.exists(pkg_json_path):
                            import json as _json
                            with open(pkg_json_path) as f:
                                pkg_data = _json.load(f)
                            existing_deps = set(pkg_data.get("dependencies", {}).keys()) | set(pkg_data.get("devDependencies", {}).keys())

                        missing = all_imports - existing_deps
                        if missing:
                            await blog.info(f"Auto-installing {len(missing)} missing deps: {', '.join(sorted(missing)[:10])}...")
                            try:
                                install_cmd = ["npx", "expo", "install"] + sorted(missing)
                                proc = await asyncio.create_subprocess_exec(
                                    *install_cmd, cwd=app_dir,
                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                )
                                await asyncio.wait_for(proc.communicate(), timeout=120)
                            except Exception as dep_err:
                                await blog.warn(f"Dep reconciliation partial failure: {dep_err}")
                        else:
                            await blog.info("All imports accounted for in package.json")
                except Exception as recon_err:
                    await blog.warn(f"Dependency reconciliation skipped: {recon_err}")

                # ── 7c: Bundle validation + comprehensive auto-repair (up to 8 rounds) ──
                await blog.info("Validating web bundle compilation...")
                bundle_ok = False
                self._last_bundle_errors = []
                MAX_REPAIR_ROUNDS = 8
                repaired_files_history = []  # Track which files we've already tried to repair (list for .count())
                try:
                    for repair_round in range(MAX_REPAIR_ROUNDS):
                        # Check if web server is alive
                        managed_app = self._app_manager._running.get(app_id)
                        web_alive = (managed_app and managed_app.web_process
                                     and managed_app.web_process.returncode is None)

                        if web_alive:
                            await self._wait_for_server(web_port, timeout=25)
                            bundle_ok, bundle_errors = await self._validate_bundle(web_port, blog)
                            if bundle_errors:
                                self._last_bundle_errors = [e.get("error", "") for e in bundle_errors]
                        else:
                            bundle_ok = False
                            await blog.warn("Web server crashed — extracting errors...")
                            bundle_errors = self._extract_errors_from_log_buffer(app_id)
                            if bundle_errors:
                                self._last_bundle_errors = [e.get("error", "") for e in bundle_errors]
                            if not bundle_errors:
                                await blog.warn("No specific errors in process output")
                                if managed_app and managed_app.log_buffer:
                                    for line in list(managed_app.log_buffer)[-15:]:
                                        await blog.info(f"  {line}")

                        if bundle_ok:
                            label = "Bundle compiles cleanly" if repair_round == 0 else f"Bundle repaired (round {repair_round})"
                            await blog.success(label)
                            break

                        # ── Categorize and fix errors ──
                        await blog.warn(f"Bundle has issues — repair round {repair_round + 1}/{MAX_REPAIR_ROUNDS}...")

                        fixed_something = False

                        # Strategy 1: Missing dependencies (npm install)
                        if bundle_errors:
                            dep_fixed = await self._fix_missing_deps(app_id, app_dir, bundle_errors, blog)
                            if dep_fixed:
                                fixed_something = True

                        # Strategy 2: Code errors — regenerate broken files via LLM
                        code_errors = [e for e in bundle_errors if e.get("file", "unknown") != "unknown"]
                        # Filter out files we've already repaired 2+ times (diminishing returns)
                        fresh_errors = [e for e in code_errors if repaired_files_history.count(e.get("file", "")) < 2]
                        if not fresh_errors and code_errors:
                            fresh_errors = code_errors  # If all are stale, retry anyway

                        if fresh_errors:
                            repaired = await self._repair_bundle_errors(
                                app_id, app_dir, fresh_errors, generated_files,
                                description, name, deps, db_type, blog
                            )
                            if repaired:
                                fixed_something = True
                                for e in fresh_errors:
                                    repaired_files_history.append(e.get("file", ""))

                        # Strategy 3: Server not alive — restart
                        if not web_alive:
                            await blog.info("Restarting web server...")
                            try:
                                web_port = await self._app_manager.start_web(app_id, app_dir=app_dir)
                                await blog.info(f"Web server restarted on port {web_port}")
                                await asyncio.sleep(5)
                                fixed_something = True
                            except Exception as restart_err:
                                await blog.warn(f"Web server restart failed: {restart_err}")

                        # Strategy 4: If nothing else worked, try clearing Metro cache
                        if not fixed_something and repair_round >= 2:
                            await blog.info("Clearing Metro cache and restarting...")
                            try:
                                clear_proc = await asyncio.create_subprocess_exec(
                                    "npx", "expo", "start", "--clear", "--web", "--port", str(web_port),
                                    cwd=app_dir,
                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                )
                                # Don't wait for it to finish — just start it and let the next round check
                                await asyncio.sleep(8)
                                fixed_something = True
                            except Exception:
                                pass

                        if not fixed_something:
                            await blog.warn(f"No progress in round {repair_round + 1} — trying next round anyway")
                            # Don't break — keep trying, the next round may catch different errors

                        # Restart server after code/dep fixes if it was alive (Metro needs to rebundle)
                        if web_alive and fixed_something:
                            await asyncio.sleep(4)  # Wait for Metro hot-reload
                        elif not web_alive and fixed_something:
                            await blog.info("Restarting web server after repairs...")
                            try:
                                web_port = await self._app_manager.start_web(app_id, app_dir=app_dir)
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
                                    web_port = await self._app_manager.start_web(app_id, app_dir=app_dir)
                                    await self._wait_for_server(web_port, timeout=20)
                                    # Verify the restarted server actually serves a valid bundle
                                    bundle_ok, _ = await self._validate_bundle(web_port, blog)
                                    if bundle_ok:
                                        await blog.success(f"Web server running on port {web_port} after static export verification")
                                    else:
                                        await blog.warn(f"Web server on port {web_port} started but bundle still has errors — preview may not render")
                                except Exception as _restart_err:
                                    await blog.warn(f"Web server restart failed after static export: {_restart_err} — preview will not be available")
                            else:
                                await blog.error(f"Static export also failed: {output[-300:]}")
                        except asyncio.TimeoutError:
                            await blog.warn("Static export timed out (120s)")
                        except Exception as e:
                            await blog.warn(f"Static export failed: {e}")

                    # ── 7e: Auto-patch known infra files if they're the sole error source ──
                    if not bundle_ok and self._last_bundle_errors:
                        _infra_stubs = {
                            "agentBridge": (
                                "/lib/agentBridge.ts",
                                "// AgentBridge — delegates to platform-injected bridge.\n"
                                "const injected = typeof window !== 'undefined' && (window as any).__TOUP_AGENT_BRIDGE;\n"
                                "const agentBridge = injected || {\n"
                                "  isConnected: false, currentScreen: 'Home',\n"
                                "  sendMessage: async () => {}, onAgentMessage: () => () => {},\n"
                                "  onToolActivity: () => () => {}, setNavigationRef: () => {},\n"
                                "  navigate: () => {}, getScreens: () => [], getActions: () => [],\n"
                                "  setScreens: () => {}, setActions: () => {}, destroy: () => {},\n"
                                "};\n"
                                "export type AgentScreen = { name: string; description: string };\n"
                                "export type AgentActionMeta = { id: string; label: string; handler: string };\n"
                                "export { agentBridge };\n"
                                "export const AgentBridge = agentBridge;\n"
                                "export default agentBridge;\n"
                            ),
                            "agentActions": (
                                "/lib/agentActions.ts",
                                "// Agent action registry — screens register actions via registerAction().\n"
                                "type AgentAction = { id: string; label: string; description: string; handler: (...args: any[]) => Promise<any> };\n"
                                "const registry: Record<string, AgentAction[]> = {};\n"
                                "export function registerAction(screen: string, action: AgentAction) {\n"
                                "  if (!registry[screen]) registry[screen] = [];\n"
                                "  registry[screen].push(action);\n"
                                "}\n"
                                "export function getActions(screen?: string): AgentAction[] {\n"
                                "  if (!screen) return Object.values(registry).flat();\n"
                                "  return registry[screen] || [];\n"
                                "}\n"
                                "export default { registerAction, getActions };\n"
                            ),
                            "ErrorBoundary": (
                                "/components/ErrorBoundary.tsx",
                                None,  # already handled by infra injection above
                            ),
                        }
                        _patched_infra = False
                        for _infra_name, (_infra_path, _infra_code) in _infra_stubs.items():
                            if _infra_code is None:
                                continue
                            # Check if any bundle error points to this infra file
                            _errors_in_file = [
                                e for e in (bundle_errors if 'bundle_errors' in dir() else [])
                                if _infra_name.lower() in e.get("file", "").lower()
                                or _infra_name.lower() in e.get("error", "").lower()
                            ]
                            if not _errors_in_file:
                                # Also check _last_bundle_errors text
                                _errors_in_file = [
                                    e for e in self._last_bundle_errors
                                    if _infra_name.lower() in e.lower()
                                ]
                            if _errors_in_file:
                                _abs = os.path.join(app_dir, _infra_path.lstrip("/"))
                                with open(_abs, "w") as f:
                                    f.write(_infra_code)
                                _patched_infra = True
                                await blog.warn(
                                    f"[AUTO-PATCH] Replaced broken {_infra_path} with reliable stub"
                                )
                                logger.info(f"[AUTO-PATCH] {_infra_path} patched job={job_id[:8]}")

                        if _patched_infra:
                            await blog.info("Restarting web server after infra patches...")
                            try:
                                web_port = await self._app_manager.start_web(app_id, app_dir=app_dir)
                                await self._wait_for_server(web_port, timeout=25)
                                bundle_ok, _ = await self._validate_bundle(web_port, blog)
                                if bundle_ok:
                                    await blog.success("Bundle compiles after infra auto-patch")
                            except Exception as _patch_err:
                                await blog.warn(f"Server restart after auto-patch failed: {_patch_err}")

                    if not bundle_ok:
                        await blog.error("Bundle has compilation errors after all repair attempts")
                except Exception as e:
                    await blog.warn(f"Bundle validation error: {e}")

                # ── 7f: Runtime validation — catch errors that compile but crash on load ──
                if bundle_ok and web_port:
                    runtime_errors = await self._validate_runtime(web_port, blog)
                    if runtime_errors:
                        await blog.warn(f"Runtime errors detected: {runtime_errors[0][:200]}")
                        # Feed runtime errors to LLM repair (up to 3 rounds)
                        MAX_RUNTIME_REPAIRS = 3
                        for rt_round in range(MAX_RUNTIME_REPAIRS):
                            await blog.info(f"[RUNTIME-FIX] Repair round {rt_round + 1}/{MAX_RUNTIME_REPAIRS}...")
                            # Extract file from error if possible
                            import re as _rt_re
                            _rt_errors = [{"file": "unknown", "error": e} for e in runtime_errors]
                            for _rte in _rt_errors:
                                _fm = _rt_re.search(r'(/[^\s:]+\.tsx?)', _rte["error"])
                                if _fm:
                                    _rte["file"] = _fm.group(1)

                            repaired = await self._repair_bundle_errors(
                                app_id, app_dir, _rt_errors, generated_files,
                                description, name, deps, db_type, blog
                            )
                            if repaired:
                                # Restart and re-check
                                await asyncio.sleep(3)
                                runtime_errors = await self._validate_runtime(web_port, blog)
                                if not runtime_errors:
                                    await blog.success(f"Runtime errors fixed (round {rt_round + 1})")
                                    break
                            else:
                                await blog.warn(f"[RUNTIME-FIX] LLM could not fix runtime errors")
                                break

                        if runtime_errors:
                            await blog.error(f"App has runtime errors after {MAX_RUNTIME_REPAIRS} repair attempts")
                            bundle_ok = False  # Treat as failed build

                qr_url = await self._app_manager.get_qr_url(app_id)
                web_url = await self._app_manager.get_web_url(app_id)
                await blog.info(f"QR URL: {qr_url}")
                await blog.info(f"Web URL: {web_url}")

                # Set status based on bundle + runtime validation result
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
                    # Collect last known errors for the failure message
                    last_errors = []
                    if hasattr(self, '_last_bundle_errors'):
                        last_errors = self._last_bundle_errors[:3]
                    error_detail = "; ".join(last_errors) if last_errors else "Unknown bundle error"
                    await blog.error(f"Build completed with errors: {error_detail}")
                    await blog.persist()
                    await self._fail_job(job_id, app_id, f"Bundle compilation failed: {error_detail[:200]}")
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
            _actual_model = getattr(self, '_last_llm_provider', None) or "claude-opus-4-6"
            async with async_session_maker() as db:
                job = await db.get(BuildJob, job_id)
                if job:
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    job.model = _actual_model
                    await db.commit()

            # Broadcast job completion so Jobs panel updates without polling
            if self._ws_broadcast:
                await self._ws_broadcast(user_id, {
                    "type": "job_update",
                    "job_id": job_id,
                    "app_id": app_id,
                    "name": name,
                    "status": "completed",
                })

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
                    "job_id": job_id,
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
        await blog.info(f"app_dir={app_dir} slug={slug}")
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
            def _esc_fmt(s: str) -> str:
                return s.replace("{", "{{").replace("}", "}}")

            analysis_response = await self._call_llm(
                MODIFY_ANALYSIS_PROMPT.format(
                    app_name=_esc_fmt(app_name),
                    file_list=_esc_fmt(json.dumps(file_list[:100])),
                    changes=_esc_fmt(changes),
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
            await self._app_manager.write_app_files(app_id, generated, app_dir=app_dir)
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
                    await self._app_manager.install_deps(app_id, new_deps, app_dir=app_dir)
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
                metro_port = await self._app_manager.start_metro(app_id, app_dir=app_dir)
                web_port = await self._app_manager.start_web(app_id, app_dir=app_dir)
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
                                web_port = await self._app_manager.start_web(app_id, app_dir=app_dir)
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
                                web_port = await self._app_manager.start_web(app_id, app_dir=app_dir)
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
            _actual_model = getattr(self, '_last_llm_provider', None) or "claude-opus-4-6"
            async with async_session_maker() as db:
                job = await db.get(BuildJob, job_id)
                if job:
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    job.model = _actual_model
                    await db.commit()

            if self._ws_broadcast:
                qr_url = await self._app_manager.get_qr_url(app_id) if self._app_manager else ""
                web_url = await self._app_manager.get_web_url(app_id) if self._app_manager else ""
                await self._ws_broadcast(user_id, {
                    "type": "app_ready",
                    "job_id": job_id,
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
            _desc_words = len(description.split())
            if blog:
                await blog.info(f"Planning with Claude Opus 4.6 ({_desc_words} word description)...")
            def _esc_plan(s: str) -> str:
                return s.replace("{", "{{").replace("}", "}}")

            llm_response = await self._call_llm(
                PLANNING_PROMPT.format(description=_esc_plan(description), extra_context=_esc_plan(extra_context)),
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
        """Generate code for all files using LLM — parallel with semaphore.

        Includes auto-compaction: when session token usage exceeds 80%,
        the logger's session counters are reset so the build can continue
        without hitting context/rate limits.
        """
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

                # Escape curly braces in LLM-generated content to prevent
                # KeyError from .format() (e.g. "{ return null; }" in design notes)
                def _esc(s: str) -> str:
                    return s.replace("{", "{{").replace("}", "}}")

                prompt = CODE_GEN_PROMPT.format(
                    file_path=file_path,
                    description=_esc(description),
                    app_name=_esc(app_name),
                    all_files=_esc(all_files),
                    all_deps=_esc(all_deps),
                    db_type=db_type,
                    design_notes=_esc(design_section),
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

                    # Reject garbage/stub responses (e.g. "return null;")
                    def _is_garbage(c: str) -> bool:
                        s = c.strip().rstrip(";").strip()
                        return len(s) < 50 or s in ("return null", "null", "undefined", "export default null")

                    if _is_garbage(code):
                        if blog:
                            await blog.warn(f"Garbage response for {file_path}: '{code.strip()[:60]}' — retrying")
                        try:
                            code = await self._call_llm(
                                prompt + (
                                    "\n\nCRITICAL: You returned a stub/garbage response like 'return null'. "
                                    "You MUST generate the COMPLETE, real implementation for this file. "
                                    "Output ONLY the full source code, no placeholders."
                                ),
                                f"Generate code for {file_path} (retry — garbage response)",
                                blog=blog, purpose=f"Retry {file_path} (garbage)",
                                max_tokens=32000,
                            )
                            code = self._strip_fences(code)
                        except Exception as retry_err:
                            if blog:
                                await blog.warn(f"Retry also failed for {file_path}: {retry_err}")

                        # If still garbage after retry, use error fallback
                        if _is_garbage(code):
                            if blog:
                                await blog.warn(f"Still garbage after retry for {file_path} — using fallback")
                            code = self._make_error_fallback(file_path)

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
                        # Auto-compaction check: if session usage > 80%, compact and continue
                        if blog.needs_compaction:
                            async with lock:
                                _state = {
                                    "current_step": "writing",
                                    "completed_steps": ["planning", "scaffolding"],
                                    "files_done": list(generated.keys()),
                                    "files_remaining": [f for f in files if f not in generated],
                                }
                            await blog.compact_session(_state)
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

    async def _write_infra_files(self, app_id: str, generated_files: dict, blog=None, app_dir: str = "", job_id: str = ""):
        """Write essential infrastructure files that every app needs.

        1. metro.config.js — stubs out expo-sqlite WASM on web (prevents crash)
        2. Patches database files to use conditional import (web-safe)
        """
        from app.agent.app_manager import APPS_DIR as _apps_dir
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
            _apps_dir,
            app_id, "web", "index.html"
        )
        # Also try the root index.html (Expo web template)
        root_index = os.path.join(
            _apps_dir,
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
        self._fix_named_import_mismatches(generated_files, infra_files)
        self._fix_default_export_misuse(generated_files, infra_files)

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

        # Force AgentPlaceholder to null stub — LLMs sometimes generate visible agent UI
        # despite prompt instructions. The platform shell renders all agent UI externally.
        _agent_placeholder_stub = (
            "import React from 'react';\n"
            "// Platform renders agent UI externally — this component intentionally renders nothing.\n"
            "export default function AgentPlaceholder() { return null; }\n"
        )
        _placeholder_path = "/components/AgentPlaceholder.tsx"
        _existing = generated_files.get(_placeholder_path, infra_files.get(_placeholder_path, ""))
        if _existing and _existing.strip() != _agent_placeholder_stub.strip():
            infra_files[_placeholder_path] = _agent_placeholder_stub
            if blog:
                await blog.warn(f"[STRIP] Overwrote AgentPlaceholder.tsx with null stub (LLM generated visible agent UI) job={job_id[:8]}")
                logger.info(f"[STRIP] AgentPlaceholder overwritten job={job_id[:8]} app={app_id[:8]}")
        elif _placeholder_path not in generated_files and _placeholder_path not in infra_files:
            # Not generated at all — add the stub so imports don't break
            infra_files[_placeholder_path] = _agent_placeholder_stub

        # Force agentBridge.ts to the known-working delegation stub.
        # LLMs generate massive, complex bridge implementations (400+ lines) with
        # TypeScript type errors that waste 8 repair rounds. The platform injects
        # window.__TOUP_AGENT_BRIDGE at runtime — the generated file just delegates.
        _bridge_stub = (
            "// AgentBridge — delegates to platform-injected bridge, falls back to no-op.\n"
            "// The platform proxy sets window.__TOUP_AGENT_BRIDGE before this code runs.\n"
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
            "  setScreens: () => {},\n"
            "  setActions: () => {},\n"
            "  destroy: () => {},\n"
            "};\n"
            "export type AgentScreen = { name: string; description: string };\n"
            "export type AgentActionMeta = { id: string; label: string; handler: string };\n"
            "export { agentBridge };\n"
            "export const AgentBridge = agentBridge;\n"
            "export default agentBridge;\n"
        )
        _bridge_path = "/lib/agentBridge.ts"
        _existing_bridge = generated_files.get(_bridge_path, "")
        if _existing_bridge and len(_existing_bridge) > 500:
            # LLM generated a complex bridge — replace with the reliable stub
            infra_files[_bridge_path] = _bridge_stub
            if blog:
                await blog.warn(
                    f"[STRIP] Overwrote agentBridge.ts ({len(_existing_bridge)} chars → stub) — "
                    f"LLM generated complex bridge that often has type errors job={job_id[:8]}"
                )
                logger.info(f"[STRIP] agentBridge overwritten job={job_id[:8]} app={app_id[:8]} orig_len={len(_existing_bridge)}")
        elif _bridge_path not in generated_files and _bridge_path not in infra_files:
            infra_files[_bridge_path] = _bridge_stub

        # Strip "Agent ready" cards from screen files — LLMs insert visible agent UI
        # blocks despite instructions. Preserve registerAction() and AgentBridge plumbing.
        import re as _strip_re
        for fp in list(generated_files.keys()) + list(infra_files.keys()):
            if '/screens/' not in fp and fp != '/App.tsx':
                continue
            code = infra_files.get(fp, generated_files.get(fp, ""))
            if not code:
                continue
            # Match JSX blocks containing "Agent ready" or "floating agent orb" text
            # These are visual-only — registerAction lives in useEffect, not in JSX
            _original = code
            # Remove View/div blocks containing agent card text
            code = _strip_re.sub(
                r'(?s)\{?/\*\s*[Aa]gent\s*(ready|card|placeholder|section).*?\*/\}?\s*'
                r'<(?:View|div)[^>]*>.*?(?:Agent ready|floating agent orb|agent support).*?</(?:View|div)>\s*',
                '', code
            )
            if code != _original:
                if fp in infra_files:
                    infra_files[fp] = code
                else:
                    generated_files[fp] = code
                if blog:
                    await blog.warn(f"[STRIP] Removed 'Agent ready' card from {fp} job={job_id[:8]}")
                    logger.info(f"[STRIP] Agent card removed from {fp} job={job_id[:8]} app={app_id[:8]}")

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
            await self._app_manager.write_app_files(app_id, infra_files, app_dir=app_dir or None)
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
            # Pattern b: const ComponentName = async () => { (arrow functions used as components)
            # Any .tsx/.jsx file with a PascalCase name is likely a component — don't touch lib/utils
            is_component_file = (
                '/screens/' in fp or '/components/' in fp
                or fp.endswith('/App.tsx') or fp.endswith('/App.jsx')
                or (fp.endswith(('.tsx', '.jsx')) and not any(
                    d in fp for d in ('/lib/', '/utils/', '/hooks/', '/services/', '/api/')
                ))
            )
            if is_component_file:
                # Catch any PascalCase arrow function (starts with uppercase letter)
                patched = _re.sub(
                    r'((?:export\s+(?:default\s+)?)?const\s+[A-Z]\w*\s*=\s*)async\s*\(',
                    r'\1(',
                    patched,
                )
                patched = _re.sub(
                    r'((?:export\s+(?:default\s+)?)?const\s+[A-Z]\w*\s*=\s*)async\s*\(\)',
                    r'\1()',
                    patched,
                )
                # Pattern c: async function ComponentName (without export, PascalCase = component)
                patched = _re.sub(
                    r'^(\s*)async\s+function\s+([A-Z]\w*)\s*\(',
                    r'\1function \2(',
                    patched,
                    flags=_re.MULTILINE,
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
    def _fix_named_import_mismatches(generated_files: dict, infra_files: dict):
        """Detect and fix named imports that don't match any export in the target module.

        Example: Screen imports `{ calcPress }` from '../lib/calcEngine', but the module
        only exports `pressKey`. This compiles but crashes at runtime (`calcPress` is undefined).

        Fix strategy: find the closest matching export name and rewrite the import.
        """
        import re as _re

        # Build a map of file_path -> set of exported names
        all_files = {**generated_files, **infra_files}
        exports_by_file: dict[str, set[str]] = {}
        for fp, code in all_files.items():
            exports = set()
            # Named exports: export { X, Y } or export { X as Y }
            for m in _re.finditer(r'export\s*\{([^}]+)\}', code):
                for name in m.group(1).split(','):
                    name = name.strip().split(' as ')[-1].strip()
                    if name:
                        exports.add(name)
            # export function/const/class/type/interface NAME
            for m in _re.finditer(r'export\s+(?:default\s+)?(?:async\s+)?(?:function|const|let|var|class|type|interface|enum)\s+(\w+)', code):
                exports.add(m.group(1))
            # export default NAME;
            for m in _re.finditer(r'export\s+default\s+(\w+)\s*;', code):
                exports.add(m.group(1))
            exports_by_file[fp] = exports

        # For each file, check all named imports and verify they exist in the target
        for fp, code in list(all_files.items()):
            patched = infra_files.get(fp, code)
            changed = False
            for m in _re.finditer(r'import\s*\{([^}]+)\}\s*from\s*[\'"]([^\'"]+)[\'"]', patched):
                import_names = [n.strip().split(' as ')[0].strip() for n in m.group(1).split(',') if n.strip()]
                module_path = m.group(2)

                # Resolve relative module path to a file path
                target_fp = None
                for candidate in exports_by_file:
                    # Match ./lib/calcEngine -> /lib/calcEngine.ts
                    clean_module = module_path.lstrip('.')
                    clean_candidate = candidate.replace('.tsx', '').replace('.ts', '')
                    if clean_candidate.endswith(clean_module):
                        target_fp = candidate
                        break

                if not target_fp:
                    continue  # External package import, skip

                target_exports = exports_by_file.get(target_fp, set())
                if not target_exports:
                    continue

                # Check each imported name
                for imp_name in import_names:
                    if imp_name.startswith('type '):
                        continue  # Type imports are handled differently
                    if imp_name in target_exports:
                        continue  # Import is valid

                    # Try to find a close match in exports (case-insensitive prefix/suffix)
                    imp_lower = imp_name.lower()
                    best_match = None
                    for exp_name in target_exports:
                        exp_lower = exp_name.lower()
                        # Check common patterns: calcPress -> pressKey, CalcState -> CalcEngineState
                        if imp_lower in exp_lower or exp_lower in imp_lower:
                            best_match = exp_name
                            break
                        # Check suffix match (calcPress -> pressKey — both end with "press" variant)
                        if imp_lower[-4:] == exp_lower[-4:] and len(imp_lower) > 4:
                            best_match = exp_name
                            break

                    if best_match:
                        logger.info(f"[IMPORT-FIX] {fp}: Rewriting import '{imp_name}' -> '{best_match}' (from {module_path})")
                        patched = patched.replace(
                            f'{{ {imp_name}' if f'{{ {imp_name}' in patched else f'{{{imp_name}',
                            f'{{ {best_match}' if f'{{ {imp_name}' in patched else f'{{{best_match}',
                            1
                        )
                        # Also fix all usages of the old name in this file
                        patched = _re.sub(r'\b' + _re.escape(imp_name) + r'\b', best_match, patched)
                        changed = True
                    else:
                        logger.warning(f"[IMPORT-FIX] {fp}: '{imp_name}' not found in {target_fp} exports: {target_exports}")

            if changed:
                infra_files[fp] = patched

    @staticmethod
    def _fix_default_export_misuse(generated_files: dict, infra_files: dict):
        """Detect when a default-exported object is called as a function.

        Pattern: lib/format.ts has `export default { formatNumber, ... }` (object),
        but screen does `const result = Format(calc)` — calling the object.
        Fix: rewrite to use the appropriate method, e.g. `Format.formatNumber(calc)`.
        """
        import re as _re

        all_files = {**generated_files, **infra_files}

        # Find modules whose default export is an object literal: `export default { ... }`
        object_default_exports: dict[str, list[str]] = {}  # file -> list of method names
        for fp, code in all_files.items():
            m = _re.search(r'export\s+default\s+\{([^}]+)\}', code)
            if m:
                methods = [n.strip().split(':')[0].strip() for n in m.group(1).split(',') if n.strip()]
                object_default_exports[fp] = methods

        if not object_default_exports:
            return

        # For each file, check if a default import from an object-exporting module is called as a function
        for fp, code in list(all_files.items()):
            patched = infra_files.get(fp, code)
            changed = False

            for m in _re.finditer(r'import\s+(\w+)\s+from\s*[\'"]([^\'"]+)[\'"]', patched):
                import_name = m.group(1)
                module_path = m.group(2)

                # Resolve to a file path
                target_fp = None
                for candidate in object_default_exports:
                    clean_module = module_path.lstrip('.')
                    clean_candidate = candidate.replace('.tsx', '').replace('.ts', '')
                    if clean_candidate.endswith(clean_module):
                        target_fp = candidate
                        break

                if not target_fp:
                    continue

                methods = object_default_exports[target_fp]

                # Check if the import is called as a function: `ImportName(` or `ImportName (`
                call_pattern = _re.compile(r'\b' + _re.escape(import_name) + r'\s*\(')
                # But exclude `ImportName.method(` which is correct usage
                method_pattern = _re.compile(r'\b' + _re.escape(import_name) + r'\.\w+\s*\(')

                for call_match in call_pattern.finditer(patched):
                    pos = call_match.start()
                    # Check it's not a method call
                    if method_pattern.match(patched, pos):
                        continue
                    # Check it's not in an import statement
                    line_start = patched.rfind('\n', 0, pos) + 1
                    line = patched[line_start:patched.find('\n', pos)]
                    if 'import' in line:
                        continue

                    # Found: object being called as function. Try to pick the right method.
                    # Heuristic: use the first method or one that looks like a main entry
                    best_method = methods[0] if methods else None
                    for method_name in methods:
                        if any(kw in method_name.lower() for kw in ['format', 'calc', 'eval', 'process', 'get', 'create']):
                            best_method = method_name
                            break

                    if best_method:
                        old_call = call_match.group(0)
                        new_call = f"{import_name}.{best_method}("
                        logger.info(f"[IMPORT-FIX] {fp}: Rewriting '{old_call.strip()}' -> '{new_call.strip()}' (default export is object, not function)")
                        patched = patched[:pos] + patched[pos:].replace(old_call, new_call, 1)
                        changed = True

            if changed:
                infra_files[fp] = patched

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
        self, system_prompt: str, user_message: str, model: str = "",
        blog=None, purpose: str = "", max_tokens: int = 32000,
    ) -> str:
        """Call the best available LLM based on the user's configured API keys.

        Routing logic:
          1. If Anthropic key is available and valid → use Claude
          2. If Anthropic fails (auth, rate limit, overload) → fall through to OpenAI
          3. If OpenAI key is available → use OpenAI
          4. If both fail or neither key exists → raise with clear message
        """
        from app.services.key_provider import keys

        anthropic_key = keys.anthropic or ""
        openai_key = keys.openai or ""
        errors: list[str] = []

        if not anthropic_key and not openai_key:
            raise RuntimeError(
                "No LLM API key configured. Add ANTHROPIC_API_KEY or OPENAI_API_KEY in Settings."
            )

        # ── Try Anthropic first (if key exists) ──────────────────────
        if anthropic_key:
            try:
                result = await self._call_anthropic(
                    system_prompt, user_message,
                    model=model or "claude-sonnet-4-6",
                    anthropic_key=anthropic_key,
                    max_tokens=max_tokens, blog=blog, purpose=purpose,
                )
                self._last_llm_provider = f"anthropic/{model or 'claude-sonnet-4-6'}"
                return result
            except TokenLimitError:
                # Rate-limit / overload — only pause if no OpenAI fallback
                if not openai_key:
                    raise
                print("[LLM] Anthropic rate-limited, trying OpenAI fallback", flush=True)
                if blog:
                    await blog.warn("Anthropic rate-limited — switching to OpenAI")
                errors.append("Anthropic: rate limited")
            except Exception as _anth_err:
                _reason = f"{type(_anth_err).__name__}: {_anth_err}"
                errors.append(f"Anthropic: {_reason}")
                if openai_key:
                    print(f"[LLM] Anthropic failed ({_reason}), trying OpenAI", flush=True)
                    if blog:
                        await blog.warn(f"Anthropic unavailable — switching to OpenAI")
                else:
                    raise

        # ── Try OpenAI (primary or fallback) ─────────────────────────
        if openai_key:
            try:
                result = await self._call_openai(
                    system_prompt, user_message,
                    openai_key=openai_key,
                    max_tokens=max_tokens, blog=blog, purpose=purpose,
                )
                self._last_llm_provider = f"openai/{self._resolve_openai_model()}"
                return result
            except Exception as _oai_err:
                errors.append(f"OpenAI: {type(_oai_err).__name__}: {_oai_err}")

        raise RuntimeError(
            f"All LLM providers failed — please check your API keys in Settings.\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    async def _call_anthropic(
        self, system_prompt: str, user_message: str, model: str,
        anthropic_key: str, max_tokens: int, blog=None, purpose: str = "",
    ) -> str:
        """Call Anthropic Claude API with streaming.

        Raises TokenLimitError on rate limit / overload (caller decides whether to pause or fallback).
        Re-raises auth errors and other failures for the caller to handle.
        """
        import time as _time
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

        _sys = ([
            {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}},
        ] if is_oauth else system_prompt)

        t0 = _time.time()
        text = ""
        _last_progress = t0
        print(f"[LLM] Calling Anthropic ({model})...", flush=True)

        try:
            async with client.messages.stream(
                model=model, max_tokens=max_tokens, system=_sys,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                async for chunk in stream.text_stream:
                    text += chunk
                    _now = _time.time()
                    if blog and (_now - _last_progress) >= 10:
                        await blog.info(f"LLM generating... {int(_now - t0)}s elapsed, ~{len(text.split())} words")
                        _last_progress = _now
                response = await stream.get_final_message()
                input_tok = getattr(response.usage, 'input_tokens', 0)
                output_tok = getattr(response.usage, 'output_tokens', 0)
                stop_reason = getattr(response, 'stop_reason', 'end_turn')

                # Safety net: recover from thinking-block stubs
                if len(text.strip()) < 100:
                    parts = []
                    for block in response.content:
                        if hasattr(block, 'text') and block.text:
                            parts.append(block.text)
                        elif hasattr(block, 'thinking') and block.thinking:
                            parts.append(block.thinking)
                    combined = "\n".join(parts)
                    if len(combined.strip()) > len(text.strip()):
                        if blog:
                            await blog.warn(f"text_stream short ({len(text)} chars), recovered {len(combined)} chars")
                        text = combined
        except anthropic.RateLimitError as e:
            retry_after = 300
            try:
                retry_after = int(getattr(e.response, 'headers', {}).get('retry-after', 300))
            except (ValueError, TypeError, AttributeError):
                pass
            raise TokenLimitError(retry_after_seconds=retry_after, message=f"Anthropic rate limit. Resets in {retry_after}s")
        except anthropic.APIStatusError as e:
            if e.status_code == 529:
                raise TokenLimitError(retry_after_seconds=120, message="Anthropic API overloaded, retry in 2min")
            raise

        elapsed = _time.time() - t0
        print(f"[LLM] Anthropic ({model}) done in {elapsed:.1f}s — {len(text)} chars", flush=True)

        if blog:
            await blog.llm_call(model=model, purpose=purpose or user_message[:50],
                                input_tokens=input_tok, output_tokens=output_tok, duration_s=elapsed)
        if stop_reason == "max_tokens" and blog:
            await blog.warn(f"LLM output truncated (hit {max_tokens} token limit)")

        return text

    @staticmethod
    def _resolve_openai_model() -> str:
        """Pick the right OpenAI model name. Never pass a Claude model to OpenAI."""
        from app.config import settings
        model = settings.agent_model or ""
        # If the configured model is a Claude model (or blank), use gpt-5.4 (best for code gen)
        if not model or "claude" in model.lower() or "haiku" in model.lower() or "sonnet" in model.lower() or "opus" in model.lower():
            return "gpt-5.4"
        return model

    async def _call_openai(
        self, system_prompt: str, user_message: str,
        openai_key: str, max_tokens: int, blog=None, purpose: str = "",
    ) -> str:
        """Call OpenAI API with streaming and proper error handling."""
        import time as _time
        from openai import AsyncOpenAI, AuthenticationError, RateLimitError, APIStatusError

        client = AsyncOpenAI(api_key=openai_key)
        model = self._resolve_openai_model()
        print(f"[LLM] Calling OpenAI ({model})...", flush=True)

        t0 = _time.time()
        _last_progress = t0
        text = ""

        try:
            stream = await client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            input_tok = 0
            output_tok = 0
            finish_reason = None
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    text += chunk.choices[0].delta.content
                    _now = _time.time()
                    if blog and (_now - _last_progress) >= 10:
                        await blog.info(f"LLM generating... {int(_now - t0)}s elapsed, ~{len(text.split())} words")
                        _last_progress = _now
                if chunk.choices and chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                if chunk.usage:
                    input_tok = chunk.usage.prompt_tokens or 0
                    output_tok = chunk.usage.completion_tokens or 0
        except AuthenticationError as e:
            err_str = str(e).lower()
            if any(kw in err_str for kw in ("quota", "billing", "credit", "balance", "exceeded")):
                raise RuntimeError(
                    "OpenAI API credit limit reached — add more credits at "
                    "platform.openai.com/account/billing or switch API key in Settings"
                ) from e
            raise RuntimeError(
                "OpenAI API key is invalid or expired — check your API key in Settings"
            ) from e
        except RateLimitError as e:
            err_str = str(e).lower()
            if any(kw in err_str for kw in ("quota", "billing", "credit", "insufficient")):
                raise RuntimeError(
                    "OpenAI API credit limit reached — add more credits at "
                    "platform.openai.com/account/billing or switch API key in Settings"
                ) from e
            raise TokenLimitError(
                retry_after_seconds=60,
                message=f"OpenAI rate limit reached — too many requests. Retrying in 60s.",
            )
        except APIStatusError as e:
            if e.status_code == 529 or e.status_code >= 500:
                raise TokenLimitError(
                    retry_after_seconds=60,
                    message=f"OpenAI API error ({e.status_code}) — retrying in 60s.",
                )
            raise

        elapsed = _time.time() - t0
        print(f"[LLM] OpenAI ({model}) done in {elapsed:.1f}s — {len(text)} chars, finish_reason={finish_reason}", flush=True)

        if blog:
            await blog.llm_call(model=model, purpose=purpose or user_message[:50],
                                input_tokens=input_tok, output_tokens=output_tok, duration_s=elapsed)
        if finish_reason == "length" and blog:
            await blog.warn(f"OpenAI output truncated (hit {max_tokens} token limit) — code may be incomplete")
        if finish_reason == "length":
            logger.warning(f"[LLM] OpenAI output TRUNCATED (finish_reason=length, {len(text)} chars, purpose={purpose})")
        return text

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
            # Count completed vs total steps for progress tracking
            total = len(steps) if steps else 0
            completed = sum(1 for s in steps if s.get("status") == "done") if steps else 0
            payload = {
                "type": "job_update",
                "job_id": job_id,
                "name": job.title.replace("Build: ", "") if job and job.title else "App Build",
                "status": job.status if job else "running",
                "step": step_dict.get("label") or step_dict.get("name") or step_dict.get("type", ""),
                "total_steps": total,
                "completed_steps": completed,
            }
            print(f"[BUILD_STEP] Broadcasting job_update: job={job_id[:8]} step={step_type}/{status} user={user_id[:8]}", flush=True)
            sent = await self._ws_broadcast(user_id, payload)
            print(f"[BUILD_STEP] Broadcast sent to {sent} connections", flush=True)
        elif not self._ws_broadcast:
            print(f"[BUILD_STEP] _ws_broadcast is None! Cannot send job_update for job={job_id[:8]}", flush=True)
        elif not step_dict:
            print(f"[BUILD_STEP] step_dict is None for step_type={step_type} in job={job_id[:8]}", flush=True)

    async def _fail_job(self, job_id: str, app_id: str, error_msg: str):
        """Mark a job as failed, stop processes, and clean up stuck step statuses."""
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob

        logger.error(f"[BUILD] Job {job_id} failed: {error_msg}")

        # Stop Metro/web processes — no reason to keep them running for a failed build
        if self._app_manager:
            try:
                await self._app_manager.stop_app(app_id)
            except Exception:
                pass

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
                # Create stub directory so Explorer can show the failed app
                if app.app_dir:
                    import os
                    os.makedirs(app.app_dir, exist_ok=True)
                await db.commit()

        # Broadcast failure so frontend updates without a refresh
        if self._ws_broadcast:
            try:
                from app.config import settings
                await self._ws_broadcast(settings.user_id, {
                    "type": "job_update",
                    "job_id": job_id,
                    "app_id": app_id,
                    "status": "failed",
                    "error_message": error_msg,
                })
            except Exception:
                pass

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
        """Strip markdown code fences and other wrapper artifacts from generated code.

        Handles GPT-5.4 output formats that may include:
        - Standard triple backticks (```tsx ... ```)
        - Backticks with language identifier (```typescript ... ```)
        - Leading/trailing explanation text before/after code blocks
        - HTML code tags
        """
        import re as _re
        code = code.strip()

        # If the response contains a fenced code block, extract ONLY the code inside
        fence_match = _re.search(r'```(?:\w+)?\s*\n(.*?)```', code, _re.DOTALL)
        if fence_match:
            code = fence_match.group(1).strip()
            return code

        # Fallback: strip leading/trailing triple backticks (original logic)
        if code.startswith("```"):
            first_newline = code.index("\n") if "\n" in code else len(code)
            code = code[first_newline + 1:]
        if code.endswith("```"):
            code = code[:-3].rstrip()

        # Strip HTML code tags (GPT sometimes wraps in <code>)
        if code.startswith("<code>"):
            code = code[6:]
        if code.endswith("</code>"):
            code = code[:-7]

        # Strip leading prose before actual code (e.g., "Here is the code:\n\nimport React...")
        lines = code.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('export ') or stripped.startswith('const ') or \
               stripped.startswith('function ') or stripped.startswith('//') or stripped.startswith('/*') or \
               stripped.startswith('import{') or stripped.startswith('"use') or stripped == '{':
                if i > 0:
                    code = '\n'.join(lines[i:])
                break

        return code.strip()

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

        # Build rich error context for the repair LLM
        error_context_parts = [description, "\n\n## BUNDLE ERRORS TO FIX"]
        for e in bundle_errors[:5]:
            error_context_parts.append(f"\n### Error in {e.get('file', 'unknown')}:")
            error_context_parts.append(e.get("error", "")[:500])

        # Include the current broken code so the LLM can see what went wrong
        if existing:
            error_context_parts.append("\n## CURRENT FILE CONTENTS (broken, for reference)")
            for fp, code in existing.items():
                error_context_parts.append(f"\n### {fp} (first 3000 chars):")
                error_context_parts.append(code[:3000])

        # Include package.json for dependency awareness
        pkg_json_path = os.path.join(app_dir, "package.json")
        if os.path.exists(pkg_json_path):
            try:
                with open(pkg_json_path) as f:
                    error_context_parts.append(f"\n## package.json:\n{f.read()[:2000]}")
            except Exception:
                pass

        error_context_parts.append(
            "\n\nIMPORTANT: Generate COMPLETE working code that fixes the above errors. "
            "Ensure all imports resolve to installed packages. Use emoji icons (not @expo/vector-icons). "
            "Include NavigationContainer theme with fonts property."
        )

        repaired = await self._generate_code(
            "\n".join(error_context_parts),
            app_name, list(files_to_repair), deps, db_type,
            blog=blog, existing_files=existing,
        )

        if repaired:
            # Apply infra fixes to repaired files
            await self._write_infra_files(app_id, repaired, blog, app_dir=app_dir)
            await self._app_manager.write_app_files(app_id, repaired, app_dir=app_dir if app_dir else None)
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
                "npx", "expo", "install", *list(missing),
                cwd=app_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "EXPO_NO_TELEMETRY": "1"},
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            output = stdout.decode("utf-8", errors="replace") if stdout else ""
            if proc.returncode == 0:
                if blog:
                    await blog.success(f"Installed {len(missing)} missing deps: {', '.join(missing)}")
                return True
            else:
                if blog:
                    await blog.warn(f"npx expo install failed (code {proc.returncode}): {output[:200]}")
                return False
        except asyncio.TimeoutError:
            if blog:
                await blog.warn("npx expo install timed out (120s)")
            return False
        except Exception as e:
            if blog:
                await blog.warn(f"npx expo install failed: {e}")
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
                "npx", "expo", "install", *new_deps,
                cwd=app_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "EXPO_NO_TELEMETRY": "1"},
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

    # Error keywords to search for in Metro output / log buffer.
    # These match ANYWHERE in a line — must be specific enough to avoid
    # false positives on compiled JS (e.g. object properties like `error: null`).
    BUNDLE_ERROR_KEYWORDS = (
        'SyntaxError', 'TransformError', 'TypeError', 'ReferenceError',
        'Cannot find module', 'Module not found', 'Unexpected token',
        'Unable to resolve', 'ENOENT', 'EACCES', 'EADDRINUSE',
        'Invariant Violation', 'Fatal error', 'CompileError', 'ParseError',
    )
    # These only match when they appear at the start of a line (after stripping
    # whitespace).  Metro/Expo CLI prefixes error lines with "error:" or "Error:"
    # but compiled JS contains `error:` mid-line as object property keys — those
    # must NOT be treated as bundle errors.
    BUNDLE_ERROR_LINE_PREFIXES = (
        'error:', 'Error:',
    )
    async def _validate_runtime(self, web_port: int, blog=None) -> list:
        """Load the app in a headless browser and check for runtime errors.

        Returns a list of error strings (empty = app runs cleanly).
        Uses Playwright to detect JavaScript errors, unhandled rejections,
        and ErrorBoundary activations that compile fine but crash on load.
        """
        errors = []
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            if blog:
                await blog.info("Runtime validation skipped (Playwright not available)")
            return []

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
                page = await browser.new_page()

                # Collect JS errors and console.error calls
                page.on("pageerror", lambda err: errors.append(str(err)))
                page.on("console", lambda msg: (
                    errors.append(msg.text)
                    if msg.type == "error" and not any(skip in msg.text for skip in [
                        "favicon", "404", "manifest", "service-worker", "hot-update",
                    ])
                    else None
                ))

                url = f"http://localhost:{web_port}"
                try:
                    await page.goto(url, timeout=20000, wait_until="networkidle")
                except Exception:
                    # Page might not reach networkidle with WebSocket connections
                    try:
                        await page.goto(url, timeout=20000, wait_until="load")
                    except Exception as nav_err:
                        if blog:
                            await blog.warn(f"Runtime check: page navigation failed: {nav_err}")
                        await browser.close()
                        return []

                # Wait for React to render and any async effects to fire
                await page.wait_for_timeout(4000)

                # Check if ErrorBoundary activated
                try:
                    error_boundary = await page.query_selector("text=Something went wrong")
                    if error_boundary:
                        # Get the error message from the ErrorBoundary
                        error_msg = await page.evaluate("""() => {
                            const el = document.body.innerText;
                            const match = el.match(/Something went wrong[\\s\\S]*?(?=Reload App|$)/);
                            return match ? match[0].trim() : 'ErrorBoundary activated';
                        }""")
                        errors.append(f"ErrorBoundary: {error_msg[:500]}")
                except Exception:
                    pass

                await browser.close()

                # Filter out noise
                errors = [e for e in errors if e and len(e) > 5 and "ErrorBoundary" not in e or "ErrorBoundary:" in e]
                # Deduplicate
                seen = set()
                unique = []
                for e in errors:
                    key = e[:100]
                    if key not in seen:
                        seen.add(key)
                        unique.append(e)
                errors = unique[:5]  # Max 5 errors

                if errors and blog:
                    await blog.warn(f"Runtime validation found {len(errors)} error(s)")
                    for e in errors:
                        await blog.error(f"  Runtime: {e[:200]}")
                elif not errors and blog:
                    await blog.success("Runtime validation passed — app loads without errors")

        except Exception as e:
            if blog:
                await blog.info(f"Runtime validation skipped: {e}")
            return []  # Don't block build if Playwright fails

        return errors

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

                    # Determine bundle size to distinguish successful compilation
                    # from error responses.
                    #
                    # Metro + Expo compiled bundles are ALWAYS large (200KB+) because
                    # they include React Native, the module system, and the HMR client.
                    # Error responses are small (<50KB) — either a JSON error body or
                    # a minimal error overlay page.
                    #
                    # The compiled bundle's preamble, runtime, and application code ALL
                    # contain error-related keywords (TypeError, SyntaxError, CompileError,
                    # "Cannot find module", etc.) as class names, variable names, and
                    # string literals.  Scanning ANY part of a successful bundle for
                    # keywords will always produce false positives.
                    #
                    # Strategy:
                    #   HTTP 200 + large response (>=50KB) → compilation succeeded.
                    #   HTTP 200 + small response (<50KB)  → likely error page, scan it.
                    chunks = []
                    total = 0
                    async for chunk in resp.content.iter_chunked(65536):
                        chunks.append(chunk)
                        total += len(chunk)
                        if total > 5 * 1024 * 1024:
                            break

                    if total >= 50 * 1024:
                        # Large bundle = successful compilation. Don't scan.
                        if blog:
                            await blog.info(f"Bundle compiled successfully ({total // 1024}KB)")
                        return True, []

                    # Small response — likely an error page. Scan it.
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
            stripped = line.lstrip()
            is_error = (
                any(kw in line for kw in AppBuilderSkill.BUNDLE_ERROR_KEYWORDS)
                or any(stripped.startswith(pfx) for pfx in AppBuilderSkill.BUNDLE_ERROR_LINE_PREFIXES)
            )
            if is_error:
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
            stripped = line.lstrip()
            is_error = (
                any(kw in line for kw in self.BUNDLE_ERROR_KEYWORDS)
                or any(stripped.startswith(pfx) for pfx in self.BUNDLE_ERROR_LINE_PREFIXES)
            )
            if is_error and any(fp in line for fp in self.BUNDLE_FALSE_POSITIVE_PATTERNS):
                is_error = False
            if is_error:
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
