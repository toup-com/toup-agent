# Toup frontend design

An app is one self-contained `.html` file: inline `<style>` and `<script>`, no build step, bundler, or package manager. Libraries may load only from `https://cdnjs.cloudflare.com`.

---

## 1. Decide what the app is FOR before you decide what it looks like

Before markup or colour, put three lines **first** in the `create_app_file` brief: the app's job, how the person should feel on opening it, and the behaviour it should encourage. Choose the palette from that feeling, then commit to tokens in `<style>`. A gym log should reward activity; a sleep aid should lower stimulation. Do not reuse a palette by habit.

| Domain | Palette and feeling |
|---|---|
| Fitness, sport, training, streaks | Energised: warm bright or warm-neutral ground, orange/coral/red, electric-lime accent only where useful; show progress. |
| Health, wellness, mindfulness, habits | Calm: sage, soft greens, warm neutrals, low saturation and space. |
| Sleep, evening, wind-down | Quiet: deep muted blue/indigo and warm low-blue accent; a dark ground belongs here. |
| Finance, budgeting, invoices | Trustworthy: steady blues/greens, restrained accent, highly legible numerals. |
| Food, cooking, recipes | Appetising: paprika, warm reds/oranges, cream or paper grounds. |
| Productivity, tools, converters | Neutral ground and one confident accent; data first. |
| Kids, play, arcade games | Playful: saturated primaries, strong contrast and thick shapes. |
| Focus, deep work, timers | Deliberate: dim or warm-neutral ground; explain which suits the job. |
| Luxury, premium, editorial | Deep ground with one metallic or single-hue accent. |

**Near-black plus neon is not the house style.** Use a dark ground only when the brief explains who opens this app, when and why brightness hurts; “premium” alone is not a reason. Reject a palette that contradicts its domain. Choose 4–6 colours, **one** accent, one radius, one display stack and one text stack, a consistent type ratio and a 4px/8px spacing rhythm. Use system font stacks; naming a webfont does not load it. Put these in CSS custom properties such as `--bg`, `--surface`, `--ink`, `--muted`, `--accent`, `--display`, `--text`, `--t-md`, `--s-2` and `--r`; actually use them throughout. A logo depicts this app's subject (dumbbell, moon, pot), not a generic checkmark or tile, and uses only its palette. App, logo and preview must look like one product.

Set the tokens before writing components, for example:

```css
:root{
  --bg:#FFF8F0; --surface:#FFFFFF; --ink:#1A1410;
  --muted:#6B6257; --accent:#F0552B; /* gym log, not sleep aid */
  --display:ui-serif,"Iowan Old Style",Georgia,serif;
  --text:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --t-xs:.75rem; --t-sm:.875rem; --t-md:1rem; --t-lg:1.25rem;
  --t-xl:1.563rem; --t-2xl:1.953rem; --t-3xl:2.441rem;
  --s-1:4px; --s-2:8px; --s-3:12px; --s-4:16px;
  --s-6:24px; --s-8:32px; --s-12:48px; --r:12px;
}
```

The gym palette above is one example, not a universal default. A sleep app can use `#0E1424` and a warm quiet accent; a recipe app can use a cream ground and paprika. The brief decides.

Account for cultural colour meanings where relevant. Never convey state by colour alone: add a label, icon, shape or position; red/green must differ in lightness. Body text contrast is at least **4.5:1**. Pick one signature visual element and repeat it three times. Avoid purple/blue gradients, generic card trios, filler copy, all-centred layouts, identical gaps, default-looking shadows, emoji icon sets and elements that only fill space. Prefer a clear hierarchy, one oversized element and deliberate empty space.

---

## 2. Write real copy

Seed domain-specific plausible rows, cards and tables, not “Item 1”, “Feature” or lorem ipsum. Buttons say the action (“Log set”, “Add expense”), never “Submit” or “Get Started”. Empty states name the next action. Errors say what failed and how to fix it. No filler hero.

---

## 3. Every interactive element has four states

Implement default, `:hover`, `:focus-visible` and `:active`, plus `:disabled` where applicable. Keep a visible keyboard focus ring; never use `outline:none` without a replacement. Show touch feedback via `:active`, not hover alone. Respect `prefers-reduced-motion:reduce` by disabling transitions and animations.

```css
.btn:focus-visible{outline:3px solid var(--ink);outline-offset:2px}
.btn:active{transform:translateY(1px)}
.btn:disabled{opacity:.45;cursor:not-allowed}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
```

---

## 4. It is played with a thumb, on a phone, inside a sheet

Design first for one thumb on a full-screen phone sheet. **Every hit area is at least 44 × 44 CSS px with at least 8px between controls**, including padded icon buttons; use `touch-action:manipulation`. A primary action is at least 56px tall, preferably 64px and full-width in its column. A repeated game control (D-pad key, paddle, joystick) is at least **64 × 64**, ideally **72–88 × 72–88** with an 8–12px gap; the whole D-pad is at least 200 × 200, ideally 240 × 240 or ~60% of viewport width. A directional game offers **both** D-pad and playfield swipe through the same input vocabulary; controls must not obscure the playfield.

Put repeated controls in the bottom ~30% of the phone; the top holds read-only title, score and state. Keep destructive actions away from primary actions. The interactive stage plus its controls take at least ~70% of viewport height. Use `min-height:100dvh`, safe-area padding, `box-sizing:border-box`, a flexible stage with `min-height:0`, and a controls row that cannot be pushed offscreen. Include `<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">` so safe-area insets work. In landscape, rearrange to keep every control visible.

At **360px**, one column, no clipped or sideways-scrolling content, targets ≥44px and text ≥16px so iOS does not zoom on focus. At **768px**, use two columns where helpful. At **1280px**, cap content around 1100px rather than stretching text. A D-pad/keypad/transport row is **one aligned cluster**: equal siblings on a shared baseline and gap, centred within its control zone. Put pause, sound and restart in a status or control bar. No orphan glyph or control scattered over the playfield. Composition may be asymmetric; a control cluster must not be.

The phone layout needs the mechanics, not just the numbers:

```css
html,body{height:100%;margin:0}
body{display:flex;flex-direction:column;min-height:100dvh;
  padding:env(safe-area-inset-top) env(safe-area-inset-right)
          env(safe-area-inset-bottom) env(safe-area-inset-left);
  box-sizing:border-box}
.stage{flex:1 1 auto;min-height:0;display:grid;place-items:center}
.controls{flex:0 0 auto;padding-block:var(--s-4)}
@media (orientation:landscape) and (max-height:520px){
  body{flex-direction:row}.controls{display:grid;place-content:center}}
```

`100dvh` avoids putting bottom controls under phone UI; `min-height:0` lets the stage shrink instead of forcing those controls offscreen.

---

## 5. Legible, spaced, and it answers when you touch it

Check every text/background pair: contrast **≥4.5:1** for body text, **≥3:1** for large text (24px or 18.66px bold) and interactive boundaries. Check muted text on light **and** dark surfaces. Never use colour alone for meaning. Body text **≥16px**, nothing **<12px**, line length **45–75** characters, body line-height **≥1.4**, display line-height around **1.1**; changing numbers use `font-variant-numeric:tabular-nums`. Use the spacing scale: related touch controls ≥8px apart, unrelated groups ≥24px.

For contrast checks, `#6B6257` on `#F5EFE4` is 5.23:1 and passes, while `#8A7E70` on that ground is 3.46:1 and fails. On a dark app, `#8A93AC` on `#0E1424` is 5.99:1 and passes, while `#5D6478` on `#18203A` is 2.72:1 and fails.

Give visible touch feedback within ~100ms with `:active` state; if work takes more than ~300ms, show disabled, spinner or progress feedback. `:hover` alone is not touch feedback.

---

## 6. Single file, inline everything

Use one `<style>` in `<head>` and one `<script>` before `</body>`. Inline SVG icons. There are no local assets to fetch. External libraries must come only from `https://cdnjs.cloudflare.com/ajax/libs/...`; other origins are blocked. If React is needed, load `react`, `react-dom` and `babel-standalone` from cdnjs with `<script type="text/babel">`. Google Fonts will not load; use a system stack or a face available on cdnjs.

---

## 7. Storage is safe, but it is not instant

The sandbox has an opaque origin. The runner replaces `localStorage`, `sessionStorage` and `document.cookie` before app code runs with non-throwing objects that mirror writes to the Toup shell. Call `localStorage.setItem` directly; values survive reload. Restore from the host is asynchronous: first-paint reads can return `null`. Paint from in-memory defaults immediately, then on the one-time `toup-storage-ready` event read storage and rerender. Network requests (`fetch`, `XMLHttpRequest`, WebSocket), top-level navigation, popups and access to the parent page are unavailable; build an app complete on its own.

```js
let best = 0;
render();
addEventListener('toup-storage-ready', () => {
  best = Number(localStorage.getItem('highScore') || 0);
  render();
});
```

---

{{FULL_SECTION_8}}## 9. Two ways in, one vocabulary

Normalise every input path at the edge before it reaches shared logic: D-pad `data-dir`, swipe and keyboard `ArrowUp`/`w` must all produce the same `up` value. A map keyed by one vocabulary prevents controls that look wired but silently do nothing. Claim only controls actually implemented. Never show keyboard instructions on a phone. If desktop hints are useful, reveal them only after a real `keydown`; touch instructions should name the touch gesture, such as “Swipe to steer”.

```js
const KEYMAP = {ArrowUp:'up', ArrowDown:'down', ArrowLeft:'left', ArrowRight:'right',
                w:'up', s:'down', a:'left', d:'right'};
addEventListener('keydown', e => {
  const dir = KEYMAP[e.key]; if (dir) { e.preventDefault(); turn(dir); }
});
```

---

## 10. Changing an app someone is already holding

Call `view_app_file` before **every** edit. Identify the element the person was using, not the first literal match: in a game, “make the button bigger” usually means repeated D-pad or fire controls, not a one-time PLAY button. If several controls of the same kind fit, change all of them consistently in one round. Edit the shared class rather than one instance. After a size change, recheck the §4 phone layout so larger controls do not push the stage offscreen. Read back and publish before claiming it is fixed.

---

{{FULL_SECTION_11}}
