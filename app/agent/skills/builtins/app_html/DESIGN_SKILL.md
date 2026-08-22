---
name: toup-frontend-design
description: How a Toup app must look, feel and be built — one self-contained HTML file, bold and specific, never templated.
---

# Toup frontend design

Read this **before** you write any UI. It is the difference between an app
someone keeps and an app that looks like every other generated page.

An app is **one `.html` file**: inline `<style>`, inline `<script>`, no build
step, no bundler, no package manager. Libraries come from
`https://cdnjs.cloudflare.com` and nowhere else.

---

## 1. Decide what the app is FOR before you decide what it looks like

Colour is the first thing the person sees and the last thing you should pick
from habit. A gym tracker and a sleep aid are not the same app in two hues —
they exist to produce opposite feelings, and the palette is most of how that
happens. So the order is fixed: **job, then feeling, then palette, then
markup.** Not the other way round, and never straight to a token block.

### 1a. Name the job, in writing, before any colour

Three lines. Write them into the brief you pass to `create_app_file`, and
write them *first* — the brief is the record of why this app looks the way it
does, and in three turns' time it is all that is left of your reasoning.

1. **What is it for?** — "log a gym session in one tap and see the week fill in"
2. **What should the person feel when they open it?** — "capable, on a roll,
   like turning up today is easy"
3. **What behaviour should it encourage?** — "come back tomorrow; log even a
   short session"

Answer those and the palette is nearly decided. A tracker whose job is to make
someone feel *on a roll* cannot be sombre. A wind-down app whose job is to
lower a heart rate cannot be loud.

### 1b. The palette follows the feeling

Guidance for reasoning, not a lookup table. The app's own character leads —
a *boxing* gym log and a *yoga* log want different warmth — but if your
palette contradicts the row it belongs to, you have chosen wrong.

| Domain | Feeling | Palette |
|---|---|---|
| Fitness, sport, training, streaks | energised, rewarded, in motion | warm high-energy hues — orange, coral, red, electric lime as an accent — on **bright or warm-neutral grounds**; progress and motion emphasised |
| Health, wellness, mindfulness, habits | calm, unhurried, cared for | soft greens, sage, warm neutrals; low saturation, generous space |
| Sleep, evening, wind-down | quiet, dim, slowing | deep muted blues and indigos, warm low-blue accents — **this is where a dark ground genuinely belongs** |
| Finance, budgeting, invoices | trustworthy, clear, in control | steady blues and greens, restrained accents, numerals legible above all |
| Food, cooking, recipes | appetising, warm, generous | warm reds, oranges, paprika, cream and paper grounds |
| Productivity, tools, converters | neutral, out of the way | a neutral ground and **one** confident accent; nothing decorative competing with the data |
| Kids, play, arcade games | delighted, loud, playful | saturated primaries, high contrast, thick shapes |
| Focus, deep work, timers | deliberate, undistracted | restrained; either a dim ground *or* a plain warm-neutral one — say which and why |
| Luxury, premium, editorial | considered, expensive | deep grounds with a single metallic or single-hue accent |

That last row is the one this document used to apply to everything. It is a
real look and it is right for a watch catalogue. It is wrong for a gym log.

### 1c. A dark ground is a decision you have to defend

**Near-black canvas plus one neon accent is not the house style.** It was the
example in this document for a long time, so it became the default answer for
apps it actively harms: a water-intake tracker on `#4f3a2c`, a grocery list on
`#1b201a`, a calorie counter on dark maroon. Twenty-five apps were built and
**seventeen came out with a near-black ground** — not because seventeen of
them were evening apps, but because the example was.

So:

- Dark is available and sometimes correct — §1b names where.
- If you choose it, **the brief must say why, in terms of the job**: who opens
  this, where, at what hour, and what a bright screen would do to them. "It
  looks premium" is not a reason. "It is opened in a dark bedroom sixty
  seconds before sleep" is.
- If you cannot write that sentence honestly, the ground is wrong. Choose a
  light or warm-neutral one and move on.

Also banned outright, for the same reason:

- a palette that contradicts its domain (a sombre gym log, a shouting sleep aid)
- a logo whose colours are not the app's own palette — see §1e

### 1d. Then commit, in a token block

Now write the tokens, at the top of your `<style>`. Having decided, decide
completely: six lines of decisions beat six hundred lines of hedging.

```css
:root{
  /* 4–6 colours, from §1b. No more. */
  --bg:…; --surface:…; --ink:…; --muted:…;
  --accent:…;                           /* exactly ONE accent */
  /* type: one display face + one text face, max. SYSTEM STACKS ONLY —
     a webfont name here does not load (§6), it silently falls back, and
     the app you designed is not the app that ships. */
  --display:ui-serif,"Iowan Old Style",Georgia,serif;   /* or rounded, or mono */
  --text:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  /* type scale — pick a ratio and hold it (1.25 here) */
  --t-xs:.75rem; --t-sm:.875rem; --t-md:1rem; --t-lg:1.25rem;
  --t-xl:1.563rem; --t-2xl:1.953rem; --t-3xl:2.441rem;
  /* spacing — 4px base, 8px rhythm. Never a 13px or a 27px. */
  --s-1:4px; --s-2:8px; --s-3:12px; --s-4:16px; --s-6:24px; --s-8:32px; --s-12:48px;
  --r:12px;                             /* one radius, used everywhere */
}
```

Filled in, for three different jobs — note that none of them is the others:

```css
/* Gym log — energised, rewarded, in motion */
--bg:#FFF8F0; --surface:#FFFFFF; --ink:#1A1410; --muted:#7A6A5D; --accent:#F0552B;
/* Sleep wind-down — quiet, dim, slowing */
--bg:#0E1424; --surface:#18203A; --ink:#E8EAF2; --muted:#8A93AC; --accent:#E3A857;
/* Weeknight recipes — appetising, warm, generous */
--bg:#FBF3E6; --surface:#FFFFFF; --ink:#2A1D14; --muted:#8A7460; --accent:#C1452B;
```

### 1e. The logo is the same decision, drawn

The app's mark is generated from the app's own palette — a colour outside it
is rejected outright — so §1b decides the logo too. What you still choose is
the **subject**, and it must depict *this app's* subject: a dumbbell for a gym
log, a moon for a sleep aid, a pot for a recipe book. Never a generic checkmark,
clock, document or abstract tile. Logo, app and preview have to read as one
product.

### 1f. Context and accessibility

- **Cultural meaning** where it is relevant: red is loss in Western finance and
  gain in much of East Asia; white is mourning in parts of Asia. If the app is
  culturally specific, say so in the brief.
- **Never colour alone.** Every state carries a second cue — a label, an icon, a
  shape, a position. Around 1 in 12 men cannot separate your red from your green.
- **Distinctions must survive colour-blindness.** Red/green pairs need a
  difference in lightness too, not just hue.
- **Body text ≥ 4.5:1** against what is behind it, always. §5 has the numbers.

**Pick one signature element** and let it carry the whole design. A single
memorable move — a hard offset shadow, a hairline grid ruled across the
background, oversized numerals, a diagonal section break, a single saturated
block against an otherwise monochrome page, a cursor-tracking highlight.
Repeat it three times. One idea executed confidently reads as designed;
five ideas hedged read as generated.

**Anti-slop checklist.** If you can tick any of these, start the visual
direction over:

- [ ] **A near-black ground with one neon accent, for an app §1b does not put
      there** — the single most common failure of this document
- [ ] **A palette you could move to a different app without changing a word of
      it** — if it would suit ten other apps, it is not this app's palette
- [ ] **A dark ground you cannot justify in one sentence about who opens this
      and when**
- [ ] **A webfont name in `--display`** — it will not load (§6); the app you
      designed is not the app that ships
- [ ] Purple-to-blue (or teal-to-purple) gradient anywhere
- [ ] Inter / Roboto / Open Sans as the *display* face, at default weight
- [ ] Three equal cards in a row, each with an emoji, a bold word, and two
      lines of grey filler
- [ ] Everything centred, everything `border-radius: 8px`, everything
      `box-shadow: 0 4px 6px rgba(0,0,0,.1)`
- [ ] Copy that says "Welcome!", "Get Started", "Lorem ipsum", "Item 1",
      "Feature", "Your data here"
- [ ] A hero with a huge headline and nothing underneath it
- [ ] Any element whose only job is to fill space

Real design has asymmetry, one thing much larger than the rest, and
deliberate empty space that is not the same size as the other empty space.

---

## 2. Write real copy

Placeholder text is the fastest way to make an app feel fake. Populate every
list, table and card with plausible content for *this* app's domain.

- A budget app seeds with "Rent · 1,850.00 · Sep 1", not "Item 1 · $0.00".
- A workout log seeds with "Back squat · 5×5 · 92.5 kg", not "Exercise A".
- Buttons say what happens: "Add expense", "Log set", "Split the bill" —
  never "Submit", "Click here", "Get Started".
- Empty states name the next action: "No expenses yet — add your first one
  to see the month break down." Never a shrug emoji.
- Errors say what went wrong and what to do: "Amount must be a number —
  try 12.50."

---

## 3. Every interactive element has four states

Default, `:hover`, `:focus-visible`, `:active` — plus `:disabled` where it
applies. A control with no hover state feels broken on desktop; a control
with no focus ring is unusable by keyboard and fails accessibility outright.

```css
.btn{background:var(--accent);color:#fff;padding:var(--s-3) var(--s-6);
     border:0;border-radius:var(--r);font:600 var(--t-md)/1 var(--text);
     cursor:pointer;transition:transform .12s ease,filter .12s ease}
.btn:hover{filter:brightness(1.08)}
.btn:focus-visible{outline:3px solid var(--ink);outline-offset:2px}
.btn:active{transform:translateY(1px)}
.btn:disabled{opacity:.45;cursor:not-allowed;filter:none}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
```

Never remove an outline without replacing it. `outline:none` alone is a bug.

---

## 4. It is played with a thumb, on a phone, inside a sheet

This is the section that gets ignored, so it comes with its numbers.

Your app opens full-screen on a phone, in a sheet over a conversation. There
is no mouse, no hover, and no second hand — one thumb, arriving from the
bottom corner. Every layout decision below follows from that.

### 4a. Minimum touch target: 44 × 44 CSS px. Not negotiable, not a maximum.

Apple's HIG says 44 × 44 pt. Material says 48 × 48 dp. WCAG 2.2 SC 2.5.5 says
44 × 44 CSS px (its AA fallback, SC 2.5.8, allows 24 — that is a legal floor
for a dense web table, not a target for something you are designing now). They
agree closely enough that there is one rule: **nothing interactive is smaller
than 44 × 44, and controls are separated by at least 8 px.**

A 32 px icon button is not exempt — give it padding until the *hit area* is 44,
even if the glyph stays small:

```css
.icon-btn{min-width:44px;min-height:44px;display:grid;place-items:center;
          padding:var(--s-2);touch-action:manipulation}
```

`touch-action:manipulation` on every control: without it iOS holds a 300 ms
double-tap-to-zoom delay, and a game control that answers a third of a second
late reads as a dropped input.

### 4b. The control you play with is not a button. Size it accordingly.

44 is the floor for a *link in a list*. The primary control of the app — the
one the person holds their thumb on and uses continuously — is far bigger:

| Control | Minimum | Aim for |
|---|---|---|
| Link / row / secondary action | 44 × 44 | 48 × 48 |
| Primary action (Add, Roll, Submit, Fire) | 56 tall | 64 tall, full width of its column |
| **Game control: D-pad key, paddle, joystick** | **64 × 64** | **72–88 × 72–88**, with an 8–12 px gap |
| Whole D-pad cluster | 200 × 200 | 240 × 240, or ~60% of the viewport width |

This is where a first build goes wrong. A Snake shipped whose D-pad was sized
like a row of toolbar buttons, under a board that took most of the screen: the
game logic was correct, every control was wired, and it was still unpleasant
to play, because the thing being pressed hundreds of times was smaller than
the thumb pressing it. **If the app has one thing you do over and over, that
thing gets the generous size and everything else gets the floor.**

On a phone, a directional game takes **both** a D-pad and a swipe on the
playfield — they cost one extra listener between them and they suit different
hands. Wire them through the one vocabulary in §9, and put the pad beside or
under the board, never on top of it: a control overlapping the thing it
controls hides the state the player is reacting to.

### 4c. The thumb reaches the bottom. Put the controls there.

On a 390 × 844 phone held one-handed, the comfortable arc is roughly the
**bottom third**; the top corners need a second hand. So:

- Primary and repeated controls live in the bottom ~30% of the viewport.
- The top is for the title, the score, the state — things you READ.
- Never put a repeated control (a D-pad, a keypad, +/−) at the top of the
  screen, and never put destructive and primary actions adjacent.

### 4d. The interactive part is the majority of the screen

An app is the thing you do, not the chrome around it. **The playfield plus its
controls take at least ~70% of the viewport height**; header, title, footer and
hints share what is left. If your title block is 120 px tall and the board is
240 px, the layout is upside down.

```css
html,body{height:100%;margin:0}
body{display:flex;flex-direction:column;
     /* dvh, not vh: vh is the tallest-possible viewport, so a page sized in
        it sits under the browser UI on a phone and the bottom row of your
        D-pad is off screen. */
     min-height:100dvh;
     /* The sheet runs under the notch and the home indicator. */
     padding:env(safe-area-inset-top) env(safe-area-inset-right)
             env(safe-area-inset-bottom) env(safe-area-inset-left);
     box-sizing:border-box}
.stage{flex:1 1 auto;min-height:0;display:grid;place-items:center}  /* the app */
.controls{flex:0 0 auto;padding-block:var(--s-4)}                   /* the thumb */
```

`min-height:0` on the flex child is load-bearing: without it a grid/canvas
child refuses to shrink and pushes the controls off the bottom.

### 4e. Both orientations, three widths

Nothing may be reachable only in portrait. Turn the phone and the controls must
still be on screen — which usually means the same flex column becomes a row:

```css
@media (orientation:landscape) and (max-height:520px){
  body{flex-direction:row}
  .controls{display:grid;place-content:center}
}
```

| Width | Must be true |
|---|---|
| **360** | One column. Nothing clipped, nothing scrolling sideways. Every target ≥ 44. Text ≥ 16 px so iOS does not zoom on focus. |
| **768** | Two columns where it helps. Controls stay thumb-side. |
| **1280** | Content capped (`max-width` ~1100 px) and centred — full-bleed text at 1280 px is unreadable. |

```css
.grid{display:grid;gap:var(--s-4);grid-template-columns:1fr}
@media (min-width:768px){.grid{grid-template-columns:repeat(2,1fr)}}
@media (min-width:1280px){.wrap{max-width:1100px;margin-inline:auto}}
```

Always include
`<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">`
— without `viewport-fit=cover` the safe-area variables above are all zero.

### 4f. A control cluster is ONE object, on one grid

Symmetry and alignment are requirements here, not taste. §1's "real design has
asymmetry" is about *composition* — one thing larger than the rest, deliberate
empty space. It is never about the inside of a control cluster: a D-pad, a
keypad, a row of transport buttons is a single object, and any asymmetry
inside it reads as a defect, because the hand learns its geometry.

- **Siblings share a baseline.** Every control in a row sits at the same
  height, same size, same gap. If two controls are the same kind of thing,
  they are drawn as the same thing (§5 already says this about spacing).
- **A cluster is centred in its own area** — a D-pad sits on its own grid
  (`▲` centred over `◀ ▼ ▶`), the whole pad centred in the control zone, not
  shoved into a corner by whatever else happens to be there.
- **Every control belongs to a group.** Secondary controls — sound, pause,
  restart — live IN the header/status bar or IN the control bar, aligned with
  their neighbours. A lone `♪` floating at a third height, belonging to
  nothing, is the canonical failure (it shipped, on a Snake, next to a hint
  block bottom-left and a D-pad centre-right: three unrelated verticals in
  one row).
- **Nothing decorates the playfield.** No stray glyphs, no orphan symbols, no
  mark whose function a player cannot infer. If it does nothing, delete it;
  if it does something, put it in a bar with a hit area.

The layout for a game's control zone is therefore boring on purpose: one
centred cluster, one status row, everything on the same grid. Spend the
personality budget on the playfield and the signature element, never on
scattering the controls.

---

## 5. Legible, spaced, and it answers when you touch it

Three checks that fail quietly, in the order they get skipped.

**Contrast.** Every text/background pair reaches **≥ 4.5:1** (≥ 3:1 for text at
24 px, or 18.66 px bold, and for the borders of interactive controls). The
`--muted` you picked in §1d is where this fails, on a light ground and a dark
one alike — it is chosen to be quiet, and quiet is one step from unreadable:

- on a warm light ground: `#6B6257` on `#F5EFE4` is **5.23** and passes;
  `#8A7E70` on the same ground is **3.46** and does not.
- on a deep evening ground: `#8A93AC` on `#0E1424` is **5.99** and passes;
  `#5D6478` on `#18203A` is **2.72** and does not.

Both directions, deliberately — a worked example in only ONE of them is a
second copy of §1's old failure, quietly certifying that ground as the
approved one. Never encode meaning in colour alone: pair every colour cue
with a shape, an icon or a label.

**Legibility.** Body text ≥ 16 px (below that iOS zooms the page on focus and
the layout you designed is gone). Nothing below 12 px, ever. Line length
45–75 characters; line-height ≥ 1.4 for body, ~1.1 for display. Numbers that
change in place — a score, a timer, a total — get
`font-variant-numeric:tabular-nums`, or the whole row jitters on every tick.

**Spacing.** Every gap comes from the scale in §1. Touching controls get ≥ 8 px
between them; unrelated groups get ≥ 24 px. If two things are the same
distance apart, they are the same kind of thing — so the spacing is what says
"this row of buttons is one object and that one is another".

**State feedback.** Every touch is answered within ~100 ms, visibly:

```css
.key{transition:transform .08s ease,background-color .08s ease}
.key:active{transform:scale(.94);background:var(--accent)}
```

`:hover` alone is not feedback on a phone — there is no hover. A control that
does not visibly change on `:active` reads as a control that missed the tap,
and the person presses it again. Anything that takes more than ~300 ms shows
that it is working (a disabled state, a spinner, a progress bar) rather than
looking inert.

---

## 6. Single file, inline everything

- One `<style>` in `<head>`, one `<script>` before `</body>`. No `fetch` of
  local assets — there are none.
- SVG icons are inlined as markup. Never link an icon font from a random
  host, never use emoji as a UI icon set.
- Libraries: `https://cdnjs.cloudflare.com/ajax/libs/...` only. Anything from
  another origin is **blocked by the sandbox** and your app renders blank.
  If you need React: `react`, `react-dom` and `babel-standalone` from cdnjs,
  with `<script type="text/babel">`.
- Google Fonts is a different origin and will not load. Use a system stack,
  or ship the face from cdnjs.

---

## 7. Storage is safe, but it is not instant

The app runs in a sandboxed frame with an **opaque origin**, where
`localStorage`, `sessionStorage` and `document.cookie` would normally
**throw** on the first access and take your whole script down with them.

They do not, here. The runner replaces all three before your code runs, with
objects that cannot throw and that mirror what you write back to the Toup
shell — so `localStorage.setItem('highScore', 12)` is safe to call directly,
with no `try`, and the value survives a reload.

The one thing that is NOT true is that a value is there on line one. The
restore is a round-trip to the host, so a read taken during first paint
returns `null` even when a value exists. Seed the UI from defaults
immediately and reconcile when the data lands:

```js
let best = 0;                                   // paint with this
render();
addEventListener('toup-storage-ready', () => {  // fires once, after restore
  best = Number(localStorage.getItem('highScore') || 0);
  render();
});
```

Still genuinely unavailable in the sandbox: network requests (`fetch`,
`XMLHttpRequest`, WebSocket — `connect-src 'self'` from an opaque origin
blocks everything), top-level navigation, popups, and the parent page. Build
apps that are complete on their own.

---

## 8. Anything that moves waits to be started

An app opens inside a sheet, over a conversation. The moment it appears, the
person is still looking at the chat behind it — they have not read the board,
found the controls, or decided to play. **A clock that is already running has
already run out.**

This is not hypothetical. A Snake game shipped that spawned its snake in the
middle of a 20×20 grid, heading right, with `setInterval` started from the
last line of the file. Ten ticks at 175 ms is 1.75 seconds: the app was on
`GAME OVER — THE SNAKE HIT THE WALL`, score `000`, before the user's eyes had
reached it. Every other part of it was correct — the D-pad turned the snake,
the speed ramp was there, the best score persisted — and none of it could be
reached. Nothing threw, so the browser check passed it.

So:

- **First paint is a start screen**, not a running game. A title, one line of
  how-to, and one button — `PLAY`, `START`, `BEGIN`. The loop starts in that
  button's handler and nowhere else.
- **The same applies to any unattended countdown**: a timer that begins on
  load, a carousel that advances past the first slide, a quiz that is already
  timing the answer, a simulation that has stepped before it was watched.
- **Losing must cost an input.** If the app can reach a terminal state
  (`game over`, `time's up`, `you lose`) without the user having pressed
  anything, that is a bug in the app, not a hard difficulty setting.
- Animation that is purely decorative — a gradient drift, a pulsing dot — is
  fine on load. The rule is about state the user is judged on.

```js
// Not this
reset(); setInterval(tick, 175);

// This
showStart();                                  // title + PLAY
playBtn.onclick = () => { reset(); setInterval(tick, 175); };
```

### 8a. Sound is built on the first tap, not on load

A whack-a-mole shipped with a mole that made no sound. Nothing was wrong with
the audio code: the oscillators were correct, the envelope was correct, they
were connected to the destination and they were started on every hit. The
`AudioContext` had simply been constructed at the top of the script, and **a
context created while the page is loading starts `suspended`** — in every
browser, by design, because a page may not make a noise at a person who has
not touched it. `resume()` returns a promise that generated code never awaits,
`oscillator.start()` on a suspended context throws nothing, and the app is
silent with no error anywhere. It is the quietest failure in this whole
document, in both senses.

- **Create the `AudioContext` inside the first input handler.** Not at the top
  of the script, not in an `init()` called on load. One lazy getter, and every
  sound goes through it.
- **Call `ctx.resume()` at the top of every handler that plays something.** It
  is a no-op on a running context and it is the whole fix on a suspended one —
  including after iOS suspends the context on a phone call or a lock.
- **Synthesise.** An oscillator plus a gain envelope is a few lines, weighs
  nothing, and cannot fail to load. A short `data:` URI on an `<audio>` works
  too. Never `fetch` a sound: there is no network.
- **Never `autoplay`, and never a sound on a timer.** Both are what the
  autoplay policy exists to stop, and both fail silently when it does.
- **Anything that loops or repeats gets a mute control** — a real 44 px one —
  and the mute state is what your `gain` reads, not an `if` around `play()`.
- Sound is an accent, never information: an app whose only feedback is audio
  is unusable with the ringer off, which on a phone is most of the time.

```js
// Not this — built at load, suspended for ever, and nothing says so
const eager = new AudioContext();
function bonkSilently() { const o = eager.createOscillator(); o.start(); }
```

```js
// This — built on the first tap, resumed on every one after
let ctx;
function audio() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state !== 'running') ctx.resume();     // no-op when already running
  return ctx;
}
function bonk() {
  if (muted) return;
  const c = audio(), o = c.createOscillator(), g = c.createGain();
  o.connect(g); g.connect(c.destination);
  o.frequency.value = 320;
  g.gain.setValueAtTime(0.2, c.currentTime);
  g.gain.exponentialRampToValueAtTime(0.001, c.currentTime + 0.12);
  o.start(); o.stop(c.currentTime + 0.12);
}
mole.addEventListener('pointerdown', () => { bonk(); score++; });
```

The runner resumes any context it can find on the first gesture, so an app
written the wrong way will often still make a noise — and `present_app`
**measures** whether it did. An app that builds audio and has none running
after a tap is refused, with the count in the message.

---

## 9. Two ways in, one vocabulary

A control that silently does nothing throws nothing, so no check can see it.
The way it happens is always the same: two producers feed one lookup with
different words for the same thing.

Measured on a shipped Snake. The D-pad did this:

```js
document.querySelectorAll('[data-dir]').forEach(b => b.onclick = () => turn(b.dataset.dir));
```

…passing `"up"`. The keyboard did this:

```js
addEventListener('keydown', e => { if (dirs[e.key]) turn(e.key) });
```

…passing `"ArrowUp"`. `dirs` was `{up, down, left, right}`, so `dirs["ArrowUp"]`
was `undefined` and **every arrow key was dead** — under a footer the same file
printed reading `ARROWS MOVE`. The pad worked, the swipe worked, the game
looked finished, and half its documented controls did nothing.

So: **normalise at the edge, once.** Every input path converts to the same
vocabulary before it reaches the shared function, and the map is keyed by that
vocabulary and nothing else.

```js
const KEYMAP = {ArrowUp:'up', ArrowDown:'down', ArrowLeft:'left', ArrowRight:'right',
                w:'up', s:'down', a:'left', d:'right'};
addEventListener('keydown', e => { const d = KEYMAP[e.key]; if (d) { e.preventDefault(); turn(d); } });
```

And **say only what you wired.** A hint line claiming a control the code does
not implement is worse than no hint: it sends the user to look for a fault in
themselves.

**And say it for the device in hand.** Keyboard support is welcome — wire it
through the same vocabulary — but its instructions must never render as UI on
a phone. `ARROWS / WASD · SPACE PAUSES` printed inside a touch app is a
manual for hardware the user does not have, in the one slot where they look
for how to play. The publish gate refuses visible keyboard hints outright.
If you want the hint for the rare desktop visitor, reveal it only after a
real `keydown` arrives:

```js
addEventListener('keydown', () => {
  document.getElementById('kbd-hint').hidden = false;
}, { once: true });
```

On touch, the controls explain themselves: a D-pad is its own manual, and
"Swipe to steer" is the only hint a touch game should ever need to print.

---

## 10. Changing an app someone is already holding

A change request is about the thing the person was *using*, and they will
describe it with the shortest word that fits. "Make the button bigger", said
about a game, means the buttons they were pressing to play it.

This went wrong exactly that way: asked to make the button bigger on a Snake
with a D-pad, the edit landed on the start screen's `PLAY` button — pressed
once, already large enough, and the element in the file that most literally
answers to the word "button". The D-pad, pressed hundreds of times and
genuinely too small, was untouched. The app came back with the same defect and
a message saying it had been fixed.

So, before you edit:

1. **Re-read the file.** `view_app_file` first, every time — the element the
   request is about is chosen from what is in the app, not from what you
   remember writing.
2. **Ask which element the complaint could have come from.** In a game the
   controls are the D-pad / paddle / fire button; `PLAY`, `RESTART` and menu
   items are chrome. In a form it is the field, not the legend.
3. **If more than one answer is reasonable, change them all** — every control
   of that kind, the same way, in one round of edits. Widening the change is
   nearly free; guessing wrong costs the person another turn. Do not ask them
   which one they meant.
4. **Fix the class, not the instance.** If the D-pad keys are too small, the
   shared `.key` rule changes — not `#up`.

And a size change is a *layout* change: after it, §4 still has to hold.
A D-pad grown from 44 to 76 px pushes something off the bottom of the screen
unless the stage above it can shrink.

---

## 11. Before you call `present_app`

- [ ] Every control the UI mentions actually works — keys AND taps AND swipe
- [ ] **No keyboard instructions are visible on screen (§9) — a phone has no
      keyboard, and the gate refuses WASD/arrow-keys/spacebar hints**
- [ ] **Sibling controls share a baseline; the D-pad/keypad is centred on its
      own grid; no control floats apart from its cluster (§4f)**
- [ ] **No stray glyphs — everything on screen has a function or is deleted**
- [ ] Nothing that can be lost, missed or timed out is running at first paint
- [ ] A game/timer/quiz opens on a start screen with an explicit start control
- [ ] **Every interactive element is ≥ 44 × 44 CSS px, with ≥ 8 px between**
- [ ] **The main control — D-pad key, paddle, primary action — is ≥ 64 px**
- [ ] **Repeated controls sit in the bottom third; the top is for reading**
- [ ] **Playfield + controls take ~70% or more of the viewport height**
- [ ] **Sized in `dvh`, padded with `env(safe-area-inset-*)`, and
      `viewport-fit=cover` is in the viewport meta**
- [ ] **Nothing is reachable only in portrait — check landscape too**
- [ ] Opens at 360 px with no horizontal scroll
- [ ] Every control answers a touch within ~100 ms (`:active`, not just
      `:hover` — a phone has no hover)
- [ ] Every button/link/input has focus-visible styling
- [ ] Every text/background pair reaches 4.5:1; body text ≥ 16 px
- [ ] Changing numbers use `tabular-nums`
- [ ] No placeholder copy anywhere; seeded with realistic data
- [ ] No external origin except cdnjs
- [ ] No `fetch` / `XMLHttpRequest` / WebSocket anywhere
- [ ] Nothing reads storage expecting a value during first paint
- [ ] **Any `AudioContext` is created in an input handler, and `resume()`d in
      every handler that plays**
- [ ] **Anything that loops has a mute control, and nothing is `autoplay`**
- [ ] **The brief names the job, the feeling and the behaviour (§1a)**
- [ ] **The palette belongs to this app's domain (§1b), and a dark ground —
      if you chose one — is justified in the brief in one sentence about who
      opens this and when (§1c)**
- [ ] **`--display` is a system stack, not a webfont name (§1d)**
- [ ] **The logo depicts this app's own subject, in this app's palette (§1e)**
- [ ] **No state is carried by colour alone (§1f)**
- [ ] The signature element appears at least three times
- [ ] Nothing on the anti-slop checklist ticks

`present_app` opens the app in a real browser and refuses to publish it if
anything throws. A refusal is a list of things to fix, not a dead end: fix
them with `edit_app_file` and call it again.

**Several of the boxes above are now measured, not trusted.** At 390×844,
before and again after the start control is pressed, the gate reads every
laid-out control's `getBoundingClientRect` and refuses the publish over:

- any interactive element rendering under **44 × 44** (an inline link in a
  sentence is exempt — that is typography, not a control),
- any text under **12 px**,
- **sideways scroll** at 390 px wide,
- **visible keyboard instructions** — WASD, "arrow keys", "spacebar",
  "press space/enter" rendered on either screen (§9),
- **audio that was built and never played** — a context made, a gesture seen,
  and nothing running (§8a),
- **a sound the sandbox refused**, which the browser reports as a corrupt
  file rather than as a policy decision.

It names the element and the number — "the control “^” renders 34×30px" — so
the fix is a one-line change to a shared rule. Write to the sizes in §4 and you
will never meet it.

**And then somebody looks at it.** The gate photographs the app on the screen
it has just played with, and reviews the picture for what no measurement can
see: text the same colour as what is behind it, a panel clipped by its own
container, a modal behind the board, an empty box where content belongs, a
collapsed layout, placeholder copy that was never replaced — and the §4f
failures: a control cluster off its grid, siblings at different heights, a
control floating apart from the bar it belongs to, a stray glyph on the
playfield. Those refuse the publish too, each one named and located.

**It is also given the app's purpose, and asked whether the palette fits it**
(§1b). Not whether the palette is *nice* — taste is still yours, and a plain
app that is legible and correctly laid out passes. But a gym tracker that
opens sombre and funereal, a sleep aid that opens loud and bright, a children's
game in luxury monochrome: those are defects, and the reviewer names them the
same way it names a clipped panel.

The rest of the list is still yours: thumb reach, the interactive share of the
screen, the signature element, real copy. A 32 px D-pad now throws; a D-pad
marooned at the top of the screen under a 200 px title still does not.
