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

## 1. Commit to a look before you write markup

Write the token block first, at the top of your `<style>`. Six lines of
decisions beat six hundred lines of hedging.

```css
:root{
  /* 4–6 colours. No more. */
  --bg:#0B0B0F; --surface:#16161D; --ink:#F4F4F5; --muted:#A1A1AA;
  --accent:#FF5C39;                     /* exactly ONE accent */
  /* type: one display face + one text face, max */
  --display:"Bricolage Grotesque",Georgia,serif;
  --text:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  /* type scale — pick a ratio and hold it (1.25 here) */
  --t-xs:.75rem; --t-sm:.875rem; --t-md:1rem; --t-lg:1.25rem;
  --t-xl:1.563rem; --t-2xl:1.953rem; --t-3xl:2.441rem;
  /* spacing — 4px base, 8px rhythm. Never a 13px or a 27px. */
  --s-1:4px; --s-2:8px; --s-3:12px; --s-4:16px; --s-6:24px; --s-8:32px; --s-12:48px;
  --r:12px;                             /* one radius, used everywhere */
}
```

**Pick one signature element** and let it carry the whole design. A single
memorable move — a hard offset shadow, a hairline grid ruled across the
background, oversized numerals, a diagonal section break, a single saturated
block against an otherwise monochrome page, a cursor-tracking highlight.
Repeat it three times. One idea executed confidently reads as designed;
five ideas hedged read as generated.

**Anti-slop checklist.** If you can tick any of these, start the visual
direction over:

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

## 4. Mobile-first, three real breakpoints

Write the 360 px layout first, then widen. Test all three mentally before you
finish:

| Width | Must be true |
|---|---|
| **360** | One column. Nothing clipped, nothing horizontally scrolling. Tap targets ≥ 44×44 px. Text ≥ 16 px so iOS does not zoom on focus. |
| **768** | Two columns where it helps. Navigation can go horizontal. |
| **1280** | Content capped (`max-width` ~1100 px) and centred — full-bleed text at 1280 px is unreadable. |

```css
.grid{display:grid;gap:var(--s-4);grid-template-columns:1fr}
@media (min-width:768px){.grid{grid-template-columns:repeat(2,1fr)}}
@media (min-width:1280px){.wrap{max-width:1100px;margin-inline:auto}}
```

Always include `<meta name="viewport" content="width=device-width,initial-scale=1">`.

---

## 5. Contrast is non-negotiable

Every text/background pair must reach **≥ 4.5:1** (≥ 3:1 for text at 24 px or
18.66 px bold, and for the borders of interactive controls). Muted greys are
where this fails: `#A1A1AA` on `#0B0B0F` passes; `#71717A` on `#16161D` does
not. Check before you ship, and never encode meaning in colour alone — pair
every colour cue with a shape, an icon or a label.

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

## 7. Never assume `localStorage`

The app runs in a sandboxed frame with an **opaque origin**. `localStorage`,
`sessionStorage`, `indexedDB` and `document.cookie` all **throw** there.
Touching them unguarded kills your script on line one and the page renders
blank.

Always wrap, always keep an in-memory fallback, and prefer the host bridge
for anything that must survive a reload:

```html
<script>
const Store = (() => {
  const mem = new Map();
  let local = null;
  try { localStorage.setItem('__t','1'); localStorage.removeItem('__t'); local = localStorage; }
  catch (_) { local = null; }                    // sandboxed: expected, not an error

  // Host bridge — the Toup shell persists on our behalf over postMessage.
  const pending = new Map(); let seq = 0;
  addEventListener('message', e => {
    const m = e.data;
    if (!m || m.source !== 'toup-storage-host') return;
    const r = pending.get(m.id); if (!r) return;
    pending.delete(m.id); r(m.ok ? m.value : null);
  });
  const ask = (op, key, value) => new Promise(res => {
    if (parent === self) return res(null);
    const id = ++seq; pending.set(id, res);
    parent.postMessage({ source:'toup-storage', v:1, id, op, key, value }, '*');
    setTimeout(() => { if (pending.delete(id)) res(null); }, 1500);
  });

  return {
    get(k){ if (mem.has(k)) return mem.get(k);
            try { return local ? JSON.parse(local.getItem(k)) : null } catch(_) { return null } },
    set(k,v){ mem.set(k,v);
              try { local && local.setItem(k, JSON.stringify(v)) } catch(_) {}
              ask('set', k, v); },
    async load(k){ const v = await ask('get', k); if (v != null) mem.set(k, v);
                   return v != null ? v : this.get(k); },
  };
})();
</script>
```

Seed the UI from in-memory defaults immediately, then reconcile with
`await Store.load(key)` — never block first paint on storage.

Also unavailable in the sandbox: network requests (`connect-src 'self'` from
an opaque origin blocks everything), top-level navigation, popups, and the
parent page. Build apps that are complete on their own.

---

## 8. Before you call `present_app`

- [ ] Opens at 360 px with no horizontal scroll
- [ ] Every button/link/input has hover **and** focus-visible styling
- [ ] Every text/background pair reaches 4.5:1
- [ ] No placeholder copy anywhere; seeded with realistic data
- [ ] No external origin except cdnjs
- [ ] No unguarded `localStorage` / `cookie` / `fetch`
- [ ] The signature element appears at least three times
- [ ] Nothing on the anti-slop checklist ticks
