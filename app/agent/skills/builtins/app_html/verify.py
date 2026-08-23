"""Does the app actually run? — the check that has to pass before present_app.

Round 18. The snake game shipped, the card said "100% · App built", and the
page opened to a perfectly styled board that did nothing at all: the CSS had
rendered and the script had died on its first statement. Nothing in the
pipeline had ever looked at the JavaScript. ``validate_html`` checks that the
document is a document and that it loads code only from cdnjs; ``bash_app``
checks whatever the model thought to check, which after a confident
generation is usually ``wc -c``. A build could therefore be reported
finished, in green, by machinery none of whose steps had any opinion about
whether the thing worked.

Two checks, in cost order:

* :func:`check_syntax` — extract every inline ``<script>`` and run
  ``node --check`` over it. Milliseconds, no network, catches the whole
  truncated-generation class (an unclosed brace, a stray backtick, a
  half-written function) which is what a "renders but does nothing" page
  usually is.
* :func:`smoke_test` — load the document in a headless browser, exactly as
  the runner serves it (:func:`runtime.wrap_for_runtime`), and collect
  uncaught exceptions for a second and a half. This is the only check that
  can see a ``ReferenceError`` on line 214, and it is the one that proves the
  storage shim is doing its job.

Both are honest about not having run, which matters more here than the checks
themselves. A missing ``node``, a browser that will not launch, a cdnjs that
cannot be reached: each downgrades the gate to the passes that did complete
(:func:`_downgrade`), and :meth:`Report.summary` says which those were —
"opened it — no errors" only when the app was genuinely opened, "the code
checks out" when only the parser ran. A green report from machinery that never
looked is the failure this module exists to end, and reproducing it one layer
up would be the easiest possible way to fail at that.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import List, Optional, Tuple

from app.agent.skills.builtins.app_html import runtime

logger = logging.getLogger(__name__)

#: Hard ceiling on the syntax pass. `node --check` on 60 KB is ~80 ms; this
#: is a hang guard, not a budget.
SYNTAX_TIMEOUT_S = 10

#: Wall-clock ceiling for the browser pass, including launch. A build must
#: never be held up for longer than a person will wait for one.
SMOKE_TIMEOUT_S = 20

#: How long the page is left running after load, for timers, rAF loops and
#: whatever the app does on its first frame.
SMOKE_SETTLE_MS = 1500

#: Script types whose body is NOT JavaScript that node can parse.
_NON_JS_TYPES = frozenset({
    "text/babel", "text/jsx", "application/json", "application/ld+json",
    "importmap", "speculationrules", "text/template", "text/x-template",
    "text/html",
})

_SCRIPT_BLOCK_RE = re.compile(
    r"<script\b([^>]*)>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL
)
_ATTR_RE = re.compile(r"""(\w[\w-]*)\s*=\s*["']([^"']*)["']""")

#: A console error that is really a network fact, not a defect in the app.
#: cdnjs being slow or blocked in the build container must not fail a build
#: that will load it fine from the user's browser.
_NETWORK_NOISE_RE = re.compile(
    r"failed to load resource|net::ERR_|ERR_NAME_NOT_RESOLVED|"
    r"the server responded with a status|favicon",
    re.IGNORECASE,
)

#: Console errors that mean the code is broken, as opposed to the app having
#: chosen to log something.
#:
#: `page.on("console")` fires for every `console.error` the app makes, and an
#: app is entitled to make them — "Could not parse that amount", a caught
#: fetch failure, React's own warnings. Refusing to publish over one would
#: make the gate a nuisance that gets turned off. These four shapes are what
#: a JavaScript engine says when the program does not work, and are worth
#: stopping a publish for even though the exception itself was swallowed.
_REAL_BREAKAGE_RE = re.compile(
    r"is not defined|is not a function|Cannot read propert|"
    r"Cannot access .* before initialization|Uncaught ",
    re.IGNORECASE,
)

#: Keyboard instructions rendered as UI on a touch device (round 23). The
#: recorded Snake printed `ARROWS / WASD · SPACE PAUSES` inside a phone app —
#: instructions for hardware the device does not have, in the slot where the
#: player looks for how to play. Deliberately narrow, because a false positive
#: here refuses a working app: `wasd` and "arrow keys" cannot mean anything
#: else, while a bare "arrows" is a word an archery game is entitled to
#: ("Arrows left: 5") and is left to the visual review.
_KEYBOARD_HINT_RE = re.compile(
    r"\bwasd\b|\barrow keys\b|\bspace ?bar\b|"
    r"\bpress (?:space|enter|any key)\b|"
    r"\bspace (?:to )?(?:pause|start|jump|shoot|fire)s?\b|"
    r"\buse (?:the |your )?keyboard\b",
    re.IGNORECASE,
)

#: Text that means the app reached a state the user LOST, as opposed to any
#: other thing an app might say. Deliberately narrow: these are terminal
#: verdicts, not moods. "Time left", "Score", "Round 2" are all states an app
#: is entitled to be in on load; "GAME OVER" is not.
_TERMINAL_TEXT_RE = re.compile(
    r"game\s*over|you\s+(?:lose|lost)|time'?s?\s+up|out\s+of\s+time|"
    r"you\s+ran\s+out|final\s+score|better\s+luck",
    re.IGNORECASE,
)

#: How long the untouched page is watched for a terminal state, measured from
#: `domcontentloaded`. Must exceed the settle above — the whole point is to
#: keep watching after the frame where a healthy app has finished painting.
#: 3 s is chosen from the failure it exists to catch: a 20-cell board at
#: 175 ms/tick is over at 1.75 s, and doubling that leaves room for a slower
#: first tick without waiting long enough to annoy anyone.
SELF_END_WATCH_MS = 3000


@dataclass
class Finding:
    """One thing wrong with the app, in words the model can act on."""
    kind: str                      # "syntax" | "runtime" | "load"
    message: str
    line: Optional[int] = None

    def as_text(self) -> str:
        where = f" (line {self.line})" if self.line else ""
        return f"{self.message}{where}"


@dataclass
class Report:
    findings: List[Finding] = field(default_factory=list)
    #: Which passes actually ran. A pass that could not run is absent, and
    #: callers must not read its absence as a pass.
    ran: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    #: PNG of the app at 390×844, after the start control was pressed —
    #: i.e. the screen a person actually plays with, not the title card.
    #: Round 20: this is what `vision.review_screenshot` looks at. None when
    #: the browser pass did not run, which is exactly when the caller must
    #: not claim the app was looked at.
    screenshot: Optional[bytes] = None
    #: PNG of the app at first paint — the start screen, i.e. the app's FACE.
    #: Round 23: this is what the card's preview shows (`store.write_preview`
    #: at publish). Separate from `screenshot` because the two answer
    #: different questions: the reviewer must judge the screen that was
    #: played; a card must show the screen a person meets first.
    cover: Optional[bytes] = None
    #: What `runtime.audio_unlock` recorded: contexts made, contexts running,
    #: whether a gesture was seen, `<audio>` play failures, CSP refusals.
    audio: Optional[dict] = None
    #: WHY the browser pass was downgraded, when it was — one short clause,
    #: e.g. "no usable browser". Round 23 on-device: the fleet ran with a
    #: stale chromium mount for days and every card said "the code checks
    #: out" — honest, but indistinguishable from a build that simply had no
    #: JS to run. The reason makes a dead gate legible on the card itself,
    #: where an operator actually looks, instead of only in a container log.
    downgrade_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.findings

    def blocking(self) -> List[Finding]:
        return list(self.findings)

    def summary(self) -> str:
        """One human line. Used as the verify step's detail.

        "no problems found" is reserved for a run where the app was actually
        OPENED. A syntax-only pass has proved the code parses and nothing
        more, and saying otherwise would recreate — one layer up — the exact
        overclaim this module exists to stop: a green report from machinery
        that never looked.
        """
        if self.findings:
            n = len(self.findings)
            return f"found {n} problem{'' if n == 1 else 's'} to fix"
        if "runtime" in self.ran:
            return "opened it — no errors"
        if "syntax" in self.ran:
            if self.downgrade_reason:
                return f"the code checks out — couldn't open it ({self.downgrade_reason})"
            return "the code checks out"
        return "could not be checked here"

    def as_error(self) -> str:
        """The model-facing block. Every line is a thing to go and fix."""
        lines = [f.as_text() for f in self.findings]
        return "\n".join(f"  - {line}" for line in lines)


# ── Browser discovery ─────────────────────────────────────────────────

#: Names tried on PATH, in preference order. Brave first: the fleet's stated
#: browser (2026-08-23 — the patchright chromium mount is gone for good; the
#: platform's internet layer is the Brave Search API, and any browser binary
#: a host carries is Brave). Chrome/Chromium follow for dev machines.
_BROWSER_BIN_NAMES = (
    "brave-browser", "brave-browser-stable", "brave",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
)

#: Absolute fallbacks for installs that do not touch PATH.
_BROWSER_BIN_PATHS = (
    "/usr/bin/brave-browser",
    "/opt/brave.com/brave/brave-browser",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def find_browser_bin() -> Optional[str]:
    """A real installed browser to hand to playwright as ``executable_path``.

    ``TOUP_BROWSER_BIN`` wins outright when it points at something that
    exists, so a mounted binary at any path works without a rebuild. Returns
    None when nothing is installed — the caller then lets playwright try its
    own bundled chromium, which is the legacy path and absent on the fleet.
    """
    env = (os.environ.get("TOUP_BROWSER_BIN") or "").strip()
    if env and os.path.exists(env):
        return env
    for name in _BROWSER_BIN_NAMES:
        p = shutil.which(name)
        if p:
            return p
    for p in _BROWSER_BIN_PATHS:
        if os.path.exists(p):
            return p
    return None


# ── Script extraction ─────────────────────────────────────────────────

def _downgrade_reason_of(e: BaseException) -> str:
    """The exception, as one clause a card can carry.

    The two shapes worth naming are the two infra faults that have actually
    taken this gate down: no launchable browser anywhere (the fleet's normal
    state since the patchright mount was retired — first seen as the
    2026-08-19 stale-revision finding), and playwright itself missing from
    the image. Both are host/image facts a build cannot fix, so the wording
    points at the machine, not the app.
    """
    text = str(e).split("\n")[0].strip()
    if "Executable doesn't exist" in text or "executable doesn't exist" in text:
        return "no browser on this host"
    if isinstance(e, ImportError):
        return "playwright not installed"
    return (text[:57] + "…") if len(text) > 58 else (text or type(e).__name__)


def _downgrade(report: "Report", pass_name: str) -> None:
    """Move a pass from *ran* to *skipped*.

    Both lists, always, and never one without the other: `ran` is what
    `Report.summary` reads to decide whether it may say "opened it — no
    errors", so a pass that stopped halfway and stayed in `ran` would report
    a clean run it did not finish.
    """
    if pass_name in report.ran:
        report.ran.remove(pass_name)
    if pass_name not in report.skipped:
        report.skipped.append(pass_name)


def inline_scripts(html: str) -> List[Tuple[int, str, str]]:
    """``(start_line, script_type, code)`` for every inline ``<script>``.

    ``start_line`` is 1-based and counted in the WHOLE document, so a syntax
    error node reports at "line 12 of this block" can be handed back to the
    model as its real line in the file — the only form of it the model can
    use, because ``edit_app_file`` matches against the file.
    """
    out: List[Tuple[int, str, str]] = []
    for m in _SCRIPT_BLOCK_RE.finditer(html):
        attrs = dict(
            (k.lower(), v.lower()) for k, v in _ATTR_RE.findall(m.group(1) or "")
        )
        if attrs.get("src"):
            continue  # external file; nothing local to parse
        stype = (attrs.get("type") or "").strip()
        if stype in _NON_JS_TYPES:
            continue
        code = m.group(2)
        if not code.strip():
            continue
        start_line = html.count("\n", 0, m.start(2)) + 1
        out.append((start_line, stype or "text/javascript", code))
    return out


def _parse_node_error(stderr: str, base_line: int) -> Finding:
    """Turn ``node --check`` output into one sentence plus a real file line.

    Node prints the offending source line, a caret, a blank line and then
    ``SyntaxError: <what>``. The path it prints is our temp file, which must
    never reach the model — the line number is the only part worth keeping,
    and only after it is translated back into the app file's own numbering.
    """
    text = (stderr or "").strip()
    line_no: Optional[int] = None
    m = re.search(r"[^\s:]+:(\d+)\b", text)
    if m:
        try:
            line_no = base_line + int(m.group(1)) - 1
        except ValueError:
            line_no = None
    msg = ""
    for candidate in text.splitlines():
        candidate = candidate.strip()
        if candidate.startswith(("SyntaxError:", "ReferenceError:", "TypeError:")):
            msg = candidate
            break
    if not msg:
        # Last resort: the final non-empty line, minus anything path-shaped.
        tail = [ln.strip() for ln in text.splitlines() if ln.strip()]
        msg = tail[-1] if tail else "the script could not be parsed"
    msg = re.sub(r"(/[^\s:]+)+", "the script", msg)
    return Finding(kind="syntax", message=f"JavaScript won't parse — {msg}", line=line_no)


async def check_syntax(html: str, *, timeout: int = SYNTAX_TIMEOUT_S) -> Report:
    """Parse every inline script with node. Fast, offline, deterministic."""
    report = Report()
    node = shutil.which("node") or shutil.which("nodejs")
    blocks = inline_scripts(html)
    if not blocks:
        report.ran.append("syntax")
        return report
    if not node:
        report.skipped.append("syntax")
        return report

    report.ran.append("syntax")
    with tempfile.TemporaryDirectory(prefix="toup-syntax-") as tmp:
        for idx, (base_line, stype, code) in enumerate(blocks):
            # `import`/`export` are a SyntaxError under the CommonJS goal, so
            # a module block has to be checked as a module — which node
            # decides from the extension, not from a flag.
            ext = ".mjs" if stype == "module" or re.search(
                r"^\s*(?:import|export)\b", code, re.MULTILINE
            ) else ".js"
            path = os.path.join(tmp, f"block{idx}{ext}")
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(code)
                proc = await asyncio.create_subprocess_exec(
                    node, "--check", path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    _out, err = await asyncio.wait_for(
                        proc.communicate(), timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    _downgrade(report, "syntax")
                    return report
            except OSError:
                logger.debug("[app_html] syntax check could not run", exc_info=True)
                _downgrade(report, "syntax")
                return report
            if proc.returncode:
                report.findings.append(
                    _parse_node_error(err.decode("utf-8", "replace"), base_line)
                )
    return report


# ── Browser smoke test ────────────────────────────────────────────────

def counts_as_breakage(message: str, *, from_console: bool) -> bool:
    """Is this browser message a reason to refuse to publish?

    Separated out so the decision can be tested without a browser — which
    matters more here than usual, because both ways of getting it wrong are
    silent. Too strict and every React app is refused on a box that cannot
    reach cdnjs; too loose and the gate passes an app that does nothing.
    """
    message = (message or "").strip()
    if not message:
        return False
    if _NETWORK_NOISE_RE.search(message):
        return False
    # An uncaught exception is always breakage. A `console.error` is only
    # breakage when it names some — apps log deliberately.
    if from_console:
        return bool(_REAL_BREAKAGE_RE.search(message))
    return True


def smoke_enabled() -> bool:
    """Off only if an operator turns it off. A build that cannot be run is
    still built, so this is a quality gate, not a dependency."""
    return (os.environ.get("TOUP_APP_SMOKE_TEST", "1") or "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


async def smoke_test(html: str, *, timeout: int = SMOKE_TIMEOUT_S) -> Report:
    """Open the app in a throwaway browser and collect what it throws.

    Deliberately NOT the agent's own browser: that one is a persistent,
    logged-in profile with the user's cookies in it, and pointing
    model-authored script at it would be the exact thing the artifact sandbox
    exists to prevent. This launches its own headless instance, uses it once
    and closes it.

    The document under test is the WRAPPED one — the same bytes the runner
    serves — so a pass means the storage shim and the error hooks are part of
    what passed.
    """
    report = Report()
    if not smoke_enabled():
        report.skipped.append("runtime")
        return report
    try:
        return await asyncio.wait_for(_smoke(html, report), timeout=timeout)
    except asyncio.TimeoutError:
        # A page that never finished loading is not a verdict on the app —
        # cdnjs may simply be slow from here.
        _downgrade(report, "runtime")
        report.downgrade_reason = "page never settled"
        return report
    except Exception as e:  # noqa: BLE001 — a missing browser must not fail a build
        # WARNING, not debug. A container where this never runs has silently
        # downgraded its strongest gate to a syntax check, and the only way
        # anyone finds out is if the line is loud enough to notice. (It is
        # also how a HOME that does not hold the browser cache announces
        # itself, which is a real way to lose this pass without breaking
        # anything else.)
        logger.warning(
            "[app_html] the browser check could not run — publishing on the "
            "syntax check alone", exc_info=True,
        )
        _downgrade(report, "runtime")
        report.downgrade_reason = _downgrade_reason_of(e)
        return report


async def _visible_text(page) -> str:
    """`innerText`, i.e. what a person can actually read.

    `innerText` and not `textContent`: a game-over panel is in the DOM from
    the first byte and hidden with `display:none`, so `textContent` says
    "GAME OVER" on a board nobody has touched and would fail every app that
    has an end state at all. `innerText` respects the cascade, which is the
    whole distinction being drawn.
    """
    try:
        return str(await page.evaluate("() => document.body ? document.body.innerText : ''"))
    except Exception:  # noqa: BLE001
        return ""


async def _ended_itself(page, first_text: str) -> Optional[str]:
    """Did the app declare the user finished, with nothing pressed?

    Returns the phrase it reached, or None. Polls rather than sleeping the
    full window so a fast self-end is reported with its own timing intact and
    a healthy app costs only the first poll.

    Two things keep this from firing on a working app. The text must be
    absent at first paint (an app may legitimately *display* "final score"),
    and it must be a terminal verdict rather than any word a game uses —
    "score", "lives", "round" are all fine to show at rest.
    """
    if _TERMINAL_TEXT_RE.search(first_text or ""):
        return None
    elapsed = SMOKE_SETTLE_MS
    while elapsed < SELF_END_WATCH_MS:
        hit = _TERMINAL_TEXT_RE.search(await _visible_text(page))
        if hit:
            return " ".join(hit.group(0).split())
        await page.wait_for_timeout(250)
        elapsed += 250
    hit = _TERMINAL_TEXT_RE.search(await _visible_text(page))
    return " ".join(hit.group(0).split()) if hit else None


#: Text on the control that begins an app. Anchored at the START of the label
#: so "Display settings" and "Restart tour" are not mistaken for one.
_START_CONTROL_RE = r"^\s*(play|start|begin|new game|go)\b"

#: JS that finds the start control and presses it. A real `.click()` on the
#: element, not a coordinate: the button sits inside an overlay whose position
#: depends on the app's own layout, and a blind click at a fixed point is how
#: this was being missed.
_PRESS_START_JS = """
(pattern) => {
  const re = new RegExp(pattern, 'i');
  const sel = 'button,[role="button"],a,input[type="button"],input[type="submit"]';
  for (const el of document.querySelectorAll(sel)) {
    const label = (el.innerText || el.value || el.getAttribute('aria-label') || '').trim();
    if (!label || !re.test(label)) continue;
    const box = el.getBoundingClientRect();
    if (!box.width || !box.height) continue;   // hidden: not the live one
    el.click();
    return label.slice(0, 40);
  }
  return null;
}
"""


async def _press_start(page) -> Optional[str]:
    """Press the control that begins the app, if the app has one.

    Round 18, second pass. The gate's original "one frame of input" was a
    keypress plus a click at a fixed point, which was enough while apps ran
    their loop on load. Then the design rule changed — anything that can be
    lost now opens on a START SCREEN — and the app's first real code path
    moved behind a gesture the blind click did not reliably land on. A Snake
    build published with `classList.add('snake', '')` on its first render,
    which throws in every browser, and the gate saw a clean page because
    `render()` had never been called.

    So the gate presses PLAY. A gate that got weaker the day apps got lazier
    is worse than no gate: it still reports "opened it — no errors".
    """
    try:
        return await page.evaluate(_PRESS_START_JS, _START_CONTROL_RE)
    except Exception:  # noqa: BLE001
        return None


#: Minimum rendered size, in CSS px, for anything a thumb has to hit. Apple
#: HIG 44pt, Material 48dp, WCAG 2.2 SC 2.5.5 44×44. The design skill has said
#: this since round 19 and nothing measured it — a 32 px D-pad throws nothing
#: and renders perfectly, which is exactly why it kept shipping.
MIN_TARGET_PX = 44

#: Below this, text is unreadable on a phone regardless of contrast.
MIN_TEXT_PX = 12

#: Most findings of one kind to report. A grid of forty undersized keys is one
#: mistake, and forty lines of it would bury everything else in the report.
MAX_PER_KIND = 4

#: Measures the laid-out page the way a person meets it. Everything here is
#: read from `getBoundingClientRect` and `getComputedStyle` AFTER the app has
#: rendered — the whole point is that it is the rendered geometry, not the
#: stylesheet, that decides whether a control can be hit.
_LAYOUT_JS = """
(opts) => {
  const SEL = 'button,[role="button"],a,input,select,textarea,summary,[onclick]';
  const seen = [], small = [], tiny = [];
  const vw = window.innerWidth, vh = window.innerHeight;

  const label = (el) => (
    (el.getAttribute('aria-label') || el.innerText || el.value ||
     el.getAttribute('title') || el.className || el.tagName) || ''
  ).toString().trim().replace(/\\s+/g, ' ').slice(0, 32);

  for (const el of document.querySelectorAll(SEL)) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') continue;
    const b = el.getBoundingClientRect();
    if (!b.width || !b.height) continue;              // not laid out
    if (b.bottom < 0 || b.top > vh * 3) continue;     // far off-screen
    // WCAG 2.5.8 exempts a link inline in a sentence — a 20px-tall <a> inside
    // a paragraph is typography, not a control, and failing it would make the
    // gate a nuisance that gets turned off.
    if (el.tagName === 'A' && cs.display.indexOf('inline') === 0) continue;
    if (el.tagName === 'INPUT' &&
        ['hidden','radio','checkbox'].indexOf(el.type) >= 0) continue;
    seen.push(1);
    const side = Math.min(b.width, b.height);
    if (side + 0.5 < opts.minTarget) {
      small.push({ label: label(el), w: Math.round(b.width), h: Math.round(b.height) });
    }
  }

  for (const el of document.querySelectorAll('body *')) {
    if (!el.childNodes.length) continue;
    let hasText = false;
    for (const n of el.childNodes) {
      if (n.nodeType === 3 && n.textContent.trim().length > 3) { hasText = true; break; }
    }
    if (!hasText) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    const px = parseFloat(cs.fontSize);
    if (px && px + 0.5 < opts.minText) {
      tiny.push({ label: (el.innerText || '').trim().slice(0, 24), px: Math.round(px) });
    }
  }

  const doc = document.documentElement;
  return {
    controls: seen.length,
    small: small.slice(0, opts.max),
    smallTotal: small.length,
    tiny: tiny.slice(0, opts.max),
    tinyTotal: tiny.length,
    overflowPx: Math.max(0, Math.round(doc.scrollWidth - doc.clientWidth)),
    vw, vh,
  };
}
"""


def layout_findings(m: dict) -> List[Finding]:
    """Turn one layout measurement into findings the model can act on.

    Split out from the browser so the thresholds can be tested without one —
    the same reason `counts_as_breakage` is its own function. Every message
    names the element AND the number, because "make your buttons bigger" is not
    something anyone can act on and "PLAY is 120×32, needs 44 tall" is.
    """
    out: List[Finding] = []
    for s in m.get("small") or []:
        out.append(Finding(kind="layout", message=(
            f"the control “{s['label']}” renders {s['w']}×{s['h']}px — under the "
            f"{MIN_TARGET_PX}px minimum a thumb can reliably hit. Give it "
            f"min-width/min-height:{MIN_TARGET_PX}px (a control used "
            f"continuously — a D-pad key, a paddle — wants 64-88px)."
        )))
    extra = int(m.get("smallTotal") or 0) - len(m.get("small") or [])
    if extra > 0:
        out.append(Finding(kind="layout", message=(
            f"…and {extra} more control{'' if extra == 1 else 's'} under "
            f"{MIN_TARGET_PX}px. Fix the shared rule, not each element."
        )))
    for t in m.get("tiny") or []:
        out.append(Finding(kind="layout", message=(
            f"the text “{t['label']}” renders at {t['px']}px — under {MIN_TEXT_PX}px "
            f"it is unreadable on a phone. Body text wants 16px."
        )))
    if int(m.get("overflowPx") or 0) > 2:
        out.append(Finding(kind="layout", message=(
            f"the page scrolls sideways by {m['overflowPx']}px at "
            f"{m.get('vw')}px wide — something is wider than the screen. Find the "
            f"fixed width and make it max-width:100%."
        )))
    return out


def keyboard_hint_findings(visible_text: str) -> List[Finding]:
    """Keyboard instructions on a touch screen, as findings. Testable dry.

    One finding no matter how many phrases match — the fix is one edit (hide
    or delete the hint line), and three findings for one sentence would bury
    the rest of the report.
    """
    hit = _KEYBOARD_HINT_RE.search(visible_text or "")
    if not hit:
        return []
    phrase = " ".join(hit.group(0).split())
    return [Finding(kind="layout", message=(
        f"the screen shows keyboard instructions (“{phrase}”) — this app runs "
        f"on a phone, where there is no keyboard. Keyboard support may exist, "
        f"but its instructions must not render as UI on touch: delete the "
        f"hint, or show it only after a physical key press is detected."
    ))]


#: Elements that never carry an end tag — suppressing on their start tag
#: would leak the suppression for the rest of the document.
_VOID_TAGS = frozenset(
    "area base br col embed hr img input link meta source track wbr".split()
)

_INLINE_HIDDEN_RE = re.compile(r"display\s*:\s*none", re.I)


class _StaticTextExtractor(HTMLParser):
    """The markup's at-rest text — the browserless stand-in for innerText.

    Scripts, styles and elements hidden at rest (`hidden`, inline
    `display:none`) contribute nothing, so an app that follows this gate's
    own advice — keep the keyboard hint out of view until a physical key
    press — is not re-flagged by the fallback scan. Class-based hiding is
    invisible to a parser and deliberately NOT modelled: a hint sitting in a
    class-hidden game screen still reaches the phone's screen when that
    screen shows, which is exactly what the live scan would catch.
    """

    _SKIP = frozenset(("script", "style", "template", "noscript"))

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: List[Tuple[str, bool]] = []
        self._suppress = 0
        self.parts: List[str] = []

    def handle_starttag(self, tag, attrs):  # noqa: ANN001 - HTMLParser API
        if tag in _VOID_TAGS:
            return
        hidden = any(
            k == "hidden" or (k == "style" and v and _INLINE_HIDDEN_RE.search(v))
            for k, v in attrs
        )
        suppressed = tag in self._SKIP or hidden
        self._stack.append((tag, suppressed))
        if suppressed:
            self._suppress += 1

    def handle_endtag(self, tag):  # noqa: ANN001 - HTMLParser API
        # Tolerant unwinding: generated HTML is usually well-formed, but a
        # stray close must not wedge the suppression counter forever.
        while self._stack:
            t, s = self._stack.pop()
            if s:
                self._suppress -= 1
            if t == tag:
                break

    def handle_data(self, data):  # noqa: ANN001 - HTMLParser API
        if not self._suppress and data.strip():
            self.parts.append(data.strip())


def static_visible_text(html: str) -> str:
    """Testable dry, and the only text source when no browser exists."""
    p = _StaticTextExtractor()
    try:
        p.feed(html or "")
        p.close()
    except Exception:  # noqa: BLE001 - a parse failure is not a finding
        return ""
    return "\n".join(p.parts)


def layout_enabled() -> bool:
    """Off only if an operator turns it off, like the smoke test itself."""
    return (os.environ.get("TOUP_APP_LAYOUT_GATE", "1") or "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


# ── Did the app actually make a sound? ────────────────────────────────

def audio_findings(state: Optional[dict]) -> List[Finding]:
    """Turn ``window.__TOUP_AUDIO`` into findings. Testable without a browser.

    Round 20. A silent app is the hardest defect in this pipeline to see,
    because silence is what success also looks like from every other
    instrument: the script parses, nothing throws, the layout measures fine.
    The runtime shim therefore counts what happened, and this reads the count.

    Three ways an app ends up quiet, and all three were reaching users:

    * **The sandbox refused the sound.** ``media-src`` was missing from the
      artifact CSP, so every ``data:``/``blob:`` sound was blocked and the
      element reported ``NotSupportedError`` — the browser's phrase for a
      corrupt file. Named explicitly here, because a policy problem that
      reads as a broken asset is one a model will "fix" by re-encoding the
      asset forever.
    * **Nothing ever resumed the context.** The shim now resumes on the first
      gesture, so reaching this state after the gate has pressed PLAY and
      clicked means the app is doing something the shim cannot reach —
      usually a context it builds per sound, or one it suspends and never
      restores.
    * **``play()`` was rejected** and the app never noticed, because
      generated code does not await that promise.

    Deliberately silent about an app that makes no sound at all: most apps do
    not, and inventing a finding for them would make this gate a nuisance.
    """
    if not isinstance(state, dict):
        return []
    out: List[Finding] = []
    blocked = [str(b) for b in (state.get("blocked") or []) if b]
    if blocked:
        out.append(Finding(kind="audio", message=(
            f"the sandbox blocked something this app loads: {blocked[0]}. "
            f"Sounds must be inline — a Web Audio oscillator, or a data: URI "
            f"— never a fetch."
        )))
    contexts = int(state.get("contexts") or 0)
    running = int(state.get("running") or 0)
    unlocked = bool(state.get("unlocked"))
    failures = [str(f) for f in (state.get("failures") or []) if f]
    if contexts and unlocked and not running:
        out.append(Finding(kind="audio", message=(
            f"this app builds audio ({contexts} AudioContext) but none of it "
            f"is running after a tap, so it makes no sound. Create the "
            f"AudioContext inside the first input handler, and call "
            f"`ctx.resume()` at the top of every handler that plays "
            f"something."
        )))
    if failures and not out:
        out.append(Finding(kind="audio", message=(
            f"a sound failed to play: {failures[0]}. Play sounds from a "
            f"handler for a real tap or key press, not on load or on a timer."
        )))
    return out


async def _measure_audio(page) -> Tuple[Optional[dict], List[Finding]]:
    """Read the shim's record, forcing a fresh count first."""
    try:
        state = await page.evaluate(
            "() => (window.__TOUP_AUDIO && window.__TOUP_AUDIO.check)"
            " ? JSON.parse(JSON.stringify(window.__TOUP_AUDIO.check()))"
            " : (window.__TOUP_AUDIO ? JSON.parse(JSON.stringify(window.__TOUP_AUDIO)) : null)"
        )
    except Exception:  # noqa: BLE001 - a measurement that cannot run is not a verdict
        logger.debug("[app_html] audio measurement failed", exc_info=True)
        return None, []
    if not isinstance(state, dict):
        return None, []
    return state, audio_findings(state)


#: Cap on the PNG handed onward. A 390×844 screenshot is 40–120 KB; anything
#: past this is a full-bleed photograph, and it would be base64'd into an LLM
#: request. Dropped rather than downscaled: a resized picture is not the
#: screen, and the review is supposed to be looking at the screen.
MAX_SCREENSHOT_BYTES = 3 * 1024 * 1024


async def _capture(page) -> Optional[bytes]:
    """A PNG of what is on screen, or None. Never fails the run."""
    try:
        png = await page.screenshot(type="png")
    except Exception:  # noqa: BLE001 - a picture that cannot be taken is not a verdict
        logger.debug("[app_html] screenshot failed", exc_info=True)
        return None
    if not png or len(png) > MAX_SCREENSHOT_BYTES:
        return None
    return png


async def _measure_layout(page) -> List[Finding]:
    if not layout_enabled():
        return []
    try:
        m = await page.evaluate(
            _LAYOUT_JS,
            {"minTarget": MIN_TARGET_PX, "minText": MIN_TEXT_PX, "max": MAX_PER_KIND},
        )
    except Exception:  # noqa: BLE001 - a measurement that cannot run is not a verdict
        logger.debug("[app_html] layout measurement failed", exc_info=True)
        return []
    if not isinstance(m, dict) or not m.get("controls"):
        # No laid-out controls at all: either the app is a pure display, or it
        # never rendered. Both are somebody else's finding, not this one's.
        return []
    return layout_findings(m)


async def _smoke(html: str, report: Report) -> Report:
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except ImportError:  # pragma: no cover - image always ships one of the two
        from patchright.async_api import async_playwright  # type: ignore

    # The wrapper AND the policy. `wrap_for_runtime` alone gave the gate a
    # browser more permissive than the user's, which is how a sound the runner
    # refuses could pass a check that claimed to have opened the app. See
    # `runtime.wrap_for_verification` for the measurement.
    wrapped = runtime.wrap_for_verification(html)
    errors: List[Finding] = []
    audio: List[Finding] = []
    #: External requests that never arrived. An app whose library did not load
    #: is not an app that is broken — it is a verdict this container is not in
    #: a position to give, and blaming the model for its own lack of egress
    #: would refuse every React app on a box that cannot reach cdnjs.
    blocked: List[str] = []

    def _note(kind: str, message: str, *, console: bool = False) -> None:
        message = (message or "").strip()
        if not counts_as_breakage(message, from_console=console):
            return
        if any(e.message == message for e in errors):
            return
        if len(errors) < 5:
            errors.append(Finding(kind=kind, message=message))

    pw = await async_playwright().start()
    try:
        # An installed browser (Brave on the fleet) beats the bundled
        # chromium, which no longer exists there. `executable_path=None`
        # keeps the bundle as the dev-machine fallback.
        browser = await pw.chromium.launch(
            headless=True,
            executable_path=find_browser_bin(),
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        try:
            page = await browser.new_page(viewport={"width": 390, "height": 844})
            page.on("pageerror", lambda e: _note("runtime", str(e).split("\n")[0]))
            page.on(
                "console",
                lambda m: _note("runtime", m.text, console=True)
                if m.type == "error" else None,
            )
            page.on("requestfailed", lambda r: blocked.append(r.url[:120]))
            report.ran.append("runtime")
            # `domcontentloaded`, not `load`: `load` waits on every cdnjs
            # fetch, and a slow CDN would spend the whole budget before a
            # single line of the app's own script had run.
            await page.set_content(wrapped, wait_until="domcontentloaded")
            # First paint, before anything has had time to run itself out.
            # An app is allowed to SAY "final score" on a leaderboard; the
            # defect is a terminal verdict that ARRIVES while nobody is
            # touching it, so the baseline is what makes the two separable.
            first_text = await _visible_text(page)
            await page.wait_for_timeout(SMOKE_SETTLE_MS)
            # Everything up to here has been UNTOUCHED — no key, no click.
            # That is the only window in which this question can be asked, so
            # it is asked before the input frame below and never after.
            self_ended = await _ended_itself(page, first_text)
            # Measure the START SCREEN's geometry before pressing anything —
            # its PLAY button is a control a thumb has to hit too — then
            # measure again below, once the app proper is on screen, because
            # that is where the D-pad lives.
            layout = await _measure_layout(page)
            # The cover: the settled, untouched first screen — the one a
            # person meets when the app opens, which is what a card preview
            # must show. Taken here, before any input, for exactly that
            # reason (the played screenshot below shows a mid-game state).
            report.cover = await _capture(page)
            # Now DRIVE it. An app that opens on a start screen has run almost
            # none of its own code yet, so everything above this line has
            # verified a title and a button.
            if await _press_start(page):
                # Long enough for a game loop to take several ticks, which is
                # where a first-render throw actually lands.
                await page.wait_for_timeout(1200)
            try:
                await page.keyboard.press("ArrowRight")
                await page.mouse.click(195, 500)
                await page.wait_for_timeout(400)
            except Exception:  # noqa: BLE001
                pass
            # The playing screen. Its controls are the ones used hundreds of
            # times, so this measurement is the one that matters most — and it
            # is only reachable after the start control has been pressed.
            for f in await _measure_layout(page):
                if not any(f.message == g.message for g in layout):
                    layout.append(f)
            # Keyboard hints, on BOTH screens: the recorded Snake printed its
            # `ARROWS / WASD` line on the playing screen, but a start screen
            # ("press SPACE to begin") is just as wrong on a phone.
            playing_text = await _visible_text(page)
            layout.extend(
                keyboard_hint_findings(f"{first_text}\n{playing_text}")
            )
            # Sound, measured rather than assumed — see `audio_findings`. Read
            # AFTER the gestures above, because "was a gesture seen" is half of
            # what the answer depends on.
            report.audio, audio = await _measure_audio(page)
            # And the picture. This is the same screen the two measurements
            # above were taken from, which is the property that matters: a
            # reviewer arguing with a layout finding must be looking at the
            # layout that produced it.
            report.screenshot = await _capture(page)
            if self_ended:
                _note("behaviour", (
                    f"the app reached “{self_ended}” on its own, "
                    f"{SELF_END_WATCH_MS // 1000}s after opening, with nothing "
                    "pressed — so the user sees it already over. Do not start a "
                    "clock, a loop or a countdown at load: open on a start "
                    "screen and begin in the start control's handler."
                ))
        finally:
            await browser.close()
    finally:
        await pw.stop()

    if blocked:
        # Inconclusive, not clean and not broken. Half the page never ran, so
        # neither an absence of errors nor a presence of them means anything
        # about the app the user will load.
        logger.warning(
            "[app_html] the browser check could not reach %s — not counting "
            "its result", blocked[0],
        )
        _downgrade(report, "runtime")
        # The picture goes too. A screenshot of a page whose stylesheet never
        # arrived would be handed to the visual review as though it were the
        # app, and every finding it produced would be about this container's
        # egress rather than about anything the model wrote. The cover for
        # the same reason: a card must not wear a picture of a page whose
        # stylesheet never arrived.
        report.screenshot = None
        report.cover = None
        return report

    report.findings.extend(errors)
    # Layout findings ride the same report, so an undersized D-pad refuses the
    # publish exactly the way a ReferenceError does. Added AFTER the blocked-CDN
    # downgrade above: a page whose stylesheet never arrived has no geometry
    # worth measuring, and blaming the model for this container's lack of
    # egress is the one way to make the gate worth turning off.
    report.findings.extend(layout)
    report.findings.extend(audio)
    return report


async def verify_app(html: str, *, deep: bool = True) -> Report:
    """Syntax first, then the browser — and stop at the first that fails.

    A page whose script does not parse will throw the same SyntaxError in the
    browser, so running both would report one defect twice and charge a
    browser launch for the privilege.
    """
    report = await check_syntax(html)
    if report.findings or not deep:
        return report
    deep_report = await smoke_test(html)
    report.findings.extend(deep_report.findings)
    report.ran.extend(deep_report.ran)
    report.skipped.extend(deep_report.skipped)
    # The picture and the audio record belong to the run that produced them.
    # `_smoke` clears the screenshot when its own result was downgraded, so a
    # None here is the honest "there is nothing to look at" the visual review
    # needs in order not to approve a screen it never saw.
    report.screenshot = deep_report.screenshot
    report.cover = deep_report.cover
    report.audio = deep_report.audio
    report.downgrade_reason = deep_report.downgrade_reason
    if "runtime" not in report.ran:
        # The hint scan lived inside the browser pass, so a browserless fleet
        # (the current one: no playwright mount, Brave Search API is the
        # whole internet layer) shipped keyboard-hint apps unchallenged. The
        # markup's at-rest text answers the same question dry.
        report.findings.extend(keyboard_hint_findings(static_visible_text(html)))
    return report
