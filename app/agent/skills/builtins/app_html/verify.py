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
            return "the code checks out"
        return "could not be checked here"

    def as_error(self) -> str:
        """The model-facing block. Every line is a thing to go and fix."""
        lines = [f.as_text() for f in self.findings]
        return "\n".join(f"  - {line}" for line in lines)


# ── Script extraction ─────────────────────────────────────────────────

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
        return report
    except Exception:  # noqa: BLE001 — a missing browser must not fail a build
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


async def _smoke(html: str, report: Report) -> Report:
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except ImportError:  # pragma: no cover - image always ships one of the two
        from patchright.async_api import async_playwright  # type: ignore

    wrapped = runtime.wrap_for_runtime(html)
    errors: List[Finding] = []
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
        browser = await pw.chromium.launch(
            headless=True,
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
            # One frame of input. A game that only wires its handlers on
            # keydown would otherwise be declared healthy without ever
            # having executed its loop.
            try:
                await page.keyboard.press("ArrowRight")
                await page.mouse.click(195, 500)
                await page.wait_for_timeout(400)
            except Exception:  # noqa: BLE001
                pass
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
        return report

    report.findings.extend(errors)
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
    return report
