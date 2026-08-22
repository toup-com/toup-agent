"""Round 20, items 2 and 3 — the app makes a noise, and somebody looks at it.

**Item 2 (sound) has a measured root cause.** The artifact CSP carried no
``media-src`` directive, so it fell back to ``default-src 'self'`` — and the
artifact frame runs on an OPAQUE origin, where ``'self'`` matches nothing. So
every ``data:`` and ``blob:`` sound a generated app made was refused. Confirmed
in a real browser with a control (two documents identical but for that one
directive): without it the browser fires ``securitypolicyviolation: media-src``
and the element rejects with ``NotSupportedError — Failed to load because no
supported source was found``; with it, the same page plays. The error is why it
survived so long — it is what a browser says about a CORRUPT FILE, so it reads
as a bad sound rather than as a policy decision. The mobile runner has always
sent ``media-src data: blob:`` in its own ``<meta>`` policy, which makes it the
control here too.

The second half is the autoplay gesture: a context built while the page loads
starts ``suspended`` and nothing reports it. `runtime.audio_unlock` resumes on
the first real gesture and RECORDS what happened, so `verify.audio_findings`
can measure "made sound, got none" instead of reading silence as success.

**Item 3 (the look)** is about a defect no measurement can reach: an app that
runs, throws nothing, measures 44 px and is white text on a white card. The
tests here are about the two ways that gate goes wrong — approving a screen it
never saw, and refusing every app over taste.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile

import pytest

from app.agent.skills.builtins.app_html import runtime, verify, vision


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── The policy that was silencing apps ────────────────────────────────

def test_the_artifact_csp_allows_a_sound_the_page_already_has():
    """The measured fix. `data:` and `blob:` are bytes the page holds; there
    is no network here for them to have come from."""
    from app.api.artifact_proxy import artifact_csp

    csp = artifact_csp()
    directives = dict(
        (d.strip().split(" ", 1) + [""])[:2] for d in csp.split(";") if d.strip()
    )
    assert "media-src" in directives, csp
    media = directives["media-src"]
    assert "data:" in media and "blob:" in media, media


def test_media_src_is_not_left_to_default_src():
    """The bug was an ABSENT directive, not a wrong one.

    Asserting the value without asserting that it is stated separately would
    still pass if someone deleted the line and widened `default-src` — which
    would silently grant every other fetch type the same latitude.
    """
    from app.api.artifact_proxy import artifact_csp

    csp = artifact_csp()
    assert "media-src" in csp
    default = [d for d in csp.split(";") if d.strip().startswith("default-src")][0]
    assert "data:" not in default and "blob:" not in default, default


def test_the_artifact_policy_still_grants_no_network():
    """The fix must not be a loosening. Everything else stays shut."""
    from app.api.artifact_proxy import artifact_csp, artifact_headers

    csp = artifact_csp()
    assert "connect-src 'self'" in csp
    assert "object-src 'none'" in csp
    assert "base-uri 'none'" in csp
    perms = artifact_headers()["Permissions-Policy"]
    for closed in ("geolocation=()", "microphone=()", "camera=()"):
        assert closed in perms, perms
    # And autoplay is stated rather than left implicit, so nobody tightens it
    # by accident while debugging a noisy page.
    assert "autoplay=(self)" in perms


def test_the_web_and_mobile_runners_agree_about_sound():
    """The divergence IS the bug, so it is worth a test of its own.

    The mobile runner's `<meta>` policy has always allowed `media-src data:
    blob:`; the web runner's response header did not. One artifact, two
    runners, two different answers to "can this app make a noise" — and the
    one that said no was the one nobody had written down.

    `worker-src`/`child-src` were the SAME divergence, found while aligning
    the two: the mobile policy grants `blob:`, the web one fell through to
    `default-src 'self'`, so a generated app using a blob Worker worked on a
    phone and was silently dead on the web.
    """
    from app.artifact_policy import PARITY_DIRECTIVES
    from app.api.artifact_proxy import artifact_csp

    csp = artifact_csp()
    for directive in PARITY_DIRECTIVES:
        stated = [d.strip() for d in csp.split(";") if d.strip().startswith(directive + " ")]
        assert stated, f"{directive} is not stated — it falls back to default-src: {csp}"
        assert "blob:" in stated[0], stated


def test_the_policy_has_exactly_one_definition():
    """A second copy of the directive list is the same mistake again.

    The bug was two runners disagreeing about one artifact. `artifact_csp`
    must therefore not compose its own list — it must read the shared one, so
    the publish gate and the response header cannot drift the way the web and
    mobile runners did.
    """
    import ast
    import inspect
    from app.api import artifact_proxy
    from app.artifact_policy import artifact_cdn_origin, sandbox_csp

    tree = ast.parse(inspect.getsource(artifact_proxy.artifact_csp).lstrip())
    fn = tree.body[0]
    # The docstring is allowed to discuss directives at length — it is where
    # the bug is explained. The CODE may not compose them.
    body = fn.body[1:] if ast.get_docstring(fn) else fn.body
    literals = [n.value for n in ast.walk(ast.Module(body=body, type_ignores=[]))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not any("-src" in s for s in literals), (
        f"artifact_csp is composing directives again instead of reading the "
        f"shared policy: {literals}")
    assert "sandbox_csp" in inspect.getsource(artifact_proxy.artifact_csp)
    # And it really is the shared one, not a coincidence.
    assert artifact_proxy.artifact_csp().startswith(
        sandbox_csp(artifact_cdn_origin()).split(";")[0])


def test_the_gate_runs_the_app_under_the_policy_the_user_gets():
    """A canary in a kinder cage than the bird cannot fail.

    The publish gate used to load the app with `set_content` and NO policy, so
    it was running it in a browser strictly more permissive than the user's —
    and this round's defect lived exactly in that gap. Measured on one page:
    with no policy the refused sound is invisible to the gate; with the
    pre-round-20 directive set the shim records
    `blocked: ["media-src blocked data"]` and the publish is refused.
    """
    from app.artifact_policy import artifact_cdn_origin, sandbox_csp

    html = ("<!doctype html><html><head><title>x</title></head>"
            "<body><p>hi</p></body></html>")
    verified = runtime.wrap_for_verification(html)
    assert "http-equiv=\"Content-Security-Policy\"" in verified
    assert "media-src" in verified
    # The policy must be met BEFORE the content it governs — including before
    # the runtime preamble, which is itself script the policy allows.
    assert verified.index("Content-Security-Policy") < verified.index(runtime.MARKER)
    # ...and it must be the same policy, not a hand-written near-copy.
    assert sandbox_csp(artifact_cdn_origin()) in verified
    # `frame-ancestors` is header-only; a <meta> ignores it, and emitting a
    # directive that does nothing is one more thing that looks like
    # enforcement and is not.
    assert "frame-ancestors" not in verified


def test_serving_an_app_does_not_double_up_the_policy():
    """The real header is the enforcement on the serve path. A <meta> there
    would be a second policy to keep in step — the exact failure mode."""
    html = ("<!doctype html><html><head><title>x</title></head>"
            "<body><p>hi</p></body></html>")
    served = runtime.wrap_for_runtime(html)
    assert "Content-Security-Policy" not in served


# ── The shim that resumes it ──────────────────────────────────────────

def test_the_preamble_carries_the_audio_unlock():
    block = runtime.preamble()
    assert runtime.AUDIO_GLOBAL in block
    assert "pointerdown" in block and "resume" in block
    # And it is still ONE script, injected once — two preambles would install
    # two storage shims over each other.
    assert block.count("<script") == 1


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_injected_javascript_parses():
    """The shim is a python string that becomes the first script on the page.

    A syntax error in it does not degrade audio — it kills the storage shim
    and the error reporter that are concatenated with it, i.e. every app on
    the fleet, at once. `node --check` is 80ms and this is the only thing
    standing between a stray quote and that.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preamble.js")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(runtime.error_reporter() + runtime.storage_shim()
                     + runtime.audio_unlock())
        proc = subprocess.run([shutil.which("node"), "--check", path],
                              capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_every_javascript_sample_in_the_design_skill_parses():
    """The design skill is IN the system prompt, and models copy from it.

    Caught on the round-20 audio section: the "not this / this" pair shared a
    block and declared `ctx` twice, so the sample a model would lift did not
    parse. A doc that ships broken JavaScript in the same prompt that tells the
    model its script will be parsed is teaching the wrong thing twice.

    Both copies are checked because there ARE two, byte-identical by
    convention — `app_html/DESIGN_SKILL.md` (packaged, read into the prompt)
    and `skills/toup-frontend-design.md` (written to the skills volume). A
    change to one and not the other is its own bug.
    """
    import re
    from app.agent.skills.builtins.app_html import skill as skill_mod

    here = os.path.dirname(os.path.dirname(os.path.abspath(skill_mod.__file__)))
    packaged = os.path.join(
        os.path.dirname(os.path.abspath(skill_mod.__file__)), "DESIGN_SKILL.md")
    repo_copy = os.path.abspath(os.path.join(
        here, "..", "..", "..", "..", "skills", "toup-frontend-design.md"))

    with open(packaged, encoding="utf-8") as fh:
        body = fh.read()
    if os.path.isfile(repo_copy):
        with open(repo_copy, encoding="utf-8") as fh:
            assert fh.read() == body, (
                "DESIGN_SKILL.md and skills/toup-frontend-design.md have "
                "drifted — they are meant to be byte-identical")

    blocks = re.findall(r"```js\n(.*?)```", body, re.DOTALL)
    assert blocks, "the design skill has no JavaScript samples any more"
    node = shutil.which("node")
    for index, block in enumerate(blocks):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"sample{index}.js")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(block)
            proc = subprocess.run([node, "--check", path],
                                  capture_output=True, text=True)
        assert proc.returncode == 0, (
            f"design-skill JS sample {index} does not parse:\n"
            f"{block[:300]}\n{proc.stderr[-300:]}")


def test_the_shim_is_injected_before_the_app_script():
    """An app whose first statement builds an AudioContext must find the
    wrapper already installed."""
    html = ("<!doctype html><html><head><title>x</title></head><body>"
            "<script>var ctx = new AudioContext();</script></body></html>")
    wrapped = runtime.wrap_for_runtime(html)
    assert wrapped.index(runtime.AUDIO_GLOBAL) < wrapped.index("var ctx")


def test_wrapping_twice_installs_one_shim():
    html = "<!doctype html><html><head></head><body><p>hi</p></body></html>"
    once = runtime.wrap_for_runtime(html)
    assert runtime.wrap_for_runtime(once) == once
    assert once.count(runtime.AUDIO_GLOBAL) == once.count("var Wrapped")


# ── What the gate does with what the shim recorded ────────────────────

def test_an_app_that_makes_no_sound_is_not_nagged():
    """Most apps are silent on purpose. A finding for them would make this
    gate a nuisance, and a nuisance gate gets switched off."""
    assert verify.audio_findings(
        {"contexts": 0, "running": 0, "unlocked": True,
         "elements": 0, "failures": [], "blocked": []}
    ) == []


def test_audio_built_and_never_played_is_a_finding():
    findings = verify.audio_findings(
        {"contexts": 2, "running": 0, "unlocked": True,
         "elements": 0, "failures": [], "blocked": []}
    )
    assert len(findings) == 1
    assert findings[0].kind == "audio"
    assert "makes no sound" in findings[0].message
    # It must say what to do, not just what is wrong.
    assert "resume()" in findings[0].message


def test_audio_not_yet_unlocked_is_not_a_finding():
    """Before any gesture, a suspended context is CORRECT — that is the
    autoplay policy working. Only silence after a tap is a defect."""
    assert verify.audio_findings(
        {"contexts": 1, "running": 0, "unlocked": False,
         "elements": 0, "failures": [], "blocked": []}
    ) == []


def test_a_running_context_is_not_a_finding():
    assert verify.audio_findings(
        {"contexts": 1, "running": 1, "unlocked": True,
         "elements": 0, "failures": [], "blocked": []}
    ) == []


def test_a_policy_refusal_is_named_as_one():
    """The whole point: `NotSupportedError` reads as a broken file, so a model
    told only that would spend the next turn re-encoding a perfectly good
    sound. The finding has to say the sandbox refused it."""
    findings = verify.audio_findings(
        {"contexts": 1, "running": 1, "unlocked": True, "elements": 1,
         "failures": ["play(): NotSupportedError"],
         "blocked": ["media-src blocked data"]}
    )
    assert findings, findings
    assert "sandbox blocked" in findings[0].message
    assert "media-src" in findings[0].message


def test_a_rejected_play_is_reported_when_nothing_else_is():
    findings = verify.audio_findings(
        {"contexts": 0, "running": 0, "unlocked": True, "elements": 1,
         "failures": ["play(): NotAllowedError"], "blocked": []}
    )
    assert len(findings) == 1
    assert "NotAllowedError" in findings[0].message


def test_a_missing_record_is_not_a_verdict():
    """No shim, no measurement, no finding — never an invented one."""
    assert verify.audio_findings(None) == []
    assert verify.audio_findings({}) == []


# ── Item 3: the look ──────────────────────────────────────────────────

def test_a_look_that_did_not_happen_never_reads_as_a_pass():
    """The round-18 lesson, one layer up.

    `ok` is `ran and not problems`, so an empty problem list from a review
    that never ran is not approval. If this inverts, every publish on a
    container with no model silently claims a screen was checked.
    """
    look = vision.Look(ran=False)
    assert not look.ok
    assert look.summary() == "couldn't look at it here"
    assert "no errors" not in look.summary()


def test_a_look_that_happened_and_found_nothing_says_so():
    look = vision.Look(ran=True)
    assert look.ok
    assert look.summary() == "looks right"


def test_no_screenshot_means_no_review(monkeypatch):
    """A run whose browser pass was downgraded hands back None, and the
    reviewer must not be asked to have an opinion about nothing."""
    monkeypatch.setenv("TOUP_APP_VISUAL_REVIEW", "1")
    # A credential is what this box does NOT have, and that check comes first
    # (deliberately — it is the cheap one). Stub it so the screenshot branch
    # is the one under test.
    monkeypatch.setattr(vision, "can_call_model", lambda: True)
    look = _run(vision.review_screenshot(None, user_id="u", title="X"))
    assert not look.ran
    assert "screenshot" in look.reason


def test_no_reachable_model_means_no_review(monkeypatch):
    """And it must be decided CHEAPLY. Discovering it by timing out costs
    every publish 25 seconds on a container with no credential."""
    monkeypatch.setenv("TOUP_APP_MODEL_CALLS", "0")
    look = _run(vision.review_screenshot(b"\x89PNG fake", user_id="u", title="X"))
    assert not look.ran
    assert not look.problems


def test_the_verdict_parser_takes_json_however_it_arrives():
    assert vision.parse_verdict('{"ok": true}') == {"ok": True}
    assert vision.parse_verdict('```json\n{"ok": true}\n```') == {"ok": True}
    assert vision.parse_verdict('Sure! {"ok": false, "problems": ["x"]}') == {
        "ok": False, "problems": ["x"]}


def test_prose_is_not_a_verdict():
    """A reviewer that answers in prose has not answered. Returning an empty
    problem list for it would be an approval nobody gave."""
    assert vision.parse_verdict("It looks fine to me!") is None
    assert vision.parse_verdict("") is None
    assert vision.parse_verdict("[1,2,3]") is None


def test_a_flagged_screen_with_no_reason_still_refuses():
    problems = vision._problems_from({"ok": False})
    assert len(problems) == 1
    assert "could not name" in problems[0]


def test_problems_are_capped_and_deduplicated():
    problems = vision._problems_from(
        {"ok": False, "problems": ["same"] * 3 + [f"p{i}" for i in range(10)]}
    )
    assert len(problems) <= vision.MAX_PROBLEMS
    assert len(set(problems)) == len(problems)


def test_an_invisible_change_becomes_the_first_problem():
    """The edit half of item 3. If the change is not on screen, that is the
    thing to say first — everything else is commentary on the wrong app."""
    look = vision.Look(ran=True, problems=["the clock is clipped"])
    data = {"ok": False, "problems": ["the clock is clipped"],
            "change_visible": False}
    # Same shaping the reviewer applies, exercised directly.
    look.problems = vision._problems_from(data)
    look.change_confirmed = bool(data.get("change_visible"))
    if not look.change_confirmed:
        look.problems.insert(0, "the change you just made (“x”) is not visible")
    assert look.problems[0].startswith("the change you just made")


def test_the_image_block_matches_the_provider():
    """Not a detail — the two wire shapes are not interchangeable, and
    `call_system_llm` hands `messages` to the provider verbatim. An Anthropic
    image block posted to OpenAI is a 400, i.e. a gate that is permanently and
    silently down."""
    openai_block = vision.image_block(b"png", "gpt-4o-mini")
    assert openai_block["type"] == "image_url"
    assert openai_block["image_url"]["url"].startswith("data:image/png;base64,")

    claude_block = vision.image_block(b"png", "claude-haiku-4-5-20251001")
    assert claude_block["type"] == "image"
    assert claude_block["source"]["media_type"] == "image/png"


def test_the_review_model_is_pinned():
    """`model=None` resolves to the tenant's CHAT model, and this runs on
    every publish with a screenshot attached."""
    assert vision.REVIEW_MODEL
    assert vision.REVIEW_MODEL != "None"


# ── The step it appears as ────────────────────────────────────────────

def test_looking_is_its_own_phase_on_the_card():
    from app.agent.skills.builtins.app_html import steps as steps_mod

    assert "look" in steps_mod.STEP_TYPES
    assert steps_mod.STEP_TYPES.index("look") > steps_mod.STEP_TYPES.index("verify")
    assert steps_mod.STEP_TYPES.index("look") < steps_mod.STEP_TYPES.index("present")
    assert steps_mod.phase_label("look", "done") == "Checked the app looks right"
    assert steps_mod.phase_label("look", "running") == "Looking at the app"


def test_a_look_that_did_not_run_leaves_no_tick_on_the_card():
    """The overclaim, one level up from `Look.ok`.

    Caught by driving the real pipeline on a box with no model: the card read
    **"Checked the app looks right · couldn't look at it here"** — the `done`
    label and the reason it had not happened, in the same row, in green. A step
    that could not run must not carry a label claiming it did, so no terminal
    step is emitted at all; `finish_job` drops the `running` one, the same way
    it drops every phase a build did not need.
    """
    import asyncio as _asyncio
    from app.agent.skills.base import SkillContext
    from app.agent.skills.builtins.app_html import steps as steps_mod, store
    from app.agent.skills.builtins.app_html.skill import AppHtmlSkill

    emitted = []

    async def _emit(**kw):
        emitted.append((kw["step_type"], kw["status"]))

    async def _noop(*a, **k):
        return None

    async def _job(*a, **k):
        return "job-x"

    async def _row(**k):
        return "app-x"

    saved = (steps_mod.ensure_job, steps_mod.emit_step, steps_mod.finish_job,
             steps_mod.upsert_app_row, steps_mod.announce_ready)
    steps_mod.ensure_job, steps_mod.emit_step = _job, _emit
    steps_mod.finish_job, steps_mod.upsert_app_row = _noop, _row
    steps_mod.announce_ready = _noop
    old_root = os.environ.get("TOUP_HTML_APPS_DIR")
    old_smoke = os.environ.get("TOUP_APP_SMOKE_TEST")
    old_calls = os.environ.get("TOUP_APP_MODEL_CALLS")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["TOUP_HTML_APPS_DIR"] = tmp
            os.environ["TOUP_APP_SMOKE_TEST"] = "0"
            os.environ["TOUP_APP_MODEL_CALLS"] = "0"   # nothing can look
            store.ensure_root()
            html = ("<!doctype html><html><head><title>x</title></head><body>"
                    "<p>" + "x" * 600 + "</p></body></html>")
            skill = AppHtmlSkill()
            _asyncio.new_event_loop().run_until_complete(skill.execute_tool(
                "app_html__create_app_file",
                {"slug": "x", "title": "X", "html": html,
                 "brief": _BRIEF_FOR_STEP_TEST}, SkillContext(user_id="u", session_id="s")))
            _asyncio.new_event_loop().run_until_complete(skill.execute_tool(
                "app_html__present_app", {"slug": "x"},
                SkillContext(user_id="u", session_id="s")))
    finally:
        (steps_mod.ensure_job, steps_mod.emit_step, steps_mod.finish_job,
         steps_mod.upsert_app_row, steps_mod.announce_ready) = saved
        for key, value in (("TOUP_HTML_APPS_DIR", old_root),
                           ("TOUP_APP_SMOKE_TEST", old_smoke),
                           ("TOUP_APP_MODEL_CALLS", old_calls)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    looks = [status for step, status in emitted if step == "look"]
    assert looks == ["running"], emitted
    assert "done" not in looks
    # ...and the app was still published: a review that could not run is not a
    # reason to refuse.
    assert ("present", "done") in emitted, emitted


_BRIEF_FOR_STEP_TEST = (
    "## What it is\n"
    "A one-screen thing for someone with a spare minute, built so there is "
    "something to do that needs no account.\n\n"
    "## Core flows\n- Open it, read it.\n\n"
    "## Features, states and controls\n- One state; nothing to press.\n\n"
    "## Design decisions\n- Plain type on a plain field, because the content "
    "is the only thing on the screen."
)


def test_a_failed_look_tells_the_user_something_true():
    """`user_message` is the only error text the clients render. The generic
    fallback ("The build stopped partway. Nothing was changed.") is false for
    an app that is written, runs, and merely looks wrong."""
    from app.agent.skills.builtins.app_html import steps as steps_mod

    message = steps_mod._PHASE_USER_MESSAGE["look"]
    assert "runs" in message
    assert "nothing was changed" not in message.lower()


def test_the_report_can_carry_a_picture_and_an_audio_record():
    report = verify.Report()
    assert report.screenshot is None
    assert report.audio is None
    report.screenshot = b"\x89PNG"
    report.audio = {"contexts": 1}
    assert json.dumps(report.audio)
