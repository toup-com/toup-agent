"""Round 15 — the exact request, end to end, through the shipped pipeline.

    "Build me an app like the snake game for nokia"

On 2026-08-20 that produced a cross-platform Expo project, 12 generated files,
"Installing dependencies", and `npm install failed (exit 243): npm error code
EACCES`. Nothing playable.

This walks the same request through what the container now actually loads:
the five `app_html__*` tools, driven in the order the system prompt tells the
model to drive them, with the real store, the real HTML validator and the real
jailed shell. It asserts the OUTCOME — one self-contained file, tens of KB, no
build step, and a game whose logic runs — not that some functions were called.

`test_the_wire_array_offers_only_the_html_pipeline` is the other half: no
amount of working HTML tooling matters if the Expo tools are still on the wire
beside it, because that is the state the model was already choosing wrong in.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from app.agent.skills.base import SkillContext
from app.agent.skills.builtins.app_html import steps as steps_mod, store
from app.agent.skills.builtins.app_html.skill import AppHtmlSkill

USER = "871bac24-c366-42b5-b224-8802c73aef3a"
SLUG = "nokia-snake-classic"
TITLE = "Nokia Snake Classic"


# The kind of file the pipeline exists to produce: one document, inline CSS,
# inline JS, no CDN, no build step. The game loop is real — the test runs it.
SNAKE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nokia Snake Classic</title>
<style>
  :root { --lcd:#9ead86; --ink:#1b2416; }
  * { box-sizing:border-box; }
  body { margin:0; min-height:100dvh; display:grid; place-items:center;
         background:#111; font-family:"Courier New",monospace; color:var(--ink); }
  .shell { width:min(92vw,360px); padding:18px; border-radius:22px;
           background:linear-gradient(#3a3f33,#20241c); box-shadow:0 18px 40px #0009; }
  canvas { width:100%; image-rendering:pixelated; background:var(--lcd);
           border:3px solid #2b3124; border-radius:4px; display:block; }
  .hud { display:flex; justify-content:space-between; font-size:13px;
         letter-spacing:.14em; color:var(--lcd); margin-bottom:10px; }
  .pad { margin-top:14px; display:grid; grid-template-columns:repeat(3,1fr);
         gap:8px; }
  button { font:inherit; padding:14px 0; border:0; border-radius:10px;
           background:#2b3124; color:var(--lcd); }
  @media (min-width:768px) { .shell { width:420px; } }
</style>
</head>
<body>
<div class="shell">
  <div class="hud"><span>SNAKE II</span><span id="score">0000</span></div>
  <canvas id="board" width="200" height="200"></canvas>
  <div class="pad">
    <span></span><button data-dir="up">▲</button><span></span>
    <button data-dir="left">◀</button><button data-dir="down">▼</button>
    <button data-dir="right">▶</button>
  </div>
</div>
<script>
// ── Game state ─────────────────────────────────────────────────────
var CELL = 10, COLS = 20, ROWS = 20;
var snake, dir, pending, food, score, alive;

function reset() {
  snake = [{x:10,y:10},{x:9,y:10},{x:8,y:10}];
  dir = {x:1,y:0}; pending = null; score = 0; alive = true;
  placeFood();
}
function placeFood() {
  do {
    food = {x:(Math.random()*COLS)|0, y:(Math.random()*ROWS)|0};
  } while (snake.some(function(s){ return s.x===food.x && s.y===food.y; }));
}
function turn(nx, ny) {
  if (nx === -dir.x && ny === -dir.y) return;   // no 180° reversal
  pending = {x:nx, y:ny};
}
function step() {
  if (!alive) return;
  if (pending) { dir = pending; pending = null; }
  var head = {x:(snake[0].x+dir.x+COLS)%COLS, y:(snake[0].y+dir.y+ROWS)%ROWS};
  if (snake.some(function(s){ return s.x===head.x && s.y===head.y; })) {
    alive = false; return;
  }
  snake.unshift(head);
  if (head.x===food.x && head.y===food.y) { score += 10; placeFood(); }
  else snake.pop();
}
// ── Rendering ──────────────────────────────────────────────────────
function draw() {
  var c = document.getElementById('board').getContext('2d');
  c.clearRect(0,0,COLS*CELL,ROWS*CELL);
  c.fillStyle = '#1b2416';
  snake.forEach(function(s){ c.fillRect(s.x*CELL+1,s.y*CELL+1,CELL-2,CELL-2); });
  c.fillRect(food.x*CELL+3,food.y*CELL+3,CELL-6,CELL-6);
  document.getElementById('score').textContent =
    ('000'+score).slice(-4);
}
// ── Wiring. Guarded: the frame is sandboxed on an opaque origin. ────
try {
  document.addEventListener('keydown', function(e){
    var m = {ArrowUp:[0,-1],ArrowDown:[0,1],ArrowLeft:[-1,0],ArrowRight:[1,0]}[e.key];
    if (m) { e.preventDefault(); turn(m[0], m[1]); }
  });
  Array.prototype.forEach.call(document.querySelectorAll('[data-dir]'), function(b){
    var m = {up:[0,-1],down:[0,1],left:[-1,0],right:[1,0]}[b.dataset.dir];
    b.addEventListener('click', function(){ turn(m[0], m[1]); });
  });
  reset();
  setInterval(function(){ step(); draw(); }, 120);
} catch (e) { /* opaque origin — never let wiring take the page down */ }
</script>
</body>
</html>
"""


@pytest.fixture()
def pipeline(tmp_path, monkeypatch):
    """The real skill over a real app root; only the job/WS reporting is
    stubbed, because a broadcast failure must never be why a file assertion
    fails (and `steps.py` is covered by its own suite)."""
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(tmp_path / "apps"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    store.ensure_root()

    async def _noop(*_a, **_k):
        return None

    async def _job(*_a, **_k):
        return "job-e2e"

    async def _row(*_a, **_k):
        return "app-e2e"

    monkeypatch.setattr(steps_mod, "ensure_job", _job)
    monkeypatch.setattr(steps_mod, "emit_step", _noop)
    monkeypatch.setattr(steps_mod, "finish_job", _noop)
    monkeypatch.setattr(steps_mod, "announce_ready", _noop)
    monkeypatch.setattr(steps_mod, "upsert_app_row", _row)
    return AppHtmlSkill(), SkillContext(user_id=USER, workspace=str(tmp_path))


async def _build(skill, ctx, slug=SLUG, title=TITLE, html=SNAKE_HTML):
    """The loop the system prompt prescribes: create → verify → present."""
    out = {}
    out["create"] = await skill.execute_tool(
        "app_html__create_app_file",
        {"slug": slug, "title": title, "html": html}, ctx,
    )
    out["verify"] = await skill.execute_tool(
        "app_html__bash_app",
        {"slug": slug, "command": f"grep -c 'function step' {slug}.html"}, ctx,
    )
    out["present"] = await skill.execute_tool(
        "app_html__present_app", {"slug": slug}, ctx,
    )
    return out


# ═════════════════════════════════════════════════════════════════════
# 1. The request produces a playable single file
# ═════════════════════════════════════════════════════════════════════

async def test_the_snake_request_produces_one_playable_html_file(pipeline):
    skill, ctx = pipeline
    out = await _build(skill, ctx)

    for phase, text in out.items():
        assert not text.startswith("ERROR:"), f"{phase}: {text}"

    path = store.app_path(SLUG)
    body = open(path).read()
    size = os.path.getsize(path)

    # ONE file. The Expo path wrote 12 source files and then 27,133 more.
    assert sorted(
        f for f in os.listdir(store.apps_root()) if not f.startswith(".")
    ) == ["manifest.json", f"{SLUG}.html"]

    # Tens of KB, not hundreds of MB.
    assert 2_000 < size < 200_000, size

    # Self-contained: nothing to fetch, nothing to install.
    assert "<script" in body and "<style" in body
    assert "src=" not in body.split("<body")[0].replace('src="#"', "")
    assert "cdnjs" not in body and "unpkg" not in body
    assert "npm" not in body and "node_modules" not in body

    # A game, not a placeholder: a loop, input, food, collision, score.
    for token in ("function step", "placeFood", "keydown", "score", "canvas"):
        assert token in body, token

    # State is guarded — the frame runs on an opaque origin where storage
    # throws and the network is closed.
    assert "try {" in body


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
async def test_the_game_logic_actually_runs(pipeline, tmp_path):
    """A file that parses is not a game. Extract the script the browser would
    run, drive 200 ticks of it headlessly, and assert the snake moves, eats,
    grows and dies — the difference between "an app was produced" and "an app
    works", which is the entire complaint this round answers."""
    skill, ctx = pipeline
    await _build(skill, ctx)
    body = open(store.app_path(SLUG)).read()

    script = body.split("<script>")[1].split("</script>")[0]
    # Everything above the DOM wiring: pure state + rules.
    logic = script.split("// ── Rendering")[0]

    harness = logic + """
reset();
if (snake.length !== 3) throw new Error('bad start: ' + snake.length);
// Park the food where the head is about to be and confirm growth + score.
food = {x: snake[0].x + 1, y: snake[0].y};
step();
if (snake.length !== 4) throw new Error('did not grow: ' + snake.length);
if (score !== 10) throw new Error('did not score: ' + score);
// A 180-degree reversal is refused, as on the real handset.
turn(-1, 0);
step();
if (!alive) throw new Error('reversal killed the snake');
// Grow long, then drive it into itself: the snake must die.
for (var i = 0; i < 40; i++) { food = {x: snake[0].x + dir.x, y: snake[0].y + dir.y}; step(); }
var grew = snake.length;
if (grew < 20) throw new Error('never grew: ' + grew);
turn(0, 1); step(); turn(-1, 0); step(); turn(0, -1); step(); step();
if (alive) throw new Error('self-collision did not end the game');
console.log('OK ' + grew);
"""
    js = tmp_path / "harness.js"
    js.write_text(harness)
    res = subprocess.run(["node", str(js)], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert res.stdout.startswith("OK "), res.stdout


async def test_the_pipeline_is_reliable_across_repeats(pipeline):
    """Ten builds, four of them the same slug. The manifest, the revision
    counter and the file all have to agree at the end — a pipeline that works
    once is not a pipeline."""
    skill, ctx = pipeline
    rebuilds = 0
    for i in range(10):
        rebuild = bool(i % 3)
        slug = SLUG if rebuild else f"{SLUG}-{i}"
        rebuilds += rebuild
        out = await _build(skill, ctx, slug=slug, title=f"{TITLE} {i}")
        for phase, text in out.items():
            assert not text.startswith("ERROR:"), f"run {i} {phase}: {text}"

    manifest = store.read_manifest()
    assert len(manifest) == 5, sorted(manifest)
    # Rebuilding the same slug replaces the file and bumps the revision —
    # it never stacks a second app under a near-identical name.
    assert manifest[SLUG].revision == rebuilds, manifest[SLUG].revision
    for slug, rec in manifest.items():
        assert rec.presented_at, slug
        assert os.path.getsize(store.app_path(slug)) == rec.size_bytes


async def test_an_edit_round_trips_without_a_rebuild(pipeline):
    """The iterate half: view → exact-string edit → present, no build step."""
    skill, ctx = pipeline
    await _build(skill, ctx)

    seen = await skill.execute_tool("app_html__view_app_file", {"slug": SLUG}, ctx)
    assert "--lcd:#9ead86" in seen

    edited = await skill.execute_tool("app_html__edit_app_file", {
        "slug": SLUG, "old_string": "--lcd:#9ead86", "new_string": "--lcd:#c7d4a8",
        "reason": "brighten the LCD",
    }, ctx)
    assert not edited.startswith("ERROR:"), edited

    body = open(store.app_path(SLUG)).read()
    assert "--lcd:#c7d4a8" in body and "#9ead86" not in body
    assert store.read_manifest()[SLUG].revision == 2


async def test_a_stub_is_refused_before_anything_is_persisted(pipeline):
    """The x.pdf lesson in this lane: a prompt-mandated tool called with
    nothing must not produce a real, persisted, billed artifact."""
    skill, ctx = pipeline
    out = await skill.execute_tool("app_html__create_app_file", {
        "slug": "x", "title": "x", "html": "<html><body>x</body></html>",
    }, ctx)
    assert out.startswith("ERROR:")
    assert not os.path.exists(os.path.join(store.apps_root(), "x.html"))


# ═════════════════════════════════════════════════════════════════════
# 2. And the Expo path is not there to be chosen
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_the_wire_array_offers_only_the_html_pipeline(tmp_path, monkeypatch):
    """Working HTML tools do not help if 19 Expo tools sit beside them: that
    IS the state the model was choosing wrong in. Loaded through the real
    SkillLoader at the shipped defaults."""
    from app.agent import tool_entitlements as te
    from app.agent.skills.loader import SkillLoader

    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(tmp_path / "apps"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    te.reset_cache_for_tests()

    loader = SkillLoader(extra_dirs=[str(tmp_path)])
    await loader.load_all()
    names = [t["name"] for s in loader.skills.values() for t in s.get_tools()]

    assert not [n for n in names if n.startswith(("app_builder__", "app__"))], \
        [n for n in names if n.startswith(("app_builder__", "app__"))]
    assert sorted(n for n in names if n.startswith("app_html__")) == [
        "app_html__bash_app",
        "app_html__create_app_file",
        "app_html__edit_app_file",
        "app_html__present_app",
        "app_html__view_app_file",
    ]
    te.reset_cache_for_tests()


@pytest.mark.asyncio
async def test_scaffolding_is_redirected_to_the_tools_that_exist(tmp_path, monkeypatch):
    """A model that reaches for `npx create-expo-app` anyway must be pointed
    at a tool it can actually see — being blocked and redirected to something
    absent is the dead end this guard was fixed for once already."""
    from app.agent import tool_entitlements as te
    from app.agent.tool_executor import _pipeline_redirect_msg, _is_app_building_exec

    te.reset_cache_for_tests()
    assert _is_app_building_exec("npx create-expo-app@latest my-app")
    msg = _pipeline_redirect_msg()
    assert "app_html__create_app_file" in msg
    assert "app_builder__build_app" not in msg
    te.reset_cache_for_tests()
