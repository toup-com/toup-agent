"""What the browser gets that the model did not write.

An artifact runs in a frame with ``sandbox="allow-scripts"`` and no
``allow-same-origin``, i.e. on an **opaque origin**. Three things follow, and
all three were reaching users as the same symptom — a page whose CSS painted
perfectly and whose JavaScript did nothing:

1.  ``localStorage``, ``sessionStorage`` and ``document.cookie`` **throw** on
    an opaque origin. Not "return null" — throw, on property ACCESS. One
    unguarded ``localStorage.getItem('highScore')`` at the top of a script
    kills the whole script, and everything after it (the render, the key
    handlers, the game loop) never runs. The design guidance has told the
    model to guard them since round 12; a rule that has to be obeyed on every
    line of every generated app is not a fix. :func:`storage_shim` replaces
    them with objects that cannot throw, so the class is closed no matter what
    the model wrote.

2.  A script loaded from cdnjs **without** ``crossorigin`` reports every error
    it raises as the string ``"Script error."`` with no file, no line and no
    stack — the browser's cross-origin masking. That is the exact opaque
    message users were shown. :func:`add_crossorigin` marks those tags
    ``crossorigin="anonymous"`` (cdnjs serves ``Access-Control-Allow-Origin:
    *``), which un-masks them WITHOUT relaxing the sandbox or the CSP.

3.  Nothing was listening. An uncaught error in the frame went to a console
    nobody opens. :func:`error_reporter` installs ``error`` /
    ``unhandledrejection`` / ``console.error`` hooks that postMessage the real
    message, file and line to the host, which is what lets the shell say
    "ReferenceError: draw is not defined (line 214)" instead of "Script
    error.".

4.  **Round 20: the app made no sound.** An ``AudioContext`` created while
    the page is loading starts ``suspended`` in every browser, and only a
    user gesture may resume it. A generated game builds its oscillators at
    the top of its script, plays them on a hit, and is silent forever —
    ``play()`` and ``start()`` both return without complaining, so nothing
    anywhere reports a failure. :func:`audio_unlock` resumes every context
    the app makes on the first press, whenever that comes, which closes the
    class for code that was written without thinking about it. It also
    RECORDS what happened (``window.__TOUP_AUDIO``) so the publish gate can
    measure whether an app that makes sound actually got any, instead of
    taking the absence of an error as an answer.

Everything here is injected at SERVE time, never written into the file. The
model's file stays exactly what the model wrote — ``view_app_file`` shows it,
``edit_app_file`` matches against it and the byte count in the card is its
real size. Injecting at write time would make every one of those lie.
"""

from __future__ import annotations

import re
from typing import List

#: Wire protocol, app → shell, for a runtime failure. The storage protocol
#: (``toup-storage``) is the sibling of this one and is spoken by the same
#: host listener; see ``AppArtifactFrame.tsx``.
ERROR_SOURCE = "toup-app-error"

#: Marker on the injected block. Lets the serve path stay idempotent (a
#: re-served or double-wrapped document must not carry two shims) and gives
#: tests something exact to assert on.
MARKER = "toup-runtime-v1"


def storage_shim() -> str:
    """Storage that degrades instead of throwing.

    Reads and writes hit an in-memory map, so they are synchronous and always
    succeed. Writes are ALSO mirrored to the host over the existing
    ``toup-storage`` bridge, and the map is hydrated from the host once at
    boot — so an app that saves a high score still has it after a reload, and
    an app that never persists anything is unaffected.

    Hydration is necessarily asynchronous (the frame has no synchronous
    channel to anything), so a read taken before the reply lands returns
    ``null`` rather than blocking first paint. The shim dispatches a
    ``toup-storage-ready`` event and fires a ``storage`` event per hydrated
    key when the reply arrives, which is enough for an app that seeds from
    defaults and reconciles — the pattern the design guidance asks for.
    """
    return (
        "(function(){"
        "var mem={},ready=false,seq=0,pending={};"
        "function post(op,key,value){"
        "try{if(parent===self)return;"
        "var id=++seq;pending[id]=op;"
        "parent.postMessage({source:'toup-storage',v:1,id:id,op:op,key:key,value:value},'*')}"
        "catch(e){}}"
        "function mk(name){"
        "return{"
        "getItem:function(k){k=String(k);return Object.prototype.hasOwnProperty.call(mem,k)?mem[k]:null},"
        "setItem:function(k,v){k=String(k);mem[k]=String(v);post('set',k,mem[k])},"
        "removeItem:function(k){k=String(k);delete mem[k];post('remove',k)},"
        "clear:function(){for(var k in mem){if(Object.prototype.hasOwnProperty.call(mem,k))post('remove',k)}mem={}},"
        "key:function(i){var ks=Object.keys(mem);return i<ks.length?ks[i]:null},"
        "get length(){return Object.keys(mem).length},"
        # Name is carried so a debugger can tell the two apart; both share the
        # map on purpose — an app that writes a key to one and reads it from
        # the other is doing something odd, and agreeing is friendlier than
        # half-working.
        "__toupKind:name"
        "}}"
        "var shim=mk('local'),sshim=mk('session');"
        # defineProperty, not assignment: `localStorage` is an accessor on the
        # window prototype whose GETTER throws here, so `window.localStorage =
        # x` is a no-op (and in strict mode a TypeError). Redefining the
        # property is the only thing that actually takes.
        "function install(n,v){try{Object.defineProperty(window,n,"
        "{value:v,configurable:true,writable:false})}catch(e){try{window[n]=v}catch(e2){}}}"
        "install('localStorage',shim);install('sessionStorage',sshim);"
        # document.cookie on an opaque origin is either inert or throws
        # depending on the engine. Make it inert everywhere, backed by the
        # same map so a cookie round-trips within the session.
        "try{Object.defineProperty(document,'cookie',{configurable:true,"
        "get:function(){var o=[];for(var k in mem){if(k.indexOf('c:')===0)"
        "o.push(k.slice(2)+'='+mem[k])}return o.join('; ')},"
        "set:function(s){try{var p=String(s).split(';')[0],i=p.indexOf('=');"
        "if(i>0)shim.setItem('c:'+p.slice(0,i).trim(),p.slice(i+1))}catch(e){}}})}catch(e){}"
        # Hydrate. `all` returns the whole persisted blob in one round-trip.
        "window.addEventListener('message',function(ev){"
        "var m=ev&&ev.data;"
        "if(!m||m.source!=='toup-storage-host'||ready)return;"
        "if(pending[m.id]!=='all')return;"
        "ready=true;delete pending[m.id];"
        "var v=m.ok&&m.value&&typeof m.value==='object'?m.value:{};"
        "for(var k in v){if(Object.prototype.hasOwnProperty.call(v,k)&&"
        "!Object.prototype.hasOwnProperty.call(mem,k)){"
        "mem[k]=typeof v[k]==='string'?v[k]:JSON.stringify(v[k]);"
        "try{window.dispatchEvent(new StorageEvent('storage',{key:k,newValue:mem[k]}))}catch(e){}}}"
        "try{window.dispatchEvent(new Event('toup-storage-ready'))}catch(e){}"
        "});"
        "post('all');"
        "})();"
    )


def error_reporter() -> str:
    """Hooks that turn a dead page into a reported failure.

    ``window.onerror``'s 5-argument form carries the real file and line for a
    same-document script; the 6th argument is the Error object, which carries
    the stack. Both are forwarded. A cross-origin script that still reports
    ``"Script error."`` is labelled as such rather than passed off as the
    app's own message — the honest thing to say is "an external library
    failed and would not say why".
    """
    return (
        "(function(){"
        "var seen=[],MAX=8;"
        "window.__TOUP_ERRORS=seen;"
        "function report(kind,message,file,line,col,stack){"
        "try{"
        "message=String(message||'').slice(0,600);"
        "if(!message)return;"
        "if(seen.length>=MAX)return;"
        "for(var i=0;i<seen.length;i++)if(seen[i].message===message)return;"
        "var rec={kind:kind,message:message,file:String(file||'').slice(0,300),"
        "line:line||0,col:col||0,stack:String(stack||'').slice(0,1200),"
        "opaque:message==='Script error.'};"
        "seen.push(rec);"
        "if(parent!==self)parent.postMessage("
        f"{{source:'{ERROR_SOURCE}',v:1,error:rec}},'*')"
        "}catch(e){}}"
        "window.addEventListener('error',function(ev){"
        # A failed <script src>/<img> fires an `error` event on the ELEMENT
        # with no message — report it as a load failure, which is a different
        # and much more actionable fact than an exception.
        "if(ev&&ev.target&&ev.target!==window&&ev.target.tagName){"
        "var t=ev.target,u=t.src||t.href;"
        "if(u)report('load','Could not load '+String(u).slice(0,200),u,0,0,'');"
        "return}"
        "report('error',ev&&ev.message,ev&&ev.filename,ev&&ev.lineno,ev&&ev.colno,"
        "ev&&ev.error&&ev.error.stack)"
        "},true);"
        "window.addEventListener('unhandledrejection',function(ev){"
        "var r=ev&&ev.reason;"
        "report('rejection',(r&&(r.message||r))||'Promise rejected',"
        "'',0,0,r&&r.stack)"
        "});"
        "try{var ce=console.error;console.error=function(){"
        "try{report('console',Array.prototype.map.call(arguments,function(a){"
        "return a&&a.message?a.message:String(a)}).join(' '),'',0,0,'')}catch(e){}"
        "return ce.apply(console,arguments)}}catch(e){}"
        "})();"
    )


#: Where the audio state lives, for the publish gate and for anyone
#: debugging a silent app in a real browser.
AUDIO_GLOBAL = "__TOUP_AUDIO"


def audio_unlock() -> str:
    """Make the app's sound work, and record whether it did.

    Two jobs, and the second is the one that keeps the first honest.

    **Unlocking.** Autoplay policy is per-frame and gesture-based: an
    ``AudioContext`` constructed during load is ``suspended``, and only a
    ``resume()`` inside (or after) a user gesture starts it. Generated apps
    almost always build their audio graph at the top of the script — that is
    where a person writes it — so the graph exists, the oscillators start,
    and nothing comes out. ``resume()`` returns a promise nobody awaits, so
    there is no error to see. The constructor is wrapped so every context the
    app makes is remembered, and a capture-phase listener on the first
    ``pointerdown``/``touchend``/``keydown``/``click`` resumes all of them;
    contexts made AFTER the first gesture are resumed on creation. This never
    starts audio the user did not ask for — it only resumes on a real
    gesture, which is exactly the condition the policy is expressing.

    ``<audio>``/``<video>`` elements get the same treatment: a ``play()``
    that was rejected is retried once on the first gesture. The rejection is
    recorded either way.

    **Recording.** ``window.__TOUP_AUDIO`` carries ``{contexts, running,
    unlocked, elements, failures, blocked}``. Without it the gate can only
    observe that nothing threw, and "nothing threw" is precisely what a
    silent app looks like. ``blocked`` comes from
    ``securitypolicyviolation``, which is how a ``media-src`` mistake
    announces itself: a data:-URI sound refused by the CSP fails with
    ``NotSupportedError``, which reads exactly like a corrupt file.
    """
    return (
        "(function(){"
        f"var S={{contexts:0,running:0,unlocked:false,elements:0,failures:[],blocked:[]}};"
        f"try{{window.{AUDIO_GLOBAL}=S}}catch(e){{}}"
        "var ctxs=[],pending=[];"
        "function note(list,msg){try{msg=String(msg||'').slice(0,200);"
        "if(msg&&list.length<6&&list.indexOf(msg)<0)list.push(msg)}catch(e){}}"
        # `running` is a COUNT OF THE CURRENT STATE, recomputed, never
        # incremented. Adding one per successful resume made it climb past
        # `contexts` — resumeAll() resumes every context including the ones
        # already running — and the gate reads this number to decide whether
        # an app that makes sound got any. A tally that can exceed its own
        # denominator is not a measurement.
        "function recount(){var n=0;"
        "for(var i=0;i<ctxs.length;i++){try{if(ctxs[i].state==='running')n++}catch(e){}}"
        "S.running=n;return S}"
        # `S.check()` re-reads the contexts and returns the record. The
        # counters are only refreshed on creation and on resume, so a reader
        # arriving at an arbitrary moment — the publish gate does exactly
        # that — must ask for a fresh count rather than trust the last event's.
        "S.check=recount;"
        "function resumeAll(){"
        "for(var i=0;i<ctxs.length;i++){(function(c){try{"
        "var p=c.resume&&c.resume();if(p&&p.then)p.then(recount,"
        "function(e){note(S.failures,'AudioContext.resume: '+(e&&e.name||e))})"
        "}catch(e){note(S.failures,'AudioContext.resume: '+e.message)}})(ctxs[i])}"
        "recount();"
        "for(var j=0;j<pending.length;j++){(function(el){try{"
        "var q=el.play&&el.play();if(q&&q.catch)q.catch(function(){})"
        "}catch(e){}})(pending[j])}"
        "pending.length=0;}"
        "function unlock(){if(S.unlocked)return;S.unlocked=true;resumeAll()}"
        # Capture phase, so a handler that stops propagation cannot cost the
        # app its sound. Not `once`: several gesture types are registered and
        # only the first to fire matters, which `S.unlocked` already handles.
        "var evs=['pointerdown','touchend','mousedown','keydown','click'];"
        "for(var k=0;k<evs.length;k++){try{"
        "window.addEventListener(evs[k],unlock,true)}catch(e){}}"
        "var AC=window.AudioContext||window.webkitAudioContext;"
        "if(AC){"
        # `var`, not a block-level `function` declaration: those are Annex-B
        # in sloppy mode and hoist differently under strict, and this string
        # is injected into documents we do not control the mode of.
        "var Wrapped=function(a,b,c){"
        "var ctx=arguments.length?new AC(a,b,c):new AC();"
        "S.contexts++;ctxs.push(ctx);recount();"
        # Created after the first gesture: resume it now rather than waiting
        # for a second one that may never come.
        "if(ctx.state!=='running'&&S.unlocked){try{var p=ctx.resume();"
        "if(p&&p.then)p.then(recount,"
        "function(e){note(S.failures,'AudioContext.resume: '+(e&&e.name||e))})}catch(e){}}"
        "return ctx};"
        "Wrapped.prototype=AC.prototype;"
        "try{Object.defineProperty(window,'AudioContext',{value:Wrapped,configurable:true,writable:true})}catch(e){}"
        "try{Object.defineProperty(window,'webkitAudioContext',{value:Wrapped,configurable:true,writable:true})}catch(e){}"
        "}"
        # HTMLMediaElement.play: count it, remember a rejection, and retry
        # once the gesture arrives. The original promise is still what the
        # app sees, so an app that handles its own rejection is unaffected.
        "try{var MP=HTMLMediaElement.prototype,orig=MP.play;"
        "MP.play=function(){S.elements++;var el=this;var p;"
        "try{p=orig.apply(this,arguments)}catch(e){"
        "note(S.failures,'play(): '+e.message);throw e}"
        "if(p&&p.catch)p.catch(function(e){"
        "note(S.failures,'play(): '+(e&&e.name||e));"
        "if(!S.unlocked&&pending.length<8)pending.push(el)});"
        "return p}}catch(e){}"
        # A CSP refusal is the other way a sound goes missing, and it is the
        # one that looks least like itself: the element reports
        # NotSupportedError, i.e. "bad file", for a policy decision.
        "try{window.addEventListener('securitypolicyviolation',function(ev){"
        "note(S.blocked,(ev&&ev.violatedDirective||'csp')+' blocked '+"
        "String(ev&&ev.blockedURI||'').slice(0,40))},true)}catch(e){}"
        "})();"
    )


def preamble() -> str:
    """The single ``<script>`` injected ahead of everything the model wrote."""
    return (
        f'<script data-toup="{MARKER}">'
        f"{error_reporter()}{storage_shim()}{audio_unlock()}"
        "</script>"
    )


#: A `<script src>` or stylesheet `<link href>` pointing at an absolute
#: https/protocol-relative origin. Only cdnjs can be present at all
#: (`store.validate_html` refuses anything else), so a match here is always a
#: tag we want CORS-enabled.
_EXTERNAL_TAG_RE = re.compile(
    r"""<(script|link)\b([^>]*\b(?:src|href)\s*=\s*["'](?:https?:)?//[^"']+["'][^>]*)>""",
    re.IGNORECASE,
)
_HAS_CROSSORIGIN_RE = re.compile(r"\bcrossorigin\b", re.IGNORECASE)


def add_crossorigin(html: str) -> str:
    """Mark external ``<script>``/``<link>`` tags ``crossorigin="anonymous"``.

    Without this attribute the browser refuses to hand the page any detail
    about an error raised inside the fetched file: ``window.onerror`` gets
    ``("Script error.", "", 0, 0, null)``. With it — and cdnjs does send
    ``Access-Control-Allow-Origin: *`` — the real message, file and line come
    through. Nothing about the sandbox or the CSP moves.
    """
    def _fix(m: re.Match) -> str:
        tag, attrs = m.group(1), m.group(2)
        if _HAS_CROSSORIGIN_RE.search(attrs):
            return m.group(0)
        return f"<{tag}{attrs.rstrip()} crossorigin=\"anonymous\">"

    return _EXTERNAL_TAG_RE.sub(_fix, html)


_HEAD_OPEN_RE = re.compile(r"<head\b[^>]*>", re.IGNORECASE)
_HTML_OPEN_RE = re.compile(r"<html\b[^>]*>", re.IGNORECASE)


def wrap_for_runtime(html: str) -> str:
    """Serve-time transform: crossorigin + preamble, injected exactly once.

    The preamble must be the FIRST script in the document — an app whose very
    first statement touches ``localStorage`` has to find the shim already in
    place — so it goes immediately after ``<head>``, falling back to just
    after ``<html>`` and finally to the very top for a fragment-ish document
    (``validate_html`` already guarantees a ``<body>``, not a ``<head>``).
    """
    if not html:
        return html
    if MARKER in html:
        return html
    html = add_crossorigin(html)
    block = preamble()

    m = _HEAD_OPEN_RE.search(html)
    if m:
        return html[:m.end()] + block + html[m.end():]
    m = _HTML_OPEN_RE.search(html)
    if m:
        return html[:m.end()] + block + html[m.end():]
    return block + html


_HEAD_INSERT_RE = re.compile(r"<head\b[^>]*>", re.IGNORECASE)


def wrap_for_verification(html: str) -> str:
    """The runtime wrapper PLUS the policy the browser will really enforce.

    Round 20. The publish gate loaded the app with `page.set_content` and no
    policy at all, so it was running the app in a browser strictly more
    permissive than the user's — and this round's bug lived exactly in that
    gap. Measured, three ways, on the same page: with no policy the refused
    sound is invisible to the gate (`blocked: []`, `failures: []`); with the
    current policy it plays; with the policy AS IT WAS BEFORE this round the
    shim records ``blocked: ["media-src blocked data"]`` and
    ``failures: ["play(): NotSupportedError"]`` and the publish is refused.

    So the gate would now catch the defect it previously could not see, which
    is the only version of this check worth having: a canary that runs in a
    kinder cage than the bird cannot fail.

    A ``<meta>`` rather than a header because `set_content` has no headers —
    the same mechanism the mobile runner uses, and it is honoured as long as
    the parser meets it before the content it governs, which is why it goes
    immediately after ``<head>`` and ahead of the preamble.

    ``frame-ancestors`` is omitted: a ``<meta>`` policy ignores it, and
    emitting a directive that does nothing would be one more thing that looks
    like enforcement and is not.
    """
    if not html:
        return html
    wrapped = wrap_for_runtime(html)
    try:
        from app.artifact_policy import artifact_cdn_origin, sandbox_csp
        csp = sandbox_csp(artifact_cdn_origin())
    except Exception:  # pragma: no cover - a policy that cannot be built is
        # not a reason to skip the run; the other passes still apply.
        return wrapped
    meta = f'<meta http-equiv="Content-Security-Policy" content="{csp}">'
    m = _HEAD_INSERT_RE.search(wrapped)
    if m:
        return wrapped[:m.end()] + meta + wrapped[m.end():]
    return meta + wrapped


def external_origins(html: str) -> List[str]:
    """Hosts the document loads code or styles from. Diagnostic only."""
    hosts: List[str] = []
    for m in re.finditer(
        r"""(?:src|href)\s*=\s*["'](?:https?:)?//([^/"'?#]+)""", html, re.IGNORECASE
    ):
        host = m.group(1).split("@")[-1].split(":")[0].lower()
        if host not in hosts:
            hosts.append(host)
    return hosts
