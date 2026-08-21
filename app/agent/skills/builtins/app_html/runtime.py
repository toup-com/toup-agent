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


def preamble() -> str:
    """The single ``<script>`` injected ahead of everything the model wrote."""
    return (
        f'<script data-toup="{MARKER}">'
        f"{error_reporter()}{storage_shim()}"
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
