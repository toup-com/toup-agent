"""
Apps proxy routes — platform forwards apps/jobs requests to the user's VPS agent.

App data (built apps, build jobs) lives on the user's VPS.
The platform is a passthrough proxy only.
"""

import json
import logging
import re
from typing import Optional, Tuple
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db, AgentConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/apps", tags=["Apps Proxy"])


def _build_agent_bridge_script(token: str, app_id: str) -> str:
    """Build the inline <script> that creates a deterministic agent bridge.

    This script runs in <head> BEFORE the Expo JS bundle loads.
    It creates window.__TOUP_AGENT_BRIDGE using HTTP fetch + SSE streaming
    (not WebSocket) for maximum reliability through Cloudflare/Railway.
    """
    return (
        "<script>"
        "(function(){"
        # ── Config ──
        f'var T="{token}",A="{app_id}";'
        # Chat endpoint — same-origin HTTP POST, no WebSocket needed
        'var chatUrl="/api/apps/"+A+"/chat";'
        # ── Set globals for generated code that checks them ──
        # __TOUP_WS_URL="" prevents the generated agentBridge from opening its own WS
        'window.__TOUP_APP_ID=A;'
        'window.__TOUP_WS_URL="";'
        'try{Object.defineProperty(window,"__TOUP_WS_URL",{value:"",writable:false,configurable:false})}catch(e){}'
        # ── Bridge state ──
        'var sending=false,msgCbs=[],toolCbs=[],navRef=null,screens=[],actions={};'
        # ── Send message via HTTP POST + read SSE stream ──
        'function send(text,opts){'
        'if(sending||!T)return;'
        'sending=true;'
        'var buf="";'
        'var payload={text:text,session_id:"app-"+A};'
        'if(opts&&opts.layer2)payload.layer2=true;'
        'for(var i=0;i<toolCbs.length;i++)try{toolCbs[i]("thinking",false)}catch(e){}'
        'fetch(chatUrl+"?token="+encodeURIComponent(T),{'
        'method:"POST",'
        'headers:{"Content-Type":"application/json"},'
        'body:JSON.stringify(payload)'
        '}).then(function(r){'
        'if(!r.ok)throw new Error("HTTP "+r.status);'
        'B.isConnected=true;'  # successful HTTP = connected
        'var reader=r.body.getReader(),dec=new TextDecoder(),partial="";'
        'function pump(){'
        'reader.read().then(function(result){'
        'if(result.done){'
        # Stream done — emit final text
        'sending=false;'
        'var final=buf;'
        'if(final)for(var i=0;i<msgCbs.length;i++)try{msgCbs[i](final)}catch(e){}'
        'for(var i=0;i<toolCbs.length;i++)try{toolCbs[i]("done",true)}catch(e){}'
        'return}'
        'partial+=dec.decode(result.value,{stream:true});'
        'var lines=partial.split("\\n");'
        'partial=lines.pop();'  # keep incomplete line
        'for(var li=0;li<lines.length;li++){'
        'var line=lines[li];'
        'if(line.indexOf("data: ")!==0)continue;'
        'var d;try{d=JSON.parse(line.slice(6))}catch(e){continue}'
        'if(d.type==="text_chunk"){buf+=(d.text||"")}'
        'else if(d.type==="done"){'
        'var t=d.text||buf;buf="";'
        'for(var i=0;i<msgCbs.length;i++)try{msgCbs[i](t)}catch(e){}'
        '}'
        'else if(d.type==="app_navigate"&&d.screen){B.navigate(d.screen,d.params||{})}'
        'else if(d.type==="tool_start"){'
        'for(var i=0;i<toolCbs.length;i++)try{toolCbs[i](d.tool||"",false)}catch(e){}'
        '}'
        'else if(d.type==="tool_end"){'
        'for(var i=0;i<toolCbs.length;i++)try{toolCbs[i](d.tool||"",true)}catch(e){}'
        '}'
        'else if(d.type==="error"){'
        'var m=d.text||d.message||"Error";'
        'for(var i=0;i<msgCbs.length;i++)try{msgCbs[i](m)}catch(e){}'
        '}'
        '}'
        'pump()'
        '}).catch(function(e){sending=false;console.error("[ToupBridge]",e)})'
        '}'
        'pump()'
        '}).catch(function(e){'
        'sending=false;'
        'console.error("[ToupBridge] fetch error",e);'
        'for(var i=0;i<msgCbs.length;i++)try{msgCbs[i]("Connection error. Try again.")}catch(e){}'
        '})'
        '}'
        # (no ping needed — HTTP mode is always "connected" if token exists)
        # ── Bridge API ──
        'var B={'
        'isConnected:!!T,'
        'currentScreen:"",'
        'sendMessage:function(text,opts){send(text,opts)},'
        'onAgentMessage:function(cb){msgCbs.push(cb);return function(){'
        'var i=msgCbs.indexOf(cb);if(i>=0)msgCbs.splice(i,1)}},'
        'onToolActivity:function(cb){toolCbs.push(cb);return function(){'
        'var i=toolCbs.indexOf(cb);if(i>=0)toolCbs.splice(i,1)}},'
        'setNavigationRef:function(ref){navRef=ref},'
        'navigate:function(screen,params){'
        'try{if(navRef&&navRef.current)navRef.current.navigate(screen,params||{});'
        'else if(navRef&&typeof navRef.navigate==="function")navRef.navigate(screen,params||{})'
        '}catch(e){}'
        '},'
        'getScreens:function(){return screens},'
        'setScreens:function(s){screens=s},'
        'getActions:function(s){return s?actions[s]||[]:Object.values(actions).flat()},'
        'setActions:function(a){actions=a},'
        'destroy:function(){msgCbs=[];toolCbs=[]}'
        '};'
        # ── PostMessage listener for config/token updates ──
        'window.addEventListener("message",function(ev){'
        'if(ev.data&&ev.data.type==="toup_agent_config"){'
        'if(ev.data.token)T=ev.data.token;'
        'if(ev.data.app_id){A=ev.data.app_id;chatUrl="/api/apps/"+A+"/chat"}'
        'B.isConnected=!!T'
        '}'
        '});'
        # ── Expose globally (non-writable so generated agentBridge.ts can't overwrite) ──
        'window.__TOUP_AGENT_BRIDGE=B;'
        'try{Object.defineProperty(window,"__TOUP_AGENT_BRIDGE",{value:B,writable:false,configurable:false})}catch(e){}'
        # Also lock __TOUP_AUTH_TOKEN — generated code reads it but we don't want it
        # to create its own WS connection (we set __TOUP_WS_URL="" to prevent that)
        'window.__TOUP_AUTH_TOKEN=T;'
        # ── Hide generated AgentPlaceholder as soon as it renders ──
        # The generated app creates its own chat bubble (high z-index, bottom-right).
        # We scan for it via computed styles and hide it immediately.
        # This runs from <head> so starts observing before the Expo bundle loads.
        'window.__TOUP_AGENT_UI_INJECTED=true;'
        'function _hgScan(){'
        'if(!document.body)return;'
        'var all=document.body.getElementsByTagName("div");'
        'for(var i=0;i<all.length;i++){'
        'var el=all[i];if(el.id==="taw"||el.closest&&el.closest("#taw"))continue;'
        'try{var cs=getComputedStyle(el);'
        'if((cs.position==="absolute"||cs.position==="fixed")'
        '&&parseInt(cs.zIndex)>=9000){'
        'var bt=parseInt(cs.bottom),ri=parseInt(cs.right);'
        'if(!isNaN(bt)&&!isNaN(ri)&&bt<=120&&ri<=60)'
        'el.style.setProperty("display","none","important")'
        '}}catch(e){}}}'
        # Run scan frequently for first 10s, then use MutationObserver
        'var _hgN=0,_hgT=setInterval(function(){'
        '_hgScan();if(++_hgN>50)clearInterval(_hgT)},200);'
        'try{new MutationObserver(function(){_hgScan()}).observe('
        'document.documentElement,{childList:true,subtree:true})}catch(e){}'
        # HTTP mode — connected if token exists (each message is a new request)
        'console.log("[ToupBridge] HTTP mode, app="+A+" connected="+B.isConnected);'
        "})()"
        "</script>"
    )



def _hex_lighten(color: str, pct: int) -> str:
    c = color.lstrip('#')
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    f = pct / 100
    return f'#{int(r+(255-r)*f):02x}{int(g+(255-g)*f):02x}{int(b+(255-b)*f):02x}'


def _hex_darken(color: str, pct: int) -> str:
    c = color.lstrip('#')
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    f = 1 - pct / 100
    return f'#{int(r*f):02x}{int(g*f):02x}{int(b*f):02x}'


def _hex_rgba(color: str, alpha: float) -> str:
    c = color.lstrip('#')
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def _build_agent_widget_script(
    agent_name: str = "Agent",
    agent_color: str = "#9B59B6",
) -> str:
    """Build inline <style> + <script> for the floating Orb agent widget.

    Recreates the platform's Orb component (sphere with eyes, breathing,
    blinking, 3D shading) in vanilla CSS/JS.  Injected by the preview proxy
    so every app gets the same consistent agent UI.
    """
    # Derived colors
    cl = _hex_lighten(agent_color, 20)   # light body
    cd = _hex_darken(agent_color, 35)    # dark overlay
    cd15 = _hex_darken(agent_color, 15)
    cd30 = _hex_darken(agent_color, 30)
    cd45 = _hex_darken(agent_color, 45)
    cl40 = _hex_lighten(agent_color, 40)  # specular highlight
    glow = _hex_rgba(agent_color, 0.25)
    glow2 = _hex_rgba(agent_color, 0.15)

    # HTML template
    tpl = (
        # ── Orb button ──
        '<button id="taw-b">'
        '<div class="taw-ring"></div>'
        '<div class="taw-ob">'
        '<div class="taw-od"></div>'
        '<div class="taw-o3"></div>'
        '<div class="taw-oe">'
        '<div class="taw-ey"><div class="taw-pu"><div class="taw-hi"></div></div></div>'
        '<div class="taw-ey"><div class="taw-pu"><div class="taw-hi"></div></div></div>'
        '</div>'
        '</div>'
        '<div class="taw-dot"></div>'
        '</button>'
        # ── Chat panel ──
        '<div id="taw-p">'
        '<div class="taw-h">'
        '<div class="taw-ho"></div>'
        '<span class="taw-hn">' + agent_name + '</span>'
        '<div class="taw-hd"></div>'
        '<button class="taw-x">\u2715</button>'
        '</div>'
        '<div id="taw-m"><div class="taw-empty">'
        'Ask me anything!'
        '<div style="margin-top:12px">'
        f'<button class="taw-chip taw-l2" onclick="this.parentElement.parentElement.remove();'
        f'am(\'Customize this app for me — let\\'s build Layer 2!\',\'u\');'
        f'B.sendMessage(\'Customize this app for me — let\\'s build Layer 2!\',{{layer2:true}})">'
        f'\u2728 Customize this app</button>'
        '</div>'
        '</div></div>'
        '<div class="taw-iw">'
        '<input id="taw-i" placeholder="Type a message..." autocomplete="off">'
        '<button id="taw-s">\u2192</button></div>'
        '</div>'
    )
    tpl_js = tpl.replace('\\', '\\\\').replace("'", "\\'")

    parts = [
        # ── CSS ──
        '<style>',
        # Keyframes
        '@keyframes taw-breathe{0%,100%{transform:scale(1)}40%{transform:scale(1.06)}}',
        '@keyframes taw-dark{0%,100%{opacity:0}40%{opacity:1}}',
        '@keyframes taw-blink{0%,42%,44%,100%{transform:scaleY(1)}43%{transform:scaleY(0.05)}}',
        f'@keyframes taw-ring-k{{0%{{transform:scale(1);opacity:0.4}}100%{{transform:scale(1.8);opacity:0}}}}',
        # Root
        '#taw{position:fixed;bottom:24px;right:24px;z-index:2147483647;'
        'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;'
        'font-size:13px;line-height:1.4}',
        '@media(max-width:768px){#taw{bottom:80px;right:16px}}',
        '#taw *{box-sizing:border-box;margin:0;padding:0}',
        # Orb button
        '#taw-b{width:56px;height:56px;border-radius:50%;border:none;'
        'cursor:pointer;padding:0;background:transparent;position:relative;'
        'animation:taw-breathe 4s ease-in-out infinite}',
        '#taw-b:hover{filter:brightness(1.1)}',
        # Wave ring
        f'.taw-ring{{position:absolute;inset:-4px;border-radius:50%;'
        f'border:2px solid {agent_color};opacity:0;'
        f'animation:taw-ring-k 2.5s ease-out infinite}}',
        # Orb body
        f'.taw-ob{{width:56px;height:56px;border-radius:50%;position:relative;'
        f'overflow:hidden;'
        f'background:radial-gradient(ellipse at 35% 40%,{cl40},{agent_color} 30%,'
        f'{cd15} 55%,{cd30} 75%,{cd45} 90%);'
        f'box-shadow:0 4px 20px rgba(0,0,0,0.35),0 0 25px {glow},'
        f'inset 0 -2px 8px rgba(0,0,0,0.15)}}',
        # Dark overlay (breathing color shift)
        f'.taw-od{{position:absolute;inset:0;border-radius:50%;'
        f'background:{cd};animation:taw-dark 4s ease-in-out infinite}}',
        # 3D shading
        '.taw-o3{position:absolute;inset:0;border-radius:50%;'
        'background:'
        'radial-gradient(ellipse at 32% 25%,rgba(255,255,255,0.18),'
        'rgba(255,255,255,0.04) 30%,transparent 55%),'
        'radial-gradient(ellipse at 50% 90%,rgba(0,0,0,0.22),transparent 50%),'
        'radial-gradient(ellipse at 50% 50%,transparent 60%,'
        'rgba(0,0,0,0.12) 80%,rgba(0,0,0,0.25) 100%)}',
        # Eyes container
        '.taw-oe{position:absolute;top:21px;left:0;right:0;'
        'display:flex;justify-content:center;gap:7px;align-items:center}',
        # Eye
        '.taw-ey{width:16px;height:16px;border-radius:50%;'
        'background:radial-gradient(circle at 45% 38%,#fff,#f0eef8 50%,#ddd8f0);'
        'display:flex;align-items:center;justify-content:center;'
        'animation:taw-blink 5s ease-in-out infinite}',
        # Pupil
        '.taw-pu{width:8px;height:8px;border-radius:50%;'
        'background:radial-gradient(circle at 42% 38%,#312e81,#1e1b4b 50%,#0c0a1a);'
        'position:relative}',
        # Pupil highlight
        '.taw-hi{position:absolute;top:1px;left:1px;'
        'width:3px;height:3px;border-radius:50%;'
        'background:rgba(255,255,255,0.85)}',
        # Connection dot
        '.taw-dot{position:absolute;top:0;right:0;'
        'width:12px;height:12px;border-radius:50%;border:2px solid #12121c}',
        # ── Panel ──
        '#taw-p{display:none;position:fixed;bottom:88px;right:24px;'
        'width:360px;max-height:520px;background:#12121c;'
        'border-radius:16px;border:1px solid #1e1e2e;'
        'flex-direction:column;box-shadow:0 12px 40px rgba(0,0,0,0.5);'
        'overflow:hidden}',
        '@media(max-width:768px){#taw-p{bottom:140px;right:8px;'
        'left:8px;width:auto;max-height:55vh}}',
        '#taw-p.show{display:flex}',
        # Header
        '.taw-h{display:flex;align-items:center;padding:12px 14px;'
        'border-bottom:1px solid #1e1e2e;background:#0e0e18;gap:10px}',
        # Mini orb in header
        f'.taw-ho{{width:28px;height:28px;border-radius:50%;flex-shrink:0;'
        f'background:radial-gradient(ellipse at 35% 40%,{cl40},{agent_color} 40%,{cd30} 90%);'
        f'box-shadow:0 0 10px {glow2}}}',
        '.taw-hn{color:#fff;font-size:14px;font-weight:600;flex:1}',
        '.taw-hd{width:8px;height:8px;border-radius:50%}',
        '.taw-x{width:26px;height:26px;border-radius:50%;border:none;'
        'background:#1e1e2e;color:#9ca3af;cursor:pointer;font-size:14px;'
        'display:flex;align-items:center;justify-content:center}',
        '.taw-x:hover{background:#2a2a3e;color:#fff}',
        # Messages
        '#taw-m{flex:1;overflow-y:auto;padding:14px;display:flex;'
        'flex-direction:column;gap:8px;min-height:200px;max-height:360px}',
        '#taw-m::-webkit-scrollbar{width:4px}',
        '#taw-m::-webkit-scrollbar-thumb{background:#2a2a3e;border-radius:2px}',
        '.taw-msg{max-width:85%;padding:10px 14px;border-radius:14px;'
        'font-size:13px;line-height:1.5;word-break:break-word}',
        f'.taw-msg.u{{align-self:flex-end;background:{agent_color};color:#fff;'
        f'border-bottom-right-radius:4px}}',
        '.taw-msg.a{align-self:flex-start;background:#1e1e2e;color:#e5e7eb;'
        'border-bottom-left-radius:4px;white-space:pre-wrap}',
        '.taw-msg.t{color:#6b7280;font-style:italic;font-size:12px}',
        '.taw-empty{text-align:center;color:#6b7280;padding:40px 16px}',
        # Action button chips
        '.taw-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}',
        f'.taw-chip{{display:inline-flex;align-items:center;padding:6px 14px;'
        f'border-radius:20px;border:1px solid {agent_color};color:{cl};'
        f'font-size:12px;cursor:pointer;background:transparent;'
        f'font-family:inherit;transition:all 0.15s;-webkit-appearance:none}}',
        f'.taw-chip:hover{{background:{agent_color};color:#fff}}',
        f'.taw-l2{{background:{agent_color}15;border-color:{agent_color}60;font-weight:500}}',
        # Input
        '.taw-iw{display:flex;align-items:center;padding:10px 12px;'
        'border-top:1px solid #1e1e2e;background:#0e0e18;gap:8px}',
        '#taw-i{flex:1;background:#1e1e2e;border:1px solid #2a2a3e;'
        'border-radius:20px;padding:8px 14px;color:#fff;font-size:13px;'
        'outline:none;font-family:inherit;-webkit-appearance:none}',
        '#taw-i::placeholder{color:#6b7280}',
        f'#taw-i:focus{{border-color:{agent_color}}}',
        f'#taw-s{{width:34px;height:34px;border-radius:50%;border:none;'
        f'background:{agent_color};color:#fff;cursor:pointer;font-size:16px;'
        f'flex-shrink:0;display:flex;align-items:center;'
        f'justify-content:center;-webkit-appearance:none}}',
        f'#taw-s:hover{{background:{cd15}}}',
        '</style>',
        # ── JS ──
        '<script>',
        'document.addEventListener("DOMContentLoaded",function(){',
        'var B=window.__TOUP_AGENT_BRIDGE;if(!B)return;',
        # Hide widget if inside platform iframe — parent React panel handles chat
        'try{if(window.self!==window.top&&document.referrer.indexOf(window.location.host)>=0)return}catch(e){}',
        'window.__TOUP_AGENT_UI_INJECTED=true;',
        # Create root
        'var r=document.createElement("div");r.id="taw";',
        "r.innerHTML='", tpl_js, "';",
        'document.body.appendChild(r);',
        # Element refs
        'var btn=document.getElementById("taw-b"),'
        'pnl=document.getElementById("taw-p"),'
        'msgs=document.getElementById("taw-m"),'
        'inp=document.getElementById("taw-i"),'
        'sbtn=document.getElementById("taw-s"),'
        'd1=r.querySelector(".taw-dot"),'
        'd2=r.querySelector(".taw-hd");',
        # Toggle
        'btn.onclick=function(){'
        'pnl.classList.add("show");btn.style.display="none";inp.focus()};',
        'r.querySelector(".taw-x").onclick=function(){'
        'pnl.classList.remove("show");btn.style.display=""};',
        # Status polling
        'function us(){'
        'var c=B.isConnected?"#22c55e":"#6b7280";'
        'd1.style.backgroundColor=c;d2.style.backgroundColor=c}',
        'setInterval(us,2000);us();',
        # Add message helper — parses [[buttons]] and basic markdown
        'function am(t,s){'
        'var e=msgs.querySelector(".taw-empty");if(e)e.remove();'
        'var d=document.createElement("div");d.className="taw-msg "+s;'
        # Strip [[reaction:EMOJI]] patterns
        't=t.replace(/\\[\\[reaction:[^\\]]*\\]\\]/g,"");'
        # Extract [[button:LABEL|DATA]] and [[OPTION]] patterns
        'var btns=[];'
        't=t.replace(/\\[\\[button:([^|\\]]+)\\|([^\\]]+)\\]\\]/g,function(_,l,d){btns.push(l);return""});'
        't=t.replace(/\\[\\[([^\\]]{1,50})\\]\\]/g,function(_,l){btns.push(l);return""});'
        't=t.trim();'
        # Sanitize HTML entities then render basic markdown
        'if(s==="a"){'
        'var h=t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");'
        'h=h.replace(/\\*\\*(.+?)\\*\\*/g,"<b>$1</b>");'
        'h=h.replace(/\\*(.+?)\\*/g,"<i>$1</i>");'
        'h=h.replace(/`([^`]+)`/g,"<code style=\\"background:#2a2a3e;padding:1px 4px;border-radius:3px\\">$1</code>");'
        'h=h.replace(/\\n/g,"<br>");'
        'd.innerHTML=h;'
        # Render button chips if any
        'if(btns.length){'
        'var cw=document.createElement("div");cw.className="taw-chips";'
        'for(var i=0;i<btns.length;i++){(function(label){'
        'var b=document.createElement("button");b.className="taw-chip";'
        'b.textContent=label;b.onclick=function(){am(label,"u");B.sendMessage(label);'
        'var tp=document.createElement("div");tp.className="taw-msg a t";tp.id="taw-tp";'
        'tp.textContent="Thinking...";msgs.appendChild(tp);msgs.scrollTop=msgs.scrollHeight};'
        'cw.appendChild(b)})(btns[i])}'
        'd.appendChild(cw)}'
        '}else{d.textContent=t}'
        'msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight}',
        # Send
        'function snd(){'
        'var t=inp.value.trim();if(!t)return;'
        'am(t,"u");B.sendMessage(t);inp.value="";'
        'var tp=document.getElementById("taw-tp");if(!tp){'
        'tp=document.createElement("div");'
        'tp.className="taw-msg a t";tp.id="taw-tp";'
        'tp.textContent="Thinking...";'
        'msgs.appendChild(tp);msgs.scrollTop=msgs.scrollHeight}}',
        'inp.addEventListener("keydown",function(e){'
        'if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();snd()}});',
        'sbtn.onclick=snd;',
        # Receive messages
        'B.onAgentMessage(function(t){'
        'var tp=document.getElementById("taw-tp");if(tp)tp.remove();'
        'am(t,"a")});',
        # Tool activity → typing indicator with tool name
        'B.onToolActivity(function(tool,done){'
        'var tp=document.getElementById("taw-tp");'
        'if(!done&&!tp){'
        'tp=document.createElement("div");'
        'tp.className="taw-msg a t";tp.id="taw-tp";'
        'var tn=tool?tool.replace(/^app_[a-z0-9_]+__/,"").replace(/_/g," "):"";'
        'tp.innerHTML=tn?"&#x1f527; Using "+tn+"...":"Thinking...";'
        'msgs.appendChild(tp);msgs.scrollTop=msgs.scrollHeight}'
        'else if(!done&&tp){'
        'var tn2=tool?tool.replace(/^app_[a-z0-9_]+__/,"").replace(/_/g," "):"";'
        'if(tn2)tp.innerHTML="&#x1f527; Using "+tn2+"..."}'
        'else if(done&&tp)tp.textContent="Almost done..."});',
        # Hide generated AgentPlaceholder (scans positioned bottom-right elements)
        # Uses MutationObserver + interval to catch late-rendered elements
        'function hg(){'
        'var all=document.body.getElementsByTagName("div");'
        'for(var i=0;i<all.length;i++){'
        'var el=all[i];if(el.closest("#taw"))continue;'
        'try{var cs=getComputedStyle(el);'
        'if((cs.position==="absolute"||cs.position==="fixed")'
        '&&parseInt(cs.zIndex)>=9000){'
        'var b=parseInt(cs.bottom),ri=parseInt(cs.right);'
        'if(!isNaN(b)&&!isNaN(ri)&&b<=120&&ri<=60)'
        'el.style.setProperty("display","none","important")'
        '}}catch(e){}}}',
        # Run hg() periodically for 15s to catch late renders, plus MutationObserver
        'var hgCount=0,hgTimer=setInterval(function(){hg();if(++hgCount>15){clearInterval(hgTimer)}},1000);',
        'try{new MutationObserver(function(){hg()}).observe(document.body,{childList:true,subtree:true})}catch(e){}',
        '});',
        '</script>',
    ]
    return ''.join(parts)


# ── Agent proxy helpers ─────────────────────────────────────

async def _get_agent(user_id: str, db: AsyncSession) -> Optional[Tuple[str, str, str]]:
    """Return (agent_url, api_key, agent_color) or None."""
    result = await db.execute(
        select(
            AgentConfig.agent_url,
            AgentConfig.agent_api_key,
            AgentConfig.agent_color,
        )
        .where(
            AgentConfig.user_id == user_id,
            AgentConfig.deploy_status == "active",
        )
    )
    row = result.first()
    if row and row.agent_url and row.agent_api_key:
        return (row.agent_url, row.agent_api_key, row.agent_color or "#9B59B6")
    return None


async def _proxy(
    agent_url: str, agent_api_key: str, path: str,
    method: str = "GET", body: Optional[dict] = None,
    timeout: float = 30.0,
):
    url = f"{agent_url}/api/apps/{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            headers = {"X-Agent-Key": agent_api_key}
            if method == "GET":
                resp = await client.get(url, headers=headers)
            elif method == "POST":
                resp = await client.post(url, headers=headers, json=body or {})
            elif method == "DELETE":
                resp = await client.delete(url, headers=headers)
            else:
                return None
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        logger.warning("Apps proxy %s %s failed: %s", method, url, e)
        raise HTTPException(502, "Agent unreachable")


def _require(info):
    if not info:
        raise HTTPException(503, "Agent not deployed or not reachable.")
    return info


def _rewrite_app_urls(data):
    """Replace raw VPS web_url with platform proxy path."""
    def _fix(app: dict):
        if isinstance(app, dict) and app.get("id"):
            app["web_url"] = f"/api/apps/{app['id']}/preview/"
        return app

    if isinstance(data, list):
        return [_fix(a) for a in data]
    elif isinstance(data, dict):
        return _fix(data)
    return data


# ── Server info ─────────────────────────────────────────────

@router.get("/server")
async def get_server_info(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get VPS system info (CPU, RAM, disk, uptime, running apps)."""
    agent_info = await _get_agent(current_user.id, db)
    if not agent_info:
        return JSONResponse(content={"status": "offline"}, status_code=200)
    agent_url, key, _ = agent_info
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{agent_url}/agent/system",
                headers={"X-Agent-Key": key},
            )
            if resp.status_code == 200:
                data = resp.json()
                data["status"] = "online"
                data["ip"] = agent_url.replace("http://", "").replace("https://", "").split(":")[0]
                return JSONResponse(content=data)
    except Exception as e:
        logger.warning("Server info proxy failed: %s", e)
    return JSONResponse(content={"status": "offline"})


# ── Agent capabilities ──────────────────────────────────────

@router.get("/capabilities")
async def get_capabilities(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Fetch all loaded tools and skills from the VPS agent."""
    agent_info = await _get_agent(current_user.id, db)
    if not agent_info:
        return JSONResponse(content={"core_tools": [], "skills": [], "total_tools": 0})
    agent_url, key, _ = agent_info
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{agent_url}/agent/capabilities",
                headers={"X-Agent-Key": key},
            )
            if resp.status_code == 200:
                return JSONResponse(content=resp.json())
    except Exception as e:
        logger.warning("Capabilities proxy failed: %s", e)
    return JSONResponse(content={"core_tools": [], "skills": [], "total_tools": 0})


# ── App endpoints ───────────────────────────────────────────

@router.get("/")
async def list_apps(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key, _ = _require(await _get_agent(current_user.id, db))
    url = f"{agent_url}/api/apps/"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers={"X-Agent-Key": key})
            data = _rewrite_app_urls(resp.json())
            return JSONResponse(content=data, status_code=resp.status_code)
    except Exception as e:
        logger.warning("Apps proxy list failed: %s", e)
        raise HTTPException(502, "Agent unreachable")


@router.get("/jobs/")
async def list_jobs(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key, _ = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, "jobs/")


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key, _ = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, f"jobs/{job_id}")


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key, _ = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, f"jobs/{job_id}", method="DELETE")


# ── Web Preview Proxy ──────────────────────────────────────
# Reverse-proxies the Expo web dev server through toup.ai so the
# mobile app can load it over HTTPS without direct VPS port access.

async def _get_app_web_port(app_id: str, agent_url: str, agent_api_key: str) -> int:
    """Fetch the app's web_port from the VPS agent API."""
    url = f"{agent_url}/api/apps/{app_id}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers={"X-Agent-Key": agent_api_key})
        if resp.status_code != 200:
            raise HTTPException(404, "App not found")
        data = resp.json()
        web_port = data.get("web_port")
        if not web_port:
            raise HTTPException(503, "App web server not running")
        return web_port


async def _get_user_from_token(token: str, db: AsyncSession):
    """Validate JWT from query param for preview auth."""
    from app.services.auth_service import decode_access_token
    try:
        user_id = decode_access_token(token)
        if not user_id:
            return None
        return type("User", (), {"id": user_id})()
    except Exception:
        return None


@router.get("/{app_id}/preview/{path:path}")
@router.get("/{app_id}/preview")
async def preview_proxy(
    app_id: str, request: Request,
    path: str = "",
    token: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Reverse-proxy the Expo web dev server for in-app preview.

    Auth via ?token=JWT (SFSafariViewController can't send Bearer headers).
    Injects <base href> into HTML so sub-resources (JS bundles, etc.)
    route back through this proxy instead of hitting toup.ai root.
    """
    # Try Bearer header first, then query param token, then cookie.
    # Also resolve the *actual* JWT so the injected bridge script always
    # has a valid token for its WebSocket connection.
    user = None
    resolved_token = token  # from query param
    try:
        user = await get_current_user(request, db)
        # If authenticated via Bearer, extract the JWT from the header
        # so we can pass it to the bridge script.
        if user and not resolved_token:
            auth_hdr = request.headers.get("authorization", "")
            if auth_hdr.lower().startswith("bearer "):
                resolved_token = auth_hdr[7:].strip()
    except Exception:
        pass
    if not user and token:
        user = await _get_user_from_token(token, db)
    if not user:
        cookie_token = request.cookies.get("preview_token")
        if cookie_token:
            user = await _get_user_from_token(cookie_token, db)
            if not resolved_token:
                resolved_token = cookie_token
    if not user:
        raise HTTPException(401, "Not authenticated")

    agent_info = await _get_agent(user.id, db)
    agent_url, key, agent_color = _require(agent_info)

    from urllib.parse import urlparse
    vps_host = urlparse(agent_url).hostname

    web_port = await _get_app_web_port(app_id, agent_url, key)
    target = f"http://{vps_host}:{web_port}/{path}"

    # Forward query string (except our token param)
    params = {k: v for k, v in request.query_params.items() if k != "token"}
    if params:
        target += f"?{urlencode(params)}"

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(target)
            content_type = resp.headers.get("content-type", "text/html")

            body = resp.content

            # For HTML responses (the initial page), inject <base href> so
            # relative URLs like /index.ts.bundle resolve through the proxy
            # path instead of toup.ai root.
            # Also rewrite script src to include ?token= so sub-resource
            # requests authenticate without relying on cookies (WebView
            # may not send cookies for cross-origin sub-requests).
            if "text/html" in content_type:
                base_href = f"/api/apps/{app_id}/preview/"
                base_tag = f'<base href="{base_href}">'
                # Inject deterministic agent bridge — connects the app's
                # AgentPlaceholder to the user's real agent via WebSocket.
                # This runs BEFORE the Expo bundle, so window.__TOUP_AGENT_BRIDGE
                # is ready when the generated agentBridge.ts loads.
                agent_bridge_script = _build_agent_bridge_script(
                    resolved_token or "", app_id
                )
                # Meta charset MUST be first in <head> — WKWebView uses it to
                # decide text encoding before parsing any other content.
                meta_charset = '<meta charset="utf-8">'
                # Emoji font CSS — iOS WKWebView doesn't auto-fallback to emoji fonts.
                # react-native-web sets font via shorthand which blocks emoji fallback.
                # Use both font-family AND font shorthand override to be bulletproof.
                emoji_css = (
                    '<style id="emoji-fix">'
                    '*, *::before, *::after { '
                    'font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
                    'Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", '
                    '"Noto Color Emoji" !important; }'
                    '</style>'
                )
                agent_widget_script = _build_agent_widget_script(
                    agent_color=agent_color
                )
                html = body.decode("utf-8", errors="replace")
                html = html.replace("<head>", f"<head>\n{meta_charset}\n{base_tag}\n{agent_bridge_script}\n{agent_widget_script}\n{emoji_css}", 1)
                # Rewrite absolute src="/..." to relative so <base href>
                # routes them through the preview proxy path.
                # Also inject ?token= so bundle requests are authenticated
                # (WebView may not send cookies for sub-resource requests).
                def _rewrite_src(m):
                    src = m.group(1)
                    # Strip leading / to make relative (so <base href> applies)
                    if src.startswith("/"):
                        src = src[1:]
                    # Inject auth token
                    if resolved_token:
                        sep = "&" if "?" in src else "?"
                        src = f"{src}{sep}token={resolved_token}"
                    return f'src="{src}"'
                html = re.sub(r'src="(/[^"]*)"', _rewrite_src, html)
                body = html.encode("utf-8")

            # Ensure charset=utf-8 is in Content-Type for text responses.
            # iOS WKWebView uses the HTTP header (not <meta charset>) to decode,
            # and defaults to ASCII when charset is missing — corrupting emoji bytes.
            resp_content_type = content_type
            if "text/html" in content_type and "charset" not in content_type:
                resp_content_type = "text/html; charset=utf-8"
            elif "javascript" in content_type and "charset" not in content_type:
                resp_content_type = content_type + "; charset=utf-8"

            response = StreamingResponse(
                iter([body]),
                status_code=resp.status_code,
                media_type=resp_content_type,
            )

            # Set auth cookie so sub-resource requests (JS bundles, etc.)
            # are authenticated without needing ?token= on every URL.
            if resolved_token and "text/html" in content_type:
                response.set_cookie(
                    key="preview_token",
                    value=resolved_token,
                    max_age=3600,
                    httponly=True,
                    samesite="none",
                    secure=True,
                )

            return response
    except Exception as e:
        logger.warning("Preview proxy failed: %s → %s", target, e)
        raise HTTPException(502, "App preview unreachable")


@router.post("/{app_id}/chat")
async def app_chat_proxy(
    app_id: str, request: Request,
    token: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """HTTP→WebSocket chat proxy — bridge sends fetch(), we relay to VPS agent WS.

    Browser can't reliably open WebSocket through Cloudflare→Railway→VPS chain,
    but Railway CAN open WS to VPS directly (server-to-server).
    So: browser POSTs here → we open WS to VPS agent → collect response → SSE back.
    """
    import asyncio

    # Authenticate (same as preview_proxy)
    user = None
    try:
        user = await get_current_user(request, db)
    except Exception:
        pass
    if not user and token:
        user = await _get_user_from_token(token, db)
    if not user:
        cookie_token = request.cookies.get("preview_token")
        if cookie_token:
            user = await _get_user_from_token(cookie_token, db)
    if not user:
        raise HTTPException(401, "Not authenticated")

    agent_info = await _get_agent(user.id, db)
    agent_url, key, _ = _require(agent_info)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    text = body.get("text", "")
    if not text:
        raise HTTPException(400, "Missing 'text' field")

    # Build WS URL to VPS agent
    ws_url = agent_url.replace("https://", "wss://").replace("http://", "ws://")
    if not ws_url.endswith("/"):
        ws_url += "/api/ws/chat"
    else:
        ws_url += "api/ws/chat"
    full_ws_url = f"{ws_url}?agent_key={key}"

    session_id = body.get("session_id", f"app-{app_id}")

    async def stream_via_ws():
        try:
            import websockets
        except ImportError:
            yield f"data: {json.dumps({'type': 'error', 'text': 'Server WebSocket library not available'})}\n\n"
            return

        try:
            async with websockets.connect(
                full_ws_url,
                max_size=10 * 1024 * 1024,
                open_timeout=15,
                close_timeout=5,
            ) as ws:
                # Send the chat message
                _chat_msg = {
                    "type": "message",
                    "text": text,
                    "session_id": session_id,
                    "app_id": app_id,
                    "channel": "app",
                }
                if body.get("layer2"):
                    _chat_msg["layer2"] = True
                msg = json.dumps(_chat_msg)
                await ws.send(msg)

                # Collect streaming response
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=120)
                    except asyncio.TimeoutError:
                        yield f"data: {json.dumps({'type': 'error', 'text': 'Agent response timeout'})}\n\n"
                        break

                    if raw == "pong":
                        continue

                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    # Forward as SSE
                    yield f"data: {raw}\n\n"

                    # Stop after "done" or "error"
                    if data.get("type") in ("done", "error"):
                        break

        except Exception as e:
            logger.warning("App chat WS proxy error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'text': f'Agent connection error: {e}'})}\n\n"

    return StreamingResponse(
        stream_via_ws(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{app_id}")
async def get_app(app_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key, _ = _require(await _get_agent(current_user.id, db))
    url = f"{agent_url}/api/apps/{app_id}"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers={"X-Agent-Key": key})
            data = _rewrite_app_urls(resp.json())
            return JSONResponse(content=data, status_code=resp.status_code)
    except Exception as e:
        logger.warning("Apps proxy get failed: %s", e)
        raise HTTPException(502, "Agent unreachable")


@router.post("/{app_id}/start")
async def start_app(app_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key, _ = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, f"{app_id}/start", method="POST", timeout=60.0)


@router.post("/{app_id}/stop")
async def stop_app(app_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key, _ = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, f"{app_id}/stop", method="POST")


@router.post("/{app_id}/publish-web")
async def publish_web(app_id: str, request: Request, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key, _ = _require(await _get_agent(current_user.id, db))
    body = None
    try:
        body = await request.json()
    except Exception:
        pass
    return await _proxy(agent_url, key, f"{app_id}/publish-web", method="POST", body=body, timeout=120.0)


@router.post("/{app_id}/push-github")
async def push_github(app_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key, _ = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, f"{app_id}/push-github", method="POST", timeout=60.0)


@router.delete("/{app_id}")
async def delete_app(app_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key, _ = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, app_id, method="DELETE")
