/**
 * Toup WhatsApp Sidecar — Baileys ↔ Python bridge.
 *
 * Why this exists
 * ---------------
 * We tried `neonize` (Python wrapper around `whatsmeow` Go binary) first.
 * Pairing kept failing with "Can't link new devices right now" on the
 * scanning phone, despite the same number successfully pairing real
 * WhatsApp Desktop. Root cause: Meta's anti-abuse rejects WhatsApp Web
 * clients advertising stale `version` triples, and `neonize 0.3.17` /
 * `whatsmeow` ships a baked-in version that's now too old. Baileys
 * 6.x exposes `fetchLatestBaileysVersion()` which polls
 * `web.whatsapp.com/check-update` for the freshest accepted triple —
 * the only reliable way to stay un-fingerprinted long-term.
 *
 * Architecture
 * ------------
 * One Node child process, spawned by the Python `BaileysWhatsAppChannel`
 * adapter at agent boot. Listens on `127.0.0.1:$PORT` (default 8002,
 * container-internal only — never exposed via Caddy).
 *
 *   POST /pair/start     wipe auth + start fresh QR flow
 *   POST /pair/logout    teardown + wipe on-disk auth
 *   GET  /pair/status    snapshot for /qr-status polling
 *   POST /messages/send  body { to: E.164, text }  → outbound text
 *   GET  /events         SSE stream: { type:"message", from, text, message_id, timestamp_ms }
 *   GET  /health         liveness + diagnostics
 *
 * Auth lives at $AUTH_DIR (default /data/whatsapp/auth). Multi-file
 * Baileys layout — survives container restarts.
 *
 * The 515-restart dance is mandatory: post-QR-scan Baileys throws a
 * synthetic stream-errored 515 disconnect. Caller must close the
 * socket and recreate against the same auth dir; the second socket
 * transitions to "connection: open" and pairing is durable. Skipping
 * this step is the #2 cause of "QR scan worked but agent never
 * connects" reports.
 */

import {
  makeWASocket,
  DisconnectReason,
  fetchLatestBaileysVersion,
  makeCacheableSignalKeyStore,
  useMultiFileAuthState,
} from '@whiskeysockets/baileys';
import pino from 'pino';
import QRCode from 'qrcode';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

// ── Config ──────────────────────────────────────────────────────────
const PORT = parseInt(process.env.WHATSAPP_SIDECAR_PORT || '8002', 10);
// Allowlist of phone numbers (CSV, E.164). The sidecar uses this to
// pre-resolve corresponding @lid identifiers via `onWhatsApp` once
// connected, so inbound messages from LID-encoded senders can be
// mapped back to a phone the agent's allowlist can match against.
// WhatsApp's 2024+ LID privacy migration means many inbound users
// arrive ONLY with `<opaque>@lid` in `key.remoteJid` — no phone
// number anywhere in the message frame, no native LID→PN lookup
// in Baileys 6.7.x. Pre-resolving the allowlist forward-direction
// is the only reliable workaround until Baileys ships a LID
// mapping store.
const RAW_ALLOWLIST = process.env.WHATSAPP_BAILEYS_ALLOWLIST || '';
// Auth must live on a bind-mounted path so it survives container
// recreation (rollouts, restarts, image bumps). The bridge mounts
// `/data/agents/<id>/workspace` → `/app/workspace`, so anything
// under `/app/workspace/` is durable. `/data/whatsapp/` is NOT
// mounted in the production container — putting auth there meant
// every rollout wiped the pair and the user had to relink, which
// was the actual root cause of "agent linked but ignores all
// messages" reports. Override path via env for local/dev.
const AUTH_DIR = process.env.WHATSAPP_AUTH_DIR || '/app/workspace/.whatsapp_auth';
const BROWSER_NAME = process.env.WHATSAPP_BROWSER_NAME || 'Toup';
const BROWSER_PLATFORM = process.env.WHATSAPP_BROWSER_PLATFORM || 'Chrome';
const BROWSER_VERSION = process.env.WHATSAPP_BROWSER_VERSION || '1.0.0';
const LOG_LEVEL = process.env.WHATSAPP_LOG_LEVEL || 'info';

// ── Logger ──────────────────────────────────────────────────────────
// Pino → stderr so the parent Python process captures it via journalctl.
const logger = pino({ level: LOG_LEVEL }, pino.destination(2));
const log = (msg, fields = {}) => logger.info(fields, msg);
const warn = (msg, fields = {}) => logger.warn(fields, msg);
const error = (msg, fields = {}) => logger.error(fields, msg);

// ── State ───────────────────────────────────────────────────────────
let sock = null;                         // current Baileys socket
let restarting = false;                  // re-entrancy guard during 515 dance
let sessionStatus = 'not_linked';        // not_linked | linking | linked | logged_out
let connected = false;
let selfE164 = null;
// Message IDs we generated via handleSend. messages.upsert will fire
// for our own outbound replies with key.fromMe=true; we drop those by
// ID here so the agent doesn't reply to itself in self-chat mode.
// Bounded LRU-ish window (keep last 256 ids) so memory is flat over
// long-running sessions.
const outboundMsgIds = new Set();
const OUTBOUND_HISTORY_MAX = 256;
function rememberOutbound(id) {
  if (!id) return;
  outboundMsgIds.add(id);
  if (outboundMsgIds.size > OUTBOUND_HISTORY_MAX) {
    // Drop the oldest (Set preserves insertion order). Trim back to ~3/4 max.
    const trim = outboundMsgIds.size - Math.floor(OUTBOUND_HISTORY_MAX * 0.75);
    let i = 0;
    for (const v of outboundMsgIds) {
      outboundMsgIds.delete(v);
      if (++i >= trim) break;
    }
  }
}
let selfJid = null;
let latestQrString = null;
let latestQrPngDataUrl = null;
let latestQrAt = null;                   // ISO timestamp
let inboundCount = 0;
let lastInboundAt = null;
let lastSendAt = null;
let lastSendError = null;
let bootedAt = new Date().toISOString();

const sseClients = new Set();            // open SSE response streams

// LID ↔ phone map, built lazily by `resolveAllowlistLids` once the
// socket is open + by inbound messages we successfully resolve.
// Keys are JIDs ("digits@lid" / "digits@s.whatsapp.net"), values
// are canonical E.164 ("+digits"). When an inbound message's
// `remoteJid` is a LID and we have it in the cache, we substitute
// the phone-number form so the Python adapter's E.164 allowlist
// just works.
const lidToPhone = new Map();

// Forward mapping: phone (E.164, "+digits") → preferred outbound JID.
// WhatsApp's 2024+ LID privacy migration broke the assumption that
// `<phone>@s.whatsapp.net` is a stable address. Once a contact's
// primary device is on LID, sending to the legacy `@s.whatsapp.net`
// form encrypts against a session the recipient's primary device
// doesn't trust — first message often slips through (initial session
// fan-out), then every subsequent reply renders as "Waiting for this
// message" because the ratchet has drifted relative to the LID
// session. Symptom seen end-to-end: agent's outbound shows single ✓
// and recipient's app shows the placeholder forever.
//
// Fix: track the JID form each contact actually USES — populated by
// (a) `onWhatsApp` at connection.open (LID preferred when present)
// and (b) every inbound message's `remoteJid` so we always send back
// against the same identity the user just sent us from. Inbound
// updates trump pre-resolved values because that's the freshest
// signal of what the user's current device is using right now.
const phoneToJid = new Map();

// ── Helpers ─────────────────────────────────────────────────────────

/** Ensure the auth dir exists with restrictive perms (0700). */
function ensureAuthDir() {
  fs.mkdirSync(AUTH_DIR, { recursive: true, mode: 0o700 });
}

/** Wipe everything inside AUTH_DIR but keep the directory itself. */
function wipeAuthDir() {
  if (!fs.existsSync(AUTH_DIR)) return;
  for (const entry of fs.readdirSync(AUTH_DIR)) {
    const p = path.join(AUTH_DIR, entry);
    try {
      fs.rmSync(p, { recursive: true, force: true });
    } catch (e) {
      warn('auth.wipe.entry_failed', { path: p, err: String(e) });
    }
  }
}

/** Convert "+15555550100" → "15555550100@s.whatsapp.net" for Baileys. */
function e164ToJid(e164) {
  if (!e164) return '';
  const digits = e164.replace(/\D/g, '');
  return digits ? `${digits}@s.whatsapp.net` : '';
}

/** Convert "15555550100@s.whatsapp.net" → "+15555550100". */
function jidToE164(jid) {
  if (!jid || !jid.includes('@')) return '';
  const [user, server] = jid.split('@', 2);
  if (server !== 's.whatsapp.net' && server !== 'c.us') return '';
  const digits = user.split(':', 1)[0];
  if (!/^\d+$/.test(digits)) return '';
  return `+${digits}`;
}

/** Normalise outbound JID. Accepts E.164 or full JID. */
function toJid(target) {
  if (!target) return '';
  if (target.includes('@')) return target;
  return e164ToJid(target);
}

/** Push an event to all open SSE listeners. Drops stalled clients. */
function emitEvent(payload) {
  const data = `data: ${JSON.stringify(payload)}\n\n`;
  for (const res of sseClients) {
    try {
      res.write(data);
    } catch {
      sseClients.delete(res);
    }
  }
}

/** Render a QR string into a base64 PNG data URL for the Settings UI. */
async function renderQrDataUrl(qrString) {
  try {
    return await QRCode.toDataURL(qrString, {
      errorCorrectionLevel: 'M',
      margin: 2,
      width: 320,
    });
  } catch (e) {
    error('qr.render_failed', { err: String(e) });
    return null;
  }
}

/** Parse the comma-separated allowlist env var into canonical E.164.
 *
 * Same normalisation as the Python adapter's `_normalize_e164` so the
 * two halves agree on the canonical form: digits-only, strip "00"
 * prefix, prepend "+".
 */
function parseAllowlist(raw) {
  if (!raw) return [];
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => {
      let digits = s.replace(/\D/g, '');
      if (digits.startsWith('00')) digits = digits.slice(2);
      return digits ? `+${digits}` : '';
    })
    .filter(Boolean);
}

/** Resolve every allowlisted phone to its current @lid via `onWhatsApp`.
 *
 * Populates the `lidToPhone` cache so inbound LIDs from these users
 * decode back to their phone-number form. Idempotent — safe to call
 * on every reconnect (`connection === "open"`).
 *
 * Each `onWhatsApp` reply looks like:
 *   [{ exists: true, jid: "14155552671@s.whatsapp.net",
 *      lid: "<opaque>@lid" }]
 * Either field may be absent depending on the contact's account
 * state. We cache whichever direction(s) resolve.
 */
async function resolveAllowlistLids(s) {
  const allowlist = parseAllowlist(RAW_ALLOWLIST);
  if (allowlist.length === 0) {
    log('lid.allowlist_empty');
    return;
  }
  log('lid.resolving', { count: allowlist.length });
  for (const phone of allowlist) {
    try {
      const result = await s.onWhatsApp(phone);
      const entry = Array.isArray(result) ? result[0] : null;
      if (!entry || !entry.exists) {
        warn('lid.not_on_whatsapp', { phone });
        continue;
      }
      // Cache both directions where available.
      if (entry.jid) lidToPhone.set(entry.jid, phone);
      if (entry.lid) lidToPhone.set(entry.lid, phone);
      // Forward: prefer LID for outbound (post-migration sessions are
      // anchored to LID). Only fall back to phone JID if the contact
      // hasn't migrated yet (no `lid` in the response). An inbound
      // message will overwrite this with the actually-used JID later.
      const preferred = entry.lid || entry.jid || null;
      if (preferred && !phoneToJid.has(phone)) {
        phoneToJid.set(phone, preferred);
      }
      log('lid.resolved', {
        phone,
        jid: entry.jid || null,
        lid: entry.lid || null,
        preferred_for_send: preferred,
      });
    } catch (e) {
      warn('lid.resolve_failed', { phone, err: String(e) });
    }
  }
  log('lid.resolved_total', { cache_size: lidToPhone.size });
}

/** Extract the human-readable text from an inbound Baileys message. */
function extractText(message) {
  if (!message) return '';
  return (
    message.conversation ||
    message.extendedTextMessage?.text ||
    message.imageMessage?.caption ||
    message.videoMessage?.caption ||
    message.documentMessage?.caption ||
    ''
  );
}

// ── Socket lifecycle ───────────────────────────────────────────────

/**
 * Build a fresh Baileys socket. Reads existing creds if present
 * (resume) or starts blank (fresh QR flow).
 *
 * Returns { sock, saveCreds } — caller wires lifecycle handlers.
 */
async function createSocket() {
  ensureAuthDir();
  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version, isLatest } = await fetchLatestBaileysVersion();
  log('socket.creating', { version, isLatest, auth_dir: AUTH_DIR });

  // Bump to 'debug' temporarily — Baileys silently drops a *lot* of
  // edge cases (history-sync stalls, decryption failures, prekey
  // misses) that surface only at debug level. When messages.upsert
  // refuses to fire, this is the only way to see what Baileys is
  // actually doing on the wire.
  const baileysLogger = pino({ level: process.env.WHATSAPP_BAILEYS_LOG_LEVEL || 'silent' }, pino.destination(2));
  const newSock = makeWASocket({
    auth: {
      creds: state.creds,
      keys: makeCacheableSignalKeyStore(state.keys, baileysLogger),
    },
    version,
    logger: baileysLogger,
    printQRInTerminal: false,
    browser: [BROWSER_NAME, BROWSER_PLATFORM, BROWSER_VERSION],
    syncFullHistory: false,
    markOnlineOnConnect: false,
    keepAliveIntervalMs: 25_000,
    connectTimeoutMs: 60_000,
    defaultQueryTimeoutMs: 60_000,
    generateHighQualityLinkPreview: false,
  });

  return { sock: newSock, saveCreds };
}

/**
 * Wire all event handlers on a freshly-created socket. Runs the 515
 * restart dance internally so the caller doesn't see it.
 */
function wireSocket(s, saveCreds) {
  s.ev.on('creds.update', async () => {
    try {
      await saveCreds();
    } catch (e) {
      warn('creds.save_failed', { err: String(e) });
    }
  });

  s.ev.on('connection.update', async (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      latestQrString = qr;
      latestQrAt = new Date().toISOString();
      latestQrPngDataUrl = await renderQrDataUrl(qr);
      sessionStatus = 'linking';
      log('qr.emitted', { len: qr.length });
      emitEvent({ type: 'qr', qr_emitted_at: latestQrAt });
    }

    if (connection === 'open') {
      connected = true;
      sessionStatus = 'linked';
      latestQrString = null;
      latestQrPngDataUrl = null;
      // After socket open, our own JID is on the user creds.
      try {
        selfJid = s.user?.id || null;
        selfE164 = jidToE164(selfJid || '') || null;
      } catch {
        // ignore
      }
      log('connection.open', { self: selfE164 });
      emitEvent({ type: 'connection_open', self_e164: selfE164 });

      // Pre-resolve allowlisted phone numbers to their @lid
      // counterparts so subsequent inbound LID-encoded messages can
      // be mapped back to phones for ACL matching. Run async so the
      // open event returns immediately; we don't block on it.
      resolveAllowlistLids(s).catch((e) => {
        warn('lid.resolve_unhandled', { err: String(e) });
      });
    }

    if (connection === 'close') {
      connected = false;
      // Boom-style error shape Baileys uses: error.output.statusCode.
      // Fall back to `.status` for non-Boom errors.
      const status =
        lastDisconnect?.error?.output?.statusCode ||
        lastDisconnect?.error?.status ||
        null;
      log('connection.close', { status, message: String(lastDisconnect?.error || '') });

      // Drop the socket reference — every reconnect path below will
      // reassign it via createSocket(). Forgetting to clear this was
      // a real bug: the timer-based reconnect checked `if (!sock)`
      // and noop'd because sock still pointed at the closed instance,
      // so 408 ("QR refs attempts ended") + any other transient close
      // left the agent permanently dead.
      sock = null;

      // 515 — Baileys signals "stream-errored, restart required" right
      // after pair-success. The auth dir is now populated; closing
      // and recreating completes the link.
      if (status === 515) {
        log('connection.restart_required');
        if (!restarting) {
          restarting = true;
          try {
            const { sock: next, saveCreds: nextSave } = await createSocket();
            wireSocket(next, nextSave);
            sock = next;
          } finally {
            restarting = false;
          }
        }
        return;
      }

      if (status === DisconnectReason.loggedOut || status === 401) {
        sessionStatus = 'logged_out';
        wipeAuthDir();
        selfE164 = null;
        selfJid = null;
        warn('logged_out — auth dir wiped, awaiting fresh pair');
        emitEvent({ type: 'logged_out' });
        return;
      }

      // 408 "QR refs attempts ended" — Baileys emitted a handful of
      // QR rotations and the user never scanned. Don't let the
      // session sit dead; immediately rebuild the socket so a fresh
      // QR appears and the next time the user opens Settings (or the
      // modal is already open) it just shows up.
      // Also handles the generic transient-close path (e.g. brief
      // network blip) — Baileys creds will be reused if they exist,
      // so a `linked` session reconnects without re-pairing.
      log('connection.reconnecting', { status });
      sessionStatus = 'linking';
      latestQrString = null;
      latestQrPngDataUrl = null;
      try {
        const { sock: next, saveCreds: nextSave } = await createSocket();
        wireSocket(next, nextSave);
        sock = next;
      } catch (e) {
        error('reconnect_failed', { err: String(e) });
        // Back off briefly before the next attempt so we don't
        // tight-loop on a fundamentally broken environment.
        setTimeout(async () => {
          if (sock) return;
          try {
            const { sock: next, saveCreds: nextSave } = await createSocket();
            wireSocket(next, nextSave);
            sock = next;
          } catch (e2) {
            error('reconnect_retry_failed', { err: String(e2) });
          }
        }, 5000);
      }
    }
  });

  s.ev.on('messages.upsert', async ({ messages, type }) => {
    log('messages.upsert', { type, count: messages?.length ?? 0 });
    // Baileys 6.x emits inbound under several `type` values:
    //   - "notify"  — fresh, real-time inbound (the case we care about)
    //   - "append"  — newer-than-history append (e.g. self-sent from
    //                 another device, or messages received during a
    //                 brief disconnect that Baileys is replaying)
    //   - "prepend" — backfill from history sync
    //   - "replace" — edits / revoke
    // We process notify + append; everything else is sync noise.
    if (type !== 'notify' && type !== 'append') return;
    for (const m of messages) {
      try {
        // Log the raw shape — incl. all JID-ish fields — so any future
        // WhatsApp identity-format change surfaces from logs alone.
        // Dump ALL key fields, not just the ones we know about, so a
        // newly-introduced identity field surfaces in the very next
        // message instead of needing another debug round-trip.
        log('messages.upsert.entry', {
          allKeyFields: m.key ? Object.fromEntries(Object.entries(m.key)) : null,
          messageKeys: m.message ? Object.keys(m.message) : [],
          pushName: m.pushName,
          messageContextInfoKeys: m.message?.messageContextInfo ? Object.keys(m.message.messageContextInfo) : [],
          messageContextInfo: m.message?.messageContextInfo,
        });
        // Skip our own replies: messages.upsert fires for the message
        // we just sent, key.fromMe=true with the id rememberOutbound()
        // captured. Without this guard a self-chat reply would bounce
        // straight back to the agent.
        if (m.key?.fromMe && m.key?.id && outboundMsgIds.has(m.key.id)) {
          log('inbound.skipped_own_outbound', { id: m.key.id });
          continue;
        }
        // Self-chat ("Message Yourself"): user has only one number, so
        // they DM themselves and the agent (linked device) sees it as
        // fromMe=true. We want to process those — but only when the
        // chat is the user's own number AND the id wasn't ours. Other
        // fromMe messages (user replying to a contact from their primary
        // phone) we still ignore: those aren't directed at the agent.
        if (m.key?.fromMe) {
          const remoteE164 = jidToE164(m.key?.remoteJid || '')
            || jidToE164(m.key?.remoteJidAlt || '')
            || (lidToPhone.get(m.key?.remoteJid || '') || '')
            || (lidToPhone.get(m.key?.remoteJidAlt || '') || '');
          const selfChat = !!(selfE164 && remoteE164 && remoteE164 === selfE164);
          if (!selfChat) continue;
          log('inbound.self_chat', { id: m.key.id, self: selfE164 });
        }
        if (m.key?.remoteJid?.endsWith('@broadcast')) continue;
        if (m.key?.remoteJid?.endsWith('@g.us')) continue;  // skip groups for v1
        const text = extractText(m.message || {});
        // WhatsApp's 2024+ "LID" (Local IDentifier) privacy migration
        // means inbound messages from individuals can have
        // `remoteJid` as `<opaque>@lid` instead of the legacy
        // `<phone>@s.whatsapp.net`. Baileys 6.7+ exposes the user's
        // actual phone-number JID in alternative fields:
        //   - key.remoteJidAlt   — for direct chats via LID
        //   - key.senderPn       — explicit phone-number JID
        //   - key.participantAlt — for group/multi-user contexts
        // Fall back through them in order; if everything still
        // resolves to LID, use the LID itself as the chat key. Tests
        // can still allowlist by phone because senderPn or
        // remoteJidAlt is preserved when present.
        const candidateJids = [
          m.key?.remoteJid,
          m.key?.remoteJidAlt,
          m.key?.senderPn,
          m.key?.participant,
          m.key?.participantAlt,
        ].filter(Boolean);
        let fromJid = m.key?.remoteJid || '';
        let fromE164 = '';
        // First pass: look in the pre-resolved LID cache (populated by
        // resolveAllowlistLids on connection.open). Hits = the sender
        // is on the agent's allowlist.
        for (const j of candidateJids) {
          const cached = lidToPhone.get(j);
          if (cached) { fromE164 = cached; fromJid = j; break; }
        }
        // Second pass: maybe the JID is already in phone form (legacy
        // @s.whatsapp.net). Decode directly.
        if (!fromE164) {
          for (const j of candidateJids) {
            const e = jidToE164(j);
            if (e) { fromE164 = e; fromJid = j; break; }
          }
        }
        const messageId = m.key?.id || '';
        const timestampMs = (m.messageTimestamp ? Number(m.messageTimestamp) * 1000 : Date.now());
        if (!fromE164) {
          log('inbound.dropped_no_e164', {
            tried: candidateJids,
            remoteJid: m.key?.remoteJid,
            cache_size: lidToPhone.size,
          });
          continue;
        }
        // Pin the JID the user is actually using right now, so our
        // outbound replies travel against the same e2e session.
        // remoteJid is what Baileys uses internally as the chat key,
        // so encrypting to it stays consistent with the session
        // Baileys already established for this thread.
        if (m.key?.remoteJid) {
          const previous = phoneToJid.get(fromE164);
          if (previous !== m.key.remoteJid) {
            phoneToJid.set(fromE164, m.key.remoteJid);
            log('phone_to_jid.updated_from_inbound', {
              phone: fromE164,
              jid: m.key.remoteJid,
              previous: previous || null,
            });
          }
        }
        inboundCount += 1;
        lastInboundAt = new Date().toISOString();
        log('inbound.dispatching', { from: fromE164, text_len: text.length, msg_id: messageId });
        emitEvent({
          type: 'message',
          from: fromE164,
          from_jid: fromJid,
          text,
          message_id: messageId,
          timestamp_ms: timestampMs,
          push_name: m.pushName || null,
        });
      } catch (e) {
        warn('inbound.handle_failed', { err: String(e) });
      }
    }
  });

  // Belt-and-suspenders: log every event Baileys fires so we can see
  // if `messages.upsert` is genuinely silent or if the dispatch path
  // is broken downstream.
  const _origEmit = s.ev.emit?.bind(s.ev);
  if (_origEmit) {
    s.ev.emit = (event, ...args) => {
      if (event !== 'creds.update' && event !== 'connection.update') {
        log('baileys.event', { event });
      }
      return _origEmit(event, ...args);
    };
  }
}

/** Boot — resume if creds exist, otherwise wait for /pair/start. */
async function boot() {
  ensureAuthDir();
  const credsPath = path.join(AUTH_DIR, 'creds.json');
  const hasCreds = fs.existsSync(credsPath) && fs.statSync(credsPath).size > 1;

  if (hasCreds) {
    log('boot.resuming');
    sessionStatus = 'linking';  // until connection.open
    try {
      const { sock: s, saveCreds } = await createSocket();
      wireSocket(s, saveCreds);
      sock = s;
    } catch (e) {
      error('boot.resume_failed', { err: String(e) });
      sessionStatus = 'not_linked';
    }
  } else {
    log('boot.no_creds_awaiting_pair');
    // /pair/start will create the socket on demand.
  }
}

// ── HTTP API ───────────────────────────────────────────────────────

/** Read full request body as UTF-8 string (with 1MB cap). */
function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let total = 0;
    req.on('data', (c) => {
      total += c.length;
      if (total > 1_000_000) {
        reject(new Error('body too large'));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function sendJson(res, status, body) {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
}

async function handlePairStart(req, res) {
  log('pair.start');
  // Tear down any existing socket + wipe auth so we always emit a
  // fresh QR. Idempotent — calling while already linking is a no-op.
  try {
    if (sock) {
      try { sock.end(undefined); } catch {}
      sock = null;
    }
    wipeAuthDir();
    sessionStatus = 'linking';
    connected = false;
    selfE164 = null;
    selfJid = null;
    latestQrString = null;
    latestQrPngDataUrl = null;
    latestQrAt = null;
    const { sock: s, saveCreds } = await createSocket();
    wireSocket(s, saveCreds);
    sock = s;
    sendJson(res, 200, { ok: true });
  } catch (e) {
    error('pair.start_failed', { err: String(e) });
    sendJson(res, 500, { ok: false, error: String(e) });
  }
}

async function handlePairLogout(req, res) {
  log('pair.logout');
  try {
    if (sock) {
      try { await sock.logout(); } catch {}
      try { sock.end(undefined); } catch {}
      sock = null;
    }
    wipeAuthDir();
    sessionStatus = 'not_linked';
    connected = false;
    selfE164 = null;
    selfJid = null;
    latestQrString = null;
    latestQrPngDataUrl = null;
    latestQrAt = null;
    sendJson(res, 200, { ok: true });
  } catch (e) {
    error('pair.logout_failed', { err: String(e) });
    sendJson(res, 500, { ok: false, error: String(e) });
  }
}

function handlePairStatus(req, res) {
  sendJson(res, 200, {
    session_status: sessionStatus,
    connected,
    self_e164: selfE164,
    qr_data_url: latestQrPngDataUrl,
    qr_emitted_at: latestQrAt,
  });
}

async function handleSend(req, res) {
  let payload;
  try {
    payload = JSON.parse(await readBody(req));
  } catch (e) {
    sendJson(res, 400, { ok: false, error: 'invalid JSON body' });
    return;
  }
  const to = (payload.to || '').toString();
  const text = (payload.text || '').toString();
  // Optional override — Python adapter passes the user's chosen agent
  // name (AgentConfig.agent_name). Used as the self-chat bold header
  // so the user sees "*🤖 [theirName]*" instead of a generic label.
  const displayName = (payload.display_name || '').toString().trim() || 'Agent';
  if (!to || !text) {
    sendJson(res, 400, { ok: false, error: 'to + text required' });
    return;
  }
  if (!sock || !connected) {
    sendJson(res, 503, { ok: false, error: 'not connected' });
    return;
  }
  // Resolve the JID we should encrypt against. Order:
  //   1. caller passed a full JID — trust it
  //   2. caller passed E.164 + we have a learned/resolved JID — use it
  //   3. fall back to legacy <phone>@s.whatsapp.net
  // Path (2) is the LID-migration fix: matches the JID form the user
  // last contacted us from so the Signal session stays consistent.
  let jid = '';
  let jidSource = 'unknown';
  if (to.includes('@')) {
    jid = to;
    jidSource = 'caller_jid';
  } else {
    const phone = (() => {
      // Reuse the same E.164 normalisation the parser uses so a
      // caller-supplied "4155552671" or "+1 (415) 555-2671" hits the
      // same key the cache stores.
      let digits = to.replace(/\D/g, '');
      if (digits.startsWith('00')) digits = digits.slice(2);
      return digits ? `+${digits}` : '';
    })();
    const cached = phone ? phoneToJid.get(phone) : null;
    if (cached) {
      jid = cached;
      jidSource = 'phone_to_jid_cache';
    } else {
      jid = e164ToJid(to);
      jidSource = 'e164_fallback';
    }
  }
  if (!jid) {
    sendJson(res, 400, { ok: false, error: 'invalid to' });
    return;
  }
  // Self-chat ("Message Yourself"): everything in the thread is fromMe
  // so WhatsApp paints user-typed and agent-typed messages with the
  // same green bubble — the user can't tell who said what. Wrap the
  // agent's reply with a clear bold header so it visually stands out
  // even though we can't influence the bubble colour itself.
  const isSelfChat = !!(selfE164 && (
    jidToE164(jid) === selfE164 ||
    (lidToPhone.get(jid) || '') === selfE164
  ));
  const outboundText = isSelfChat ? `*🤖 ${displayName}*\n${text}` : text;
  log('send.dispatch', { jid, jid_source: jidSource, len: outboundText.length, self_chat: isSelfChat, display_name: isSelfChat ? displayName : undefined });
  try {
    const result = await sock.sendMessage(jid, { text: outboundText });
    lastSendAt = new Date().toISOString();
    lastSendError = null;
    // Remember our own outbound id so messages.upsert can drop the
    // echo it'll fire for this message — otherwise a self-chat reply
    // bounces back into the agent and triggers an infinite loop.
    rememberOutbound(result?.key?.id || '');
    sendJson(res, 200, { ok: true, message_id: result?.key?.id || null, jid, jid_source: jidSource });
  } catch (e) {
    lastSendError = { at: new Date().toISOString(), message: String(e).slice(0, 300) };
    error('send.failed', { err: String(e), jid, jid_source: jidSource });
    sendJson(res, 502, { ok: false, error: String(e) });
  }
}

function handleEventsStream(req, res) {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'x-accel-buffering': 'no',
  });
  res.write(`: connected\n\n`);
  // Heartbeat every 25s — keeps long-lived intermediaries from idling.
  const heartbeat = setInterval(() => {
    try { res.write(`: ping\n\n`); } catch {}
  }, 25_000);
  sseClients.add(res);
  req.on('close', () => {
    clearInterval(heartbeat);
    sseClients.delete(res);
  });
}

function handleHealth(req, res) {
  sendJson(res, 200, {
    ok: true,
    booted_at: bootedAt,
    session_status: sessionStatus,
    connected,
    self_e164: selfE164,
    has_socket: sock !== null,
    inbound_count: inboundCount,
    last_inbound_at: lastInboundAt,
    last_send_at: lastSendAt,
    last_send_error: lastSendError,
    sse_listeners: sseClients.size,
    auth_dir: AUTH_DIR,
  });
}

const server = http.createServer(async (req, res) => {
  // 127.0.0.1 only — defence in depth, in case Docker net mapping ever
  // exposes the port externally by accident.
  const remote = req.socket.remoteAddress || '';
  if (!remote.includes('127.0.0.1') && !remote.includes('::1') && !remote.startsWith('::ffff:127.')) {
    res.writeHead(403);
    res.end('forbidden');
    return;
  }

  const url = new URL(req.url, `http://127.0.0.1:${PORT}`);
  const route = `${req.method} ${url.pathname}`;

  try {
    switch (route) {
      case 'POST /pair/start':   return await handlePairStart(req, res);
      case 'POST /pair/logout':  return await handlePairLogout(req, res);
      case 'GET /pair/status':   return handlePairStatus(req, res);
      case 'POST /messages/send': return await handleSend(req, res);
      case 'GET /events':        return handleEventsStream(req, res);
      case 'GET /health':        return handleHealth(req, res);
      default:
        sendJson(res, 404, { ok: false, error: 'not found' });
    }
  } catch (e) {
    error('http.unhandled', { route, err: String(e) });
    try { sendJson(res, 500, { ok: false, error: String(e) }); } catch {}
  }
});

server.listen(PORT, '127.0.0.1', async () => {
  log('sidecar.listening', { port: PORT, auth_dir: AUTH_DIR });
  await boot();
});

// Graceful shutdown — close socket cleanly so Meta sees a controlled
// disconnect rather than a TCP RST (which can spook anti-abuse).
const shutdown = async (signal) => {
  log('shutdown.received', { signal });
  try { if (sock) { sock.end(undefined); } } catch {}
  try { server.close(); } catch {}
  setTimeout(() => process.exit(0), 1500).unref();
};
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
