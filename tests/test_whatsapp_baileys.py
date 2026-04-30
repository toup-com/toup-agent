"""Unit tests for app.agent.channels.whatsapp_baileys.

Exercises the bits of the BaileysWhatsAppChannel that don't need a
running neonize session: construction, allowlist normalisation, ACL
gate, health-surface lifecycle, active-channel reference identity,
graceful no-neonize start.

The actual neonize integration (event handlers, supervisor reconnect,
QR rendering) is out of scope here — it requires a real neonize import
and the Go runtime, exercised via the live integration test the user
runs against their own number.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock


def _isolate_store():
    tmp = tempfile.mkdtemp(prefix="wa-bail-")
    prev = os.environ.get("WHATSAPP_BAILEYS_STORE_DIR")
    os.environ["WHATSAPP_BAILEYS_STORE_DIR"] = tmp
    return tmp, prev


def _restore_store(prev):
    if prev is None:
        os.environ.pop("WHATSAPP_BAILEYS_STORE_DIR", None)
    else:
        os.environ["WHATSAPP_BAILEYS_STORE_DIR"] = prev


# ── _normalize_e164 ──────────────────────────────────────────────


class TestNormaliseE164:
    def test_empty(self):
        from app.agent.channels.whatsapp_baileys import _normalize_e164
        assert _normalize_e164("") == ""

    def test_plain_canonical(self):
        from app.agent.channels.whatsapp_baileys import _normalize_e164
        assert _normalize_e164("+14155552671") == "+14155552671"

    def test_missing_plus(self):
        from app.agent.channels.whatsapp_baileys import _normalize_e164
        assert _normalize_e164("14155552671") == "+14155552671"

    def test_with_spaces_dashes_parens(self):
        from app.agent.channels.whatsapp_baileys import _normalize_e164
        assert _normalize_e164("+1 (415) 555-2671") == "+14155552671"

    def test_double_zero_intl_prefix(self):
        from app.agent.channels.whatsapp_baileys import _normalize_e164
        assert _normalize_e164("00 1 415-555-2671") == "+14155552671"

    def test_only_garbage(self):
        from app.agent.channels.whatsapp_baileys import _normalize_e164
        assert _normalize_e164("abc def") == ""


# ── BaileysWhatsAppChannel construction ──────────────────────────


class TestConstruction:
    def setup_method(self):
        self._tmp, self._prev = _isolate_store()
    def teardown_method(self):
        _restore_store(self._prev)

    def test_empty_allowlist(self):
        from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel
        ch = BaileysWhatsAppChannel()
        assert ch.allowed_numbers == set()

    def test_allowlist_normalised(self):
        from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel
        ch = BaileysWhatsAppChannel(allowed_numbers=[
            "+14155552671",
            "14155552671",
            "+1 415-555-2671",
            "+1 (415) 555-2671",
            "00 1 415 555 2671",
        ])
        # All collapse to the same canonical form
        assert ch.allowed_numbers == {"+14155552671"}

    def test_allowlist_drops_empty_entries(self):
        from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel
        ch = BaileysWhatsAppChannel(allowed_numbers=[
            "+14155552671",
            "",
            "   ",
            "abc",  # no digits → drops
        ])
        assert ch.allowed_numbers == {"+14155552671"}

    def test_channel_type_is_whatsapp(self):
        from app.agent.channels.base import ChannelType
        from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel
        ch = BaileysWhatsAppChannel()
        assert ch.channel_type == ChannelType.WHATSAPP


# ── health() pre-start ───────────────────────────────────────────


class TestHealthPreStart:
    def setup_method(self):
        self._tmp, self._prev = _isolate_store()
    def teardown_method(self):
        _restore_store(self._prev)

    def test_shape(self):
        from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel
        ch = BaileysWhatsAppChannel(allowed_numbers=["+14155552671"])
        h = ch.health()
        assert h["mode"] == "qr_link"
        assert h["configured"] is True
        assert h["started"] is False
        assert h["connected"] is False
        assert h["self_e164"] is None
        assert h["allowed_numbers_count"] == 1
        assert h["inbound_count"] == 0
        assert h["last_inbound_at"] is None
        assert h["last_send_at"] is None
        assert h["last_send_error"] is None
        assert h["session_status"] in ("not_linked", "linked")

    def test_session_status_reflects_disk_state(self):
        from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel
        from app.agent.channels.whatsapp_baileys_session import (
            ensure_store_dir, store_path,
        )
        # Pre-existing session on disk → reported as 'linked' until
        # supervisor reports otherwise
        ensure_store_dir()
        store_path().write_bytes(b"\x00" * 16)
        ch = BaileysWhatsAppChannel()
        assert ch.health()["session_status"] == "linked"


# ── pairing snapshot ─────────────────────────────────────────────


class TestPairingSnapshot:
    def setup_method(self):
        self._tmp, self._prev = _isolate_store()
    def teardown_method(self):
        _restore_store(self._prev)

    def test_empty_when_idle(self):
        from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel
        ch = BaileysWhatsAppChannel()
        snap = ch.get_pairing_status()
        assert snap["qr_data_url"] is None
        assert snap["qr_emitted_at"] is None
        assert snap["self_e164"] is None
        assert snap["connected"] is False


# ── start() with no neonize installed ────────────────────────────


class TestGracefulStart:
    def setup_method(self):
        self._tmp, self._prev = _isolate_store()
    def teardown_method(self):
        _restore_store(self._prev)

    async def test_no_neonize_no_crash(self):
        """If neonize isn't installed, start() must log + return
        cleanly. The agent boot keeps running."""
        from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel
        ch = BaileysWhatsAppChannel()
        # neonize is not in the dev env → start() bails before any
        # state mutates
        await ch.start()
        h = ch.health()
        assert h["started"] is False


# ── active-channel ref ───────────────────────────────────────────


class TestActiveChannelRef:
    def setup_method(self):
        self._tmp, self._prev = _isolate_store()
    def teardown_method(self):
        _restore_store(self._prev)

    def test_returns_none_initially(self):
        from app.agent.channels.whatsapp_baileys import (
            BaileysWhatsAppChannel,
            get_active_baileys_channel,
        )
        # Construct ≠ register; only start() sets the global
        BaileysWhatsAppChannel()
        assert get_active_baileys_channel() is None


# ── ACL behaviour via _handle_message ────────────────────────────
#
# We can't fully drive _handle_message without a fake neonize message
# struct, but we can verify the ACL math via the public allowed_numbers
# set + the _normalize_e164 helper.


class TestAcl:
    def setup_method(self):
        self._tmp, self._prev = _isolate_store()
    def teardown_method(self):
        _restore_store(self._prev)

    def test_inbound_e164_canonical_match_passes(self):
        from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel
        from app.agent.channels.whatsapp_baileys_session import jid_to_e164
        ch = BaileysWhatsAppChannel(allowed_numbers=["+14155552671"])
        sender_e164 = jid_to_e164("14155552671:1@s.whatsapp.net")
        assert sender_e164 in ch.allowed_numbers

    def test_inbound_from_outside_allowlist_blocked(self):
        from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel
        from app.agent.channels.whatsapp_baileys_session import jid_to_e164
        ch = BaileysWhatsAppChannel(allowed_numbers=["+14155552671"])
        outsider = jid_to_e164("12025550100@s.whatsapp.net")
        assert outsider not in ch.allowed_numbers

    def test_empty_allowlist_blocks_everyone(self):
        from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel
        from app.agent.channels.whatsapp_baileys_session import jid_to_e164
        ch = BaileysWhatsAppChannel()  # empty allowlist
        # Even the most "obviously legit" number fails the gate
        e = jid_to_e164("14155552671@s.whatsapp.net")
        assert e not in ch.allowed_numbers
