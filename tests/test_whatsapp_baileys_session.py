"""Unit tests for app.agent.channels.whatsapp_baileys_session.

Pure functions — no DB, no neonize, no network. Each test class uses
``setup_method`` to allocate a fresh tempdir + set the
``WHATSAPP_BAILEYS_STORE_DIR`` env var, so the tests work both under
real pytest and under the project's direct-execution harness (which
doesn't inject pytest fixtures).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _isolate_store_dir():
    """Allocate a fresh temp dir + point the helper module at it.
    Returns (tmp_path, prev_env_value) so callers can restore env on
    teardown — preventing cross-test pollution under both harnesses.
    """
    tmp = tempfile.mkdtemp(prefix="wa-test-")
    prev = os.environ.get("WHATSAPP_BAILEYS_STORE_DIR")
    os.environ["WHATSAPP_BAILEYS_STORE_DIR"] = tmp
    return tmp, prev


def _restore_store_dir(prev):
    if prev is None:
        os.environ.pop("WHATSAPP_BAILEYS_STORE_DIR", None)
    else:
        os.environ["WHATSAPP_BAILEYS_STORE_DIR"] = prev


# ── store_dir / ensure_store_dir / store_path ─────────────────────


class TestStorePath:
    def setup_method(self):
        self._tmp, self._prev = _isolate_store_dir()

    def teardown_method(self):
        _restore_store_dir(self._prev)

    def test_store_dir_honours_env(self):
        from app.agent.channels.whatsapp_baileys_session import store_dir
        assert str(store_dir()) == self._tmp

    def test_ensure_store_dir_creates_with_700_perms(self):
        from app.agent.channels.whatsapp_baileys_session import ensure_store_dir
        Path(self._tmp).rmdir()
        p = ensure_store_dir()
        assert p.exists() and p.is_dir()
        assert p.stat().st_mode & 0o777 == 0o700

    def test_ensure_store_dir_idempotent(self):
        from app.agent.channels.whatsapp_baileys_session import ensure_store_dir
        p1 = ensure_store_dir()
        p2 = ensure_store_dir()
        assert p1 == p2

    def test_store_path_filename(self):
        from app.agent.channels.whatsapp_baileys_session import store_path
        assert store_path().name == "store.db"


# ── session_file_exists / clear_session_files ────────────────────


class TestSessionLifecycle:
    def setup_method(self):
        self._tmp, self._prev = _isolate_store_dir()

    def teardown_method(self):
        _restore_store_dir(self._prev)

    def test_session_file_exists_false_on_empty(self):
        from app.agent.channels.whatsapp_baileys_session import session_file_exists
        assert session_file_exists() is False

    def test_session_file_exists_false_for_zero_byte_file(self):
        """A truncated file is "no session" — neonize emits a fresh
        QR rather than try to use it."""
        from app.agent.channels.whatsapp_baileys_session import (
            session_file_exists, store_path, ensure_store_dir,
        )
        ensure_store_dir()
        store_path().write_bytes(b"")
        assert session_file_exists() is False

    def test_session_file_exists_true_after_write(self):
        from app.agent.channels.whatsapp_baileys_session import (
            session_file_exists, store_path, ensure_store_dir,
        )
        ensure_store_dir()
        store_path().write_bytes(b"\x00" * 16)
        assert session_file_exists() is True

    def test_clear_session_files_removes_store(self):
        from app.agent.channels.whatsapp_baileys_session import (
            clear_session_files, store_path, ensure_store_dir,
        )
        ensure_store_dir()
        store_path().write_bytes(b"data")
        clear_session_files()
        assert not store_path().exists()

    def test_clear_session_files_no_op_when_dir_missing(self):
        """Removing the store dir before clear() should not raise."""
        from app.agent.channels.whatsapp_baileys_session import clear_session_files
        Path(self._tmp).rmdir()
        clear_session_files()


# ── jid_to_e164 ──────────────────────────────────────────────────


class TestJidToE164:
    def test_canonical_user_jid(self):
        from app.agent.channels.whatsapp_baileys_session import jid_to_e164
        assert jid_to_e164("14155552671@s.whatsapp.net") == "+14155552671"

    def test_device_suffixed_jid_strips_suffix(self):
        from app.agent.channels.whatsapp_baileys_session import jid_to_e164
        assert jid_to_e164("14155552671:42@s.whatsapp.net") == "+14155552671"

    def test_legacy_c_us_server_accepted(self):
        from app.agent.channels.whatsapp_baileys_session import jid_to_e164
        assert jid_to_e164("14155552671@c.us") == "+14155552671"

    def test_group_jid_rejected(self):
        from app.agent.channels.whatsapp_baileys_session import jid_to_e164
        assert jid_to_e164("14155552671-1234567890@g.us") == ""

    def test_lid_jid_rejected_when_no_mapping(self):
        from app.agent.channels.whatsapp_baileys_session import jid_to_e164
        assert jid_to_e164("12345@lid") == ""

    def test_empty(self):
        from app.agent.channels.whatsapp_baileys_session import jid_to_e164
        assert jid_to_e164("") == ""

    def test_garbage(self):
        from app.agent.channels.whatsapp_baileys_session import jid_to_e164
        assert jid_to_e164("not a jid") == ""

    def test_non_digit_user_part(self):
        from app.agent.channels.whatsapp_baileys_session import jid_to_e164
        assert jid_to_e164("abc@s.whatsapp.net") == ""


# ── e164_to_jid ──────────────────────────────────────────────────


class TestE164ToJid:
    def test_with_plus(self):
        from app.agent.channels.whatsapp_baileys_session import e164_to_jid
        assert e164_to_jid("+14155552671") == "14155552671@s.whatsapp.net"

    def test_without_plus(self):
        from app.agent.channels.whatsapp_baileys_session import e164_to_jid
        assert e164_to_jid("14155552671") == "14155552671@s.whatsapp.net"

    def test_empty(self):
        from app.agent.channels.whatsapp_baileys_session import e164_to_jid
        assert e164_to_jid("") == ""

    def test_garbage_with_plus(self):
        from app.agent.channels.whatsapp_baileys_session import e164_to_jid
        assert e164_to_jid("+abc") == ""


# ── render_qr_png_data_url ───────────────────────────────────────


class TestRenderQr:
    def test_empty_returns_none(self):
        from app.agent.channels.whatsapp_baileys_session import render_qr_png_data_url
        assert render_qr_png_data_url("") is None

    def test_renders_png_data_url_when_qrcode_installed(self):
        """Skipped silently when qrcode isn't installed in the dev
        env. In production neonize's requirements drag qrcode in via
        our requirements.txt addition."""
        try:
            import qrcode  # noqa: F401
        except ImportError:
            return  # silent skip — see docstring
        from app.agent.channels.whatsapp_baileys_session import render_qr_png_data_url
        url = render_qr_png_data_url("2@AbCdEfGhIj==,XX==")
        assert url is not None
        assert url.startswith("data:image/png;base64,")
        assert len(url) > 200

    def test_renders_none_when_qrcode_missing(self):
        """If qrcode lib isn't installed (e.g. minimal dev env),
        render returns None and logs — caller falls back to the raw
        string. We force the path by injecting None into sys.modules
        so a fresh ``import qrcode`` raises ImportError."""
        import sys
        from app.agent.channels import whatsapp_baileys_session as mod

        saved = sys.modules.get("qrcode")
        # Force ImportError by setting sys.modules["qrcode"] = None
        sys.modules["qrcode"] = None  # type: ignore[assignment]
        try:
            assert mod.render_qr_png_data_url("hello") is None
        finally:
            if saved is not None:
                sys.modules["qrcode"] = saved
            else:
                sys.modules.pop("qrcode", None)
