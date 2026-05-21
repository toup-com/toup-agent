"""Regression test for TKT-LAT-010 — LRU cache on Fernet decrypt."""

from __future__ import annotations

import os

from cryptography.fernet import Fernet

# Generate a key BEFORE importing credential_crypto so _multi_fernet
# can build itself from settings (which reads env vars at load time).
_KEY = Fernet.generate_key().decode()
os.environ["PLATFORM_ENCRYPTION_KEY"] = _KEY

from app.services import credential_crypto as cc  # noqa: E402


def _reset():
    cc.reset_decrypt_cache()


def test_round_trip_decrypt_works():
    _reset()
    token = cc.encrypt_str("hunter2")
    assert cc.decrypt_str(token) == "hunter2"


def test_repeated_decrypt_is_cached():
    _reset()
    token = cc.encrypt_str("hunter2")
    # First call populates the cache
    assert cc.decrypt_str(token) == "hunter2"
    info_after_one = cc._decrypt_cached.cache_info()
    assert info_after_one.misses == 1
    assert info_after_one.hits == 0
    # Second + third call hit the cache
    cc.decrypt_str(token)
    cc.decrypt_str(token)
    info_after_three = cc._decrypt_cached.cache_info()
    assert info_after_three.misses == 1
    assert info_after_three.hits == 2


def test_reset_clears_cache():
    _reset()
    token = cc.encrypt_str("payload")
    cc.decrypt_str(token)
    assert cc._decrypt_cached.cache_info().misses == 1
    _reset()
    cc.decrypt_str(token)
    # After reset, the next call counts as a fresh miss.
    assert cc._decrypt_cached.cache_info().misses == 1
    assert cc._decrypt_cached.cache_info().hits == 0


def test_distinct_tokens_are_distinct_cache_entries():
    _reset()
    t1 = cc.encrypt_str("alpha")
    t2 = cc.encrypt_str("beta")
    assert cc.decrypt_str(t1) == "alpha"
    assert cc.decrypt_str(t2) == "beta"
    assert cc._decrypt_cached.cache_info().currsize == 2
