"""
In-memory rate limiter for login attempts.

Tracks failed login attempts by IP + email combination.
Allows 5 attempts per 5-minute window. Resets on successful login.
"""

import time
from collections import defaultdict
from typing import Optional


class LoginRateLimiter:
    def __init__(self, max_attempts: int = 5, window_seconds: int = 300):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: dict[str, list[float]] = defaultdict(list)

    def _key(self, ip: str, email: str) -> str:
        return f"{ip}:{email.lower()}"

    def check(self, ip: str, email: str) -> Optional[int]:
        """Returns None if allowed, or seconds until retry if rate-limited."""
        key = self._key(ip, email)
        now = time.time()
        cutoff = now - self.window_seconds
        self._attempts[key] = [t for t in self._attempts[key] if t > cutoff]
        if len(self._attempts[key]) >= self.max_attempts:
            oldest = self._attempts[key][0]
            return int(oldest + self.window_seconds - now) + 1
        return None

    def record(self, ip: str, email: str):
        """Record a failed login attempt."""
        self._attempts[self._key(ip, email)].append(time.time())

    def clear(self, ip: str, email: str):
        """Clear attempts on successful login."""
        self._attempts.pop(self._key(ip, email), None)


login_rate_limiter = LoginRateLimiter()


class SignupRateLimiter:
    """IP-keyed signup rate limiter — Phase 3 of the prewarm hardening
    work (docs/onboarding/prewarm-phase0.md). The login limiter is
    keyed on (ip, email) because brute-force testers cycle email
    addresses; signup spam is shaped differently — same IP creating
    many accounts with different emails. So we key on IP alone.

    Default: 5 signups per IP per hour. Set
    `settings.signup_rate_limit_per_hour=0` to disable (dev / CI).

    Mirrors LoginRateLimiter's interface so call sites read the same
    way (check → record after attempt) but does not expose a clear()
    method — successful signup is not a "good actor" signal here, just
    one fewer slot for the next hour.
    """

    def __init__(self, max_attempts: int = 5, window_seconds: int = 3600):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: dict[str, list[float]] = defaultdict(list)

    def check(self, ip: str) -> Optional[int]:
        """Returns None if allowed, or seconds until retry if rate-limited."""
        if self.max_attempts <= 0:
            return None  # disabled
        now = time.time()
        cutoff = now - self.window_seconds
        self._attempts[ip] = [t for t in self._attempts[ip] if t > cutoff]
        if len(self._attempts[ip]) >= self.max_attempts:
            oldest = self._attempts[ip][0]
            return int(oldest + self.window_seconds - now) + 1
        return None

    def record(self, ip: str):
        """Record a signup attempt (whether successful or not — the
        cost is paid before the user row is checked, since spam
        signups can't be distinguished from real ones at the IP layer
        before processing)."""
        if self.max_attempts <= 0:
            return
        self._attempts[ip].append(time.time())


signup_rate_limiter = SignupRateLimiter()
