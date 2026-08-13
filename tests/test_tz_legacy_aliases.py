"""Legacy IANA alias resolution (Last Mile, W-6 tail).

Found live: a real user's timezone is `US/Pacific`. python:3.12-slim's
Debian tzdata omits backward-compat links (Debian moved them to
tzdata-legacy), so inside agent containers ZoneInfo("US/Pacific") raised
ZoneInfoNotFoundError and every user-local-date computation (voice day
header, routine idempotency dates) silently fell back to UTC wording.
Red-proof was live: `docker exec <agent> python -c 'ZoneInfo("US/Pacific")'`
→ ZoneInfoNotFoundError, while America/* zones resolved.

The pip `tzdata` package bundles the FULL database including links, and
zoneinfo consults it per-key when the system path misses, so pinning it
in every requirements set kills the class for all images at once.
"""

import pytest
from zoneinfo import ZoneInfo


# Every timezone string present in production `users.timezone` on
# 2026-08-12, plus the legacy-alias families most likely to be chosen by
# clients. If this test fails, the runtime is missing tzdata links and
# user-local dates silently degrade to UTC — do not skip it.
LIVE_AND_LEGACY_ZONES = [
    "America/Toronto",
    "America/Detroit",
    "America/Port_of_Spain",
    "UTC",
    "US/Pacific",
    "US/Eastern",
    "Etc/UTC",
    "Etc/GMT+5",
]


@pytest.mark.parametrize("name", LIVE_AND_LEGACY_ZONES)
def test_zone_resolves(name):
    ZoneInfo(name)  # raises ZoneInfoNotFoundError when links are missing
