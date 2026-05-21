"""Smoke-test the email-send pipeline configured via EMAIL_PROVIDER.

Sends a single test verification email through whatever provider the
backend is currently configured for (Resend / SES / SMTP). Useful when
swapping provider credentials in Railway — run from the deployment
shell to confirm DKIM / SPF / App Password / etc. work end-to-end
before flipping the credit-system feature flag.

Usage:
    python -m app.scripts.test_email_send --to <recipient>

Returns:
    exit 0 — send succeeded
    exit 1 — send failed (logs explain why)
"""
from __future__ import annotations

import argparse
import asyncio
import sys


async def _async_main(to: str) -> int:
    try:
        from app.config import settings
        from app.services.email_service import send_email, render_verification_email
    except Exception as e:
        print(f"ERROR: could not import email_service: {e}", file=sys.stderr)
        return 1

    provider = (getattr(settings, "email_provider", "") or "").strip() or "(unset)"
    print(f"EMAIL_PROVIDER={provider}")
    print(f"EMAIL_FROM_ADDRESS={getattr(settings, 'email_from_address', '')}")
    print(f"Sending verification probe → {to}")

    subject, html, text = render_verification_email(
        name="Smoke Test",
        verify_url=f"{getattr(settings, 'app_public_base_url', 'https://toup.ai')}/verify-email?token=PROBE",
    )

    try:
        await send_email(
            to=to,
            subject=subject + " (probe)",
            html=html,
            text=text,
            reply_to=getattr(settings, "email_from_reply_to", None),
        )
        print("✅ Send succeeded.")
        return 0
    except Exception as e:
        print(f"❌ Send failed: {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", required=True, help="Recipient email address.")
    args = parser.parse_args()
    return asyncio.run(_async_main(args.to))


if __name__ == "__main__":
    raise SystemExit(main())
