"""EmailService — abstraction over Resend / SES / SMTP / none.

One module is the single email-sending surface for the whole platform.
Today only verification emails go through it; future expansion (security
alerts, weekly digests, billing receipts) reuses ``send_email``.

Provider dispatch is config-driven (``settings.email_provider``). On
``none`` we log to stderr — convenient for local dev: a developer can
register, see the verification link in the logs, paste it into a
browser, and proceed without setting up an external mail provider.

Per docs/credits/design.md F13.
"""

from __future__ import annotations

import asyncio
import html as _html
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import settings


logger = logging.getLogger(__name__)


@dataclass
class EmailResult:
    success: bool
    provider: str
    message_id: Optional[str] = None
    error: Optional[str] = None


async def send_email(
    *,
    to: str,
    subject: str,
    html: str,
    text: Optional[str] = None,
    reply_to: Optional[str] = None,
) -> EmailResult:
    """Send a transactional email via the configured provider.

    Returns an :py:class:`EmailResult`. Never raises — failures are
    logged but the caller decides how to react. The signup flow, for
    example, ignores send failures and lets the user resend via
    ``POST /api/auth/send-verification``.
    """
    provider = (settings.email_provider or "none").lower().strip()
    reply_to = reply_to or settings.email_from_reply_to or None

    if provider == "resend":
        return await _send_via_resend(to, subject, html, text, reply_to)
    if provider == "ses":
        return await _send_via_ses(to, subject, html, text, reply_to)
    if provider == "smtp":
        return await _send_via_smtp(to, subject, html, text, reply_to)

    # Dev / CI fallback: log so the developer can copy the link out.
    logger.warning(
        "[email] provider=%s (no send) — to=%s subject=%r html_preview=%r",
        provider, to, subject, html[:200],
    )
    return EmailResult(success=True, provider="logged", message_id=None)


# ── Resend ──────────────────────────────────────────────────────────


async def _send_via_resend(
    to: str, subject: str, html: str, text: Optional[str], reply_to: Optional[str],
) -> EmailResult:
    key = (settings.resend_api_key or "").strip()
    if not key:
        logger.warning("[email.resend] RESEND_API_KEY not set — falling back to log")
        logger.info("[email.resend.log] to=%s subject=%r", to, subject)
        return EmailResult(success=True, provider="resend-logged")

    payload: dict = {
        "from": settings.email_from_address,
        "to": [to],
        "subject": subject,
        "html": html,
    }
    if text:
        payload["text"] = text
    if reply_to:
        payload["reply_to"] = reply_to

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        if r.status_code >= 400:
            logger.warning("[email.resend] HTTP %d for to=%s: %s", r.status_code, to, r.text[:400])
            return EmailResult(success=False, provider="resend", error=r.text[:400])
        body = r.json()
        msg_id = body.get("id")
        logger.info("[email.resend] sent to=%s id=%s", to, msg_id)
        return EmailResult(success=True, provider="resend", message_id=msg_id)
    except Exception as e:
        logger.exception("[email.resend] send failed for to=%s: %s", to, e)
        return EmailResult(success=False, provider="resend", error=str(e))


# ── SES ─────────────────────────────────────────────────────────────


async def _send_via_ses(
    to: str, subject: str, html: str, text: Optional[str], reply_to: Optional[str],
) -> EmailResult:
    """AWS SES via boto3. boto3 is sync; we wrap in run_in_executor."""
    try:
        import boto3  # type: ignore
    except ImportError:
        logger.error("[email.ses] boto3 not installed — pip install boto3")
        return EmailResult(success=False, provider="ses", error="boto3_missing")

    if not (settings.aws_ses_access_key_id and settings.aws_ses_secret_access_key):
        logger.warning("[email.ses] credentials missing — falling back to log")
        return EmailResult(success=True, provider="ses-logged")

    def _do_send() -> EmailResult:
        try:
            client = boto3.client(
                "ses",
                region_name=settings.aws_ses_region,
                aws_access_key_id=settings.aws_ses_access_key_id,
                aws_secret_access_key=settings.aws_ses_secret_access_key,
            )
            body_kwargs: dict = {"Html": {"Data": html, "Charset": "UTF-8"}}
            if text:
                body_kwargs["Text"] = {"Data": text, "Charset": "UTF-8"}
            kwargs: dict = {
                "Source": settings.email_from_address,
                "Destination": {"ToAddresses": [to]},
                "Message": {
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": body_kwargs,
                },
            }
            if reply_to:
                kwargs["ReplyToAddresses"] = [reply_to]
            resp = client.send_email(**kwargs)
            return EmailResult(success=True, provider="ses", message_id=resp.get("MessageId"))
        except Exception as e:
            logger.exception("[email.ses] send failed for to=%s: %s", to, e)
            return EmailResult(success=False, provider="ses", error=str(e))

    return await asyncio.get_event_loop().run_in_executor(None, _do_send)


# ── SMTP (Gmail / Google Workspace / SendGrid / Brevo / any) ────────


async def _send_via_smtp(
    to: str, subject: str, html: str, text: Optional[str], reply_to: Optional[str],
) -> EmailResult:
    """RFC-822 SMTP via aiosmtplib.

    Handles Gmail / Google Workspace specifics:
      * Port → TLS mode auto-detect (465 = implicit, 587 = STARTTLS)
      * Rewrites the From: address to match the SMTP-authenticated
        user if they differ, defending against 553 Sender Mismatch
        without losing the display name.
      * Surfaces friendly hints in the error path for the three most
        common Gmail failure modes (535 wrong password, 553 sender
        mismatch, TLS handshake).
    """
    try:
        import aiosmtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
    except ImportError:
        logger.error("[email.smtp] aiosmtplib not installed — pip install aiosmtplib")
        return EmailResult(success=False, provider="smtp", error="aiosmtplib_missing")

    if not settings.smtp_host:
        logger.warning("[email.smtp] SMTP_HOST not set — falling back to log")
        return EmailResult(success=True, provider="smtp-logged")

    # Gmail / Google Workspace will reject with 553 "Username and
    # Sender mismatch" if From: ≠ SMTP-authenticated account unless
    # that account has "Send mail as" configured for the alias.
    # Easiest fix: when they differ, rewrite From: to the auth user.
    # Display-name half preserved so users still see "Toup <addr>".
    smtp_user = (settings.smtp_username or "").strip()
    from_addr = settings.email_from_address or smtp_user
    if smtp_user and "@" in smtp_user:
        bare_from = from_addr
        if "<" in from_addr and ">" in from_addr:
            bare_from = from_addr[from_addr.index("<") + 1: from_addr.index(">")]
        if bare_from.strip().lower() != smtp_user.lower():
            display = from_addr.split("<")[0].strip().strip('"') if "<" in from_addr else ""
            from_addr = f'{display} <{smtp_user}>' if display else smtp_user
            logger.info(
                "[email.smtp] rewrote From: to match auth user (%s → %s)",
                settings.email_from_address, from_addr,
            )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to
    if reply_to:
        msg["Reply-To"] = reply_to
    if text:
        msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    # Port → TLS mode mapping. 465 = implicit TLS from byte zero,
    # 587 = STARTTLS upgrade. Both Gmail and Workspace support both.
    port = settings.smtp_port
    use_tls = port == 465
    start_tls = (not use_tls) and bool(settings.smtp_use_tls)

    try:
        await aiosmtplib.send(
            msg,
            hostname=settings.smtp_host,
            port=port,
            username=smtp_user or None,
            password=settings.smtp_password or None,
            use_tls=use_tls,
            start_tls=start_tls,
            timeout=20,
        )
        logger.info("[email.smtp] sent to=%s via=%s:%d", to, settings.smtp_host, port)
        return EmailResult(success=True, provider="smtp")
    except Exception as e:
        s = str(e).lower()
        hint = ""
        if "username and password not accepted" in s or "535" in s:
            hint = (
                " — Gmail rejected the password. For Google Workspace, "
                "use an App Password (not the account password). "
                "Enable 2FA, then create one at https://myaccount.google.com/apppasswords"
            )
        elif "553" in s or "sender mismatch" in s:
            hint = (
                " — Gmail rejected the From: address. The SMTP user "
                "and the From: email must match, or you must configure "
                "Gmail Settings → Accounts → 'Send mail as' for the alias."
            )
        elif "ssl" in s or "starttls" in s:
            hint = (
                " — TLS handshake failed. Try port 587 with STARTTLS "
                "(SMTP_PORT=587, SMTP_USE_TLS=true) or 465 with implicit TLS."
            )
        logger.exception("[email.smtp] send failed for to=%s: %s%s", to, e, hint)
        return EmailResult(success=False, provider="smtp", error=f"{e}{hint}")


# ── Verification email template ────────────────────────────────────


def render_verification_email(
    *, name: Optional[str], verify_url: str,
) -> tuple[str, str, str]:
    """Build (subject, html, plain_text) for the email-verification message."""
    display_name = (name or "there").split()[0] if name else "there"
    subject = "Verify your Toup email"
    text = (
        f"Hi {display_name},\n\n"
        f"Confirm your email to finish setting up your Toup account:\n\n"
        f"{verify_url}\n\n"
        f"If you didn't sign up for Toup, you can safely ignore this email.\n\n"
        f"— The Toup team\n"
    )
    html = f"""<!DOCTYPE html>
<html><body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 480px; margin: 0 auto; padding: 32px 24px; color: #18181b; line-height: 1.5;">
  <h1 style="font-size: 22px; margin: 0 0 16px;">Verify your email</h1>
  <p>Hi {display_name},</p>
  <p>Confirm your email to finish setting up your Toup account.</p>
  <p style="margin: 28px 0;">
    <a href="{verify_url}" style="display: inline-block; background: #f97316; color: #fff; text-decoration: none; padding: 12px 24px; border-radius: 8px; font-weight: 600;">
      Verify email
    </a>
  </p>
  <p style="font-size: 13px; color: #71717a;">
    Or paste this link into your browser:<br>
    <a href="{verify_url}" style="color: #71717a; word-break: break-all;">{verify_url}</a>
  </p>
  <hr style="border: none; border-top: 1px solid #e4e4e7; margin: 32px 0;">
  <p style="font-size: 12px; color: #71717a;">
    If you didn't sign up for Toup, you can safely ignore this email.
  </p>
</body></html>
"""
    return subject, html, text


def render_support_card_email(
    *,
    issue_id: str,
    symptom: Optional[str],
    severity: str,
    channel: str,
    classification: Optional[str],
    reporter: Optional[str],
    queue_url: str,
    has_screenshot: bool,
) -> tuple[str, str, str]:
    """Build (subject, html, plain_text) for the 'a support card arrived' alert.

    This is only the alert — handling happens in the admin queue. Keep it
    short: who/what/severity + a link to the card.
    """
    sym = (symptom or "(no summary)").strip()
    cls = classification or "pending diagnosis"
    shot = "yes" if has_screenshot else "no"
    subject = f"[Toup support] {severity.upper()} · {sym[:80]}"
    text = (
        f"A support card arrived.\n\n"
        f"Summary:   {sym}\n"
        f"Severity:  {severity}\n"
        f"Channel:   {channel}\n"
        f"Reporter:  {reporter or 'unknown'}\n"
        f"Class:     {cls}\n"
        f"Screenshot:{shot}\n"
        f"Issue ID:  {issue_id}\n\n"
        f"Review & decide in the admin queue:\n{queue_url}\n"
    )
    # HTML-escape user-controlled fields (symptom ← raw_report, reporter ←
    # reporter_email, channel) before interpolating into the HTML body.
    e_sym, e_reporter = _html.escape(sym), _html.escape(reporter or "unknown")
    e_cls, e_channel, e_sev = _html.escape(cls), _html.escape(channel), _html.escape(severity)
    html = f"""<!DOCTYPE html>
<html><body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 480px; margin: 0 auto; padding: 32px 24px; color: #18181b; line-height: 1.5;">
  <h1 style="font-size: 20px; margin: 0 0 8px;">A support card arrived</h1>
  <p style="margin: 0 0 16px; color: #71717a; font-size: 13px;">Severity <b>{e_sev}</b> · {e_channel}</p>
  <p style="font-size: 15px;">{e_sym}</p>
  <table style="font-size: 13px; color: #3f3f46; border-collapse: collapse; margin: 12px 0;">
    <tr><td style="padding: 2px 12px 2px 0; color:#71717a;">Reporter</td><td>{e_reporter}</td></tr>
    <tr><td style="padding: 2px 12px 2px 0; color:#71717a;">Class</td><td>{e_cls}</td></tr>
    <tr><td style="padding: 2px 12px 2px 0; color:#71717a;">Screenshot</td><td>{shot}</td></tr>
    <tr><td style="padding: 2px 12px 2px 0; color:#71717a;">Issue</td><td style="font-family: monospace;">{issue_id}</td></tr>
  </table>
  <p style="margin: 24px 0;">
    <a href="{queue_url}" style="display: inline-block; background: #f97316; color: #fff; text-decoration: none; padding: 12px 24px; border-radius: 8px; font-weight: 600;">
      Open in admin queue
    </a>
  </p>
  <hr style="border: none; border-top: 1px solid #e4e4e7; margin: 28px 0;">
  <p style="font-size: 12px; color: #71717a;">
    Handling happens in the queue — this is only an alert. The card already carries an
    auto-diagnosis; approve only after review.
  </p>
</body></html>
"""
    return subject, html, text
