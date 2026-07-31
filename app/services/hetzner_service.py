"""
Hetzner Cloud VPS provisioning service.

Provisions a cloud server via the Hetzner Cloud API, runs a setup script
via cloud-init user_data, and updates the VPSInstance record.

Hetzner Cloud API: https://docs.hetzner.cloud/
Base URL: https://api.hetzner.cloud/v1
Auth: Bearer token
"""

import asyncio
import logging
import secrets
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import settings
from app.db.models import VPSInstance, VPSPlan

logger = logging.getLogger(__name__)

HETZNER_API_BASE = "https://api.hetzner.cloud/v1"


def _get_headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.hetzner_api_token}",
        "Content-Type": "application/json",
    }


def _build_cloud_init(
    ssh_password: str,
    user_id: str,
    agent_api_key: str,
) -> str:
    """Cloud-init script for Hetzner VPS.

    Hetzner supports cloud-init natively via user_data on server creation.
    This installs everything from scratch on a fresh Ubuntu 24.04 image.
    """
    return f"""#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive

# ── SSH password for root ─────────────────────────────────────
echo "root:{ssh_password}" | chpasswd
sed -i 's/^#\\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/^#\\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
systemctl restart sshd

# ── System dependencies ──────────────────────────────────────
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv python3-pip git build-essential curl

# ── Clone agent ───────────────────────────────────────────────
mkdir -p /opt/toup-agent
git clone https://github.com/toup-com/toup-agent.git /opt/toup-agent

# ── Python venv ───────────────────────────────────────────────
cd /opt/toup-agent
python3.12 -m venv venv
source venv/bin/activate
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet -r requirements.txt

# ── .env ──────────────────────────────────────────────────────
cat > /opt/toup-agent/.env << 'ENVEOF'
DATABASE_URL={settings.supabase_url or settings.database_url}
RUN_MODE=agent

USER_ID={user_id}
AGENT_API_KEY={agent_api_key}
PLATFORM_API_URL={settings.platform_api_url}

OPENAI_API_KEY={settings.openai_api_key or ""}
ANTHROPIC_API_KEY={settings.anthropic_api_key or ""}
AGENT_MODEL={settings.agent_model}

EMBEDDING_PROVIDER={settings.embedding_provider}
EMBEDDING_MODEL={settings.embedding_model}
EMBEDDING_DIMENSION={settings.embedding_dimension}
ENVEOF

# ── Systemd service ──────────────────────────────────────────
cat > /etc/systemd/system/toup-agent.service << 'SVCEOF'
[Unit]
Description=Toup Agent Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/toup-agent
ExecStart=/opt/toup-agent/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=5
EnvironmentFile=/opt/toup-agent/.env

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable toup-agent
systemctl restart toup-agent
"""


async def _hetzner_request(
    method: str,
    path: str,
    json_data: dict | None = None,
    timeout: float = 30.0,
) -> dict:
    """Make an authenticated request to the Hetzner Cloud API."""
    url = f"{HETZNER_API_BASE}{path}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.request(
            method=method,
            url=url,
            headers=_get_headers(),
            json=json_data,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}


async def provision_instance(
    vps_instance_id: str,
    db_session_factory,
) -> None:
    """
    Background task: provision a Hetzner Cloud server for the given VPSInstance.

    Hetzner supports cloud-init natively, so the setup script runs automatically
    on first boot — no need to SSH in separately.
    """
    async with db_session_factory() as db:
        result = await db.execute(
            select(VPSInstance).where(VPSInstance.id == vps_instance_id)
        )
        vps: VPSInstance | None = result.scalar_one_or_none()

        if not vps:
            logger.error("VPSInstance %s not found", vps_instance_id)
            return

        plan_result = await db.execute(
            select(VPSPlan).where(VPSPlan.id == vps.plan_id)
        )
        plan: VPSPlan | None = plan_result.scalar_one_or_none()

        if not plan:
            await _mark_error(db, vps, "Plan not found")
            return

        if not settings.hetzner_api_token:
            await _mark_error(db, vps, "HETZNER_API_TOKEN not configured")
            return

        # Generate credentials
        ssh_password = secrets.token_urlsafe(16)
        agent_api_key = f"toup_ak_{secrets.token_urlsafe(32)}"

        cloud_init_script = _build_cloud_init(
            ssh_password=ssh_password,
            user_id=vps.user_id,
            agent_api_key=agent_api_key,
        )

        try:
            # ── Create server via Hetzner API ──────────────────────
            # Map our plan instance_type to Hetzner server type name
            server_type = plan.hetzner_server_type or plan.instance_type  # e.g. "cx23"

            create_payload = {
                "name": f"toup-{vps.user_id[:8]}-{vps_instance_id[:6]}",
                "server_type": server_type,
                "image": "ubuntu-24.04",
                "location": settings.hetzner_location,  # e.g. "ash" for Ashburn
                "user_data": cloud_init_script,
                "start_after_create": True,
                "labels": {
                    "toup-user": vps.user_id[:63],
                    "toup-vps": vps_instance_id[:63],
                },
            }

            # Add SSH key if configured
            if settings.hetzner_ssh_key_id:
                create_payload["ssh_keys"] = [int(settings.hetzner_ssh_key_id)]

            resp = await _hetzner_request(
                "POST", "/servers",
                json_data=create_payload,
                timeout=60.0,
            )

            server = resp.get("server", {})
            hetzner_server_id = str(server.get("id", ""))

            if not hetzner_server_id:
                await _mark_error(db, vps, "Hetzner API returned no server ID")
                return

            # Store the Hetzner server ID
            vps.hetzner_vm_id = hetzner_server_id
            await db.commit()

            logger.info(
                "Hetzner server %s created for VPS %s",
                hetzner_server_id, vps_instance_id,
            )

            # ── Extract IP immediately (Hetzner returns it in create response)
            public_ip = _extract_ip(server)

            if not public_ip:
                # Poll until we get an IP
                for attempt in range(30):  # up to 5 minutes
                    await asyncio.sleep(10)
                    try:
                        server_resp = await _hetzner_request(
                            "GET", f"/servers/{hetzner_server_id}"
                        )
                        server = server_resp.get("server", {})
                        status = server.get("status", "")

                        if status == "running":
                            public_ip = _extract_ip(server)
                            if public_ip:
                                break

                        if status in ("deleting", "off"):
                            await _mark_error(db, vps, f"Server entered state: {status}")
                            return
                    except Exception:
                        continue

            if not public_ip:
                await _mark_error(db, vps, "Timed out waiting for Hetzner server IP")
                return

            # ── Persist final state ────────────────────────────────
            vps.status = "active"
            vps.public_ip = public_ip
            vps.ssh_password = ssh_password
            vps.agent_api_key = agent_api_key
            vps.provisioned_at = datetime.utcnow()
            await db.commit()

            logger.info(
                "VPS %s provisioned via Hetzner: %s (server_id=%s)",
                vps_instance_id, public_ip, hetzner_server_id,
            )

        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text[:500] if exc.response else str(exc)
            logger.exception("Hetzner API error provisioning VPS %s", vps_instance_id)
            await _mark_error(db, vps, f"Hetzner API error: {error_body}")
        except Exception as exc:
            logger.exception("Unexpected error provisioning VPS %s", vps_instance_id)
            await _mark_error(db, vps, str(exc)[:500])


def _extract_ip(server: dict) -> Optional[str]:
    """Extract public IPv4 from Hetzner server response."""
    try:
        public_net = server.get("public_net", {})
        ipv4 = public_net.get("ipv4", {})
        ip = ipv4.get("ip", "")
        if ip:
            return str(ip)
    except (KeyError, TypeError):
        pass
    return None


async def terminate_instance(hetzner_server_id: str) -> bool:
    """Delete a Hetzner Cloud server."""
    try:
        await _hetzner_request("DELETE", f"/servers/{hetzner_server_id}")
        logger.info("Hetzner server %s terminated", hetzner_server_id)
        return True
    except Exception as exc:
        logger.exception("Failed to terminate Hetzner server %s: %s", hetzner_server_id, exc)
        return False


async def _mark_error(db: AsyncSession, vps: VPSInstance, message: str) -> None:
    vps.status = "error"
    vps.error_message = message[:500]
    await db.commit()
