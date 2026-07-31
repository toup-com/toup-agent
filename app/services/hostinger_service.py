"""
Hostinger VPS provisioning service.

Provisions a VPS via the Hostinger REST API, configures it with
a setup script over SSH, and updates the VPSInstance record.

Hostinger API docs: https://developers.hostinger.com
Base URL: https://developers.hostinger.com/api
Auth: Bearer token via Authorization header
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

HOSTINGER_API_BASE = "https://developers.hostinger.com/api"


def _get_headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.hostinger_api_token}",
        "Content-Type": "application/json",
    }


def _build_setup_script(
    ssh_password: str,
    user_id: str,
    agent_api_key: str,
) -> str:
    """Post-provisioning setup script.

    Hostinger VPS comes with a fresh Ubuntu install. This script:
    1. Installs Python 3.12 + dependencies
    2. Clones toup-agent
    3. Creates venv and installs packages
    4. Writes .env
    5. Creates and starts systemd service
    """
    return f"""#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive

# ── SSH password ──────────────────────────────────────────────
echo "root:{ssh_password}" | chpasswd

# ── System dependencies ──────────────────────────────────────
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv python3-pip git build-essential curl

# ── Clone agent ───────────────────────────────────────────────
mkdir -p /opt/toup-agent
if [ ! -d /opt/toup-agent/.git ]; then
    git clone https://github.com/toup-com/toup-agent.git /opt/toup-agent
else
    cd /opt/toup-agent && git pull origin main --ff-only 2>/dev/null || true
fi

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
BRAVE_API_KEY={settings.brave_api_key or ""}

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


async def _hostinger_request(
    method: str,
    path: str,
    json_data: dict | None = None,
    timeout: float = 30.0,
) -> dict:
    """Make an authenticated request to the Hostinger API."""
    url = f"{HOSTINGER_API_BASE}{path}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.request(
            method=method,
            url=url,
            headers=_get_headers(),
            json=json_data,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}


async def list_hostinger_plans() -> list[dict]:
    """Fetch available VPS plans from Hostinger catalog."""
    try:
        data = await _hostinger_request("GET", "/billing/v1/catalog")
        # Filter to VPS hosting category
        vps_items = [
            item for item in data
            if item.get("category") == "vps"
        ]
        return vps_items
    except Exception as exc:
        logger.exception("Failed to fetch Hostinger catalog: %s", exc)
        return []


async def provision_instance(
    vps_instance_id: str,
    db_session_factory,
) -> None:
    """
    Background task: provision a Hostinger VPS for the given VPSInstance record.

    Steps:
    1. Create VPS via Hostinger API (POST /api/vps/v1/virtual-machines)
    2. Poll until VPS is running and has an IP
    3. SSH in and run setup script
    4. Update VPSInstance record with IP, credentials, status=active
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
            logger.error("VPSPlan %s not found", vps.plan_id)
            await _mark_error(db, vps, "Plan not found")
            return

        if not settings.hostinger_api_token:
            await _mark_error(db, vps, "HOSTINGER_API_TOKEN not configured")
            return

        # Generate credentials
        ssh_password = secrets.token_urlsafe(16)
        agent_api_key = f"toup_ak_{secrets.token_urlsafe(32)}"

        try:
            # ── Create VPS via Hostinger API ───────────────────────
            setup_script = _build_setup_script(
                ssh_password=ssh_password,
                user_id=vps.user_id,
                agent_api_key=agent_api_key,
            )

            create_payload = {
                "plan_id": plan.hostinger_plan_id or plan.instance_type,
                "data_center_id": settings.hostinger_data_center_id or 1,
                "payment_method_id": settings.hostinger_payment_method_id or "default",
                "setup": {
                    "hostname": f"toup-{vps.user_id[:8]}",
                    "template_id": settings.hostinger_template_id,  # Ubuntu 24.04
                    "password": ssh_password,
                    "post_install_script_id": None,
                    "install_monarx": False,
                },
            }

            resp = await _hostinger_request(
                "POST",
                "/vps/v1/virtual-machines",
                json_data=create_payload,
                timeout=60.0,
            )

            hostinger_vm_id = str(resp.get("id", ""))
            if not hostinger_vm_id:
                await _mark_error(db, vps, "Hostinger API returned no VM ID")
                return

            # Store the Hostinger VM ID
            vps.hostinger_vm_id = hostinger_vm_id
            await db.commit()

            logger.info(
                "Hostinger VPS %s created for VPS %s", hostinger_vm_id, vps_instance_id
            )

            # ── Poll for running status + IP ───────────────────────
            public_ip = None
            max_attempts = 60  # 10 minutes (10s intervals)
            for attempt in range(max_attempts):
                await asyncio.sleep(10)

                try:
                    vm_data = await _hostinger_request(
                        "GET", f"/vps/v1/virtual-machines/{hostinger_vm_id}"
                    )
                except Exception:
                    continue

                vm_state = vm_data.get("state", "")
                if vm_state == "running":
                    # Extract IP from the response
                    public_ip = _extract_ip(vm_data)
                    if public_ip:
                        break

                if vm_state in ("error", "failed", "destroyed"):
                    await _mark_error(
                        db, vps,
                        f"Hostinger VPS entered state: {vm_state}"
                    )
                    return

            if not public_ip:
                await _mark_error(db, vps, "Timed out waiting for Hostinger VPS IP")
                return

            # ── Run setup script via SSH ───────────────────────────
            # Wait a bit for SSH to become available
            await asyncio.sleep(30)

            ssh_success = await _run_setup_via_ssh(
                host=public_ip,
                password=ssh_password,
                script=setup_script,
            )

            if not ssh_success:
                logger.warning(
                    "SSH setup failed for VPS %s — agent may need manual setup",
                    vps_instance_id,
                )

            # ── Persist final state ────────────────────────────────
            vps.status = "active"
            vps.public_ip = public_ip
            vps.ssh_password = ssh_password
            vps.agent_api_key = agent_api_key
            vps.provisioned_at = datetime.utcnow()
            await db.commit()

            logger.info(
                "VPS %s provisioned via Hostinger: %s (vm_id=%s)",
                vps_instance_id, public_ip, hostinger_vm_id,
            )

        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text[:500] if exc.response else str(exc)
            logger.exception("Hostinger API error provisioning VPS %s", vps_instance_id)
            await _mark_error(db, vps, f"Hostinger API error: {error_body}")
        except Exception as exc:
            logger.exception("Unexpected error provisioning VPS %s", vps_instance_id)
            await _mark_error(db, vps, str(exc)[:500])


def _extract_ip(vm_data: dict) -> Optional[str]:
    """Extract the public IPv4 address from Hostinger VM data."""
    # Hostinger returns IPs in different possible locations
    # Try direct ip_address field
    ip = vm_data.get("ip_address") or vm_data.get("ipv4")
    if ip:
        return str(ip)

    # Try nested firewall/network structure
    ips = vm_data.get("ips", [])
    for ip_entry in ips:
        if ip_entry.get("version") == 4:
            return str(ip_entry.get("address", ""))

    # Try public_ip field
    if vm_data.get("public_ip"):
        return str(vm_data["public_ip"])

    return None


async def _run_setup_via_ssh(
    host: str,
    password: str,
    script: str,
    max_retries: int = 3,
) -> bool:
    """Run the setup script on the VPS via SSH using asyncssh."""
    try:
        import asyncssh
    except ImportError:
        logger.warning("asyncssh not installed — skipping SSH setup. Install with: pip install asyncssh")
        return False

    for attempt in range(max_retries):
        try:
            async with asyncssh.connect(
                host,
                username="root",
                password=password,
                known_hosts=None,
                connect_timeout=30,
            ) as conn:
                result = await conn.run(script, timeout=600)
                if result.exit_status == 0:
                    logger.info("SSH setup completed for %s", host)
                    return True
                else:
                    logger.warning(
                        "SSH setup script failed (attempt %d/%d) exit=%d: %s",
                        attempt + 1, max_retries, result.exit_status,
                        result.stderr[:500] if result.stderr else "no stderr",
                    )
        except Exception as exc:
            logger.warning(
                "SSH connection failed (attempt %d/%d) to %s: %s",
                attempt + 1, max_retries, host, exc,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(15)

    return False


async def terminate_instance(hostinger_vm_id: str) -> bool:
    """Delete a Hostinger VPS instance."""
    try:
        await _hostinger_request(
            "DELETE", f"/vps/v1/virtual-machines/{hostinger_vm_id}"
        )
        logger.info("Hostinger VPS %s terminated", hostinger_vm_id)
        return True
    except Exception as exc:
        logger.exception("Failed to terminate Hostinger VPS %s: %s", hostinger_vm_id, exc)
        return False


async def _mark_error(db: AsyncSession, vps: VPSInstance, message: str) -> None:
    vps.status = "error"
    vps.error_message = message[:500]
    await db.commit()
