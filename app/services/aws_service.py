"""
AWS EC2 provisioning service.

Provisions an EC2 instance from the configured custom AMI, sets an
auto-generated SSH password via cloud-init UserData, allocates an Elastic IP,
and updates the VPSInstance record in the database.
"""

import asyncio
import logging
import secrets
import uuid
from base64 import b64encode
from datetime import datetime

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import settings
from app.db.models import VPSInstance, VPSPlan

logger = logging.getLogger(__name__)


def _get_ec2_client():
    return boto3.client(
        "ec2",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )


def _build_user_data(
    ssh_password: str,
    username: str,
    user_id: str,
    agent_api_key: str,
) -> str:
    """Cloud-init script that configures the Agent service on first boot.

    The custom AMI already has /opt/toup-agent with the Agent code + venv.
    This UserData writes the .env, pulls latest code, and starts the service.
    """
    script = f"""#!/bin/bash
set -e

# ── SSH access ─────────────────────────────────────────────────
echo "ubuntu:{ssh_password}" | chpasswd
sed -i 's/^#\\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/^#\\?ChallengeResponseAuthentication.*/ChallengeResponseAuthentication yes/' /etc/ssh/sshd_config
systemctl restart sshd

# ── Agent .env ─────────────────────────────────────────────────
cat > /opt/toup-agent/.env << 'ENVEOF'
# Database — Supabase direct connection (not pooler, for pgvector)
DATABASE_URL={settings.supabase_url or settings.database_url}
RUN_MODE=agent

# Identity
USER_ID={user_id}
AGENT_API_KEY={agent_api_key}
PLATFORM_API_URL={settings.platform_api_url}

# LLM
OPENAI_API_KEY={settings.openai_api_key or ""}
ANTHROPIC_API_KEY={settings.anthropic_api_key or ""}
AGENT_MODEL={settings.agent_model}
BRAVE_API_KEY={settings.brave_api_key or ""}

# Embedding (same as platform for vector compatibility)
EMBEDDING_PROVIDER={settings.embedding_provider}
EMBEDDING_MODEL={settings.embedding_model}
EMBEDDING_DIMENSION={settings.embedding_dimension}
ENVEOF

echo "TOUP_USER={username}" >> /etc/environment

# ── Pull latest agent code & restart ──────────────────────────
cd /opt/toup-agent
if [ -d .git ]; then
    git pull origin main --ff-only 2>/dev/null || true
fi
systemctl restart toup-agent
"""
    return script


async def provision_instance(
    vps_instance_id: str,
    db_session_factory,
) -> None:
    """
    Background task: provision an EC2 instance for the given VPSInstance record.

    Called after Stripe payment is confirmed. Runs in a FastAPI BackgroundTask
    so it doesn't block the webhook response.
    """
    async with db_session_factory() as db:
        # Load the VPSInstance and its plan
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

        # Generate credentials
        ssh_password = secrets.token_urlsafe(16)
        agent_api_key = f"toup_ak_{secrets.token_urlsafe(32)}"

        # Extract a display username from user_id (first 8 chars of UUID)
        display_name = vps.user_id[:8]

        user_data_script = _build_user_data(
            ssh_password=ssh_password,
            username=display_name,
            user_id=vps.user_id,
            agent_api_key=agent_api_key,
        )
        user_data_b64 = b64encode(user_data_script.encode()).decode()

        ami_id = settings.aws_ami_id
        if not ami_id:
            await _mark_error(db, vps, "AWS_AMI_ID not configured")
            return

        ec2 = _get_ec2_client()

        try:
            # ── Launch EC2 instance ────────────────────────────────
            run_kwargs = {
                "ImageId": ami_id,
                "InstanceType": plan.instance_type,
                "MinCount": 1,
                "MaxCount": 1,
                "UserData": user_data_b64,
                "TagSpecifications": [
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": f"toup-{display_name}"},
                            {"Key": "ToupUser", "Value": vps.user_id},
                            {"Key": "ToupVPSInstance", "Value": vps_instance_id},
                        ],
                    }
                ],
            }

            if settings.aws_key_pair_name:
                run_kwargs["KeyName"] = settings.aws_key_pair_name

            if settings.aws_security_group_id:
                run_kwargs["SecurityGroupIds"] = [settings.aws_security_group_id]

            resp = await asyncio.to_thread(ec2.run_instances, **run_kwargs)
            instance = resp["Instances"][0]
            aws_instance_id = instance["InstanceId"]

            # Update DB with instance ID
            vps.aws_instance_id = aws_instance_id
            vps.ami_id = ami_id
            await db.commit()

            logger.info("EC2 instance %s launched for VPS %s", aws_instance_id, vps_instance_id)

            # ── Wait for instance to be running ───────────────────
            waiter = ec2.get_waiter("instance_running")
            await asyncio.to_thread(
                waiter.wait,
                InstanceIds=[aws_instance_id],
                WaiterConfig={"Delay": 10, "MaxAttempts": 36},  # up to 6 minutes
            )

            # ── Allocate Elastic IP ───────────────────────────────
            alloc = await asyncio.to_thread(
                ec2.allocate_address, Domain="vpc"
            )
            allocation_id = alloc["AllocationId"]
            public_ip = alloc["PublicIp"]

            await asyncio.to_thread(
                ec2.associate_address,
                InstanceId=aws_instance_id,
                AllocationId=allocation_id,
            )

            # ── Get public DNS ────────────────────────────────────
            desc = await asyncio.to_thread(
                ec2.describe_instances, InstanceIds=[aws_instance_id]
            )
            public_dns = (
                desc["Reservations"][0]["Instances"][0].get("PublicDnsName", "") or ""
            )

            # ── Persist final state ───────────────────────────────
            vps.status = "active"
            vps.public_ip = public_ip
            vps.public_dns = public_dns
            vps.ssh_password = ssh_password
            vps.agent_api_key = agent_api_key
            vps.provisioned_at = datetime.utcnow()
            await db.commit()

            logger.info(
                "VPS %s provisioned: %s (%s)", vps_instance_id, public_ip, aws_instance_id
            )

        except (BotoCoreError, ClientError) as exc:
            logger.exception("AWS error provisioning VPS %s", vps_instance_id)
            await _mark_error(db, vps, str(exc))
        except Exception as exc:
            logger.exception("Unexpected error provisioning VPS %s", vps_instance_id)
            await _mark_error(db, vps, str(exc))


async def _mark_error(db: AsyncSession, vps: VPSInstance, message: str) -> None:
    vps.status = "error"
    vps.error_message = message[:500]
    await db.commit()
