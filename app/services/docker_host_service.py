"""
Managed Docker Host Service — provisions and manages agent containers
on a shared Docker host for the Quick Setup (managed hosting) flow.

Each user gets:
- An isolated Docker container running agent_main.py
- A dedicated PostgreSQL database on the shared host
- A unique port mapped on the Docker host
- Persistent volumes for workspace and skills

The platform connects to containers via the existing ws_chat_proxy
mechanism — it just stores the agent_url as http://docker_host_ip:port.
"""

import asyncio
import logging
import secrets
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import ManagedContainer, AgentConfig

logger = logging.getLogger(__name__)

# Port range for agent containers
PORT_START = settings.docker_port_range_start
PORT_END = settings.docker_port_range_end


_ssh_key_file: Optional[str] = None


def _get_ssh_key_file() -> Optional[str]:
    """Write the SSH key to a temp file (once) and return the path."""
    global _ssh_key_file
    if _ssh_key_file:
        return _ssh_key_file
    if not settings.docker_host_ssh_key:
        return None
    import tempfile, os
    fd, path = tempfile.mkstemp(prefix="toup_ssh_", suffix=".key")
    with os.fdopen(fd, 'w') as f:
        f.write(settings.docker_host_ssh_key)
        if not settings.docker_host_ssh_key.endswith('\n'):
            f.write('\n')
    os.chmod(path, 0o600)
    _ssh_key_file = path
    return path


async def _run_ssh(cmd: str) -> tuple[int, str, str]:
    """Run a command on the Docker host via SSH. Passes command via stdin to avoid quoting issues."""
    key_file = _get_ssh_key_file()
    if key_file:
        ssh_args = [
            "ssh", "-i", key_file,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "PasswordAuthentication=no",
            f"root@{settings.docker_host_ip}",
            "bash", "-s",
        ]
    else:
        ssh_args = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"root@{settings.docker_host_ip}",
            "bash", "-s",
        ]
    proc = await asyncio.create_subprocess_exec(
        *ssh_args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(
        proc.communicate(input=cmd.encode()),
        timeout=120,
    )
    return proc.returncode or 0, stdout.decode(), stderr.decode()


async def allocate_port(db: AsyncSession) -> int:
    """Find the next available port in the range."""
    result = await db.execute(
        select(ManagedContainer.host_port)
        .where(ManagedContainer.status.in_(["provisioning", "running", "stopped"]))
        .order_by(ManagedContainer.host_port)
    )
    used_ports = {row[0] for row in result.all()}

    for port in range(PORT_START, PORT_END + 1):
        if port not in used_ports:
            return port

    raise RuntimeError(f"No available ports in range {PORT_START}-{PORT_END}")


async def create_user_database(user_id: str) -> str:
    """Create a PostgreSQL database for the user on the Docker host."""
    db_name = f"toup_agent_{user_id[:8].replace('-', '')}"

    # Create database via SSH on the Docker host
    create_sql = (
        f"SELECT 'CREATE DATABASE {db_name}' "
        f"WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '{db_name}');"
    )
    code, out, err = await _run_ssh(
        f'sudo -u postgres psql -tc "{create_sql}" | sudo -u postgres psql'
    )
    if code != 0:
        logger.warning(f"DB create output: {out} {err}")

    # Enable pgvector extension
    code, out, err = await _run_ssh(
        f'sudo -u postgres psql -d {db_name} -c "CREATE EXTENSION IF NOT EXISTS vector;"'
    )
    if code != 0:
        logger.warning(f"pgvector extension: {out} {err}")

    # Set timezone to UTC (prevents naive/aware datetime conflicts)
    await _run_ssh(
        f'sudo -u postgres psql -d {db_name} -c "ALTER DATABASE {db_name} SET timezone TO \'UTC\';"'
    )

    return db_name


def _build_env(
    user_id: str,
    db_name: str,
    agent_api_key: str,
    agent_config: Optional[AgentConfig],
    host_port: int = 8001,
) -> str:
    """Build the .env content for an agent container."""
    lines = [
        f"RUN_MODE=agent",
        f"USER_ID={user_id}",
        f"AGENT_API_KEY={agent_api_key}",
        f"DATABASE_URL=postgresql+asyncpg://postgres:postgres@host.docker.internal:5432/{db_name}",
        f"AGENT_WORKSPACE_DIR=/app/workspace",
        f"SKILLS_DIR=/app/skills",
        f"PLATFORM_API_URL={settings.platform_api_url if settings.platform_api_url.endswith('/api') else settings.platform_api_url + '/api'}",
        f"TOUP_TOKEN={agent_config.connect_token or '' if agent_config else ''}",
        f"AGENT_PORT={host_port}",
    ]

    # Forward LLM keys from agent config
    if agent_config:
        if agent_config.openai_api_key:
            lines.append(f"OPENAI_API_KEY={agent_config.openai_api_key}")
        if agent_config.anthropic_api_key:
            lines.append(f"ANTHROPIC_API_KEY={agent_config.anthropic_api_key}")
        if agent_config.google_api_key:
            lines.append(f"GOOGLE_API_KEY={agent_config.google_api_key}")
        if agent_config.mistral_api_key:
            lines.append(f"MISTRAL_API_KEY={agent_config.mistral_api_key}")
        if agent_config.xai_api_key:
            lines.append(f"XAI_API_KEY={agent_config.xai_api_key}")
        if agent_config.deepseek_api_key:
            lines.append(f"DEEPSEEK_API_KEY={agent_config.deepseek_api_key}")
        if agent_config.telegram_bot_token:
            lines.append(f"TELEGRAM_BOT_TOKEN={agent_config.telegram_bot_token}")
        if agent_config.discord_bot_token:
            lines.append(f"DISCORD_BOT_TOKEN={agent_config.discord_bot_token}")
        if agent_config.agent_model:
            lines.append(f"AGENT_MODEL={agent_config.agent_model}")
        if agent_config.agent_color:
            lines.append(f"AGENT_COLOR={agent_config.agent_color}")

    return "\n".join(lines)


async def provision_container(
    db: AsyncSession,
    user_id: str,
    agent_config: Optional[AgentConfig] = None,
) -> ManagedContainer:
    """Provision a new Docker container for a user's agent."""

    # Check if container already exists
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    existing = result.scalar_one_or_none()
    if existing and existing.status in ("running", "provisioning"):
        return existing

    # Allocate port and create database
    port = await allocate_port(db)
    db_name = await create_user_database(user_id)
    agent_api_key = secrets.token_urlsafe(32)
    container_name = f"toup-agent-{user_id[:8]}"

    # Create or update the record
    if existing:
        container = existing
        container.host_port = port
        container.db_name = db_name
        container.status = "provisioning"
        container.error_message = None
    else:
        container = ManagedContainer(
            id=str(uuid.uuid4()),
            user_id=user_id,
            container_name=container_name,
            host_port=port,
            db_name=db_name,
            status="provisioning",
            image_tag=settings.docker_agent_image,
        )
        db.add(container)

    await db.commit()
    await db.refresh(container)

    try:
        # Build env file content
        env_content = _build_env(user_id, db_name, agent_api_key, agent_config, host_port=port)

        # Write env file on Docker host
        escaped_env = env_content.replace("'", "'\\''")
        code, _, err = await _run_ssh(
            f"mkdir -p /data/agents/{user_id[:8]} && "
            f"echo '{escaped_env}' > /data/agents/{user_id[:8]}/.env"
        )
        if code != 0:
            raise RuntimeError(f"Failed to write env: {err}")

        # Create workspace and skills directories
        await _run_ssh(
            f"mkdir -p /data/agents/{user_id[:8]}/workspace "
            f"/data/agents/{user_id[:8]}/skills"
        )

        # Remove existing container if any
        await _run_ssh(f"docker rm -f {container_name} 2>/dev/null || true")

        # Run the container
        docker_run = (
            f"docker run -d "
            f"--name {container_name} "
            f"--restart unless-stopped "
            f"--env-file /data/agents/{user_id[:8]}/.env "
            f"-p {port}:8001 "
            f"-v /data/agents/{user_id[:8]}/workspace:/app/workspace "
            f"-v /data/agents/{user_id[:8]}/skills:/app/skills "
            f"--add-host host.docker.internal:host-gateway "
            f"--memory=512m --cpus=0.5 "
            f"{settings.docker_agent_image}"
        )
        code, out, err = await _run_ssh(docker_run)
        if code != 0:
            raise RuntimeError(f"Docker run failed: {err}")

        container_id = out.strip()[:12]
        container.container_id = container_id
        container.status = "running"
        container.started_at = datetime.utcnow()

        # Update AgentConfig with the container's URL
        if not agent_config:
            # Fetch it if not passed
            result = await db.execute(
                select(AgentConfig).where(AgentConfig.user_id == user_id)
            )
            agent_config = result.scalar_one_or_none()
        if agent_config:
            agent_config.agent_url = f"http://{settings.docker_host_ip}:{port}"
            agent_config.agent_api_key = agent_api_key
            agent_config.hosting_mode = "managed"
            agent_config.deploy_status = "active"

        await db.commit()
        await db.refresh(container)
        logger.info(f"Provisioned container {container_name} on port {port} for user {user_id[:8]}")

        # Auto-sync soul config to the new container (background)
        asyncio.create_task(_sync_soul_after_start(
            user_id, f"http://{settings.docker_host_ip}:{port}", agent_api_key,
        ))

        return container

    except Exception as e:
        container.status = "error"
        container.error_message = str(e)[:500]
        await db.commit()
        logger.error(f"Failed to provision container for user {user_id[:8]}: {e}")
        raise


async def stop_container(db: AsyncSession, user_id: str) -> Optional[ManagedContainer]:
    """Stop a user's agent container."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return None

    await _run_ssh(f"docker stop {container.container_name}")
    container.status = "stopped"
    container.stopped_at = datetime.utcnow()
    await db.commit()
    return container


async def start_container(db: AsyncSession, user_id: str) -> Optional[ManagedContainer]:
    """Start a stopped container."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return None

    code, _, err = await _run_ssh(f"docker start {container.container_name}")
    if code != 0:
        container.status = "error"
        container.error_message = f"Start failed: {err}"
    else:
        container.status = "running"
        container.started_at = datetime.utcnow()
        container.error_message = None

    await db.commit()
    return container


async def restart_container(db: AsyncSession, user_id: str) -> Optional[ManagedContainer]:
    """Restart a container (e.g. after config change)."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return None

    await _run_ssh(f"docker restart {container.container_name}")
    container.status = "running"
    container.started_at = datetime.utcnow()
    container.error_message = None
    await db.commit()
    return container


async def destroy_container(db: AsyncSession, user_id: str) -> bool:
    """Remove a container and optionally its data."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return False

    await _run_ssh(f"docker rm -f {container.container_name} 2>/dev/null || true")
    container.status = "deleted"
    container.stopped_at = datetime.utcnow()
    await db.commit()
    return True


async def get_container_status(db: AsyncSession, user_id: str) -> Optional[dict]:
    """Get the status of a user's managed container."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return None

    # Check actual Docker status
    code, out, _ = await _run_ssh(
        f"docker inspect --format='{{{{.State.Status}}}}' {container.container_name} 2>/dev/null || echo 'not_found'"
    )
    docker_status = out.strip()

    return {
        "id": container.id,
        "status": container.status,
        "docker_status": docker_status,
        "port": container.host_port,
        "container_name": container.container_name,
        "created_at": container.created_at.isoformat() if container.created_at else None,
        "started_at": container.started_at.isoformat() if container.started_at else None,
        "error": container.error_message,
    }


async def _sync_soul_after_start(
    user_id: str, agent_url: str, agent_api_key: str, db_session_maker=None,
):
    """Wait for container health, then push soul config.

    Runs as a background task after provision or recreate.
    """
    import httpx
    from app.db.database import async_session_maker
    from app.db.models import SoulConfig
    from sqlalchemy import select

    # Wait up to 30s for the container to be healthy
    for _ in range(10):
        await asyncio.sleep(3)
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{agent_url}/agent/health")
                if resp.status_code == 200:
                    break
        except Exception:
            continue
    else:
        logger.warning(f"[DOCKER] Container not healthy after 30s for {user_id[:8]} — skipping soul sync")
        return

    # Load soul config from platform DB
    try:
        async with async_session_maker() as db:
            result = await db.execute(
                select(SoulConfig).where(SoulConfig.user_id == user_id)
            )
            soul = result.scalar_one_or_none()
            if not soul:
                return  # No soul configured yet

        # Push to agent
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.put(
                f"{agent_url}/api/soul/sync",
                json={
                    "user_id": user_id,
                    "name": soul.name,
                    "compiled_text": soul.compiled_text,
                    "deactivate_agent_soul_memories": False,
                    "agent_config_updates": {
                        "agent_name": soul.name,
                        "agent_color": soul.color,
                    },
                },
                headers={"X-Agent-Key": agent_api_key},
            )
            if resp.status_code == 200:
                logger.info(f"[DOCKER] Auto-synced soul to container for {user_id[:8]}")
            else:
                logger.warning(f"[DOCKER] Soul sync failed for {user_id[:8]}: {resp.status_code}")
    except Exception as e:
        logger.warning(f"[DOCKER] Soul sync error for {user_id[:8]}: {e}")


async def update_container_env(
    db: AsyncSession,
    user_id: str,
    agent_config: AgentConfig,
) -> Optional[ManagedContainer]:
    """Update a container's environment and restart it."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return None

    agent_api_key = agent_config.agent_api_key or secrets.token_urlsafe(32)
    env_content = _build_env(user_id, container.db_name, agent_api_key, agent_config, host_port=container.host_port)

    escaped_env = env_content.replace("'", "'\\''")
    await _run_ssh(f"echo '{escaped_env}' > /data/agents/{user_id[:8]}/.env")

    # Must recreate (not restart) so new --env-file is read.
    # docker restart keeps the original env from docker run.
    port = container.host_port
    name = container.container_name
    prefix = user_id[:8]
    await _run_ssh(f"docker rm -f {name} 2>/dev/null || true")
    await _run_ssh(
        f"docker run -d --name {name} --restart unless-stopped "
        f"--env-file /data/agents/{prefix}/.env "
        f"-p {port}:8001 "
        f"-v /data/agents/{prefix}/workspace:/app/workspace "
        f"-v /data/agents/{prefix}/skills:/app/skills "
        f"--add-host host.docker.internal:host-gateway "
        f"--memory=512m --cpus=0.5 "
        f"{settings.docker_agent_image}"
    )

    container.status = "running"
    container.started_at = datetime.utcnow()
    await db.commit()

    # Auto-sync soul config to the new/recreated container (background)
    asyncio.create_task(_sync_soul_after_start(
        user_id, agent_config.agent_url or f"http://{settings.docker_host_ip}:{port}",
        agent_api_key, db_session_maker=None,
    ))

    return container
