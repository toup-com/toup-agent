"""
SSH-based agent deployment service.

Routes SSH operations through AWS Lambda because Railway blocks outbound port 22.
The Lambda function (toup-ssh-proxy) connects to the target machine and executes commands.

For local machine deployments, generates a setup script the user runs directly.
"""

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable, Optional

import boto3

from app.config import settings

logger = logging.getLogger(__name__)

AGENT_REPO = "https://github.com/toup-com/toup-agent.git"
AGENT_DIR = "/opt/toup-agent"
AGENT_PORT = 8001
SYSTEMD_UNIT = "toup-agent"


# ── Lambda invocation helpers ────────────────────────────────────

def _get_lambda_client():
    """Get a boto3 Lambda client using AWS credentials from settings."""
    from botocore.config import Config as BotoConfig
    return boto3.client(
        "lambda",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=BotoConfig(
            read_timeout=120,    # 2 min read timeout (for poll responses)
            connect_timeout=10,
        ),
    )


async def _invoke_lambda(payload: dict) -> dict:
    """Invoke the SSH proxy Lambda function and return parsed response."""
    function_name = settings.ssh_lambda_function_name
    if not function_name:
        raise RuntimeError("SSH Lambda not configured (SSH_LAMBDA_FUNCTION_NAME is empty)")

    loop = asyncio.get_event_loop()
    client = _get_lambda_client()

    # Run synchronous boto3 call in thread pool
    response = await loop.run_in_executor(
        None,
        lambda: client.invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload).encode(),
        ),
    )

    # Parse response
    response_payload = json.loads(response["Payload"].read().decode())

    # Check for Lambda-level errors
    if "FunctionError" in response:
        error_msg = response_payload.get("errorMessage", "Lambda execution failed")
        raise RuntimeError(f"Lambda error: {error_msg}")

    return response_payload


# ── SSH Test ─────────────────────────────────────────────────────

async def test_ssh_connection(
    ssh_host: str,
    ssh_port: int,
    ssh_user: str,
    ssh_password: Optional[str] = None,
    ssh_key: Optional[str] = None,
) -> dict:
    """Test SSH connectivity via Lambda proxy and return machine info."""
    logger.info(f"SSH test via Lambda: {ssh_host}:{ssh_port} as {ssh_user}")

    try:
        result = await _invoke_lambda({
            "action": "test",
            "ssh_host": ssh_host,
            "ssh_port": ssh_port,
            "ssh_user": ssh_user,
            "ssh_password": ssh_password,
            "ssh_key": ssh_key,
        })

        if "error" in result:
            logger.error(f"SSH test failed: {result['error']}")
            return {"connected": False, "error": result["error"]}

        logger.info(f"SSH test success: {result.get('os')}, {result.get('python')}")
        return result

    except RuntimeError as e:
        logger.error(f"SSH test Lambda error: {e}")
        return {"connected": False, "error": str(e)}
    except Exception as e:
        logger.error(f"SSH test unexpected error: {type(e).__name__}: {e}")
        return {"connected": False, "error": f"SSH proxy error: {str(e)}"}


# ── Deploy via Lambda ────────────────────────────────────────────

async def deploy_agent(
    ssh_host: str,
    ssh_port: int,
    ssh_user: str,
    ssh_password: Optional[str],
    ssh_key: Optional[str],
    env_content: str,
    on_log: Callable[[str, str], Awaitable[None]],
    db_mode: str = "auto",
    target_os: str = "linux",
) -> bool:
    """
    Deploy the Toup Agent to a remote machine via Lambda SSH proxy.

    Uses async deploy: Lambda starts the script with nohup and returns
    immediately. Backend polls for log output every few seconds, streaming
    lines to the frontend in real-time. No more timeout issues.

    Args:
        ssh_host: Target machine hostname/IP
        ssh_port: SSH port
        ssh_user: SSH username
        ssh_password: SSH password (or None for key auth)
        ssh_key: PEM private key content (or None for password auth)
        env_content: Generated .env file content
        on_log: Async callback (line, level) for real-time log streaming.
                Levels: "info", "success", "error", "cmd", "step"
        db_mode: Database mode ("auto", "local_postgres", "own_supabase")
        target_os: Target OS ("linux", "macos", "windows")

    Returns:
        True if deployment succeeded, False otherwise
    """
    await on_log(f"Connecting via SSH proxy (target: {target_os})...", "step")

    # Build the deploy script that will run on the remote machine
    deploy_script = _build_deploy_script(env_content, db_mode=db_mode, target_os=target_os)

    ssh_creds = {
        "ssh_host": ssh_host,
        "ssh_port": ssh_port,
        "ssh_user": ssh_user,
        "ssh_password": ssh_password,
        "ssh_key": ssh_key,
    }

    try:
        await on_log(f"Deploying to {ssh_host}:{ssh_port}...", "step")

        # Step 1: Start deploy in background (returns immediately)
        start_result = await _invoke_lambda({
            **ssh_creds,
            "action": "deploy_start",
            "script": deploy_script,
        })

        if "error" in start_result:
            await on_log(f"Connection failed: {start_result['error']}", "error")
            return False

        if not start_result.get("started"):
            await on_log("Failed to start deploy script on server", "error")
            return False

        pid = start_result.get("pid", "?")
        await on_log(f"Deploy started (PID {pid}), streaming logs...", "info")

        # Step 2: Poll for status every 3 seconds
        offset = 0
        max_polls = 200  # 200 * 3s = 10 minutes max
        consecutive_errors = 0

        for poll_num in range(max_polls):
            await asyncio.sleep(3)

            try:
                status = await _invoke_lambda({
                    **ssh_creds,
                    "action": "deploy_status",
                    "offset": offset,
                })
                consecutive_errors = 0  # Reset on success
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    await on_log(f"Lost connection to server: {e}", "error")
                    return False
                await on_log(f"Poll hiccup ({e}), retrying...", "info")
                continue

            if "error" in status:
                await on_log(f"SSH error: {status['error']}", "error")
                return False

            # Stream new output lines
            new_output = status.get("output", "")
            if new_output:
                for line in new_output.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("[STEP]"):
                        await on_log(line[6:].strip(), "step")
                    elif line.startswith("[OK]"):
                        await on_log(line[4:].strip(), "success")
                    elif line.startswith("[ERR]"):
                        await on_log(line[5:].strip(), "error")
                    else:
                        await on_log(line, "info")

            offset = status.get("offset", offset)

            # Check if deploy finished
            if not status.get("running", True):
                exit_code = status.get("exit_code")
                if exit_code is not None:
                    if status.get("success"):
                        await on_log("Deployment completed successfully!", "success")
                        return True
                    else:
                        await on_log(
                            f"Deployment failed (exit code: {exit_code})", "error"
                        )
                        return False
                else:
                    # Process ended but no exit code file yet — wait one more cycle
                    await asyncio.sleep(2)
                    try:
                        final = await _invoke_lambda({
                            **ssh_creds,
                            "action": "deploy_status",
                            "offset": offset,
                        })
                        # Stream any remaining output
                        remaining = final.get("output", "")
                        if remaining:
                            for line in remaining.split("\n"):
                                line = line.strip()
                                if not line:
                                    continue
                                if line.startswith("[STEP]"):
                                    await on_log(line[6:].strip(), "step")
                                elif line.startswith("[OK]"):
                                    await on_log(line[4:].strip(), "success")
                                elif line.startswith("[ERR]"):
                                    await on_log(line[5:].strip(), "error")
                                else:
                                    await on_log(line, "info")

                        if final.get("success"):
                            await on_log("Deployment completed successfully!", "success")
                            return True
                        else:
                            ec = final.get("exit_code", "unknown")
                            await on_log(f"Deployment failed (exit code: {ec})", "error")
                            return False
                    except Exception:
                        await on_log("Deploy process ended — could not read exit code", "error")
                        return False

        # If we get here, we exceeded max polls
        await on_log("Deployment timed out (exceeded 10 minutes)", "error")
        return False

    except RuntimeError as e:
        await on_log(f"Lambda error: {e}", "error")
        return False
    except Exception as e:
        logger.exception("Deploy failed: %s", e)
        await on_log(f"Deployment error: {e}", "error")
        return False


def _build_deploy_script(env_content: str, db_mode: str = "auto", target_os: str = "linux") -> str:
    """Build the bash script that runs on the remote machine to install the agent."""
    if target_os == "macos":
        return _build_deploy_script_macos(env_content, db_mode)
    return _build_deploy_script_linux(env_content, db_mode)


def _build_deploy_script_linux(env_content: str, db_mode: str = "auto") -> str:
    """Build Linux deploy script with distro-aware package management."""

    # PostgreSQL + pgvector installation section (only for local_postgres mode)
    pg_section = ""
    if db_mode == "local_postgres":
        pg_section = """
# ── Install PostgreSQL + pgvector ────────────────────────────
echo "[STEP] Installing PostgreSQL + pgvector..."

if [ "$PKG_MGR" = "apt" ]; then
    export DEBIAN_FRONTEND=noninteractive
    for i in $(seq 1 10); do
        if ! flock -n /var/lib/dpkg/lock-frontend true 2>/dev/null; then
            echo "Waiting for apt lock to clear... ($i/10)"
            sleep 3
        else
            break
        fi
    done
fi

if ! command -v psql &>/dev/null; then
    echo "Installing PostgreSQL..."
    if [ "$PKG_MGR" = "apt" ]; then
        apt-get update -qq 2>&1
        apt-get install -y postgresql postgresql-contrib curl ca-certificates 2>&1
    elif [ "$PKG_MGR" = "dnf" ]; then
        dnf install -y postgresql-server postgresql-contrib curl ca-certificates 2>&1
        postgresql-setup --initdb 2>/dev/null || true
    elif [ "$PKG_MGR" = "yum" ]; then
        yum install -y postgresql-server postgresql-contrib curl ca-certificates 2>&1
        postgresql-setup initdb 2>/dev/null || true
    fi
    systemctl start postgresql
    systemctl enable postgresql
    echo "[OK] PostgreSQL installed: $(psql --version)"
else
    echo "[OK] PostgreSQL already installed: $(psql --version)"
    systemctl start postgresql 2>/dev/null || true
fi

# Install pgvector extension
echo "Installing pgvector extension..."
PG_MAJOR=$(psql --version | grep -oP '\\d+' | head -1)

if sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null; then
    echo "[OK] pgvector already available"
else
    if [ "$PKG_MGR" = "apt" ]; then
        apt-get install -y postgresql-${PG_MAJOR}-pgvector 2>/dev/null || {
            echo "Building pgvector from source..."
            apt-get install -y postgresql-server-dev-all build-essential git 2>&1
            cd /tmp && rm -rf pgvector && git clone https://github.com/pgvector/pgvector.git 2>&1
            cd pgvector && make 2>&1 && make install 2>&1
            cd / && rm -rf /tmp/pgvector
            echo "[OK] pgvector built from source"
        }
    elif [ "$PKG_MGR" = "dnf" ] || [ "$PKG_MGR" = "yum" ]; then
        $PKG_MGR install -y pgvector_${PG_MAJOR} 2>/dev/null || {
            echo "Building pgvector from source..."
            $PKG_MGR install -y postgresql-devel gcc make git 2>&1
            cd /tmp && rm -rf pgvector && git clone https://github.com/pgvector/pgvector.git 2>&1
            cd pgvector && make 2>&1 && make install 2>&1
            cd / && rm -rf /tmp/pgvector
            echo "[OK] pgvector built from source"
        }
    fi
fi

# Create database user and database (drop existing for clean slate)
echo "Setting up database..."
sudo -u postgres psql -c "CREATE USER toup WITH PASSWORD 'toup';" 2>/dev/null || \\
    sudo -u postgres psql -c "ALTER USER toup WITH PASSWORD 'toup';" 2>/dev/null || true
sudo -u postgres psql -c "DROP DATABASE IF EXISTS toup_brain;" 2>/dev/null || true
sudo -u postgres psql -c "CREATE DATABASE toup_brain OWNER toup;" 2>/dev/null || true
sudo -u postgres psql -d toup_brain -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true
echo "[OK] Database toup_brain ready (user: toup, pgvector enabled, clean slate)"
"""

    return f"""#!/bin/bash
set -e

# Kill any previous deploy scripts still running
for pid in $(pgrep -f toup_deploy.sh 2>/dev/null); do
    if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
        kill -9 "$pid" 2>/dev/null || true
    fi
done

# ── Detect package manager ──────────────────────────────
if command -v apt-get &>/dev/null; then
    PKG_MGR="apt"
elif command -v dnf &>/dev/null; then
    PKG_MGR="dnf"
elif command -v yum &>/dev/null; then
    PKG_MGR="yum"
elif command -v pacman &>/dev/null; then
    PKG_MGR="pacman"
else
    echo "[ERR] No supported package manager found (apt/dnf/yum/pacman)"
    exit 1
fi
echo "[OK] Package manager: $PKG_MGR"

echo "[STEP] Checking prerequisites..."

# Install Python + Git + venv if missing
if [ "$PKG_MGR" = "apt" ]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq 2>&1
    if ! command -v python3 &>/dev/null; then
        echo "[STEP] Installing Python 3..."
        apt-get install -y python3 python3-venv python3-pip git curl 2>&1
    else
        apt-get install -y python3-venv 2>&1
    fi
elif [ "$PKG_MGR" = "dnf" ]; then
    if ! command -v python3 &>/dev/null; then
        echo "[STEP] Installing Python 3..."
        dnf install -y python3 python3-pip git curl 2>&1
    fi
elif [ "$PKG_MGR" = "yum" ]; then
    if ! command -v python3 &>/dev/null; then
        echo "[STEP] Installing Python 3..."
        yum install -y python3 python3-pip git curl 2>&1
    fi
elif [ "$PKG_MGR" = "pacman" ]; then
    if ! command -v python3 &>/dev/null; then
        echo "[STEP] Installing Python 3..."
        pacman -Sy --noconfirm python python-pip git curl 2>&1
    fi
fi
echo "[OK] Python: $(python3 --version 2>&1)"

if ! command -v git &>/dev/null; then
    if [ "$PKG_MGR" = "apt" ]; then apt-get install -y git 2>&1;
    elif [ "$PKG_MGR" = "dnf" ]; then dnf install -y git 2>&1;
    elif [ "$PKG_MGR" = "yum" ]; then yum install -y git 2>&1;
    elif [ "$PKG_MGR" = "pacman" ]; then pacman -Sy --noconfirm git 2>&1;
    fi
fi
echo "[OK] Git: $(git --version)"
{pg_section}
# Clone or update agent code
echo "[STEP] Setting up agent code..."
if [ -d "{AGENT_DIR}/.git" ]; then
    echo "Updating existing agent code..."
    cd {AGENT_DIR}
    git checkout main 2>/dev/null || true
    git branch --set-upstream-to=origin/main main 2>/dev/null || true
    git pull origin main --ff-only 2>&1 || true
else
    echo "Cloning agent repository..."
    mkdir -p {AGENT_DIR}
    git clone -b main {AGENT_REPO} {AGENT_DIR} 2>&1
fi
echo "[OK] Agent code ready"

# Create virtualenv
echo "[STEP] Setting up Python environment..."
if [ ! -d "{AGENT_DIR}/venv/bin/python" ]; then
    python3 -m venv {AGENT_DIR}/venv
fi
echo "Installing dependencies (this may take a minute)..."
{AGENT_DIR}/venv/bin/pip install -q -r {AGENT_DIR}/requirements.txt 2>&1
echo "[OK] Dependencies installed"

# Install headless Chromium for browser API (search, page reading)
echo "Installing browser engine..."
{AGENT_DIR}/venv/bin/python -m patchright install chromium 2>&1 || {AGENT_DIR}/venv/bin/python -m playwright install chromium 2>&1 || echo "[WARN] Browser install failed — web search will use fallback"
echo "[OK] Browser engine ready"

# Write .env
echo "[STEP] Writing configuration..."
cat > {AGENT_DIR}/.env << 'TOUP_ENV_EOF'
{env_content}
TOUP_ENV_EOF
echo "[OK] Configuration written"

# Create systemd service
echo "[STEP] Configuring systemd service..."
cat > /etc/systemd/system/{SYSTEMD_UNIT}.service << 'TOUP_SVC_EOF'
[Unit]
Description=Toup Agent Service
After=network.target postgresql.service

[Service]
Type=simple
WorkingDirectory={AGENT_DIR}
EnvironmentFile={AGENT_DIR}/.env
Environment=AGENT_DIR={AGENT_DIR}
ExecStart={AGENT_DIR}/venv/bin/uvicorn agent_main:app --host 0.0.0.0 --port {AGENT_PORT}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
TOUP_SVC_EOF

systemctl daemon-reload
systemctl enable {SYSTEMD_UNIT}
systemctl restart {SYSTEMD_UNIT}
echo "[OK] Service started"

# ── Whoogle (self-hosted Google search proxy) ──
echo "[STEP] Setting up Whoogle search..."
if ! [ -d /opt/whoogle-venv ]; then
    python3 -m venv /opt/whoogle-venv
    /opt/whoogle-venv/bin/pip install -q whoogle-search cachetools 2>&1
fi
cat > /etc/systemd/system/whoogle.service << 'WHOOGLE_EOF'
[Unit]
Description=Whoogle Search (self-hosted Google proxy)
After=network.target

[Service]
Type=simple
Environment=WHOOGLE_PORT=5000
Environment=WHOOGLE_HOST=127.0.0.1
ExecStart=/opt/whoogle-venv/bin/whoogle-search
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
WHOOGLE_EOF
systemctl daemon-reload
systemctl enable whoogle
systemctl start whoogle
echo "[OK] Whoogle search running on port 5000"

# Verify agent health
echo "[STEP] Verifying agent health..."
for i in $(seq 1 6); do
    sleep 5
    if curl -sf http://localhost:{AGENT_PORT}/agent/health > /dev/null 2>&1; then
        echo "[OK] Agent is running and healthy!"

        # Register with platform
        AGENT_API_KEY=$(grep '^AGENT_API_KEY=' {AGENT_DIR}/.env 2>/dev/null | cut -d= -f2)
        PLATFORM_URL=$(grep '^PLATFORM_API_URL=' {AGENT_DIR}/.env 2>/dev/null | cut -d= -f2)
        case "$PLATFORM_URL" in */api) ;; *) PLATFORM_URL="${{PLATFORM_URL%/}}/api" ;; esac
        PUBLIC_IP=$(curl -sf --max-time 5 https://api.ipify.org 2>/dev/null || hostname -I 2>/dev/null | awk '{{print $1}}')
        if [ -n "$AGENT_API_KEY" ] && [ -n "$PLATFORM_URL" ] && [ -n "$PUBLIC_IP" ]; then
            curl -sf -X POST "$PLATFORM_URL/agent-setup/register" \
                -H "Content-Type: application/json" \
                -d "{{\\"agent_api_key\\":\\"$AGENT_API_KEY\\",\\"agent_url\\":\\"http://$PUBLIC_IP:{AGENT_PORT}\\"}}" > /dev/null 2>&1 && \
                echo "[OK] Registered with platform" || true
        fi
        exit 0
    fi
    echo "Waiting for agent to start... (attempt $i/6)"
done

echo "[ERR] Agent did not respond to health check after 30s"
echo "Recent logs:"
journalctl -u {SYSTEMD_UNIT} --no-pager -n 20 2>/dev/null || true
exit 1
"""


LAUNCHD_PLIST = "com.toup.agent"


def _build_deploy_script_macos(env_content: str, db_mode: str = "auto") -> str:
    """Build macOS deploy script using Homebrew + launchd."""

    pg_section = ""
    if db_mode == "local_postgres":
        pg_section = """
# ── Install PostgreSQL + pgvector ────────────────────────────
echo "[STEP] Installing PostgreSQL + pgvector..."
if ! command -v psql &>/dev/null; then
    brew install postgresql@16 pgvector 2>&1
    brew services start postgresql@16
else
    echo "[OK] PostgreSQL already installed: $(psql --version)"
    brew services start postgresql@16 2>/dev/null || brew services start postgresql 2>/dev/null || true
fi

# Create database and user
createuser toup 2>/dev/null || true
psql postgres -c "ALTER USER toup WITH PASSWORD 'toup';" 2>/dev/null || true
createdb -O toup toup_brain 2>/dev/null || true
psql -d toup_brain -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true
echo "[OK] Database toup_brain ready"
"""

    return f"""#!/bin/bash
set -e

AGENT_DIR="{AGENT_DIR}"

# Kill any previous deploy scripts still running
for pid in $(pgrep -f toup_deploy.sh 2>/dev/null); do
    if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
        kill -9 "$pid" 2>/dev/null || true
    fi
done

echo "[STEP] Checking prerequisites..."

# Check for Homebrew
if ! command -v brew &>/dev/null; then
    echo "[ERR] Homebrew is required on macOS but not installed."
    echo "[ERR] Install it: /bin/bash -c \\\"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\\\""
    exit 1
fi
echo "[OK] Homebrew found"

# Install Python + Git if missing
if ! command -v python3 &>/dev/null; then
    echo "[STEP] Installing Python 3..."
    brew install python3 2>&1
fi
echo "[OK] Python: $(python3 --version 2>&1)"

if ! command -v git &>/dev/null; then
    echo "[STEP] Installing Git..."
    brew install git 2>&1
fi
echo "[OK] Git: $(git --version)"
{pg_section}
# Clone or update agent code
echo "[STEP] Setting up agent code..."
sudo mkdir -p "$AGENT_DIR" 2>/dev/null || mkdir -p "$AGENT_DIR"
sudo chown $(whoami) "$AGENT_DIR" 2>/dev/null || true
if [ -d "$AGENT_DIR/.git" ]; then
    echo "Updating existing agent code..."
    cd "$AGENT_DIR"
    git checkout main 2>/dev/null || true
    git branch --set-upstream-to=origin/main main 2>/dev/null || true
    git pull origin main --ff-only 2>&1 || true
else
    echo "Cloning agent repository..."
    git clone -b main {AGENT_REPO} "$AGENT_DIR" 2>&1
fi
echo "[OK] Agent code ready"

# Create virtualenv
echo "[STEP] Setting up Python environment..."
if [ ! -d "$AGENT_DIR/venv/bin/python" ]; then
    python3 -m venv "$AGENT_DIR/venv"
fi
echo "Installing dependencies (this may take a minute)..."
"$AGENT_DIR/venv/bin/pip" install -q -r "$AGENT_DIR/requirements.txt" 2>&1
echo "[OK] Dependencies installed"

# Write .env
echo "[STEP] Writing configuration..."
cat > "$AGENT_DIR/.env" << 'TOUP_ENV_EOF'
{env_content}
TOUP_ENV_EOF
echo "[OK] Configuration written"

# Stop existing service
echo "[STEP] Configuring launchd service..."
launchctl unload "$HOME/Library/LaunchAgents/{LAUNCHD_PLIST}.plist" 2>/dev/null || true
lsof -ti :{AGENT_PORT} 2>/dev/null | xargs kill -9 2>/dev/null || true

# Create LaunchAgent plist
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$HOME/Library/LaunchAgents/{LAUNCHD_PLIST}.plist" << TOUP_PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_PLIST}</string>
    <key>ProgramArguments</key>
    <array>
        <string>$AGENT_DIR/venv/bin/uvicorn</string>
        <string>agent_main:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>{AGENT_PORT}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$AGENT_DIR</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>AGENT_DIR</key>
        <string>$AGENT_DIR</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$AGENT_DIR/agent.log</string>
    <key>StandardErrorPath</key>
    <string>$AGENT_DIR/agent_err.log</string>
</dict>
</plist>
TOUP_PLIST_EOF

launchctl load "$HOME/Library/LaunchAgents/{LAUNCHD_PLIST}.plist"
echo "[OK] Service started via launchd"

# Verify agent health
echo "[STEP] Verifying agent health..."
for i in $(seq 1 6); do
    sleep 5
    if curl -sf http://localhost:{AGENT_PORT}/agent/health > /dev/null 2>&1; then
        echo "[OK] Agent is running and healthy!"

        # Register with platform
        AGENT_API_KEY=$(grep '^AGENT_API_KEY=' "$AGENT_DIR/.env" 2>/dev/null | cut -d= -f2)
        PLATFORM_URL=$(grep '^PLATFORM_API_URL=' "$AGENT_DIR/.env" 2>/dev/null | cut -d= -f2)
        case "$PLATFORM_URL" in */api) ;; *) PLATFORM_URL="${{PLATFORM_URL%/}}/api" ;; esac
        PUBLIC_IP=$(curl -sf --max-time 5 https://api.ipify.org 2>/dev/null || ipconfig getifaddr en0 2>/dev/null)
        if [ -n "$AGENT_API_KEY" ] && [ -n "$PLATFORM_URL" ] && [ -n "$PUBLIC_IP" ]; then
            curl -sf -X POST "$PLATFORM_URL/agent-setup/register" \
                -H "Content-Type: application/json" \
                -d "{{\\"agent_api_key\\":\\"$AGENT_API_KEY\\",\\"agent_url\\":\\"http://$PUBLIC_IP:{AGENT_PORT}\\"}}" > /dev/null 2>&1 && \
                echo "[OK] Registered with platform" || true
        fi
        exit 0
    fi
    echo "Waiting for agent to start... (attempt $i/6)"
done

echo "[ERR] Agent did not respond to health check after 30s"
echo "Recent logs:"
cat "$AGENT_DIR/agent_err.log" 2>/dev/null | tail -20 || true
exit 1
"""


# ── Local Machine Scripts (no Lambda needed) ─────────────────────

def generate_setup_script(env_content: str) -> str:
    """Generate a self-contained bash setup script for local machines (Mac/Linux).

    Installs the agent and starts it in the foreground:
    - Clones repo, creates venv, installs deps, writes .env
    - Optionally installs PostgreSQL + pgvector for local_postgres db_mode
    - Installs CLI helper (~/toup-agent/toup) for start/stop/update/status
    - Starts uvicorn in the foreground — user sees output, Ctrl+C stops it
    """
    # Detect if local PostgreSQL install is needed
    needs_local_pg = "postgresql+asyncpg://toup:toup@localhost" in env_content

    pg_install_section = ""
    if needs_local_pg:
        pg_install_section = r'''
# ── Install PostgreSQL + pgvector (local mode) ──────────
echo ""
echo "[4/7] Setting up local PostgreSQL + pgvector..."

if ! command -v psql &>/dev/null; then
  echo "  Installing PostgreSQL..."
  if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &>/dev/null; then
      brew install postgresql@16 </dev/null >/dev/null 2>&1 || true
      brew services start postgresql@16 </dev/null >/dev/null 2>&1 || true
    else
      echo "  Homebrew is required to install PostgreSQL on macOS."
      echo "  Install with: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
      exit 1
    fi
  else
    sudo apt-get update -qq
    sudo apt-get install -y postgresql postgresql-contrib
    sudo systemctl start postgresql
    sudo systemctl enable postgresql
  fi
else
  echo "  PostgreSQL already installed: $(psql --version)"
fi

# Install pgvector (needed regardless of whether PostgreSQL was just installed)
echo "  Ensuring pgvector is installed..."
_pgvector_ok=false
if [[ "$OSTYPE" == "darwin"* ]]; then
  # Find pg_config for the installed PostgreSQL version
  PG_CONFIG=$(find /opt/homebrew/Cellar/postgresql* /usr/local/Cellar/postgresql* -name pg_config 2>/dev/null | head -1)
  if [ -z "$PG_CONFIG" ]; then
    PG_CONFIG=$(command -v pg_config 2>/dev/null || true)
  fi

  # Try brew first — but verify the .so ends up in the right PG lib dir
  if command -v brew &>/dev/null; then
    brew install pgvector </dev/null >/dev/null 2>&1 || true
    # Check if vector.so exists in the PG lib directory for this pg_config
    _pg_lib_dir=$("$PG_CONFIG" --pkglibdir 2>/dev/null || true)
    if [ -n "$_pg_lib_dir" ] && [ -f "$_pg_lib_dir/vector.so" ]; then
      _pgvector_ok=true
      echo "  pgvector installed via Homebrew"
    fi
  fi

  # If brew didn't work (e.g. user has postgresql@14), build from source
  if [ "$_pgvector_ok" = false ] && [ -n "$PG_CONFIG" ]; then
    echo "  Building pgvector from source for $(${PG_CONFIG} --version)..."
    cd /tmp && rm -rf pgvector_build
    if git clone --branch v0.8.0 --depth 1 https://github.com/pgvector/pgvector.git pgvector_build 2>/dev/null; then
      cd pgvector_build
      if make PG_CONFIG="$PG_CONFIG" 2>/dev/null && make PG_CONFIG="$PG_CONFIG" install 2>/dev/null; then
        _pgvector_ok=true
      fi
    fi
    cd / && rm -rf /tmp/pgvector_build
  fi
else
  PG_MAJOR=$(psql --version 2>/dev/null | grep -oE '[0-9]+' | head -1)
  sudo apt-get install -y "postgresql-${PG_MAJOR}-pgvector" 2>/dev/null || \
  sudo apt-get install -y postgresql-16-pgvector 2>/dev/null || \
  sudo apt-get install -y postgresql-15-pgvector 2>/dev/null || \
  sudo apt-get install -y postgresql-14-pgvector 2>/dev/null || {
    echo "  Building pgvector from source..."
    sudo apt-get install -y postgresql-server-dev-all build-essential
    cd /tmp && rm -rf pgvector_build
    git clone --branch v0.8.0 --depth 1 https://github.com/pgvector/pgvector.git pgvector_build
    cd pgvector_build && make && sudo make install
    cd / && rm -rf /tmp/pgvector_build
  }
  _pgvector_ok=true
fi

if [ "$_pgvector_ok" = true ]; then
  echo "  pgvector installed"
else
  echo "  ⚠ pgvector could not be installed (vector search will be unavailable)"
fi

# Create database and user
echo "  Creating database..."
if [[ "$OSTYPE" == "darwin"* ]]; then
  createuser toup 2>/dev/null || true
  psql postgres -c "ALTER USER toup WITH PASSWORD 'toup';" 2>/dev/null || true
  dropdb toup_brain 2>/dev/null || true
  createdb -O toup toup_brain 2>/dev/null || true
  psql -d toup_brain -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true
else
  sudo -u postgres psql -c "CREATE USER toup WITH PASSWORD 'toup';" 2>/dev/null || true
  sudo -u postgres psql -c "DROP DATABASE IF EXISTS toup_brain;" 2>/dev/null || true
  sudo -u postgres psql -c "CREATE DATABASE toup_brain OWNER toup;" 2>/dev/null || true
  sudo -u postgres psql -d toup_brain -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true
fi
echo "  PostgreSQL + pgvector ready"
'''

    return f'''#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Toup Agent — Local Setup Script
#  Generated by Toup Platform
# ═══════════════════════════════════════════════════════════════

# Wrap in a function so bash reads the entire script before executing.
# This prevents pipe-buffer issues when running via: curl ... | bash
_toup_setup() {{

set -e

AGENT_DIR="$HOME/toup-agent"
AGENT_PORT={AGENT_PORT}
REPO="{AGENT_REPO}"

echo ""
echo "  Toup Agent Setup"
echo "  ════════════════════════"
echo ""

# ── Check & auto-install prerequisites ───────────────────
echo "[1/7] Checking prerequisites..."

if [[ "$OSTYPE" == "darwin"* ]]; then
  # ── macOS: ensure Xcode Command Line Tools ──
  if ! xcode-select -p &>/dev/null; then
    echo "  Installing Xcode Command Line Tools (this may take a few minutes)..."
    xcode-select --install 2>/dev/null || true
    # Wait for the installer to finish
    echo "  Waiting for installation to complete..."
    until xcode-select -p &>/dev/null; do
      sleep 5
    done
    echo "  Xcode Command Line Tools installed"
  fi

  # ── macOS: ensure Homebrew ──
  if ! command -v brew &>/dev/null; then
    echo "  Installing Homebrew (this may take a few minutes)..."
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for Apple Silicon and Intel
    if [ -f /opt/homebrew/bin/brew ]; then
      eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -f /usr/local/bin/brew ]; then
      eval "$(/usr/local/bin/brew shellenv)"
    fi
    echo "  Homebrew installed"
  fi

  # ── macOS: ensure Python 3.10+ ──
  _need_python=false
  if ! command -v python3 &>/dev/null; then
    _need_python=true
  else
    _pyver_minor=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
    if [ "$_pyver_minor" -lt 10 ]; then
      echo "  Python 3.$_pyver_minor is too old (need 3.10+), upgrading..."
      _need_python=true
    fi
  fi
  if [ "$_need_python" = true ]; then
    echo "  Installing Python 3.12..."
    brew install python@3.12 </dev/null
    # Ensure brew python is on PATH
    export PATH="$(brew --prefix python@3.12)/libexec/bin:$(brew --prefix)/bin:$PATH"
    echo "  Python 3.12 installed"
  fi

  # ── macOS: ensure Git ──
  if ! command -v git &>/dev/null; then
    echo "  Installing Git..."
    brew install git </dev/null
    echo "  Git installed"
  fi

else
  # ── Linux: auto-install missing prerequisites ──
  if ! command -v python3 &>/dev/null || ! command -v git &>/dev/null; then
    echo "  Installing prerequisites..."
    sudo apt-get update -qq
  fi
  if ! command -v python3 &>/dev/null; then
    echo "  Installing Python 3..."
    sudo apt-get install -y python3 python3-venv python3-pip
  fi
  _pyver_minor=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
  if [ "$_pyver_minor" -lt 10 ]; then
    echo "  Python 3.$_pyver_minor is too old (need 3.10+)."
    echo "  Installing Python 3.12..."
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y python3.12 python3.12-venv
    # Use python3.12 explicitly
    alias python3=python3.12
  fi
  if ! command -v git &>/dev/null; then
    echo "  Installing Git..."
    sudo apt-get install -y git
  fi
fi

# Pick the best python3 binary (prefer 3.12, then 3.11, then whatever is on PATH)
PYTHON3=""
for _candidate in python3.12 python3.11 python3; do
  if command -v "$_candidate" &>/dev/null; then
    _ver=$("$_candidate" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
    if [ "$_ver" -ge 10 ]; then
      PYTHON3=$(command -v "$_candidate")
      break
    fi
  fi
done

# Final check — if somehow still missing, bail with clear message
if [ -z "$PYTHON3" ]; then
  echo "  ERROR: Python 3.10+ could not be found. Please install Python 3.10+ manually and re-run."
  exit 1
fi
if ! command -v git &>/dev/null; then
  echo "  ERROR: Git could not be installed. Please install Git manually and re-run."
  exit 1
fi

PYVER=$($PYTHON3 --version 2>&1)
echo "  $PYVER ($(which $PYTHON3))"
echo "  $(git --version)"

# ── Clone or update ──────────────────────────────────────
echo ""
echo "[2/7] Setting up agent code..."

if [ -d "$AGENT_DIR/.git" ]; then
  echo "  Updating existing installation..."
  cd "$AGENT_DIR"
  git checkout main 2>/dev/null || true
  git branch --set-upstream-to=origin/main main 2>/dev/null || true
  git pull origin main --ff-only
else
  echo "  Cloning agent repository..."
  git clone -b main "$REPO" "$AGENT_DIR"
fi

# Auto-detect repo structure (flat vs nested)
if [ -f "$AGENT_DIR/agent_main.py" ]; then
  BACKEND_ROOT="$AGENT_DIR"
elif [ -f "$AGENT_DIR/backend/agent_main.py" ]; then
  BACKEND_ROOT="$AGENT_DIR/backend"
else
  echo "  ERROR: agent_main.py not found in repo"
  exit 1
fi

# ── Virtualenv ───────────────────────────────────────────
echo ""
echo "[3/7] Setting up Python environment..."

if [ ! -d "$AGENT_DIR/venv" ]; then
  "$PYTHON3" -m venv "$AGENT_DIR/venv"
  echo "  Virtualenv created"
else
  # Recreate venv if existing Python is too old
  _existing_ver=$("$AGENT_DIR/venv/bin/python" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
  if [ "$_existing_ver" -lt 10 ]; then
    echo "  Existing venv uses Python 3.$_existing_ver, recreating with $PYVER..."
    rm -rf "$AGENT_DIR/venv"
    "$PYTHON3" -m venv "$AGENT_DIR/venv"
    echo "  Virtualenv recreated"
  fi
fi

echo "  Installing dependencies (this may take a minute)..."
"$AGENT_DIR/venv/bin/pip" install -q -r "$BACKEND_ROOT/requirements.txt"
echo "  Dependencies installed"

echo "  Installing browser engine..."
"$AGENT_DIR/venv/bin/python" -m patchright install chromium 2>/dev/null || "$AGENT_DIR/venv/bin/python" -m playwright install chromium 2>/dev/null || echo "  (browser engine skipped — web search will use fallback)"
echo "  Browser engine ready"
{pg_install_section}
# ── Write .env ───────────────────────────────────────────
echo ""
echo "[5/7] Writing configuration..."

cat > "$BACKEND_ROOT/.env" << 'TOUP_ENV_EOF'
{env_content}
TOUP_ENV_EOF
echo "  Configuration saved to $BACKEND_ROOT/.env"

# ── Stop existing agent ──────────────────────────────────
echo ""
echo "[6/7] Stopping existing agent..."

# Stop old PID-based process if exists
if [ -f "$AGENT_DIR/agent.pid" ]; then
  OLD_PID=$(cat "$AGENT_DIR/agent.pid" 2>/dev/null)
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    kill "$OLD_PID" 2>/dev/null || true
    echo "  Stopped old agent (PID $OLD_PID)"
    sleep 2
  fi
  rm -f "$AGENT_DIR/agent.pid"
fi

# Stop existing background service if running
if [[ "$OSTYPE" == "darwin"* ]]; then
  launchctl unload "$HOME/Library/LaunchAgents/com.toup.agent.plist" 2>/dev/null || true
else
  systemctl --user stop toup-agent 2>/dev/null || true
fi

# Kill any uvicorn on AGENT_PORT
lsof -ti :$AGENT_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# ── Create CLI helper ─────────────────────────────────────
echo ""
echo "[7/7] Setting up CLI..."

UVICORN="$AGENT_DIR/venv/bin/uvicorn"
BACKEND_DIR="$BACKEND_ROOT"

cat > "$AGENT_DIR/toup" << 'CLI_EOF'
#!/bin/bash
# ═══════════════════════════════════════════════════════
#  toup — Toup Agent CLI
#  Usage: ~/toup-agent/toup [command]
# ═══════════════════════════════════════════════════════

AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Auto-detect repo structure
if [ -f "$AGENT_DIR/agent_main.py" ]; then
  BACKEND_DIR="$AGENT_DIR"
elif [ -f "$AGENT_DIR/backend/agent_main.py" ]; then
  BACKEND_DIR="$AGENT_DIR/backend"
else
  BACKEND_DIR="$AGENT_DIR"
fi
AGENT_PORT=$(grep '^AGENT_PORT=' "$BACKEND_DIR/.env" 2>/dev/null | cut -d= -f2)
AGENT_PORT=${{AGENT_PORT:-8001}}
UVICORN="$AGENT_DIR/venv/bin/uvicorn"

_run() {{
  echo ""
  echo "  Starting Toup Agent (foreground)..."
  echo "  Press Ctrl+C to stop."
  echo ""
  cd "$BACKEND_DIR" && exec "$UVICORN" agent_main:app --host 0.0.0.0 --port "$AGENT_PORT"
}}

_status() {{
  if curl -sf "http://localhost:$AGENT_PORT/agent/health" > /dev/null 2>&1; then
    HEALTH=$(curl -sf "http://localhost:$AGENT_PORT/agent/health")
    echo "Agent is running"
    echo "   URL:     http://localhost:$AGENT_PORT"
    echo "   Health:  $HEALTH"
  else
    echo "Agent is not responding"
  fi
}}

_update() {{
  echo "Updating Toup Agent..."
  cd "$AGENT_DIR"
  git branch --set-upstream-to=origin/main main 2>/dev/null || true
  git pull origin main --ff-only
  echo "Installing dependencies..."
  "$AGENT_DIR/venv/bin/pip" install -q -r "$BACKEND_DIR/requirements.txt"
  echo "Updated! Run: ~/toup-agent/toup start"
}}

_stop() {{
  lsof -ti :$AGENT_PORT 2>/dev/null | xargs kill 2>/dev/null || true
  echo "Agent stopped"
}}

_logs() {{
  tail -f "$AGENT_DIR/agent.log" 2>/dev/null || echo "No log file found. Run the agent first."
}}

case "${{1:-help}}" in
  start|run) _run ;;
  stop)      _stop ;;
  status)    _status ;;
  update)    _update ;;
  logs)      _logs ;;
  *)
    echo "Usage: toup <command>"
    echo ""
    echo "Commands:"
    echo "  start    Start the agent (foreground, Ctrl+C to stop)"
    echo "  stop     Stop the agent"
    echo "  status   Check if the agent is running"
    echo "  update   Pull latest code (then run 'toup start')"
    echo "  logs     Tail the agent logs"
    ;;
esac
CLI_EOF
chmod +x "$AGENT_DIR/toup"

echo "  CLI installed: ~/toup-agent/toup"

# ── Register with platform ────────────────────────────────
AGENT_API_KEY=$(grep '^AGENT_API_KEY=' "$BACKEND_ROOT/.env" | cut -d= -f2)
PLATFORM_URL=$(grep '^PLATFORM_API_URL=' "$BACKEND_ROOT/.env" | cut -d= -f2)
# Ensure /api suffix
case "$PLATFORM_URL" in */api) ;; *) PLATFORM_URL="${{PLATFORM_URL%/}}/api" ;; esac

echo ""
echo "  ════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Starting your agent now..."
echo "  After you see 'Toup Agent ready', go to https://toup.ai"
echo "  and click 'Verify Connection'."
echo ""
echo "  Press Ctrl+C to stop the agent."
echo "  To restart later: ~/toup-agent/toup start"
echo "  To update:        ~/toup-agent/toup update"
echo "  ════════════════════════════════════════════"
echo ""

# Register with platform in the background (after agent starts)
(
  sleep 10
  # Detect public IP for registration
  PUBLIC_IP=$(curl -sf --max-time 5 https://api.ipify.org 2>/dev/null || hostname -I 2>/dev/null | awk '{{print $1}}')
  if [ -n "$AGENT_API_KEY" ] && [ -n "$PLATFORM_URL" ] && [ -n "$PUBLIC_IP" ]; then
    # Retry up to 3 times with backoff
    for attempt in 1 2 3; do
      RESP=$(curl -sf -w "%{{http_code}}" -X POST "$PLATFORM_URL/agent-setup/register" \
        -H "Content-Type: application/json" \
        -d "{{\\"agent_api_key\\":\\"$AGENT_API_KEY\\",\\"agent_url\\":\\"http://$PUBLIC_IP:$AGENT_PORT\\"}}" 2>/dev/null)
      HTTP_CODE="${{RESP: -3}}"
      if [ "$HTTP_CODE" = "200" ]; then
        echo "Registered with platform (http://$PUBLIC_IP:$AGENT_PORT)"
        break
      fi
      echo "Platform register attempt $attempt failed (HTTP $HTTP_CODE), retrying..."
      sleep $((attempt * 5))
    done
  fi
) &

# Start agent in foreground — user sees output, Ctrl+C stops it
cd "$BACKEND_DIR" && exec "$UVICORN" agent_main:app --host 0.0.0.0 --port $AGENT_PORT

}}
# Call the wrapper function — ensures entire script is read before execution
_toup_setup
'''


def generate_setup_script_windows(env_content: str) -> str:
    """Generate a PowerShell setup script for Windows."""
    return f'''# ═══════════════════════════════════════════════════════════════
#  Toup Agent — Windows Setup Script (PowerShell)
#  Generated by Toup Platform
#  Run in PowerShell: paste this entire script
# ═══════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

$AGENT_DIR = "$env:USERPROFILE\\toup-agent"
$AGENT_PORT = {AGENT_PORT}
$REPO = "{AGENT_REPO}"

Write-Host ""
Write-Host "  Toup Agent Setup (Windows)" -ForegroundColor Cyan
Write-Host "  =========================="
Write-Host ""

# ── Check prerequisites ──────────────────────────────────
Write-Host "[1/6] Checking prerequisites..."

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {{
    $py = Get-Command python3 -ErrorAction SilentlyContinue
}}
if (-not $py) {{
    Write-Host "  Python is required but not installed." -ForegroundColor Red
    Write-Host "  Download from: https://www.python.org/downloads/"
    Write-Host "  Make sure to check 'Add Python to PATH' during install."
    exit 1
}}
$PYTHON = $py.Source
Write-Host "  $(&$PYTHON --version 2>&1)"

$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {{
    Write-Host "  Git is required but not installed." -ForegroundColor Red
    Write-Host "  Download from: https://git-scm.com/download/win"
    exit 1
}}
Write-Host "  $(git --version)"

# ── Clone or update ──────────────────────────────────────
Write-Host ""
Write-Host "[2/6] Setting up agent code..."

if (Test-Path "$AGENT_DIR\\.git") {{
    Write-Host "  Updating existing installation..."
    Push-Location $AGENT_DIR
    git checkout main 2>$null
    git branch --set-upstream-to=origin/main main 2>$null
    git pull origin main --ff-only
    Pop-Location
}} else {{
    Write-Host "  Cloning agent repository..."
    git clone $REPO $AGENT_DIR
}}

# ── Virtualenv ───────────────────────────────────────────
Write-Host ""
Write-Host "[3/6] Setting up Python environment..."

if (-not (Test-Path "$AGENT_DIR\\venv")) {{
    & $PYTHON -m venv "$AGENT_DIR\\venv"
    Write-Host "  Virtualenv created"
}}

Write-Host "  Installing dependencies (this may take a minute)..."
& "$AGENT_DIR\\venv\\Scripts\\pip.exe" install -q -r "$AGENT_DIR\\requirements.txt"
Write-Host "  Dependencies installed"

# ── Write .env ───────────────────────────────────────────
Write-Host ""
Write-Host "[4/6] Writing configuration..."

@"
{env_content}
"@ | Set-Content -Path "$AGENT_DIR\\.env" -Encoding UTF8
Write-Host "  Configuration saved to $AGENT_DIR\\.env"

# ── Stop existing agent if running ───────────────────────
Write-Host ""
Write-Host "[5/6] Starting agent..."

$pidFile = "$AGENT_DIR\\agent.pid"
if (Test-Path $pidFile) {{
    $oldPid = Get-Content $pidFile -ErrorAction SilentlyContinue
    if ($oldPid) {{
        $proc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
        if ($proc) {{
            Write-Host "  Stopping existing agent (PID $oldPid)..."
            Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
        }}
    }}
}}

# Start agent
Push-Location "$AGENT_DIR"
$agentProc = Start-Process -FilePath "$AGENT_DIR\\venv\\Scripts\\uvicorn.exe" `
    -ArgumentList "agent_main:app","--host","0.0.0.0","--port","$AGENT_PORT" `
    -RedirectStandardOutput "$AGENT_DIR\\agent.log" `
    -RedirectStandardError "$AGENT_DIR\\agent_err.log" `
    -PassThru -NoNewWindow:$false -WindowStyle Hidden
Pop-Location

$agentProc.Id | Set-Content $pidFile
Write-Host "  Agent started (PID: $($agentProc.Id))"

# ── Verify ───────────────────────────────────────────────
Write-Host ""
Write-Host "[6/6] Verifying..."

for ($i = 1; $i -le 6; $i++) {{
    Start-Sleep -Seconds 3
    try {{
        $resp = Invoke-WebRequest -Uri "http://localhost:$AGENT_PORT/agent/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {{
            Write-Host ""
            Write-Host "  Agent is running at http://localhost:$AGENT_PORT" -ForegroundColor Green
            Write-Host ""
            Write-Host "  ════════════════════════════════════════════"
            Write-Host "  Setup complete!" -ForegroundColor Green
            Write-Host ""
            Write-Host "  Logs:    Get-Content $AGENT_DIR\\agent.log -Tail 50"
            Write-Host "  Stop:    Stop-Process -Id (Get-Content $AGENT_DIR\\agent.pid)"
            Write-Host "  ════════════════════════════════════════════"
            exit 0
        }}
    }} catch {{}}
    Write-Host "  Waiting for agent to start... ($i/6)"
}}

Write-Host ""
Write-Host "  Agent did not respond after 18 seconds." -ForegroundColor Red
Write-Host "  Check logs: Get-Content $AGENT_DIR\\agent.log -Tail 50"
Write-Host "  Check errors: Get-Content $AGENT_DIR\\agent_err.log -Tail 50"
exit 1
'''


def generate_env_content(
    user_id: str,
    agent_api_key: str,
    openai_api_key: str = "",
    anthropic_api_key: str = "",
    google_api_key: str = "",
    mistral_api_key: str = "",
    groq_api_key: str = "",
    xai_api_key: str = "",
    deepseek_api_key: str = "",
    agent_model: str = "claude-sonnet-4-6",
    llm_mode: str = "manual",
    telegram_bot_token: str = "",
    discord_bot_token: str = "",
    slack_bot_token: str = "",
    slack_app_token: str = "",
    whatsapp_phone_number_id: str = "",
    whatsapp_access_token: str = "",
    brave_api_key: str = "",
    elevenlabs_api_key: str = "",
    toup_token: str = "",
    db_mode: str = "auto",
    supabase_url: str = "",
) -> str:
    """Generate .env file content for the agent service."""
    lines = [
        "# Toup Agent Configuration",
        "# Generated by Toup Platform",
        "",
        "# --- Runtime ---",
        "RUN_MODE=agent",
        "",
        "# --- Database ---",
    ]
    if db_mode == "own_supabase" and supabase_url:
        lines.append(f"DATABASE_URL={supabase_url}")
    elif db_mode == "local_postgres":
        lines.append("DATABASE_URL=postgresql+asyncpg://toup:toup@localhost:5432/toup_brain")
    else:
        lines.append(f"DATABASE_URL={settings.database_url}")
    lines += [
        "",
        "# --- Platform ---",
        f"PLATFORM_API_URL={settings.platform_api_url}",
        f"USER_ID={user_id}",
        f"AGENT_API_KEY={agent_api_key}",
    ]
    if toup_token:
        lines.append(f"TOUP_TOKEN={toup_token}")
    lines.append("")
    lines += [
        "# --- Auth (must match platform for JWT validation) ---",
        f"JWT_SECRET={settings.jwt_secret}",
        "",
        "# --- LLM ---",
        f"LLM_MODE={llm_mode}",
    ]

    if openai_api_key:
        lines.append(f"OPENAI_API_KEY={openai_api_key}")
    if anthropic_api_key:
        lines.append(f"ANTHROPIC_API_KEY={anthropic_api_key}")
    if google_api_key:
        lines.append(f"GOOGLE_API_KEY={google_api_key}")
    if mistral_api_key:
        lines.append(f"MISTRAL_API_KEY={mistral_api_key}")
    if groq_api_key:
        lines.append(f"GROQ_API_KEY={groq_api_key}")
    if xai_api_key:
        lines.append(f"XAI_API_KEY={xai_api_key}")
    if deepseek_api_key:
        lines.append(f"DEEPSEEK_API_KEY={deepseek_api_key}")
    lines.append(f"AGENT_MODEL={agent_model}")

    # Workspace (local mode uses agent dir, not /app)
    lines.extend([
        "",
        "# --- Workspace ---",
        "AGENT_WORKSPACE_DIR=./workspace",
        "SKILLS_DIR=./skills",
    ])

    # Embedding — use OpenAI if user has key, otherwise local
    if openai_api_key:
        _emb_provider, _emb_model, _emb_dim = "openai", "text-embedding-3-small", 1536
    else:
        _emb_provider, _emb_model, _emb_dim = "local", "all-MiniLM-L6-v2", 384
    lines.extend([
        "",
        "# --- Embedding ---",
        f"EMBEDDING_PROVIDER={_emb_provider}",
        f"EMBEDDING_MODEL={_emb_model}",
        f"EMBEDDING_DIMENSION={_emb_dim}",
    ])

    # Channels
    channel_lines = []
    if telegram_bot_token:
        channel_lines.append(f"TELEGRAM_BOT_TOKEN={telegram_bot_token}")
    if discord_bot_token:
        channel_lines.append(f"DISCORD_BOT_TOKEN={discord_bot_token}")
    if slack_bot_token:
        channel_lines.append(f"SLACK_BOT_TOKEN={slack_bot_token}")
    if slack_app_token:
        channel_lines.append(f"SLACK_APP_TOKEN={slack_app_token}")
    if whatsapp_phone_number_id:
        channel_lines.append(f"WHATSAPP_PHONE_NUMBER_ID={whatsapp_phone_number_id}")
    if whatsapp_access_token:
        channel_lines.append(f"WHATSAPP_ACCESS_TOKEN={whatsapp_access_token}")
    if channel_lines:
        lines.extend(["", "# --- Channels ---"] + channel_lines)

    # Services
    service_lines = []
    if brave_api_key:
        service_lines.append(f"BRAVE_API_KEY={brave_api_key}")
    if elevenlabs_api_key:
        service_lines.append(f"ELEVENLABS_API_KEY={elevenlabs_api_key}")
    if service_lines:
        lines.extend(["", "# --- Services ---"] + service_lines)

    lines.append("")
    return "\n".join(lines)
