"""
SSH-based agent deployment service.

Connects to a target machine via SSH and installs/configures the Toup Agent.
Yields log lines in real-time for WebSocket streaming to the frontend.
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

import asyncssh
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

AGENT_REPO = "https://github.com/toup-com/toup-platform.git"
AGENT_DIR = "/opt/toup-agent"
AGENT_PORT = 8001
SYSTEMD_UNIT = "toup-agent"


async def _run_cmd(
    conn: asyncssh.SSHClientConnection,
    cmd: str,
    on_log: Callable[[str, str], Awaitable[None]],
    label: str = "",
    sudo: bool = False,
) -> tuple[bool, str]:
    """Run a single command via SSH and stream output."""
    full_cmd = f"sudo {cmd}" if sudo else cmd
    await on_log(f"$ {full_cmd}", "cmd")

    try:
        result = await asyncio.wait_for(
            conn.run(full_cmd, check=False),
            timeout=300,  # 5 min max per command
        )
    except asyncio.TimeoutError:
        await on_log(f"Command timed out after 300s", "error")
        return False, ""

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    if stdout:
        for line in stdout.split("\n")[-20:]:  # Last 20 lines
            await on_log(line, "info")
    if stderr and result.exit_status != 0:
        for line in stderr.split("\n")[-10:]:
            await on_log(line, "error")

    if result.exit_status != 0:
        await on_log(f"Command exited with code {result.exit_status}", "error")
        return False, stdout

    return True, stdout


async def test_ssh_connection(
    ssh_host: str,
    ssh_port: int,
    ssh_user: str,
    ssh_password: Optional[str] = None,
    ssh_key: Optional[str] = None,
) -> dict:
    """Test SSH connectivity and return machine info."""
    try:
        conn_kwargs = _build_conn_kwargs(ssh_host, ssh_port, ssh_user, ssh_password, ssh_key)
        async with asyncssh.connect(**conn_kwargs) as conn:
            # Get OS info
            os_result = await conn.run("cat /etc/os-release 2>/dev/null | head -1 || uname -s", check=False)
            os_info = (os_result.stdout or "Unknown").strip()
            if "PRETTY_NAME=" in os_info:
                os_info = os_info.split("=", 1)[1].strip('"')

            # Get Python version
            py_result = await conn.run("python3 --version 2>/dev/null || echo 'not found'", check=False)
            py_info = (py_result.stdout or "not found").strip()

            return {
                "connected": True,
                "os": os_info,
                "python": py_info,
            }
    except asyncssh.PermissionDenied:
        return {"connected": False, "error": "Authentication failed. Check username and password."}
    except asyncssh.ConnectionLost:
        return {"connected": False, "error": "Connection lost. Check the host address."}
    except OSError as e:
        return {"connected": False, "error": f"Cannot reach host: {e}"}
    except Exception as e:
        return {"connected": False, "error": str(e)}


def _build_conn_kwargs(
    ssh_host: str,
    ssh_port: int,
    ssh_user: str,
    ssh_password: Optional[str],
    ssh_key: Optional[str],
) -> dict:
    """Build asyncssh connection kwargs."""
    kwargs = {
        "host": ssh_host,
        "port": ssh_port,
        "username": ssh_user,
        "known_hosts": None,  # Skip host key verification
    }
    if ssh_key:
        kwargs["client_keys"] = [asyncssh.import_private_key(ssh_key)]
    elif ssh_password:
        kwargs["password"] = ssh_password
    return kwargs


async def deploy_agent(
    ssh_host: str,
    ssh_port: int,
    ssh_user: str,
    ssh_password: Optional[str],
    ssh_key: Optional[str],
    env_content: str,
    on_log: Callable[[str, str], Awaitable[None]],
) -> bool:
    """
    Deploy the Toup Agent to a remote machine via SSH.

    Args:
        ssh_host: Target machine hostname/IP
        ssh_port: SSH port
        ssh_user: SSH username
        ssh_password: SSH password (or None for key auth)
        ssh_key: PEM private key content (or None for password auth)
        env_content: Generated .env file content
        on_log: Async callback (line, level) for real-time log streaming.
                Levels: "info", "success", "error", "cmd", "step"

    Returns:
        True if deployment succeeded, False otherwise
    """
    await on_log("Connecting to {}:{}...".format(ssh_host, ssh_port), "step")

    try:
        conn_kwargs = _build_conn_kwargs(ssh_host, ssh_port, ssh_user, ssh_password, ssh_key)
        async with asyncssh.connect(**conn_kwargs) as conn:
            await on_log(f"Connected as {ssh_user}", "success")

            # Step 1: Check prerequisites
            await on_log("Checking prerequisites...", "step")
            ok, py_ver = await _run_cmd(conn, "python3 --version", on_log)
            if not ok:
                await on_log("Installing Python...", "step")
                ok, _ = await _run_cmd(
                    conn,
                    "apt update -qq && apt install -y python3 python3-venv python3-pip git",
                    on_log, sudo=True,
                )
                if not ok:
                    await on_log("Failed to install prerequisites", "error")
                    return False

            ok, git_ver = await _run_cmd(conn, "git --version", on_log)
            if not ok:
                ok, _ = await _run_cmd(conn, "apt install -y git", on_log, sudo=True)
                if not ok:
                    await on_log("Failed to install git", "error")
                    return False

            # Step 2: Clone or update agent code
            await on_log("Setting up agent code...", "step")
            ok, _ = await _run_cmd(conn, f"test -d {AGENT_DIR}/.git && echo exists", on_log)
            if "exists" in (_ or ""):
                await on_log("Updating existing agent code...", "info")
                ok, _ = await _run_cmd(conn, f"cd {AGENT_DIR} && git pull --ff-only", on_log, sudo=True)
            else:
                await on_log("Cloning agent repository...", "info")
                ok, _ = await _run_cmd(
                    conn,
                    f"mkdir -p {AGENT_DIR} && git clone {AGENT_REPO} {AGENT_DIR}",
                    on_log, sudo=True,
                )
            if not ok:
                await on_log("Failed to set up agent code", "error")
                return False

            # Step 3: Create virtualenv and install deps
            await on_log("Installing dependencies...", "step")
            ok, _ = await _run_cmd(
                conn,
                f"test -d {AGENT_DIR}/venv/bin/python && echo exists",
                on_log,
            )
            if "exists" not in (_ or ""):
                ok, _ = await _run_cmd(
                    conn,
                    f"python3 -m venv {AGENT_DIR}/venv",
                    on_log, sudo=True,
                )
                if not ok:
                    await on_log("Failed to create virtualenv", "error")
                    return False

            ok, _ = await _run_cmd(
                conn,
                f"{AGENT_DIR}/venv/bin/pip install -q -r {AGENT_DIR}/brain/backend/requirements.txt",
                on_log, sudo=True,
            )
            if not ok:
                await on_log("Failed to install Python dependencies", "error")
                return False

            # Step 4: Write .env
            await on_log("Writing configuration...", "step")
            # Escape single quotes in env content for echo
            safe_env = env_content.replace("'", "'\\''")
            ok, _ = await _run_cmd(
                conn,
                f"cat > {AGENT_DIR}/brain/backend/.env << 'TOUP_ENV_EOF'\n{env_content}\nTOUP_ENV_EOF",
                on_log, sudo=True,
            )
            if not ok:
                await on_log("Failed to write .env", "error")
                return False
            await on_log("Configuration written", "success")

            # Step 5: Create systemd service
            await on_log("Configuring systemd service...", "step")
            service_content = f"""[Unit]
Description=Toup Agent Service
After=network.target

[Service]
Type=simple
WorkingDirectory={AGENT_DIR}/brain/backend
EnvironmentFile={AGENT_DIR}/brain/backend/.env
Environment=AGENT_DIR={AGENT_DIR}
ExecStart={AGENT_DIR}/venv/bin/uvicorn agent_main:app --host 0.0.0.0 --port {AGENT_PORT}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target"""

            ok, _ = await _run_cmd(
                conn,
                f"cat > /etc/systemd/system/{SYSTEMD_UNIT}.service << 'TOUP_SVC_EOF'\n{service_content}\nTOUP_SVC_EOF",
                on_log, sudo=True,
            )
            if not ok:
                await on_log("Failed to create systemd service", "error")
                return False

            # Step 6: Enable and start service
            await on_log("Starting agent service...", "step")
            ok, _ = await _run_cmd(
                conn,
                f"systemctl daemon-reload && systemctl enable {SYSTEMD_UNIT} && systemctl restart {SYSTEMD_UNIT}",
                on_log, sudo=True,
            )
            if not ok:
                await on_log("Failed to start agent service", "error")
                return False

            # Step 7: Verify agent health
            await on_log("Verifying agent health...", "step")
            for attempt in range(6):  # Retry up to 30s
                await asyncio.sleep(5)
                ok, output = await _run_cmd(
                    conn,
                    f"curl -sf http://localhost:{AGENT_PORT}/agent/health 2>/dev/null || echo FAIL",
                    on_log,
                )
                if ok and "FAIL" not in (output or "FAIL"):
                    await on_log("Agent is running!", "success")
                    return True
                await on_log(f"Waiting for agent to start... (attempt {attempt + 1}/6)", "info")

            await on_log("Agent did not respond to health check after 30s", "error")
            return False

    except asyncssh.PermissionDenied:
        await on_log("SSH authentication failed. Check credentials.", "error")
        return False
    except asyncssh.ConnectionLost:
        await on_log("SSH connection lost", "error")
        return False
    except OSError as e:
        await on_log(f"Cannot reach host: {e}", "error")
        return False
    except Exception as e:
        logger.exception("Deploy failed: %s", e)
        await on_log(f"Deployment error: {e}", "error")
        return False


def generate_setup_script(env_content: str) -> str:
    """Generate a self-contained bash setup script for local machines (Mac/Linux).

    Installs the agent as a background service:
    - macOS: launchd (~/Library/LaunchAgents)
    - Linux: systemd user service (~/.config/systemd/user)

    The agent auto-starts on login and restarts on crash.
    A CLI helper (~/toup-agent/toup) provides start/stop/update/logs/status.
    """
    return f'''#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Toup Agent — Local Setup Script
#  Generated by Toup Platform
# ═══════════════════════════════════════════════════════════════

set -e

AGENT_DIR="$HOME/toup-agent"
AGENT_PORT={AGENT_PORT}
REPO="{AGENT_REPO}"

echo ""
echo "  Toup Agent Setup"
echo "  ════════════════════════"
echo ""

# ── Check prerequisites ──────────────────────────────────
echo "[1/7] Checking prerequisites..."

if ! command -v python3 &>/dev/null; then
  echo "  Python 3 is required but not installed."
  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  Install with: brew install python3"
  else
    echo "  Install with: sudo apt install python3 python3-venv"
  fi
  exit 1
fi
PYVER=$(python3 --version 2>&1)
echo "  $PYVER"

if ! command -v git &>/dev/null; then
  echo "  Git is required but not installed."
  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  Install with: brew install git"
  else
    echo "  Install with: sudo apt install git"
  fi
  exit 1
fi
echo "  $(git --version)"

# ── Clone or update ──────────────────────────────────────
echo ""
echo "[2/7] Setting up agent code..."

if [ -d "$AGENT_DIR/.git" ]; then
  echo "  Updating existing installation..."
  cd "$AGENT_DIR" && git pull --ff-only
else
  echo "  Cloning agent repository..."
  git clone "$REPO" "$AGENT_DIR"
fi

# ── Virtualenv ───────────────────────────────────────────
echo ""
echo "[3/7] Setting up Python environment..."

if [ ! -d "$AGENT_DIR/venv" ]; then
  python3 -m venv "$AGENT_DIR/venv"
  echo "  Virtualenv created"
fi

echo "  Installing dependencies (this may take a minute)..."
"$AGENT_DIR/venv/bin/pip" install -q -r "$AGENT_DIR/brain/backend/requirements.txt"
echo "  Dependencies installed"

# ── Write .env ───────────────────────────────────────────
echo ""
echo "[4/7] Writing configuration..."

cat > "$AGENT_DIR/brain/backend/.env" << 'TOUP_ENV_EOF'
{env_content}
TOUP_ENV_EOF
echo "  Configuration saved to $AGENT_DIR/brain/backend/.env"

# ── Stop existing agent ──────────────────────────────────
echo ""
echo "[5/7] Stopping existing agent..."

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

# Stop existing service if running
if [[ "$OSTYPE" == "darwin"* ]]; then
  launchctl bootout gui/$(id -u)/com.toup.agent 2>/dev/null || true
else
  systemctl --user stop toup-agent 2>/dev/null || true
fi

# ── Install background service ────────────────────────────
echo ""
echo "[6/7] Installing background service..."

UVICORN="$AGENT_DIR/venv/bin/uvicorn"
BACKEND_DIR="$AGENT_DIR/brain/backend"
LOG_FILE="$AGENT_DIR/agent.log"
ERR_FILE="$AGENT_DIR/agent-error.log"

if [[ "$OSTYPE" == "darwin"* ]]; then
  # ── macOS: launchd plist ─────────────────────────────
  PLIST_DIR="$HOME/Library/LaunchAgents"
  PLIST_FILE="$PLIST_DIR/com.toup.agent.plist"
  mkdir -p "$PLIST_DIR"

  cat > "$PLIST_FILE" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.toup.agent</string>
  <key>ProgramArguments</key>
  <array>
    <string>$UVICORN</string>
    <string>agent_main:app</string>
    <string>--host</string>
    <string>0.0.0.0</string>
    <string>--port</string>
    <string>$AGENT_PORT</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$BACKEND_DIR</string>
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
  <string>$LOG_FILE</string>
  <key>StandardErrorPath</key>
  <string>$ERR_FILE</string>
  <key>ThrottleInterval</key>
  <integer>5</integer>
</dict>
</plist>
PLIST_EOF

  # Load the service
  launchctl bootstrap gui/$(id -u) "$PLIST_FILE" 2>/dev/null || \\
    launchctl load "$PLIST_FILE" 2>/dev/null || true
  echo "  Installed launchd service (auto-starts on login)"
  SERVICE_TYPE="launchd"

else
  # ── Linux: systemd user service ─────────────────────
  SYSTEMD_DIR="$HOME/.config/systemd/user"
  UNIT_FILE="$SYSTEMD_DIR/toup-agent.service"
  mkdir -p "$SYSTEMD_DIR"

  cat > "$UNIT_FILE" << UNIT_EOF
[Unit]
Description=Toup AI Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$BACKEND_DIR
Environment=AGENT_DIR=$AGENT_DIR
ExecStart=$UVICORN agent_main:app --host 0.0.0.0 --port $AGENT_PORT
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
UNIT_EOF

  systemctl --user daemon-reload
  systemctl --user enable toup-agent
  systemctl --user start toup-agent
  # Enable lingering so the service runs even when not logged in
  loginctl enable-linger "$(whoami)" 2>/dev/null || true
  echo "  Installed systemd user service (auto-starts on boot)"
  SERVICE_TYPE="systemd"
fi

# ── Create CLI helper ─────────────────────────────────────
cat > "$AGENT_DIR/toup" << 'CLI_EOF'
#!/bin/bash
# ═══════════════════════════════════════════════════════
#  toup — Toup Agent CLI
#  Usage: ~/toup-agent/toup [command]
# ═══════════════════════════════════════════════════════

AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_PORT=$(grep '^AGENT_PORT=' "$AGENT_DIR/brain/backend/.env" 2>/dev/null | cut -d= -f2)
AGENT_PORT=${{AGENT_PORT:-8001}}

_is_mac() {{ [[ "$OSTYPE" == "darwin"* ]]; }}

_start() {{
  if _is_mac; then
    launchctl bootstrap gui/$(id -u) "$HOME/Library/LaunchAgents/com.toup.agent.plist" 2>/dev/null || \
      launchctl load "$HOME/Library/LaunchAgents/com.toup.agent.plist" 2>/dev/null
  else
    systemctl --user start toup-agent
  fi
  echo "✅ Agent started"
}}

_stop() {{
  if _is_mac; then
    launchctl bootout gui/$(id -u)/com.toup.agent 2>/dev/null || \
      launchctl unload "$HOME/Library/LaunchAgents/com.toup.agent.plist" 2>/dev/null
  else
    systemctl --user stop toup-agent
  fi
  echo "🛑 Agent stopped"
}}

_restart() {{
  _stop 2>/dev/null
  sleep 1
  _start
}}

_status() {{
  if curl -sf "http://localhost:$AGENT_PORT/agent/health" > /dev/null 2>&1; then
    HEALTH=$(curl -sf "http://localhost:$AGENT_PORT/agent/health")
    echo "✅ Agent is running"
    echo "   URL:     http://localhost:$AGENT_PORT"
    echo "   Health:  $HEALTH"
  else
    echo "❌ Agent is not responding"
    if _is_mac; then
      launchctl print gui/$(id -u)/com.toup.agent 2>/dev/null | head -5
    else
      systemctl --user status toup-agent --no-pager 2>/dev/null | head -10
    fi
  fi
}}

_update() {{
  echo "🔄 Updating Toup Agent..."
  cd "$AGENT_DIR" && git pull --ff-only
  echo "📦 Installing dependencies..."
  "$AGENT_DIR/venv/bin/pip" install -q -r "$AGENT_DIR/brain/backend/requirements.txt"
  echo "🔁 Restarting..."
  _restart
  sleep 3
  _status
}}

_logs() {{
  tail -f "$AGENT_DIR/agent.log"
}}

case "${{1:-help}}" in
  start)   _start ;;
  stop)    _stop ;;
  restart) _restart ;;
  status)  _status ;;
  update)  _update ;;
  logs)    _logs ;;
  *)
    echo "Usage: toup <command>"
    echo ""
    echo "Commands:"
    echo "  start    Start the agent"
    echo "  stop     Stop the agent"
    echo "  restart  Restart the agent"
    echo "  status   Check if the agent is running"
    echo "  update   Pull latest code & restart"
    echo "  logs     Tail the agent logs"
    ;;
esac
CLI_EOF
chmod +x "$AGENT_DIR/toup"

# ── Verify ───────────────────────────────────────────────
echo ""
echo "[7/7] Verifying..."

AGENT_API_KEY=$(grep '^AGENT_API_KEY=' "$AGENT_DIR/brain/backend/.env" | cut -d= -f2)
PLATFORM_URL=$(grep '^PLATFORM_API_URL=' "$AGENT_DIR/brain/backend/.env" | cut -d= -f2)

for i in 1 2 3 4 5 6; do
  sleep 3
  if curl -sf "http://localhost:$AGENT_PORT/agent/health" > /dev/null 2>&1; then
    echo ""
    echo "  Agent is running at http://localhost:$AGENT_PORT"

    # Register with platform
    if [ -n "$AGENT_API_KEY" ] && [ -n "$PLATFORM_URL" ]; then
      curl -sf -X POST "$PLATFORM_URL/agent-setup/register" \\
        -H "Content-Type: application/json" \\
        -d "{{\\"agent_api_key\\":\\"$AGENT_API_KEY\\",\\"agent_url\\":\\"http://localhost:$AGENT_PORT\\"}}" > /dev/null 2>&1 \\
        && echo "  Registered with platform" \\
        || echo "  (Could not register with platform — verify manually)"
    fi

    echo ""
    echo "  ════════════════════════════════════════════"
    echo "  ✅ Setup complete!"
    echo ""
    echo "  Your agent is running as a background service."
    echo "  It will auto-start on login and restart on crash."
    echo ""
    echo "  CLI commands:     ~/toup-agent/toup <command>"
    echo "    toup status     Check agent status"
    echo "    toup update     Pull latest code & restart"
    echo "    toup logs       View agent logs"
    echo "    toup stop       Stop the agent"
    echo "    toup start      Start the agent"
    echo "    toup restart    Restart the agent"
    echo ""
    echo "  Web UI:           https://toup.ai"
    echo "  ════════════════════════════════════════════"
    echo ""
    exit 0
  fi
  echo "  Waiting for agent to start... ($i/6)"
done

echo ""
echo "  Agent did not respond after 18 seconds."
echo "  Check logs: tail -50 $AGENT_DIR/agent.log"
echo "  Check errors: tail -50 $AGENT_DIR/agent-error.log"
exit 1
'''


def generate_setup_script_windows(env_content: str) -> str:
    """Generate a PowerShell setup script for Windows."""
    # Escape single quotes in env_content for PowerShell here-string
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
    git pull --ff-only
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
& "$AGENT_DIR\\venv\\Scripts\\pip.exe" install -q -r "$AGENT_DIR\\brain\\backend\\requirements.txt"
Write-Host "  Dependencies installed"

# ── Write .env ───────────────────────────────────────────
Write-Host ""
Write-Host "[4/6] Writing configuration..."

@"
{env_content}
"@ | Set-Content -Path "$AGENT_DIR\\brain\\backend\\.env" -Encoding UTF8
Write-Host "  Configuration saved to $AGENT_DIR\\brain\\backend\\.env"

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
Push-Location "$AGENT_DIR\\brain\\backend"
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
    agent_model: str = "gpt-5.2",
    llm_mode: str = "manual",
    telegram_bot_token: str = "",
    discord_bot_token: str = "",
    slack_bot_token: str = "",
    slack_app_token: str = "",
    whatsapp_phone_number_id: str = "",
    whatsapp_access_token: str = "",
    brave_api_key: str = "",
    elevenlabs_api_key: str = "",
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
        f"DATABASE_URL={settings.database_url}",
        "",
        "# --- Platform ---",
        f"PLATFORM_API_URL={settings.platform_api_url}",
        f"USER_ID={user_id}",
        f"AGENT_API_KEY={agent_api_key}",
        "",
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

    # Embedding (must match platform)
    lines.extend([
        "",
        "# --- Embedding ---",
        f"EMBEDDING_PROVIDER={settings.embedding_provider}",
        f"EMBEDDING_MODEL={settings.embedding_model}",
        f"EMBEDDING_DIMENSION={settings.embedding_dimension}",
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
