#!/bin/bash
# ============================================================
# Setup Docker Host for Managed Agent Containers
# Run this on the VPS: 76.13.116.149
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/toup-com/toup-platform/main/backend/scripts/setup_docker_host.sh | bash
#   OR
#   sshpass -p 'Nariman1212!' ssh root@76.13.116.149 'bash -s' < backend/scripts/setup_docker_host.sh
# ============================================================

set -euo pipefail

echo "═══════════════════════════════════════════════"
echo "  Toup Docker Host Setup"
echo "═══════════════════════════════════════════════"

# ── 1. Install Docker ──────────────────────────────────────
echo ""
echo "[1/6] Installing Docker..."
if command -v docker &>/dev/null; then
    echo "  ✓ Docker already installed: $(docker --version)"
else
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    echo "  ✓ Docker installed: $(docker --version)"
fi

# ── 2. Install PostgreSQL + pgvector ───────────────────────
echo ""
echo "[2/6] Setting up PostgreSQL with pgvector..."
if command -v psql &>/dev/null; then
    echo "  ✓ PostgreSQL already installed"
else
    apt-get update -qq
    apt-get install -y -qq postgresql postgresql-contrib
    systemctl enable postgresql
    systemctl start postgresql
    echo "  ✓ PostgreSQL installed"
fi

# Install pgvector extension
if ! dpkg -l | grep -q postgresql-.*-pgvector 2>/dev/null; then
    apt-get install -y -qq postgresql-*-pgvector 2>/dev/null || {
        # Build from source if package not available
        echo "  Building pgvector from source..."
        apt-get install -y -qq build-essential postgresql-server-dev-all git
        cd /tmp
        git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git 2>/dev/null || true
        cd pgvector
        make && make install
        cd /
        rm -rf /tmp/pgvector
    }
    echo "  ✓ pgvector extension installed"
else
    echo "  ✓ pgvector already installed"
fi

# Configure PostgreSQL to allow local Docker connections
PG_HBA=$(find /etc/postgresql -name pg_hba.conf | head -1)
PG_CONF=$(find /etc/postgresql -name postgresql.conf | head -1)

if [ -n "$PG_HBA" ]; then
    # Allow connections from Docker containers (172.17.0.0/16 = docker bridge)
    if ! grep -q "172.17.0.0/16" "$PG_HBA"; then
        echo "host all all 172.17.0.0/16 trust" >> "$PG_HBA"
        echo "  ✓ Added Docker network to pg_hba.conf"
    fi
fi

if [ -n "$PG_CONF" ]; then
    # Listen on all interfaces (needed for Docker containers)
    if ! grep -q "listen_addresses = '\*'" "$PG_CONF"; then
        sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" "$PG_CONF"
        echo "  ✓ PostgreSQL listening on all interfaces"
    fi
fi

systemctl restart postgresql
echo "  ✓ PostgreSQL configured for Docker containers"

# ── 3. Create data directories ─────────────────────────────
echo ""
echo "[3/6] Creating data directories..."
mkdir -p /data/agents
chmod 755 /data
echo "  ✓ /data/agents/ ready"

# ── 4. Clone and build agent image ─────────────────────────
echo ""
echo "[4/6] Building agent Docker image..."
REPO_DIR="/opt/toup-agent"

if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR"
    git pull --ff-only 2>/dev/null || true
else
    git clone https://github.com/toup-com/toup-agent.git "$REPO_DIR" 2>/dev/null || {
        # Fallback: clone from platform repo and use backend/
        git clone https://github.com/toup-com/toup-platform.git /tmp/toup-platform-build
        mkdir -p "$REPO_DIR"
        cp -r /tmp/toup-platform-build/backend/* "$REPO_DIR/"
        rm -rf /tmp/toup-platform-build
    }
    cd "$REPO_DIR"
fi

# Build the agent image
docker build -f Dockerfile.agent -t toup-agent:latest . 2>&1 | tail -5
echo "  ✓ toup-agent:latest image built"

# ── 5. Firewall — allow port range for containers ──────────
echo ""
echo "[5/6] Configuring firewall..."
if command -v ufw &>/dev/null; then
    ufw allow 9000:9999/tcp 2>/dev/null || true
    echo "  ✓ Ports 9000-9999 opened"
else
    echo "  ⚠ ufw not found — ensure ports 9000-9999 are open in your firewall"
fi

# ── 6. Create update script ────────────────────────────────
echo ""
echo "[6/6] Creating update script..."
cat > /opt/toup-update-agent-image.sh << 'SCRIPT'
#!/bin/bash
# Update the toup-agent Docker image from latest code
set -e
cd /opt/toup-agent
git pull --ff-only
docker build -f Dockerfile.agent -t toup-agent:latest .
echo "Image updated. Running containers will use the new image on next restart."
echo "To update all running containers:"
echo "  docker ps --filter 'ancestor=toup-agent:latest' -q | xargs -r docker restart"
SCRIPT
chmod +x /opt/toup-update-agent-image.sh
echo "  ✓ Update script at /opt/toup-update-agent-image.sh"

# ── Done ───────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "  Setup complete!"
echo "═══════════════════════════════════════════════"
echo ""
echo "  Docker host is ready for managed agent containers."
echo ""
echo "  Set these env vars on Railway (platform):"
echo "    DOCKER_HOST_IP=76.13.116.149"
echo "    DOCKER_HOST_SSH_PASSWORD=<your-ssh-password>"
echo "    DOCKER_HOST_PG_URL=postgresql://postgres:postgres@localhost:5432"
echo "    DOCKER_AGENT_IMAGE=toup-agent:latest"
echo "    MANAGED_HOSTING_ENABLED=true"
echo ""
echo "  To update the agent image later:"
echo "    /opt/toup-update-agent-image.sh"
echo ""
echo "  Container data lives in: /data/agents/<user_id>/"
echo "  Port range: 9000-9999 (~1000 users max per host)"
echo ""
