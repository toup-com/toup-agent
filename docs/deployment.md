# HexBrain — Production Deployment Guide

## Overview

HexBrain runs as three Docker containers:
- **db** — PostgreSQL 16 + pgvector
- **backend** — Python 3.12 FastAPI (agent, API, Telegram bot)
- **frontend** — React app served by nginx

## Deployment Options

### Option 1: Single Server (Docker Compose)

Best for: personal use, small teams, evaluation.

```bash
# 1. Clone and configure
git clone https://github.com/toup-com/hex-brain.git
cd hex-brain
cp .env.example .env
# Edit .env with your API keys

# 2. Build and start
docker compose up -d --build

# 3. Verify
docker compose ps
curl http://localhost:8000/health
```

### Option 2: VPS / Cloud VM

Best for: always-on Telegram bot, small production deployment.

**Recommended specs:**
- 2 vCPU, 4 GB RAM, 20 GB SSD
- Ubuntu 22.04+ or Debian 12+
- Docker Engine installed

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Clone and deploy
git clone https://github.com/toup-com/hex-brain.git
cd hex-brain
cp .env.example .env
# Edit .env

# Start with auto-restart
docker compose up -d
```

### Option 3: Cloud Container Services

For AWS ECS, Google Cloud Run, Azure Container Apps, etc.:

1. Build images: `docker compose build`
2. Push to your container registry
3. Deploy each service with the appropriate environment variables
4. Use a managed PostgreSQL instance with pgvector extension

## Production Configuration

### Environment Variables

```env
# Required
OPENAI_API_KEY=sk-your-production-key
TELEGRAM_BOT_TOKEN=your-bot-token

# Security — CHANGE THESE
JWT_SECRET=generate-a-random-64-char-string
DATABASE_URL=postgresql+asyncpg://hexbrain:STRONG_PASSWORD@db:5432/hexbrain

# In docker-compose.yml, also change:
# POSTGRES_PASSWORD=STRONG_PASSWORD

# Performance
DEBUG=false
AGENT_MODEL=gpt-5.2
ENABLE_SCHEDULER=true
```

### Generate a Strong JWT Secret

```bash
openssl rand -hex 32
```

### Database Passwords

Change the default PostgreSQL password in both:
1. `docker-compose.yml` → `POSTGRES_PASSWORD`
2. `.env` → `DATABASE_URL` connection string

### HTTPS / TLS

Use a reverse proxy (Caddy, Traefik, or nginx) in front of the frontend:

**Caddy (recommended — auto-HTTPS):**
```
brain.yourdomain.com {
    reverse_proxy localhost:80
}
```

**nginx with Let's Encrypt:**
```nginx
server {
    listen 443 ssl;
    server_name brain.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/brain.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/brain.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:80;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

### Custom Domain

1. Point your domain's DNS to your server IP
2. Set up HTTPS (see above)
3. Update `CORS_ORIGINS` in `.env` to include your domain:
   ```
   CORS_ORIGINS=["https://brain.yourdomain.com"]
   ```

## Backup & Restore

### Database Backup

```bash
# Backup
docker compose exec db pg_dump -U hexbrain hexbrain > backup_$(date +%Y%m%d).sql

# Restore
cat backup_20260207.sql | docker compose exec -T db psql -U hexbrain hexbrain
```

### Automated Backups (cron)

```bash
# Add to crontab
0 3 * * * cd /path/to/hex-brain && docker compose exec -T db pg_dump -U hexbrain hexbrain | gzip > /backups/hexbrain_$(date +\%Y\%m\%d).sql.gz
```

### Volume Backup

```bash
# Backup workspace and skills volumes
docker run --rm -v hex-brain_agent_workspace:/data -v $(pwd):/backup alpine tar czf /backup/workspace.tar.gz -C /data .
docker run --rm -v hex-brain_agent_skills:/data -v $(pwd):/backup alpine tar czf /backup/skills.tar.gz -C /data .
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "database": "connected",
  "embedding_model": "text-embedding-3-small",
  "chat_model": "gpt-4o",
  "platform": "HexBrain Agent Platform v5 — Toup Edition",
  "telegram_bot": "enabled",
  "agent_model": "gpt-5.2"
}
```

### Logs

```bash
# All services
docker compose logs -f

# Backend only
docker compose logs -f backend

# Last 100 lines
docker compose logs backend --tail 100
```

### Admin Dashboard

Access the admin API (requires JWT auth):

```bash
# System status
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/admin/status

# Recent errors
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/admin/errors

# Cron jobs
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/admin/cron
```

## Updating

```bash
cd hex-brain
git pull

# Rebuild and restart (zero-downtime for frontend)
docker compose build --no-cache
docker compose up -d
```

## Scaling Notes

- **Database:** For high memory counts (>100K), consider dedicated PostgreSQL with pgvector indices tuned
- **Backend:** Single-process by default. For multi-worker, set `ENABLE_SCHEDULER=false` on extra workers (only one scheduler instance)
- **Frontend:** Static files served by nginx — scales horizontally without issues
- **Skills:** Loaded in-process. Each backend instance loads its own copy from the shared volume

## Security Checklist

- [ ] Changed default database password
- [ ] Changed JWT secret
- [ ] Set `DEBUG=false`
- [ ] HTTPS enabled
- [ ] `TELEGRAM_ALLOWED_USER_IDS` restricts bot access
- [ ] API keys created for programmatic access (no shared tokens)
- [ ] Database backups configured
- [ ] Firewall: only ports 80/443 exposed publicly (not 5432 or 8000 directly)
