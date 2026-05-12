# Backend Dockerfile — multi-stage
#
# Stages:
#   base   — system deps + Python base image (shared)
#   deps   — + pip install (shared by test and prod)
#   test   — minimal test image (no Chromium install, no startup init)
#   prod   — full prod image (Chromium for patchright, init_startup in CMD)
#
# `docker build backend/Dockerfile` (no --target) builds `prod` — the last
# stage — unchanged from single-stage behavior. Railway picks prod.
# docker-compose.test.yml uses `target: test`.
#
# Shared base + deps stages mean test and prod cannot drift on Python
# version or installed packages.

# ─── Stage 1: base ─────────────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

# System deps for both test and prod. Chromium runtime libs are cheap
# (~20MB total) and included in both so a test image could opportunistically
# run browser code without re-building; the expensive part is the Chromium
# binary itself, which only the prod stage installs.
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    git \
    openssh-client \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libwayland-client0 \
    && rm -rf /var/lib/apt/lists/*

# ─── Stage 2a: test_deps ───────────────────────────────────────────
# Slim pip install for the test image — omits sentence-transformers,
# torch, CUDA, anthropic, telegram, patchright, fastmcp (collectively
# ~3GB). Caching a 3GB pip layer still costs 2+ minutes to restore from
# GHA cache and unpack; removing the layer weight is the only real lever.
# List is kept in sync with prod deps in requirements.test.docker.txt.
FROM base AS test_deps

COPY backend/requirements.test.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

RUN useradd -m -u 1000 toup && \
    mkdir -p /app/workspace && \
    chown -R toup:toup /app

# ─── Stage 2b: prod_deps ───────────────────────────────────────────
# Full pip install for prod. Adds Node.js + @anthropic-ai/claude-code
# globally for the experimental Toup Code feature (/code/*). Pinned
# Node 20 LTS via NodeSource; install adds ~110 MB to the prod image.
# If the experiment is killed, drop this RUN block to reclaim space.
FROM base AS prod_deps

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    npm install -g --no-audit --no-fund @anthropic-ai/claude-code

COPY backend/requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

RUN useradd -m -u 1000 toup && \
    mkdir -p /app/workspace && \
    chown -R toup:toup /app

# ─── Stage 3: test ─────────────────────────────────────────────────
# Used by docker-compose.test.yml. Skips:
#   - Chromium browser install (~200MB saved)
#   - Heavy ML deps (~3GB saved via requirements.test.docker.txt)
# Defaults to monolith run mode so E2E can exercise register + identity seeding.
FROM test_deps AS test

USER toup
EXPOSE 8000
ENV RUN_MODE=monolith
# init_startup creates the schema + a default user; cheap, needed for
# E2E tests that hit endpoints which query tables like vps_plans.
CMD python -m app.scripts.init_startup && uvicorn platform_main:app --host 0.0.0.0 --port 8000

# ─── Stage 4: prod ─────────────────────────────────────────────────
# Default build target (when no --target is passed).
FROM prod_deps AS prod

RUN python -m patchright install chromium || python -m playwright install chromium
RUN cp -r /root/.cache/ms-playwright /home/toup/.cache/ms-playwright 2>/dev/null || true && \
    chown -R toup:toup /home/toup/.cache 2>/dev/null || true

USER toup
EXPOSE 8000
CMD python -m app.scripts.init_startup && uvicorn platform_main:app --host 0.0.0.0 --port 8000
