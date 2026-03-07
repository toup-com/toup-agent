# Backend Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies + Chromium deps for headless browser
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    git \
    # Chromium dependencies for Playwright/Patchright
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

# Install CPU-only PyTorch first (prevents ~3GB of NVIDIA CUDA packages)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements first for better caching
# Railway build context is repo root, so paths are relative to repo root
COPY backend/requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Chromium browser for Patchright/Playwright
RUN python -m patchright install chromium || python -m playwright install chromium

# Copy backend application code
COPY backend/ .

# Create non-root user and workspace
RUN useradd -m -u 1000 hexbrain && \
    mkdir -p /app/workspace && \
    chown -R hexbrain:hexbrain /app && \
    # Copy browser binaries to hexbrain's home so non-root can access them
    cp -r /root/.cache/ms-playwright /home/hexbrain/.cache/ms-playwright 2>/dev/null || true && \
    chown -R hexbrain:hexbrain /home/hexbrain/.cache 2>/dev/null || true
USER hexbrain

# Expose port
EXPOSE 8000

# Initialize and start server
CMD python -m app.scripts.init_startup && uvicorn platform_main:app --host 0.0.0.0 --port 8000
