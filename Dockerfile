# Backend Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies + Chromium deps for headless browser
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    git \
    openssh-client \
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

# Copy requirements first for better caching
# Railway build context is repo root, so paths are relative to repo root
COPY backend/requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Chromium browser for Patchright/Playwright
RUN python -m patchright install chromium || python -m playwright install chromium

# Copy backend application code
COPY backend/ .

# Create non-root user and workspace
RUN useradd -m -u 1000 toup && \
    mkdir -p /app/workspace && \
    chown -R toup:toup /app && \
    # Copy browser binaries to toup's home so non-root can access them
    cp -r /root/.cache/ms-playwright /home/toup/.cache/ms-playwright 2>/dev/null || true && \
    chown -R toup:toup /home/toup/.cache 2>/dev/null || true
USER toup

# Expose port
EXPOSE 8000

# Initialize and start server
CMD python -m app.scripts.init_startup && uvicorn platform_main:app --host 0.0.0.0 --port 8000
