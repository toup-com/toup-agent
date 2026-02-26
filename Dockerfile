# Backend Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user and workspace
RUN useradd -m -u 1000 hexbrain && \
    mkdir -p /app/workspace && \
    chown -R hexbrain:hexbrain /app
USER hexbrain

# Expose port
EXPOSE 8000

# Initialize and start server (migrations already applied)
CMD python -m app.scripts.init_startup && uvicorn app.main:app --host 0.0.0.0 --port 8000
