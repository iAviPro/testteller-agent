# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

# Set the working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpoppler-cpp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt .

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp-dev \
    tesseract-ocr \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create necessary directories with proper permissions
RUN mkdir -p /app/chroma_data \
    /app/temp_cloned_repos \
    && chmod -R 777 /app/chroma_data \
    && chmod -R 777 /app/temp_cloned_repos

# Create entrypoint script
RUN echo '#!/bin/bash\n\
    if [ "$1" = "serve" ]; then\n\
    echo "Container is running. Use docker exec to run testteller commands."\n\
    # Keep container running\n\
    tail -f /dev/null\n\
    else\n\
    python -m testteller.main "$@"\n\
    fi' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Copy the application code
COPY . .

# Create a non-root user and switch to it
RUN useradd -m -u 1000 testteller && \
    chown -R testteller:testteller /app
USER testteller

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/heartbeat || exit 1

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]