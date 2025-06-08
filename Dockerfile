# Build stage
FROM ubuntu:22.04 AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install Python and build dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    gcc \
    libpoppler-cpp-dev \
    pkg-config \
    sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && sqlite3 --version

# Create and activate virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first for better caching
COPY requirements.txt .
COPY setup.py .
COPY README.md .
COPY MANIFEST.in .
COPY pyproject.toml .

# Install dependencies and package in development mode
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install -e .

# Final stage
FROM ubuntu:22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    libpoppler-cpp-dev \
    tesseract-ocr \
    git \
    wget \
    curl \
    sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && sqlite3 --version

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create necessary directories with proper permissions
RUN mkdir -p /app/chroma_data \
    /app/temp_cloned_repos \
    && chmod -R 777 /app/chroma_data \
    && chmod -R 777 /app/temp_cloned_repos

# Create entrypoint script with improved error handling
RUN echo '#!/bin/bash\n\
    set -e\n\
    \n\
    # Function to check ChromaDB health\n\
    check_chromadb_health() {\n\
    for i in {1..30}; do\n\
    if curl -s -f http://chromadb:8000/api/v1/heartbeat > /dev/null; then\n\
    return 0\n\
    fi\n\
    echo "Waiting for ChromaDB to be ready... ($i/30)"\n\
    sleep 2\n\
    done\n\
    echo "ChromaDB is not ready after 60 seconds"\n\
    return 1\n\
    }\n\
    \n\
    # Function to validate environment\n\
    validate_environment() {\n\
    if [ -z "$GOOGLE_API_KEY" ]; then\n\
    echo "Error: GOOGLE_API_KEY is required"\n\
    return 1\n\
    fi\n\
    }\n\
    \n\
    if [ "$1" = "serve" ]; then\n\
    echo "Container is running. Use one of the following commands:"\n\
    echo "  docker-compose exec app python -m testteller.main --help"\n\
    echo "  docker-compose exec app python -m testteller.main <command> [options]"\n\
    # Keep container running\n\
    tail -f /dev/null\n\
    else\n\
    # Validate environment\n\
    validate_environment\n\
    # Check ChromaDB health before running commands\n\
    check_chromadb_health\n\
    # Unset GITHUB_TOKEN if empty to avoid validation errors\n\
    if [ -z "$GITHUB_TOKEN" ]; then\n\
    unset GITHUB_TOKEN\n\
    fi\n\
    # Run the command\n\
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
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://chromadb:8000/api/v1/heartbeat || exit 1

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]