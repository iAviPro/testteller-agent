# TestTeller Agent - Optimized Docker Build
# Multi-stage build optimized for minimal size
FROM python:3.11-slim AS builder

# Build-time environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install only essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpoppler-cpp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml README.md MANIFEST.in ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy source and install package
COPY testteller/ testteller/
RUN pip install --no-cache-dir --user -e .

# Final stage - minimal runtime image
FROM python:3.11-slim

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/home/testteller/.local/bin:$PATH" \
    LOG_LEVEL=ERROR

WORKDIR /app

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp0v5 \
    tesseract-ocr \
    sqlite3 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get autoclean

# Create non-root user
RUN useradd -m -u 1000 testteller

# Copy Python packages from builder
COPY --from=builder --chown=testteller:testteller /root/.local /home/testteller/.local

# Copy application files
COPY --from=builder --chown=testteller:testteller /app/testteller /app/testteller
COPY --from=builder --chown=testteller:testteller /app/pyproject.toml /app/

# Create necessary directories
RUN mkdir -p /app/{chroma_data,temp_cloned_repos,testteller_generated_tests,testteller_automated_tests} \
    && chown -R testteller:testteller /app

# Create optimized entrypoint script inline
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
check_chromadb_health() {\n\
    for i in {1..15}; do\n\
        if curl -s -f http://chromadb:8000/api/v1/heartbeat > /dev/null 2>&1; then\n\
            return 0\n\
        fi\n\
        echo "Waiting for ChromaDB... ($i/15)"\n\
        sleep 2\n\
    done\n\
    echo "ChromaDB unavailable after 30s"\n\
    return 1\n\
}\n\
\n\
validate_environment() {\n\
    local provider=${LLM_PROVIDER:-gemini}\n\
    case "$provider" in\n\
        "gemini") [ -n "$GOOGLE_API_KEY" ] || { echo "Error: GOOGLE_API_KEY required"; return 1; };;\n\
        "openai") [ -n "$OPENAI_API_KEY" ] || { echo "Error: OPENAI_API_KEY required"; return 1; };;\n\
        "claude") \n\
            [ -n "$CLAUDE_API_KEY" ] || { echo "Error: CLAUDE_API_KEY required"; return 1; }\n\
            [ -n "$OPENAI_API_KEY" ] || { echo "Error: OPENAI_API_KEY also required"; return 1; };;\n\
        "llama") echo "Using Ollama at ${OLLAMA_BASE_URL:-http://host.docker.internal:11434}";;\n\
        *) echo "Error: Unsupported provider: $provider"; return 1;;\n\
    esac\n\
}\n\
\n\
if [ "$1" = "serve" ]; then\n\
    echo "🤖 TestTeller Agent - Optimized Container"\n\
    echo "Provider: ${LLM_PROVIDER:-gemini} | ChromaDB: ${CHROMA_DB_HOST:-chromadb}"\n\
    tail -f /dev/null\n\
else\n\
    validate_environment\n\
    check_chromadb_health\n\
    [ -z "$GITHUB_TOKEN" ] && unset GITHUB_TOKEN\n\
    python -m testteller.main "$@"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

USER testteller

# Minimal healthcheck
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=2 \
    CMD curl -f http://chromadb:8000/api/v1/heartbeat || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]