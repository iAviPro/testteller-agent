"""
Constants and default values for the TestTeller RAG Agent.
This module centralizes all default values and configuration constants used throughout the application.
"""

# Application Information
APP_NAME = "TestTeller RAG Agent"
# APP_VERSION is now imported from __init__.py to maintain single source of truth
# Import will be handled by __init__.py
FALLBACK_VERSION = "0.1.3"  # Fallback version when _version.py import fails
APP_DESCRIPTION = "A versatile CLI-based RAG (Retrieval Augmented Generation) agent designed to generate software test cases."

# Default Environment Settings
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "json"

# ChromaDB Settings
DEFAULT_CHROMA_HOST = "localhost"
DEFAULT_CHROMA_PORT = 8000
DEFAULT_CHROMA_USE_REMOTE = False
DEFAULT_CHROMA_PERSIST_DIRECTORY = "./chroma_data"
DEFAULT_COLLECTION_NAME = "test_collection"

# LLM Settings
DEFAULT_GEMINI_EMBEDDING_MODEL = "text-embedding-004"
DEFAULT_GEMINI_GENERATION_MODEL = "gemini-2.0-flash"

# Document Processing Settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Code Processing Settings
DEFAULT_CODE_EXTENSIONS = [
    ".py",   # Python
    ".js",   # JavaScript
    ".ts",   # TypeScript
    ".java",  # Java
    ".go",   # Go
    ".rs",   # Rust
    ".cpp",  # C++
    ".hpp",  # C++ Headers
    ".c",    # C
    ".h",    # C Headers
    ".cs",   # C#
    ".rb",   # Ruby
    ".php"   # PHP
]
DEFAULT_TEMP_CLONE_DIR = "./temp_cloned_repos"

# Output Settings
DEFAULT_OUTPUT_FILE = "testteller-testcases.md"

# API Retry Settings
DEFAULT_API_RETRY_ATTEMPTS = 3
DEFAULT_API_RETRY_WAIT_SECONDS = 2

# Docker Settings
DOCKER_HEALTHCHECK_INTERVAL = "30s"
DOCKER_HEALTHCHECK_TIMEOUT = "10s"
DOCKER_HEALTHCHECK_RETRIES = 3
DOCKER_HEALTHCHECK_START_PERIOD = "30s"
DOCKER_DEFAULT_CPU_LIMIT = "2"
DOCKER_DEFAULT_MEMORY_LIMIT = "4G"
DOCKER_DEFAULT_CPU_RESERVATION = "0.5"
DOCKER_DEFAULT_MEMORY_RESERVATION = "1G"

# Environment Variable Names
ENV_GOOGLE_API_KEY = "GOOGLE_API_KEY"
ENV_GITHUB_TOKEN = "GITHUB_TOKEN"
ENV_LOG_LEVEL = "LOG_LEVEL"
ENV_CHROMA_DB_HOST = "CHROMA_DB_HOST"
ENV_CHROMA_DB_PORT = "CHROMA_DB_PORT"
ENV_CHROMA_DB_USE_REMOTE = "CHROMA_DB_USE_REMOTE"
ENV_CHROMA_DB_PERSIST_DIRECTORY = "CHROMA_DB_PERSIST_DIRECTORY"
ENV_DEFAULT_COLLECTION_NAME = "DEFAULT_COLLECTION_NAME"
ENV_GEMINI_EMBEDDING_MODEL = "GEMINI_EMBEDDING_MODEL"
ENV_GEMINI_GENERATION_MODEL = "GEMINI_GENERATION_MODEL"
ENV_CHUNK_SIZE = "CHUNK_SIZE"
ENV_CHUNK_OVERLAP = "CHUNK_OVERLAP"
ENV_CODE_EXTENSIONS = "CODE_EXTENSIONS"
ENV_TEMP_CLONE_DIR_BASE = "TEMP_CLONE_DIR_BASE"
ENV_OUTPUT_FILE_PATH = "OUTPUT_FILE_PATH"
ENV_API_RETRY_ATTEMPTS = "API_RETRY_ATTEMPTS"
ENV_API_RETRY_WAIT_SECONDS = "API_RETRY_WAIT_SECONDS"
