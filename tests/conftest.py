"""
Pytest configuration and fixtures for TestTeller RAG Agent tests.
"""
import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Generator, Dict, Any, Optional

import asyncio
from testteller.generator_agent.agent.testteller_agent import TestTellerAgent
from testteller.core.llm.llm_manager import LLMManager
from testteller.core.vector_store.chromadb_manager import ChromaDBManager
from testteller.config import settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_collection_name() -> str:
    """Return a unique test collection name."""
    return "test_collection_pytest"


@pytest.fixture
def mock_env_vars() -> Dict[str, str]:
    """Return mock environment variables for testing."""
    return {
        "LLM_PROVIDER": "gemini",
        "GOOGLE_API_KEY": "test_google_api_key",
        "OPENAI_API_KEY": "test_openai_api_key",
        "CLAUDE_API_KEY": "test_claude_api_key",
        "GITHUB_TOKEN": "test_github_token",
        "CHROMA_DB_HOST": "localhost",
        "CHROMA_DB_PORT": "8000",
        "CHROMA_DB_USE_REMOTE": "false",
        "CHROMA_DB_PERSIST_DIRECTORY": "./test_chroma_data",
        "DEFAULT_COLLECTION_NAME": "test_collection",
        "LOG_LEVEL": "DEBUG",
        "LOG_FORMAT": "text"
    }


@pytest.fixture
def mock_llm_response() -> str:
    """Return a mock LLM response for testing."""
    return """
# Test Cases for User Authentication API

## End-to-End Tests

### Test Case 1: Successful User Registration
**Description:** Test complete user registration flow
**Prerequisites:** Clean database state
**Steps:**
1. Send POST request to /api/auth/register with valid user data
2. Verify response contains user ID and success message
3. Verify user is created in database
4. Verify confirmation email is sent
**Expected Result:** User successfully registered and confirmation email sent

### Test Case 2: User Login with Valid Credentials
**Description:** Test user login with correct credentials
**Prerequisites:** User exists in database
**Steps:**
1. Send POST request to /api/auth/login with valid credentials
2. Verify response contains authentication token
3. Verify token is valid and not expired
**Expected Result:** User successfully logged in with valid token

## Integration Tests

### Test Case 3: Password Reset Flow
**Description:** Test complete password reset functionality
**Prerequisites:** User exists in database
**Steps:**
1. Send POST request to /api/auth/forgot-password with user email
2. Verify reset email is sent
3. Use reset token to update password
4. Verify old password no longer works
5. Verify new password works for login
**Expected Result:** Password successfully reset and user can login with new password

## Technical Tests

### Test Case 4: Rate Limiting on Login Endpoint
**Description:** Test rate limiting prevents brute force attacks
**Prerequisites:** Clean rate limit state
**Steps:**
1. Send multiple login requests rapidly
2. Verify rate limiting kicks in after threshold
3. Verify proper error response is returned
4. Verify rate limit resets after time window
**Expected Result:** Rate limiting prevents excessive login attempts
"""


@pytest.fixture
def mock_embedding() -> list[float]:
    """Return a mock embedding vector for testing."""
    return [0.1] * 384  # Mock 384-dimensional embedding


@pytest.fixture
def sample_document_content() -> str:
    """Return sample document content for testing."""
    return """
# User Authentication System Requirements

## Overview
This document outlines the requirements for the user authentication system.

## Functional Requirements
1. Users must be able to register with email and password
2. Users must be able to login with email and password
3. Users must be able to reset their password
4. System must support OAuth login (Google, GitHub)

## API Endpoints
- POST /api/auth/register - Register new user
- POST /api/auth/login - Login user
- POST /api/auth/logout - Logout user
- POST /api/auth/forgot-password - Send password reset email
- POST /api/auth/reset-password - Reset password with token

## Security Requirements
- Passwords must be hashed using bcrypt
- JWT tokens must expire after 24 hours
- Rate limiting must be implemented for login attempts
- HTTPS must be used for all authentication endpoints
"""


@pytest.fixture
def sample_code_content() -> str:
    """Return sample code content for testing."""
    return """
from flask import Flask, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime

app = Flask(__name__)

class UserAuth:
    def __init__(self):
        self.users = {}
    
    def register_user(self, email, password):
        if email in self.users:
            return {"error": "User already exists"}, 400
        
        hashed_password = generate_password_hash(password)
        self.users[email] = {
            "password": hashed_password,
            "created_at": datetime.datetime.now()
        }
        return {"message": "User registered successfully"}, 201
    
    def login_user(self, email, password):
        if email not in self.users:
            return {"error": "User not found"}, 404
        
        if not check_password_hash(self.users[email]["password"], password):
            return {"error": "Invalid password"}, 401
        
        token = jwt.encode({
            "email": email,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, "secret_key", algorithm="HS256")
        
        return {"token": token}, 200

@app.route('/api/auth/register', methods=['POST'])
def register():
    auth = UserAuth()
    data = request.get_json()
    return auth.register_user(data['email'], data['password'])

@app.route('/api/auth/login', methods=['POST'])
def login():
    auth = UserAuth()
    data = request.get_json()
    return auth.login_user(data['email'], data['password'])
"""


@pytest.fixture
def mock_chromadb_manager(test_collection_name: str) -> Mock:
    """Create a mock ChromaDB manager for testing."""
    mock_manager = Mock(spec=ChromaDBManager)
    mock_manager.collection_name = test_collection_name
    mock_manager.add_documents.return_value = None
    mock_manager.query_similar.return_value = {
        "documents": [["Sample test case 1", "Sample test case 2"]],
        "metadatas": [[{"source": "test1.py"}, {"source": "test2.py"}]],
        "distances": [[0.1, 0.2]]
    }
    mock_manager.get_collection_count.return_value = 2
    mock_manager.clear_collection.return_value = None
    return mock_manager


@pytest.fixture
def mock_llm_manager(mock_llm_response: str, mock_embedding: list[float]) -> Mock:
    """Create a mock LLM manager for testing."""
    mock_manager = Mock(spec=LLMManager)
    mock_manager.provider = "gemini"
    mock_manager.get_current_provider.return_value = "gemini"
    mock_manager.generate_text_async.return_value = mock_llm_response
    mock_manager.generate_text.return_value = mock_llm_response
    mock_manager.get_embedding_async.return_value = mock_embedding
    mock_manager.get_embedding_sync.return_value = mock_embedding
    mock_manager.get_embeddings_async.return_value = [
        mock_embedding, mock_embedding]
    mock_manager.get_embeddings_sync.return_value = [
        mock_embedding, mock_embedding]
    return mock_manager


@pytest.fixture
def mock_testteller_agent(
    mock_llm_manager: Mock,
    mock_chromadb_manager: Mock,
    test_collection_name: str
) -> TestTellerAgent:
    """Create a TestTellerAgent with mocked dependencies."""
    with patch('testteller.agent.testteller_agent.LLMManager') as mock_llm_class:
        with patch('testteller.agent.testteller_agent.ChromaDBManager') as mock_chroma_class:
            mock_llm_class.return_value = mock_llm_manager
            mock_chroma_class.return_value = mock_chromadb_manager
            agent = TestTellerAgent(collection_name=test_collection_name)
            return agent


@pytest.fixture
def create_test_files(temp_dir: Path) -> Dict[str, Path]:
    """Create test files for document and code ingestion tests."""
    files = {}

    # Create test document
    doc_file = temp_dir / "test_document.md"
    doc_file.write_text("""
# Test Document

This is a test document for ingestion testing.

## Features
- Document processing
- Content extraction
- Vector storage
""")
    files["document"] = doc_file

    # Create test code file
    code_file = temp_dir / "test_code.py"
    code_file.write_text("""
def test_function():
    '''Test function for code ingestion.'''
    return "Hello, World!"

class TestClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
""")
    files["code"] = code_file

    # Create test directory with multiple files
    test_dir = temp_dir / "test_docs"
    test_dir.mkdir()

    for i in range(3):
        (test_dir / f"doc_{i}.txt").write_text(f"Test document {i} content")

    files["directory"] = test_dir

    return files


@pytest.fixture(params=["gemini", "openai", "claude", "llama"])
def llm_provider(request) -> str:
    """Parametrized fixture for testing with different LLM providers."""
    return request.param


@pytest.fixture
def provider_specific_env_vars(llm_provider: str) -> Dict[str, str]:
    """Return provider-specific environment variables."""
    base_vars = {
        "LLM_PROVIDER": llm_provider,
        "CHROMA_DB_USE_REMOTE": "false",
        "CHROMA_DB_PERSIST_DIRECTORY": "./test_chroma_data",
        "LOG_LEVEL": "DEBUG",
        "LOG_FORMAT": "text",
        "DEFAULT_COLLECTION_NAME": "test_collection"
    }

    # Add provider-specific variables
    if llm_provider == "gemini":
        base_vars.update({
            "GOOGLE_API_KEY": "test_google_api_key_valid",
            "GEMINI_EMBEDDING_MODEL": "embedding-001",
            "GEMINI_GENERATION_MODEL": "gemini-2.0-flash"
        })
    elif llm_provider == "openai":
        base_vars.update({
            "OPENAI_API_KEY": "test_openai_api_key_valid",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
            "OPENAI_GENERATION_MODEL": "gpt-3.5-turbo"
        })
    elif llm_provider == "claude":
        base_vars.update({
            "CLAUDE_API_KEY": "test_claude_api_key_valid",
            "CLAUDE_EMBEDDING_PROVIDER": "openai",  # Use OpenAI for embeddings in tests
            "OPENAI_API_KEY": "test_openai_api_key_valid",  # Required for embeddings
            "CLAUDE_GENERATION_MODEL": "claude-3-haiku-20240307"
        })
    elif llm_provider == "llama":
        base_vars.update({
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "LLAMA_EMBEDDING_MODEL": "llama3.2:1b",
            "LLAMA_GENERATION_MODEL": "llama3.2:1b"
        })

    return base_vars


@pytest.fixture
def mock_env_for_provider(provider_specific_env_vars: Dict[str, str]):
    """Mock environment variables for a specific provider."""
    # Create test data directory with proper permissions
    test_data_dir = Path("./test_chroma_data")
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)
    test_data_dir.mkdir(exist_ok=True)
    os.chmod(test_data_dir, 0o777)  # Ensure directory is writable

    # Create a .env file for testing if it doesn't exist
    env_path = Path(".env")
    if env_path.exists():
        env_path.unlink()
    env_path.write_text("\n".join([
        f"{key}=\"{value}\"" for key, value in provider_specific_env_vars.items()
    ]))

    # Clear existing environment variables
    for key in list(os.environ.keys()):
        if key.startswith(("GOOGLE_", "OPENAI_", "CLAUDE_", "OLLAMA_", "LLAMA_", "LLM_")):
            del os.environ[key]

    # Set new environment variables
    with patch.dict(os.environ, provider_specific_env_vars, clear=False):
        # Mock settings for the provider
        with patch('testteller.config.settings') as mock_settings:
            # Mock LLM settings
            mock_settings.llm = Mock()
            mock_settings.llm.provider = provider_specific_env_vars["LLM_PROVIDER"]

            # Mock API keys
            mock_settings.api_keys = Mock()
            mock_settings.api_keys.__dict__ = {
                "google_api_key": provider_specific_env_vars.get("GOOGLE_API_KEY"),
                "openai_api_key": provider_specific_env_vars.get("OPENAI_API_KEY"),
                "claude_api_key": provider_specific_env_vars.get("CLAUDE_API_KEY")
            }

            # Mock ChromaDB settings
            mock_settings.chromadb = Mock()
            mock_settings.chromadb.__dict__ = {
                "persist_directory": provider_specific_env_vars["CHROMA_DB_PERSIST_DIRECTORY"],
                "default_collection_name": provider_specific_env_vars["DEFAULT_COLLECTION_NAME"]
            }

            # Mock provider-specific settings
            if provider_specific_env_vars["LLM_PROVIDER"] == "gemini":
                mock_settings.gemini = Mock()
                mock_settings.gemini.__dict__ = {
                    "embedding_model": provider_specific_env_vars.get("GEMINI_EMBEDDING_MODEL"),
                    "generation_model": provider_specific_env_vars.get("GEMINI_GENERATION_MODEL")
                }
            elif provider_specific_env_vars["LLM_PROVIDER"] == "openai":
                mock_settings.openai = Mock()
                mock_settings.openai.__dict__ = {
                    "embedding_model": provider_specific_env_vars.get("OPENAI_EMBEDDING_MODEL"),
                    "generation_model": provider_specific_env_vars.get("OPENAI_GENERATION_MODEL")
                }
            elif provider_specific_env_vars["LLM_PROVIDER"] == "claude":
                mock_settings.claude = Mock()
                mock_settings.claude.__dict__ = {
                    "embedding_provider": provider_specific_env_vars.get("CLAUDE_EMBEDDING_PROVIDER"),
                    "generation_model": provider_specific_env_vars.get("CLAUDE_GENERATION_MODEL")
                }
            elif provider_specific_env_vars["LLM_PROVIDER"] == "llama":
                mock_settings.llama = Mock()
                mock_settings.llama.__dict__ = {
                    "base_url": provider_specific_env_vars.get("OLLAMA_BASE_URL"),
                    "embedding_model": provider_specific_env_vars.get("LLAMA_EMBEDDING_MODEL"),
                    "generation_model": provider_specific_env_vars.get("LLAMA_GENERATION_MODEL")
                }

            yield

    # Cleanup
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)
    if env_path.exists():
        env_path.unlink()


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment with default values."""
    # Set default environment variables
    monkeypatch.setenv("CHROMA_DB_HOST", "localhost")
    monkeypatch.setenv("CHROMA_DB_PORT", "8000")
    monkeypatch.setenv("CHROMA_DB_USE_REMOTE", "false")
    monkeypatch.setenv("CHROMA_DB_PERSIST_DIRECTORY", "./test_chroma_data")
    monkeypatch.setenv("DEFAULT_COLLECTION_NAME", "test_collection")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FORMAT", "text")

    # Create test data directory with proper permissions
    test_data_dir = Path("./test_chroma_data")
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)
    test_data_dir.mkdir(exist_ok=True)
    os.chmod(test_data_dir, 0o777)  # Ensure directory is writable

    # Create a .env file for testing if it doesn't exist
    env_path = Path(".env")
    if env_path.exists():
        env_path.unlink()

    yield

    # Cleanup test directories and files
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)
    if env_path.exists():
        env_path.unlink()


@pytest.fixture
def skip_if_no_api_key(llm_provider: str):
    """Skip test if required API key is not available."""
    required_keys = {
        "gemini": ["GOOGLE_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        # Embedding provider key will be checked dynamically
        "claude": ["CLAUDE_API_KEY"],
        "llama": []  # No API key required for Llama
    }

    missing_keys = []
    for key in required_keys.get(llm_provider, []):
        if not os.getenv(key) or os.getenv(key).startswith("test_"):
            missing_keys.append(key)

    if missing_keys:
        pytest.skip(
            f"Skipping {llm_provider} test - missing API keys: {missing_keys}")


@pytest.fixture
def cleanup_test_data():
    """Clean up test data after tests."""
    yield
    # Cleanup test ChromaDB data
    test_chroma_dir = Path("./test_chroma_data")
    if test_chroma_dir.exists():
        shutil.rmtree(test_chroma_dir)

    # Cleanup test output files
    test_files = [
        "testteller-testcases.md",
        "test_output.md",
        "test_collection_pytest.md"
    ]
    for file in test_files:
        if Path(file).exists():
            Path(file).unlink()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "cli: marks tests as CLI tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: marks tests that require real API keys"
    )
    config.addinivalue_line(
        "markers", "automation: marks tests for automation functionality"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "test_integration" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
        elif "test_unit" in item.fspath.basename:
            item.add_marker(pytest.mark.unit)
        elif "test_cli" in item.fspath.basename:
            item.add_marker(pytest.mark.cli)
        elif "test_automation" in item.fspath.basename or "test_parser" in item.fspath.basename or "test_generators" in item.fspath.basename:
            item.add_marker(pytest.mark.automation)

        # Add slow marker for tests that might be slow
        if any(keyword in item.name.lower() for keyword in ["generate", "ingest", "llm"]):
            item.add_marker(pytest.mark.slow)
