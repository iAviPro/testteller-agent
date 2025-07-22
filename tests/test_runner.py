#!/usr/bin/env python3
"""
Comprehensive test runner for TestTeller RAG Agent.

This script allows running tests locally with different LLM providers
and provides interactive API key prompts for local testing.
"""

import os
import sys
import subprocess
import argparse
import getpass
from pathlib import Path
from typing import Dict, List, Optional
import shutil

# Add path for testteller imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from testteller.core.constants import (
    DEFAULT_LLAMA_GENERATION_MODEL,
    DEFAULT_LLAMA_EMBEDDING_MODEL
)


class TestRunner:
    """Test runner for TestTeller RAG Agent."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.providers = ["gemini", "openai", "claude", "llama"]
        self.test_types = ["unit", "integration", "cli", "automation", "all"]

    def setup_environment(self, provider: str, interactive: bool = True) -> Dict[str, str]:
        """Set up environment variables for testing."""
        env_vars = {
            "LLM_PROVIDER": provider,
            "CHROMA_DB_HOST": "localhost",
            "CHROMA_DB_PORT": "8000",
            "CHROMA_DB_USE_REMOTE": "false",
            # Use in-memory mode for testing
            "CHROMA_DB_PERSIST_DIRECTORY": "test_chroma_data",
            "LOG_LEVEL": "INFO",
            "LOG_FORMAT": "text",
            "DEFAULT_COLLECTION_NAME": "test_collection",
            "PYTHONPATH": str(self.project_root)
        }

        if interactive:
            # Prompt for API keys interactively
            if provider == "gemini":
                api_key = self._get_api_key("Google Gemini", "GOOGLE_API_KEY")
                if api_key:
                    env_vars["GOOGLE_API_KEY"] = api_key

            elif provider == "openai":
                api_key = self._get_api_key("OpenAI", "OPENAI_API_KEY")
                if api_key:
                    env_vars["OPENAI_API_KEY"] = api_key

            elif provider == "claude":
                claude_key = self._get_api_key(
                    "Anthropic Claude", "CLAUDE_API_KEY")
                if claude_key:
                    env_vars["CLAUDE_API_KEY"] = claude_key

                # Ask for embedding provider
                embedding_provider = input(
                    "Select embedding provider for Claude (google/openai): ").strip().lower()
                if embedding_provider == "google":
                    google_key = self._get_api_key(
                        "Google (for embeddings)", "GOOGLE_API_KEY")
                    if google_key:
                        env_vars["GOOGLE_API_KEY"] = google_key
                    env_vars["CLAUDE_EMBEDDING_PROVIDER"] = "google"
                elif embedding_provider == "openai":
                    openai_key = self._get_api_key(
                        "OpenAI (for embeddings)", "OPENAI_API_KEY")
                    if openai_key:
                        env_vars["OPENAI_API_KEY"] = openai_key
                    env_vars["CLAUDE_EMBEDDING_PROVIDER"] = "openai"

            elif provider == "llama":
                # Check if Ollama is available
                try:
                    import shutil
                    ollama_path = shutil.which("ollama")
                    if not ollama_path:
                        print("❌ Ollama not found. Please install Ollama:")
                        print("   curl -fsSL https://ollama.ai/install.sh | sh")
                        print("   ollama serve")
                        print(f"   ollama pull {DEFAULT_LLAMA_EMBEDDING_MODEL}")
                        print(f"   ollama pull {DEFAULT_LLAMA_GENERATION_MODEL}")
                        return None

                    # Try to run ollama list
                    result = subprocess.run(
                        [ollama_path, "list"], capture_output=True, check=True)
                    env_vars["OLLAMA_BASE_URL"] = "http://localhost:11434"
                    env_vars["LLAMA_EMBEDDING_MODEL"] = DEFAULT_LLAMA_EMBEDDING_MODEL
                    env_vars["LLAMA_GENERATION_MODEL"] = DEFAULT_LLAMA_GENERATION_MODEL
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print("❌ Error running Ollama:", str(e))
                    print("Please ensure Ollama is installed and running:")
                    print("   ollama serve")
                    print(f"   ollama pull {DEFAULT_LLAMA_EMBEDDING_MODEL}")
                    print(f"   ollama pull {DEFAULT_LLAMA_GENERATION_MODEL}")
                    return None
        else:
            # Use test API keys for non-interactive mode
            if provider == "gemini":
                env_vars["GOOGLE_API_KEY"] = "test_google_api_key"
            elif provider == "openai":
                env_vars["OPENAI_API_KEY"] = "test_openai_api_key"
            elif provider == "claude":
                env_vars["CLAUDE_API_KEY"] = "test_claude_api_key"
                env_vars["CLAUDE_EMBEDDING_PROVIDER"] = "openai"
                env_vars["OPENAI_API_KEY"] = "test_openai_api_key"
            elif provider == "llama":
                env_vars["OLLAMA_BASE_URL"] = "http://localhost:11434"
                env_vars["LLAMA_EMBEDDING_MODEL"] = DEFAULT_LLAMA_EMBEDDING_MODEL
                env_vars["LLAMA_GENERATION_MODEL"] = DEFAULT_LLAMA_GENERATION_MODEL

        return env_vars

    def _get_api_key(self, provider_name: str, env_var: str) -> Optional[str]:
        """Get API key from environment or prompt user."""
        # First check if it's already in environment
        existing_key = os.getenv(env_var)
        if existing_key and not existing_key.startswith("test_"):
            return existing_key

        # Prompt user for API key
        print(f"\n🔑 {provider_name} API Key Required")
        print(f"   Environment variable: {env_var}")

        while True:
            response = input(
                f"   Enter {provider_name} API key (or 'skip' to skip): ").strip()

            if response.lower() == 'skip':
                print(
                    f"   ⚠️  Skipping {provider_name} - tests will be skipped")
                return None
            elif response:
                return response
            else:
                print(
                    "   ❌ API key cannot be empty. Please enter a valid key or 'skip'.")

    def run_unit_tests(self, env_vars: Dict[str, str]) -> int:
        """Run unit tests."""
        print("\n🧪 Running Unit Tests...")

        cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/test_minimal_unit.py",
            "tests/unit/test_simple_document_processing.py",
            "-v",
            "--cov=testteller",
            "--cov-report=term-missing", 
            "--cov-report=html:htmlcov",
            "-x"  # Stop on first failure to avoid cascade issues
        ]

        return subprocess.run(cmd, env={**os.environ, **env_vars}).returncode

    def run_integration_tests(self, provider: str, env_vars: Dict[str, str]) -> int:
        """Run simplified integration tests."""
        print(f"\n🔗 Running Simple Integration Tests...")

        cmd = [
            sys.executable, "-m", "pytest",
            "tests/integration/test_simple_flows.py",
            "-v",
            "--cov=testteller",
            "--cov-report=term-missing",
            "--cov-append",
            "-m", "integration"
        ]

        return subprocess.run(cmd, env={**os.environ, **env_vars}).returncode

    def run_cli_tests(self, env_vars: Dict[str, str]) -> int:
        """Run simplified CLI tests."""
        print("\n💻 Running Simple CLI Tests...")

        cmd = [
            sys.executable, "-m", "pytest",
            "tests/cli/test_simple_commands.py",
            "-v",
            "--cov=testteller",
            "--cov-report=term-missing",
            "--cov-append",
            "-m", "cli"
        ]

        return subprocess.run(cmd, env={**os.environ, **env_vars}).returncode

    def run_automation_tests(self, env_vars: Dict[str, str]) -> int:
        """Run automation tests."""
        print("\n🤖 Running Automation Tests...")

        cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/test_parser.py",
            "tests/unit/test_generators.py", 
            "tests/unit/test_cli_automation.py",
            "tests/test_automation.py",
            "-v",
            "--cov=testteller.automator_agent",
            "--cov-report=term-missing",
            "--cov-append",
            "-m", "automation"
        ]

        return subprocess.run(cmd, env={**os.environ, **env_vars}).returncode

    def run_tests(self, provider: str, test_type: str, interactive: bool = True) -> int:
        """Run tests for specific provider and type."""
        print(f"\n🚀 Running {test_type} tests with {provider} provider")
        print("=" * 60)

        # Setup environment
        env_vars = self.setup_environment(provider, interactive)
        if not env_vars:
            return 1

        total_result = 0

        if test_type in ["unit", "all"]:
            result = self.run_unit_tests(env_vars)
            total_result += result

        if test_type in ["integration", "all"]:
            result = self.run_integration_tests(provider, env_vars)
            total_result += result

        if test_type in ["cli", "all"]:
            result = self.run_cli_tests(env_vars)
            total_result += result

        if test_type in ["automation", "all"]:
            result = self.run_automation_tests(env_vars)
            total_result += result

        return total_result

    def run_all_providers(self, test_type: str, interactive: bool = True) -> int:
        """Run tests for all providers."""
        print("\n🌟 Running tests for all providers")
        print("=" * 60)

        total_result = 0
        results = {}

        for provider in self.providers:
            print(f"\n📋 Testing {provider.title()} Provider")
            print("-" * 40)

            result = self.run_tests(provider, test_type, interactive)
            results[provider] = result
            total_result += result

        # Summary
        print("\n📊 Test Results Summary")
        print("=" * 60)
        for provider, result in results.items():
            status = "✅ PASSED" if result == 0 else "❌ FAILED"
            print(f"{provider.title():10} {status}")

        return total_result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for TestTeller Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py --provider gemini --type unit
  python test_runner.py --provider openai --type integration
  python test_runner.py --provider claude --type all
  python test_runner.py --all-providers --type unit
  python test_runner.py --provider llama --type integration --no-interactive
        """
    )

    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "claude", "llama"],
        help="LLM provider to test"
    )

    parser.add_argument(
        "--type",
        choices=["unit", "integration", "cli", "automation", "all"],
        default="all",
        help="Type of tests to run (default: all)"
    )

    parser.add_argument(
        "--all-providers",
        action="store_true",
        help="Run tests for all providers"
    )

    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Don't prompt for API keys (use environment variables)"
    )

    args = parser.parse_args()

    if not args.provider and not args.all_providers:
        parser.error("Must specify either --provider or --all-providers")

    runner = TestRunner()

    print("🧪 TestTeller RAG Agent Test Runner")
    print("=" * 60)

    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Please install test dependencies:")
        print("   pip install pytest pytest-asyncio pytest-cov")
        return 1

    # Check if Docker is available
    try:
        subprocess.run(["docker", "--version"],
                       capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker not found. Please install Docker for ChromaDB testing.")
        return 1

    interactive = not args.no_interactive

    if args.all_providers:
        result = runner.run_all_providers(args.type, interactive)
    else:
        result = runner.run_tests(args.provider, args.type, interactive)

    if result == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n❌ Some tests failed (exit code: {result})")

    return result


if __name__ == "__main__":
    sys.exit(main())
