"""
CLI tests for TestTeller RAG Agent commands.
"""
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, call
from typer.testing import CliRunner
from testteller.main import app


class TestCLICommands:
    """Test cases for CLI commands."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @pytest.mark.cli
    def test_version_command(self):
        """Test version command."""
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "TestTeller RAG Agent version:" in result.stdout

    @pytest.mark.cli
    def test_help_command(self):
        """Test help command."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "TestTeller: RAG Agent for AI Test Case Generation" in result.stdout
        assert "ingest-docs" in result.stdout
        assert "ingest-code" in result.stdout
        assert "generate" in result.stdout
        assert "status" in result.stdout
        assert "clear-data" in result.stdout
        assert "configure" in result.stdout

    @pytest.mark.cli
    def test_ingest_docs_help(self):
        """Test ingest-docs command help."""
        result = self.runner.invoke(app, ["ingest-docs", "--help"])
        assert result.exit_code == 0
        assert "Path to a document file or a directory" in result.stdout
        assert "--collection-name" in result.stdout

    @pytest.mark.cli
    def test_ingest_code_help(self):
        """Test ingest-code command help."""
        result = self.runner.invoke(app, ["ingest-code", "--help"])
        assert result.exit_code == 0
        assert "source_path" in result.stdout
        assert "code folder" in result.stdout

    @pytest.mark.cli
    def test_generate_help(self):
        """Test generate command help."""
        result = self.runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "Query for test case generation" in result.stdout
        assert "--collection-name" in result.stdout
        assert "--num-retrieved" in result.stdout
        assert "--output-file" in result.stdout

    @pytest.mark.cli
    def test_status_help(self):
        """Test status command help."""
        result = self.runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "--collection-name" in result.stdout

    @pytest.mark.cli
    def test_clear_data_help(self):
        """Test clear-data command help."""
        result = self.runner.invoke(app, ["clear-data", "--help"])
        assert result.exit_code == 0
        assert "--collection-name" in result.stdout
        assert "--force" in result.stdout

    @pytest.mark.cli
    def test_configure_help(self):
        """Test configure command help."""
        result = self.runner.invoke(app, ["configure", "--help"])
        assert result.exit_code == 0

    @pytest.mark.cli
    def test_ingest_docs_missing_path(self):
        """Test ingest-docs command with missing path."""
        result = self.runner.invoke(app, ["ingest-docs"])
        assert result.exit_code == 2  # Typer uses exit code 2 for missing arguments
        assert result.stdout.strip() == ""  # Typer outputs error to stderr

    @pytest.mark.cli
    def test_ingest_code_missing_path(self):
        """Test ingest-code command with missing path."""
        result = self.runner.invoke(app, ["ingest-code"])
        assert result.exit_code == 2  # Typer uses exit code 2 for missing arguments
        assert result.stdout.strip() == ""  # Typer outputs error to stderr

    @pytest.mark.cli
    def test_generate_missing_query(self):
        """Test generate command with missing query."""
        result = self.runner.invoke(app, ["generate"])
        assert result.exit_code == 2  # Typer uses exit code 2 for missing arguments
        assert result.stdout.strip() == ""  # Typer outputs error to stderr

    @pytest.mark.cli
    @patch('testteller.main.ingest_docs_async')
    def test_ingest_docs_success(self, mock_ingest, mock_env_vars, create_test_files):
        """Test successful document ingestion."""
        with patch.dict(os.environ, mock_env_vars):
            mock_ingest.return_value = None

            doc_file = create_test_files["document"]
            result = self.runner.invoke(app, [
                "ingest-docs",
                str(doc_file),
                "--collection-name", "test_collection"
            ])

            assert result.exit_code == 0
            mock_ingest.assert_called_once()

    @pytest.mark.cli
    @patch('testteller.main.ingest_code_async')
    def test_ingest_code_success(self, mock_ingest, mock_env_vars, create_test_files):
        """Test successful code ingestion."""
        with patch.dict(os.environ, mock_env_vars):
            mock_ingest.return_value = None

            code_file = create_test_files["code"]
            result = self.runner.invoke(app, [
                "ingest-code",
                str(code_file.parent),
                "--collection-name", "test_collection"
            ])

            assert result.exit_code == 0
            mock_ingest.assert_called_once()

    @pytest.mark.cli
    @patch('testteller.main.ingest_code_async')
    def test_ingest_code_with_no_cleanup(self, mock_ingest, mock_env_vars):
        """Test code ingestion with no cleanup flag."""
        with patch.dict(os.environ, mock_env_vars):
            mock_ingest.return_value = None

            result = self.runner.invoke(app, [
                "ingest-code",
                "https://github.com/test/repo.git",
                "--collection-name", "test_collection",
                "--no-cleanup-github"
            ])

            assert result.exit_code == 0
            mock_ingest.assert_called_once()
            args = mock_ingest.call_args
            assert args[0][2] is True  # no_cleanup_github parameter

    @pytest.mark.cli
    @patch('testteller.main.generate_async')
    def test_generate_success(self, mock_generate, mock_env_vars, mock_llm_response):
        """Test successful test case generation."""
        with patch.dict(os.environ, mock_env_vars):
            mock_generate.return_value = None

            result = self.runner.invoke(app, [
                "generate",
                "Create API tests for user authentication",
                "--collection-name", "test_collection",
                "--num-retrieved", "5",
                "--output-file", "test_output.md"
            ])

            assert result.exit_code == 0
            mock_generate.assert_called_once()

    @pytest.mark.cli
    @patch('testteller.main.generate_async')
    def test_generate_with_defaults(self, mock_generate, mock_env_vars):
        """Test test case generation with default parameters."""
        with patch.dict(os.environ, mock_env_vars):
            mock_generate.return_value = None

            result = self.runner.invoke(app, [
                "generate",
                "Create API tests"
            ])

            assert result.exit_code == 0
            mock_generate.assert_called_once()

    @pytest.mark.cli
    @patch('testteller.main.status_async')
    def test_status_success(self, mock_status, mock_env_vars):
        """Test successful status command."""
        with patch.dict(os.environ, mock_env_vars):
            mock_status.return_value = None

            result = self.runner.invoke(app, [
                "status",
                "--collection-name", "test_collection"
            ])

            assert result.exit_code == 0
            mock_status.assert_called_once()

    @pytest.mark.cli
    @patch('testteller.main.status_async')
    def test_status_with_defaults(self, mock_status, mock_env_vars):
        """Test status command with default collection name."""
        with patch.dict(os.environ, mock_env_vars):
            mock_status.return_value = None

            result = self.runner.invoke(app, ["status"])

            assert result.exit_code == 0
            mock_status.assert_called_once()

    @pytest.mark.cli
    @patch('testteller.main.clear_data_async')
    def test_clear_data_with_force(self, mock_clear, mock_env_vars):
        """Test clear-data command with force flag."""
        with patch.dict(os.environ, mock_env_vars):
            mock_clear.return_value = True

            result = self.runner.invoke(app, [
                "clear-data",
                "--collection-name", "test_collection",
                "--force"
            ])

            assert result.exit_code == 0
            mock_clear.assert_called_once()

    @pytest.mark.cli
    @patch('testteller.main.clear_data_async')
    def test_clear_data_without_force(self, mock_clear, mock_env_vars):
        """Test clear-data command without force flag (interactive)."""
        with patch.dict(os.environ, mock_env_vars):
            mock_clear.return_value = True

            # Simulate user confirming deletion
            result = self.runner.invoke(app, [
                "clear-data",
                "--collection-name", "test_collection"
            ], input="y\n")

            assert result.exit_code == 0
            mock_clear.assert_called_once()

    @pytest.mark.cli
    @patch('testteller.main.clear_data_async')
    def test_clear_data_cancelled(self, mock_clear, mock_env_vars):
        """Test clear-data command cancelled by user."""
        with patch.dict(os.environ, mock_env_vars):
            mock_clear.return_value = False

            # Simulate user cancelling deletion
            result = self.runner.invoke(app, [
                "clear-data",
                "--collection-name", "test_collection"
            ], input="n\n")

            assert result.exit_code == 0
            mock_clear.assert_called_once()

    @pytest.mark.cli
    @patch('testteller.main.open', create=True)
    @patch('typer.prompt')
    @patch('typer.confirm')
    @patch('os.path.exists')
    def test_configure_command_basic(self, mock_exists, mock_confirm, mock_prompt, mock_open):
        """Test basic configure command functionality."""
        # Mock file existence checks - .env doesn't exist, .env.example doesn't exist
        mock_exists.return_value = False

        # Mock user inputs for the configuration wizard
        prompt_responses = [
            1,  # Select Gemini (1st option)
            "test_api_key",  # Google API key (required for Gemini)
            "",  # Gemini embedding model (optional, use default)
            "",  # Gemini generation model (optional, use default)
            "",  # GitHub token (optional)
            "",  # Log level (optional, use default)
            "",  # ChromaDB path (optional, use default)
            "",  # Collection name (optional, use default)
            "",  # Output file path (optional, use default)
        ]

        # Mock typer.prompt() calls
        mock_prompt.side_effect = prompt_responses

        # Mock typer.confirm() calls - no additional configs to confirm
        mock_confirm.side_effect = []

        # Mock file operations
        mock_file = mock_open.return_value.__enter__.return_value

        result = self.runner.invoke(app, ["configure"])

        # Should complete successfully
        assert result.exit_code == 0
        assert "Configuration complete!" in result.stdout

        # Verify file write operations
        mock_open.assert_called_once()
        mock_file.write.assert_called()

    @pytest.mark.cli
    @patch('testteller.main.open', create=True)
    @patch('typer.prompt')
    @patch('typer.confirm')
    @patch('os.path.exists')
    def test_configure_command_claude_with_google_embeddings(self, mock_exists, mock_confirm, mock_prompt, mock_open):
        """Test configure command with Claude provider and Google embeddings."""
        # Mock that .env.example doesn't exist and .env doesn't exist
        mock_exists.side_effect = lambda path: False

        # Mock user inputs for Claude with Google embeddings
        mock_prompt.side_effect = [
            3,  # Select Claude (3rd option)
            "test_claude_key",  # Claude API key
            1,  # Select Google for embedding provider
            "test_google_key",  # Google API key (required for embeddings)
            "claude-3-5-haiku-20241022",  # Claude generation model (default)
        ]

        # Mock confirmations
        mock_confirm.side_effect = [
            True,  # Include additional configurations (if any)
            True,  # Include provider-specific configurations (if any)
        ]

        # Mock file operations
        mock_file = mock_open.return_value.__enter__.return_value

        result = self.runner.invoke(app, ["configure"])

        # Should complete successfully
        assert result.exit_code == 0
        assert "Configuration complete!" in result.stdout
        assert "Claude configuration complete!" in result.stdout

        # Verify file write operations (only called once for writing .env)
        mock_open.assert_called_with(
            '/Users/aviral/code/github/testteller-rag-agent/.env', 'w')
        mock_file.write.assert_called()

    @pytest.mark.cli
    @patch('testteller.main.open', create=True)
    @patch('typer.prompt')
    @patch('typer.confirm')
    @patch('os.path.exists')
    def test_configure_command_llama(self, mock_exists, mock_confirm, mock_prompt, mock_open):
        """Test configure command with Llama provider."""
        # Mock that .env.example doesn't exist and .env doesn't exist
        mock_exists.side_effect = lambda path: False

        # Mock user inputs for Llama configuration
        def mock_prompt_func(*args, **kwargs):
            if "Select LLM provider" in str(args[0]):
                return 4  # Select Llama
            elif "Ollama server URL" in str(args[0]):
                return "localhost"
            elif "Ollama server Port" in str(args[0]):
                return "11434"
            elif "Llama embedding model" in str(args[0]):
                return "llama3.2:1b"
            elif "Llama generation model" in str(args[0]):
                return "llama3.2:3b"
            else:
                return ""  # Default empty for all other prompts

        mock_prompt.side_effect = mock_prompt_func

        # Mock confirmations
        mock_confirm.side_effect = [
            True,  # Customize Llama model configurations
            True,  # Include additional configurations (if any)
            True,  # Include provider-specific configurations (if any)
        ]

        # Mock file operations
        mock_file = mock_open.return_value.__enter__.return_value

        result = self.runner.invoke(app, ["configure"])

        # Should complete successfully
        assert result.exit_code == 0
        assert "Configuration complete!" in result.stdout
        assert "Configuring Ollama connection:" in result.stdout
        assert "Ollama URL configured: http://localhost:11434" in result.stdout
        assert "Llama Model Configuration:" in result.stdout

        # Verify file write operations
        mock_open.assert_called_with(
            '/Users/aviral/code/github/testteller-rag-agent/.env', 'w')
        mock_file.write.assert_called()
        written_content = ''.join(call.args[0]
                                  for call in mock_file.write.call_args_list)
        assert 'LLAMA_EMBEDDING_MODEL="llama3.2:1b"' in written_content
        assert 'LLAMA_GENERATION_MODEL="llama3.2:3b"' in written_content

    @pytest.mark.cli
    @patch('testteller.main.open', create=True)
    @patch('typer.prompt')
    @patch('typer.confirm')
    @patch('os.path.exists')
    def test_configure_command_llama_with_remote_url(self, mock_exists, mock_confirm, mock_prompt, mock_open):
        """Test configure command with Llama provider and remote Ollama URL."""
        # Mock that .env.example doesn't exist and .env doesn't exist
        mock_exists.side_effect = lambda path: False

        # Mock user inputs for Llama configuration with remote URL
        mock_prompt.side_effect = [
            4,  # Select Llama (4th option)
            "docker-host",  # Custom Ollama server URL
            "11434",  # Default port
            "llama3.2:1b",  # Llama embedding model (default)
            "llama3.2:3b",  # Llama generation model (default)
            "",  # GitHub token (skip)
            "",  # Log level (default)
            "",  # ChromaDB path (default)
            "",  # Collection name (default)
            "",  # Output file path (default)
        ]

        # Mock confirmations
        mock_confirm.side_effect = [
            True,  # Include additional configurations (if any)
            True,  # Include provider-specific configurations (if any)
        ]

        # Mock file operations
        mock_file = mock_open.return_value.__enter__.return_value

        result = self.runner.invoke(app, ["configure"])

        # Should complete successfully
        assert result.exit_code == 0
        assert "Configuration complete!" in result.stdout
        assert "Configuring Ollama connection:" in result.stdout
        assert "Ollama URL configured: http://docker-host:11434" in result.stdout
        assert "Llama Model Configuration:" in result.stdout

        # Verify file write operations
        mock_open.assert_called_with(
            '/Users/aviral/code/github/testteller-rag-agent/.env', 'w')
        mock_file.write.assert_called()
        written_content = ''.join(call.args[0]
                                  for call in mock_file.write.call_args_list)
        assert 'LLAMA_EMBEDDING_MODEL="llama3.2:1b"' in written_content
        assert 'LLAMA_GENERATION_MODEL="llama3.2:3b"' in written_content

    @pytest.mark.cli
    @patch('testteller.main.open', create=True)
    @patch('typer.prompt')
    @patch('typer.confirm')
    @patch('os.path.exists')
    def test_configure_command_llama_url_validation(self, mock_exists, mock_confirm, mock_prompt, mock_open):
        """Test configure command with Llama provider and port validation."""
        # Mock that .env.example doesn't exist and .env doesn't exist
        mock_exists.side_effect = lambda path: False

        # Mock user inputs for Llama configuration with port validation
        mock_prompt.side_effect = [
            4,  # Select Llama (4th option)
            "remote-ollama",  # Valid URL
            "invalid-port",  # Invalid port (not a number)
            "11434",  # Valid port
            "llama3.2:1b",  # Llama embedding model (default)
            "llama3.2:3b",  # Llama generation model (default)
            "",  # GitHub token (skip)
            "",  # Log level (default)
            "",  # ChromaDB path (default)
            "",  # Collection name (default)
            "",  # Output file path (default)
        ]

        # Mock confirmations
        mock_confirm.side_effect = [
            True,  # Include additional configurations (if any)
            True,  # Include provider-specific configurations (if any)
        ]

        # Mock file operations
        mock_file = mock_open.return_value.__enter__.return_value

        result = self.runner.invoke(app, ["configure"])

        # Should complete successfully after port validation
        assert result.exit_code == 0
        assert "Configuration complete!" in result.stdout
        assert "Port must be a valid number." in result.stdout
        assert "Configuring Ollama connection:" in result.stdout
        assert "Ollama URL configured: http://remote-ollama:11434" in result.stdout
        assert "Llama Model Configuration:" in result.stdout

        # Verify file write operations
        mock_open.assert_called_with(
            '/Users/aviral/code/github/testteller-rag-agent/.env', 'w')
        mock_file.write.assert_called()
        written_content = ''.join(call.args[0]
                                  for call in mock_file.write.call_args_list)
        assert 'LLAMA_EMBEDDING_MODEL="llama3.2:1b"' in written_content
        assert 'LLAMA_GENERATION_MODEL="llama3.2:3b"' in written_content

    @pytest.mark.cli
    @patch('testteller.main.open', create=True)
    @patch('typer.prompt')
    @patch('typer.confirm')
    @patch('os.path.exists')
    def test_configure_command_llama_with_defaults_only(self, mock_exists, mock_confirm, mock_prompt, mock_open):
        """Test configure command with Llama provider using only default configurations."""
        # Mock that .env.example doesn't exist and .env doesn't exist
        mock_exists.side_effect = lambda path: False

        # Mock user inputs for Llama configuration with defaults only
        def mock_prompt_func(*args, **kwargs):
            if "Select LLM provider" in str(args[0]):
                return 4  # Select Llama
            elif "Ollama server URL" in str(args[0]):
                return "localhost"
            elif "Ollama server Port" in str(args[0]):
                return "11434"
            elif "Llama embedding model" in str(args[0]):
                return "llama3.2:1b"  # Use default embedding model
            elif "Llama generation model" in str(args[0]):
                return "llama3.2:3b"  # Use default generation model
            else:
                return ""  # Default empty for all other prompts

        mock_prompt.side_effect = mock_prompt_func

        # Mock confirmations
        mock_confirm.side_effect = [
            True,   # Include additional configurations (if any)
            True,   # Include provider-specific configurations (if any)
        ]

        # Mock file operations
        mock_file = mock_open.return_value.__enter__.return_value

        result = self.runner.invoke(app, ["configure"])

        # Should complete successfully
        assert result.exit_code == 0
        assert "Configuration complete!" in result.stdout
        assert "Configuring Ollama connection:" in result.stdout
        assert "Ollama URL configured: http://localhost:11434" in result.stdout
        assert "Llama Model Configuration:" in result.stdout

        # Verify the mock was called correctly
        mock_file.write.assert_called()
        written_content = ''.join(call.args[0]
                                  for call in mock_file.write.call_args_list)
        assert 'LLAMA_EMBEDDING_MODEL="llama3.2:1b"' in written_content
        assert 'LLAMA_GENERATION_MODEL="llama3.2:3b"' in written_content

    @pytest.mark.cli
    @patch('testteller.main.open', create=True)
    @patch('typer.prompt')
    @patch('typer.confirm')
    @patch('os.path.exists')
    def test_configure_command_claude_with_openai_embeddings(self, mock_exists, mock_confirm, mock_prompt, mock_open):
        """Test configure command with Claude provider and OpenAI embeddings."""
        # Mock that .env.example doesn't exist and .env doesn't exist
        mock_exists.side_effect = lambda path: False

        # Mock user inputs for Claude with OpenAI embeddings
        prompt_responses = [
            3,  # Select Claude (3rd option)
            "test_claude_key",  # Claude API key
            2,  # Select OpenAI for embedding provider
            "test_openai_key",  # OpenAI API key (required for embeddings)
            "",  # Claude generation model (use default)
        ]

        # Mock typer.prompt() calls
        mock_prompt.side_effect = prompt_responses

        # Mock typer.confirm() calls - no additional configs to confirm
        mock_confirm.side_effect = []

        # Mock file operations
        mock_file = mock_open.return_value.__enter__.return_value

        result = self.runner.invoke(app, ["configure"])

        # Should complete successfully
        assert result.exit_code == 0
        assert "Configuration complete!" in result.stdout
        assert "Claude configuration complete!" in result.stdout

        # Verify file write operations
        mock_open.assert_called_with(
            '/Users/aviral/code/github/testteller-rag-agent/.env', 'w')
        mock_file.write.assert_called()

    @pytest.mark.cli
    def test_num_retrieved_validation(self, mock_env_vars):
        """Test num-retrieved parameter validation."""
        with patch.dict(os.environ, mock_env_vars):
            # Test with value too high
            result = self.runner.invoke(app, [
                "generate",
                "Create tests",
                "--num-retrieved", "25"
            ])
            assert result.exit_code != 0

            # Test with negative value
            result = self.runner.invoke(app, [
                "generate",
                "Create tests",
                "--num-retrieved", "-1"
            ])
            assert result.exit_code != 0

    @pytest.mark.cli
    @patch('testteller.main._get_agent')
    def test_command_with_missing_api_key(self, mock_get_agent, mock_env_vars):
        """Test command execution with missing API key."""
        # Remove API key from environment
        env_vars = mock_env_vars.copy()
        del env_vars["GOOGLE_API_KEY"]

        with patch.dict(os.environ, env_vars, clear=True):
            mock_get_agent.side_effect = ValueError("API key not found")

            result = self.runner.invoke(app, [
                "status",
                "--collection-name", "test_collection"
            ])

            assert result.exit_code != 0

    @pytest.mark.cli
    @patch('testteller.main.ingest_docs_async')
    def test_ingest_docs_error_handling(self, mock_ingest, mock_env_vars):
        """Test error handling in ingest-docs command."""
        with patch.dict(os.environ, mock_env_vars):
            mock_ingest.side_effect = Exception("Ingestion failed")

            result = self.runner.invoke(app, [
                "ingest-docs",
                "/nonexistent/path",
                "--collection-name", "test_collection"
            ])

            assert result.exit_code != 0

    @pytest.mark.cli
    @patch('testteller.main.generate_async')
    def test_generate_error_handling(self, mock_generate, mock_env_vars):
        """Test error handling in generate command."""
        with patch.dict(os.environ, mock_env_vars):
            mock_generate.side_effect = Exception("Generation failed")

            result = self.runner.invoke(app, [
                "generate",
                "Create tests",
                "--collection-name", "test_collection"
            ])

            assert result.exit_code != 0

    @pytest.mark.cli
    @patch('testteller.main.get_collection_name')
    def test_collection_name_resolution(self, mock_get_collection_name, mock_env_vars):
        """Test collection name resolution logic."""
        with patch.dict(os.environ, mock_env_vars):
            mock_get_collection_name.return_value = "resolved_collection"

            with patch('testteller.main.status_async') as mock_status:
                mock_status.return_value = None

                result = self.runner.invoke(app, ["status"])

                assert result.exit_code == 0
                mock_get_collection_name.assert_called_once()

    @pytest.mark.cli
    @patch('testteller.main.generate_async')
    def test_output_file_from_settings(self, mock_generate):
        """Test output file path resolution from settings."""
        env_vars = {
            "LLM_PROVIDER": "gemini",
            "GOOGLE_API_KEY": "test_api_key",
            "LOG_LEVEL": "INFO",
            "LOG_FORMAT": "json",
            "CHROMA_DB_HOST": "localhost",
            "CHROMA_DB_PORT": "8000",
            "CHROMA_DB_USE_REMOTE": "false",
            "CHROMA_DB_PERSIST_DIRECTORY": "./test_chroma_data",
            "DEFAULT_COLLECTION_NAME": "test_collection"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            mock_generate.return_value = None

            result = self.runner.invoke(app, [
                "generate",
                "Create API tests",
                "--output-file", "test_output.md"
            ])

            assert result.exit_code == 0
            mock_generate.assert_called_once()

    @pytest.mark.cli
    def test_invalid_command(self):
        """Test invalid command handling."""
        result = self.runner.invoke(app, ["invalid-command"])
        assert result.exit_code == 2  # Typer uses exit code 2 for invalid commands
        assert result.stdout.strip() == ""  # Typer outputs error to stderr

    @pytest.mark.cli
    @patch('testteller.main.check_settings')
    def test_settings_check_failure(self, mock_check_settings, mock_env_vars):
        """Test settings check failure handling."""
        with patch.dict(os.environ, mock_env_vars):
            mock_check_settings.side_effect = Exception("Settings error")

            result = self.runner.invoke(app, [
                "status",
                "--collection-name", "test_collection"
            ])

            assert result.exit_code != 0
