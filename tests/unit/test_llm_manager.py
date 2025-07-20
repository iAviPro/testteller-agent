"""
Unit tests for LLMManager class.
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from testteller.core.llm.llm_manager import LLMManager
from testteller.core.llm.gemini_client import GeminiClient
from testteller.core.llm.openai_client import OpenAIClient
from testteller.core.llm.claude_client import ClaudeClient
from testteller.core.llm.llama_client import LlamaClient


class TestLLMManager:
    """Test cases for LLMManager class."""

    @pytest.mark.unit
    def test_init_with_default_provider(self, mock_env_vars):
        """Test LLMManager initialization with default provider."""
        with patch.dict(os.environ, mock_env_vars):
            with patch('testteller.llm.llm_manager.settings', None):  # Force use of environment
                with patch('testteller.llm.llm_manager.GeminiClient') as mock_gemini:
                    mock_gemini.return_value = Mock()
                    manager = LLMManager()
                    assert manager.provider == "gemini"
                    mock_gemini.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.parametrize("provider,expected_client", [
        ("gemini", "GeminiClient"),
        ("openai", "OpenAIClient"),
        ("claude", "ClaudeClient"),
        ("llama", "LlamaClient")
    ])
    def test_init_with_specific_provider(self, provider, expected_client, mock_env_vars):
        """Test LLMManager initialization with specific providers."""
        env_vars = mock_env_vars.copy()
        env_vars["LLM_PROVIDER"] = provider

        with patch.dict(os.environ, env_vars):
            with patch(f'testteller.llm.llm_manager.{expected_client}') as mock_client:
                mock_client.return_value = Mock()
                manager = LLMManager(provider=provider)
                assert manager.provider == provider
                mock_client.assert_called_once()

    @pytest.mark.unit
    def test_init_with_invalid_provider(self, mock_env_vars):
        """Test LLMManager initialization with invalid provider."""
        with patch.dict(os.environ, mock_env_vars):
            with pytest.raises(ValueError, match="Unsupported LLM provider"):
                LLMManager(provider="invalid_provider")

    @pytest.mark.unit
    def test_get_supported_providers(self):
        """Test getting list of supported providers."""
        providers = LLMManager.get_supported_providers()
        expected = ["gemini", "openai", "claude", "llama"]
        assert providers == expected

    @pytest.mark.unit
    def test_get_current_provider(self, mock_env_vars):
        """Test getting current provider."""
        with patch.dict(os.environ, mock_env_vars):
            with patch('testteller.llm.llm_manager.settings', None):  # Force use of environment
                with patch('testteller.llm.llm_manager.GeminiClient') as mock_gemini:
                    mock_gemini.return_value = Mock()
                    manager = LLMManager()
                    assert manager.get_current_provider() == "gemini"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_text_async(self, mock_env_vars):
        """Test async text generation."""
        with patch.dict(os.environ, mock_env_vars):
            with patch('testteller.llm.llm_manager.settings', None):  # Force use of environment
                with patch('testteller.llm.llm_manager.GeminiClient') as mock_gemini:
                    mock_client = Mock()
                    mock_client.generate_text_async = AsyncMock(
                        return_value="Generated text")
                    mock_gemini.return_value = mock_client

                    manager = LLMManager()
                    result = await manager.generate_text_async("Test prompt")

                    assert result == "Generated text"
                    mock_client.generate_text_async.assert_called_once_with(
                        "Test prompt")

    @pytest.mark.unit
    def test_generate_text_sync(self, mock_env_vars):
        """Test sync text generation."""
        with patch.dict(os.environ, mock_env_vars):
            with patch('testteller.llm.llm_manager.settings', None):  # Force use of environment
                with patch('testteller.llm.llm_manager.GeminiClient') as mock_gemini:
                    mock_client = Mock()
                    mock_client.generate_text.return_value = "Generated text"
                    mock_gemini.return_value = mock_client

                    manager = LLMManager()
                    result = manager.generate_text("Test prompt")

                    assert result == "Generated text"
                    mock_client.generate_text.assert_called_once_with(
                        "Test prompt")

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_embedding_async(self, mock_env_vars, mock_embedding):
        """Test async embedding generation."""
        with patch.dict(os.environ, mock_env_vars):
            with patch('testteller.llm.llm_manager.settings', None):  # Force use of environment
                with patch('testteller.llm.llm_manager.GeminiClient') as mock_gemini:
                    mock_client = Mock()
                    mock_client.get_embedding_async = AsyncMock(
                        return_value=mock_embedding)
                    mock_gemini.return_value = mock_client

                    manager = LLMManager()
                    result = await manager.get_embedding_async("Test text")

                    assert result == mock_embedding
                    mock_client.get_embedding_async.assert_called_once_with(
                        "Test text")

    @pytest.mark.unit
    def test_get_embedding_sync(self, mock_env_vars, mock_embedding):
        """Test sync embedding generation."""
        with patch.dict(os.environ, mock_env_vars):
            with patch('testteller.llm.llm_manager.settings', None):  # Force use of environment
                with patch('testteller.llm.llm_manager.GeminiClient') as mock_gemini:
                    mock_client = Mock()
                    mock_client.get_embedding_sync.return_value = mock_embedding
                    mock_gemini.return_value = mock_client

                    manager = LLMManager()
                    result = manager.get_embedding_sync("Test text")

                    assert result == mock_embedding
                    mock_client.get_embedding_sync.assert_called_once_with(
                        "Test text")

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_embeddings_async(self, mock_env_vars, mock_embedding):
        """Test async batch embedding generation."""
        with patch.dict(os.environ, mock_env_vars):
            with patch('testteller.llm.llm_manager.settings', None):  # Force use of environment
                with patch('testteller.llm.llm_manager.GeminiClient') as mock_gemini:
                    mock_client = Mock()
                    mock_client.get_embeddings_async = AsyncMock(return_value=[
                        mock_embedding, mock_embedding])
                    mock_gemini.return_value = mock_client

                    manager = LLMManager()
                    result = await manager.get_embeddings_async(["Text 1", "Text 2"])

                    assert result == [mock_embedding, mock_embedding]
                    mock_client.get_embeddings_async.assert_called_once_with(
                        ["Text 1", "Text 2"])

    @pytest.mark.unit
    def test_get_embeddings_sync(self, mock_env_vars, mock_embedding):
        """Test sync batch embedding generation."""
        with patch.dict(os.environ, mock_env_vars):
            with patch('testteller.llm.llm_manager.settings', None):  # Force use of environment
                with patch('testteller.llm.llm_manager.GeminiClient') as mock_gemini:
                    mock_client = Mock()
                    mock_client.get_embeddings_sync.return_value = [
                        mock_embedding, mock_embedding]
                    mock_gemini.return_value = mock_client

                    manager = LLMManager()
                    result = manager.get_embeddings_sync(["Text 1", "Text 2"])

                    assert result == [mock_embedding, mock_embedding]
                    mock_client.get_embeddings_sync.assert_called_once_with(
                        ["Text 1", "Text 2"])

    @pytest.mark.unit
    def test_get_provider_info(self, mock_env_vars):
        """Test getting provider information."""
        with patch.dict(os.environ, mock_env_vars):
            with patch('testteller.llm.llm_manager.settings', None):  # Force use of environment
                with patch('testteller.llm.llm_manager.GeminiClient') as mock_gemini:
                    mock_client = Mock()
                    mock_client.generation_model = "gemini-2.0-flash"
                    mock_client.embedding_model = "text-embedding-004"
                    mock_gemini.return_value = mock_client

                    manager = LLMManager()
                    info = manager.get_provider_info()

                    assert info["provider"] == "gemini"
                    assert info["generation_model"] == "gemini-2.0-flash"
                    assert info["embedding_model"] == "text-embedding-004"

    @pytest.mark.unit
    def test_validate_provider_config_valid(self, mock_env_vars):
        """Test provider configuration validation - valid config."""
        with patch.dict(os.environ, mock_env_vars):
            with patch('testteller.llm.llm_manager.settings', None):  # Force use of environment
                with patch('testteller.llm.llm_manager.GeminiClient') as mock_gemini:
                    mock_gemini.return_value = Mock()

                    is_valid, message = LLMManager.validate_provider_config(
                        "gemini")

                    assert is_valid is True
                    assert message == "Configuration valid"

    @pytest.mark.unit
    def test_validate_provider_config_invalid(self):
        """Test provider configuration validation - invalid provider."""
        is_valid, message = LLMManager.validate_provider_config(
            "invalid_provider")

        assert is_valid is False
        assert "Unsupported provider" in message

    @pytest.mark.unit
    def test_validate_provider_config_missing_key(self, mock_env_vars):
        """Test provider configuration validation - missing API key."""
        env_vars = mock_env_vars.copy()
        del env_vars["GOOGLE_API_KEY"]

        with patch.dict(os.environ, env_vars, clear=True):
            with patch('testteller.llm.llm_manager.GeminiClient') as mock_gemini:
                mock_gemini.side_effect = ValueError("API key not found")

                is_valid, message = LLMManager.validate_provider_config(
                    "gemini")

                assert is_valid is False
                assert "API key not found" in message

    @pytest.mark.unit
    def test_handle_api_key_error_gemini(self, mock_env_vars):
        """Test API key error handling for Gemini."""
        env_vars = mock_env_vars.copy()
        del env_vars["GOOGLE_API_KEY"]

        with patch.dict(os.environ, env_vars, clear=True):
            with patch('testteller.llm.llm_manager.GeminiClient') as mock_gemini:
                mock_gemini.side_effect = ValueError("API key not found")

                with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                    LLMManager(provider="gemini")

    @pytest.mark.unit
    def test_handle_api_key_error_llama(self, mock_env_vars):
        """Test API key error handling for Llama (no API key required)."""
        env_vars = mock_env_vars.copy()
        env_vars["LLM_PROVIDER"] = "llama"

        with patch.dict(os.environ, env_vars):
            with patch('testteller.llm.llm_manager.LlamaClient') as mock_llama:
                mock_llama.side_effect = ValueError("Ollama connection failed")

                with pytest.raises(ValueError, match="Ollama connection failed"):
                    LLMManager(provider="llama")

    @pytest.mark.unit
    def test_provider_from_environment(self, mock_env_vars):
        """Test provider selection from environment variable."""
        env_vars = mock_env_vars.copy()
        env_vars["LLM_PROVIDER"] = "openai"

        with patch.dict(os.environ, env_vars):
            with patch('testteller.llm.llm_manager.settings', None):  # Force use of environment
                with patch('testteller.llm.llm_manager.OpenAIClient') as mock_openai:
                    mock_openai.return_value = Mock()

                    manager = LLMManager()
                    assert manager.provider == "openai"

    @pytest.mark.unit
    def test_provider_from_settings(self, mock_env_vars):
        """Test provider selection from settings."""
        with patch.dict(os.environ, mock_env_vars):
            with patch('testteller.llm.llm_manager.settings') as mock_settings:
                mock_llm_settings = Mock()
                mock_llm_settings.provider = "claude"
                mock_settings.llm = mock_llm_settings

                with patch('testteller.llm.llm_manager.ClaudeClient') as mock_claude:
                    mock_claude.return_value = Mock()

                    manager = LLMManager()
                    assert manager.provider == "claude"
