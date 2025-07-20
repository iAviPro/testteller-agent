"""
Unit tests for Claude client with hybrid embedding support.
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from testteller.core.llm.claude_client import ClaudeClient


class TestClaudeClient:
    """Test cases for Claude client with hybrid embedding support."""

    @pytest.fixture
    def mock_claude_env_vars(self):
        """Mock environment variables for Claude client."""
        return {
            "CLAUDE_API_KEY": "test_claude_key",
            "GOOGLE_API_KEY": "test_google_key",
            "OPENAI_API_KEY": "test_openai_key",
            "CLAUDE_EMBEDDING_PROVIDER": "google"
        }

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for Claude client."""
        mock_settings = Mock()
        mock_api_keys = Mock()
        mock_api_keys.claude_api_key = Mock()
        mock_api_keys.claude_api_key.get_secret_value.return_value = "test_claude_key"
        mock_settings.api_keys = mock_api_keys

        mock_llm = Mock()
        mock_llm.claude_generation_model = "claude-3-5-haiku-20241022"
        mock_llm.claude_embedding_provider = "google"
        mock_settings.llm = mock_llm

        return mock_settings

    @pytest.mark.unit
    def test_init_with_default_settings(self, mock_claude_env_vars):
        """Test Claude client initialization with default settings."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()

                    assert client.generation_model == "claude-3-5-haiku-20241022"
                    assert client.embedding_provider == "google"
                    assert client.api_key == "test_claude_key"

    @pytest.mark.unit
    def test_init_with_settings(self, mock_claude_env_vars, mock_settings):
        """Test Claude client initialization with settings."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.settings', mock_settings):
                with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                    with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                        mock_anthropic.return_value = Mock()
                        mock_async_anthropic.return_value = Mock()

                        client = ClaudeClient()

                        assert client.generation_model == "claude-3-5-haiku-20241022"
                        assert client.embedding_provider == "google"

    @pytest.mark.unit
    def test_init_without_claude_api_key(self):
        """Test Claude client initialization without Claude API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Claude API key not found"):
                ClaudeClient()

    @pytest.mark.unit
    def test_get_google_embedding_sync(self, mock_claude_env_vars):
        """Test sync Google embedding generation."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()

                    with patch('google.generativeai.embed_content') as mock_embed:
                        mock_embed.return_value = {
                            'embedding': [0.1, 0.2, 0.3]}

                        result = client._get_google_embedding_sync("test text")

                        assert result == [0.1, 0.2, 0.3]
                        mock_embed.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_google_embedding_async(self, mock_claude_env_vars):
        """Test async Google embedding generation."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()

                    with patch('google.generativeai.embed_content_async') as mock_embed:
                        mock_embed.return_value = {
                            'embedding': [0.1, 0.2, 0.3]}

                        result = await client._get_google_embedding_async("test text")

                        assert result == [0.1, 0.2, 0.3]
                        mock_embed.assert_called_once()

    @pytest.mark.unit
    def test_get_openai_embedding_sync(self, mock_claude_env_vars):
        """Test sync OpenAI embedding generation."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()

                    with patch('openai.OpenAI') as mock_openai:
                        mock_openai_client = Mock()
                        mock_response = Mock()
                        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
                        mock_openai_client.embeddings.create.return_value = mock_response
                        mock_openai.return_value = mock_openai_client

                        result = client._get_openai_embedding_sync("test text")

                        assert result == [0.1, 0.2, 0.3]
                        mock_openai_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_openai_embedding_async(self, mock_claude_env_vars):
        """Test async OpenAI embedding generation."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()

                    with patch('openai.AsyncOpenAI') as mock_openai:
                        mock_openai_client = Mock()
                        mock_response = Mock()
                        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
                        mock_openai_client.embeddings.create = AsyncMock(
                            return_value=mock_response)
                        mock_openai.return_value = mock_openai_client

                        result = await client._get_openai_embedding_async("test text")

                        assert result == [0.1, 0.2, 0.3]
                        mock_openai_client.embeddings.create.assert_called_once()

    @pytest.mark.unit
    def test_get_embedding_sync_google_primary_success(self, mock_claude_env_vars):
        """Test sync embedding with Google as primary provider - success."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()

                    with patch.object(client, '_get_google_embedding_sync') as mock_google:
                        mock_google.return_value = [0.1, 0.2, 0.3]

                        result = client.get_embedding_sync("test text")

                        assert result == [0.1, 0.2, 0.3]
                        mock_google.assert_called_once_with("test text")

    @pytest.mark.unit
    def test_get_embedding_sync_google_provider_failure(self, mock_claude_env_vars):
        """Test sync embedding with Google provider failure."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()

                    with patch.object(client, '_get_google_embedding_sync') as mock_google:
                        mock_google.side_effect = Exception("Google API error")

                        from testteller.core.utils.exceptions import EmbeddingGenerationError
                        with pytest.raises(EmbeddingGenerationError):
                            client.get_embedding_sync("test text")

                        # Should be called 3 times due to retry logic
                        assert mock_google.call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_embedding_async_google_primary_success(self, mock_claude_env_vars):
        """Test async embedding with Google as primary provider - success."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()

                    with patch.object(client, '_get_google_embedding_async') as mock_google:
                        mock_google.return_value = [0.1, 0.2, 0.3]

                        result = await client.get_embedding_async("test text")

                        assert result == [0.1, 0.2, 0.3]
                        mock_google.assert_called_once_with("test text")

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_embedding_async_google_provider_failure(self, mock_claude_env_vars):
        """Test async embedding with Google provider failure."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()

                    with patch.object(client, '_get_google_embedding_async') as mock_google:
                        mock_google.side_effect = Exception("Google API error")

                        from testteller.core.utils.exceptions import EmbeddingGenerationError
                        with pytest.raises(EmbeddingGenerationError):
                            await client.get_embedding_async("test text")

                        # Should be called 3 times due to retry logic
                        assert mock_google.call_count == 3

    @pytest.mark.unit
    def test_get_embedding_sync_openai_provider_success(self, mock_claude_env_vars):
        """Test sync embedding with OpenAI provider - success."""
        env_vars = mock_claude_env_vars.copy()
        env_vars["CLAUDE_EMBEDDING_PROVIDER"] = "openai"

        with patch.dict(os.environ, env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()

                    with patch.object(client, '_get_openai_embedding_sync') as mock_openai:
                        mock_openai.return_value = [0.1, 0.2, 0.3]

                        result = client.get_embedding_sync("test text")

                        assert result == [0.1, 0.2, 0.3]
                        mock_openai.assert_called_once_with("test text")

    @pytest.mark.unit
    def test_get_embeddings_sync_multiple_texts(self, mock_claude_env_vars):
        """Test sync batch embedding generation."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()

                    with patch.object(client, 'get_embedding_sync') as mock_get_embedding:
                        mock_get_embedding.side_effect = [
                            [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

                        result = client.get_embeddings_sync(["text1", "text2"])

                        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
                        assert mock_get_embedding.call_count == 2

    @pytest.mark.unit
    def test_get_embedding_sync_empty_text(self, mock_claude_env_vars):
        """Test sync embedding with empty text."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()

                    result = client.get_embedding_sync("")

                    assert result is None

    @pytest.mark.unit
    def test_generate_text_sync(self, mock_claude_env_vars):
        """Test sync text generation."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_client = Mock()
                    mock_response = Mock()
                    mock_response.content = [Mock(text="Generated text")]
                    mock_client.messages.create.return_value = mock_response
                    mock_anthropic.return_value = mock_client
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()
                    result = client.generate_text("Test prompt")

                    assert result == "Generated text"
                    mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_text_async(self, mock_claude_env_vars):
        """Test async text generation."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_client = Mock()
                    mock_response = Mock()
                    mock_response.content = [Mock(text="Generated text")]
                    mock_client.messages.create = AsyncMock(
                        return_value=mock_response)
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = mock_client

                    client = ClaudeClient()
                    result = await client.generate_text_async("Test prompt")

                    assert result == "Generated text"
                    mock_client.messages.create.assert_called_once()

    @pytest.mark.unit
    def test_unknown_embedding_provider(self, mock_claude_env_vars):
        """Test handling of unknown embedding provider."""
        env_vars = mock_claude_env_vars.copy()
        env_vars["CLAUDE_EMBEDDING_PROVIDER"] = "unknown"

        with patch.dict(os.environ, env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()

                    from testteller.core.utils.exceptions import EmbeddingGenerationError
                    with pytest.raises(EmbeddingGenerationError):
                        client.get_embedding_sync("test text")

    @pytest.mark.unit
    def test_missing_google_api_key_error_message(self):
        """Test that a helpful error message is shown when Google API key is missing."""
        env_vars = {
            "CLAUDE_API_KEY": "test_claude_key",
            "OPENAI_API_KEY": "test_openai_key"
            # No GOOGLE_API_KEY set
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()
                    client.embedding_provider = "google"

                    from testteller.core.utils.exceptions import EmbeddingGenerationError
                    with pytest.raises(EmbeddingGenerationError) as exc_info:
                        client.get_embedding_sync("test text")

                    error_msg = str(exc_info.value)
                    assert "Google API key is required for Gemini embeddings when using Claude" in error_msg
                    assert "Please set GOOGLE_API_KEY in your .env file" in error_msg
                    assert "testteller configure" in error_msg

    @pytest.mark.unit
    def test_missing_openai_api_key_error_message(self):
        """Test that a helpful error message is shown when OpenAI API key is missing."""
        env_vars = {
            "CLAUDE_API_KEY": "test_claude_key",
            "GOOGLE_API_KEY": "test_google_key"
            # No OPENAI_API_KEY set
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()
                    client.embedding_provider = "openai"

                    from testteller.core.utils.exceptions import EmbeddingGenerationError
                    with pytest.raises(EmbeddingGenerationError) as exc_info:
                        client.get_embedding_sync("test text")

                    error_msg = str(exc_info.value)
                    assert "OpenAI API key is required for embeddings when using Claude" in error_msg
                    assert "Please set OPENAI_API_KEY in your .env file" in error_msg
                    assert "testteller configure" in error_msg

    @pytest.mark.unit
    def test_embedding_provider_fail_with_better_error(self, mock_claude_env_vars):
        """Test that embedding provider failing results in a helpful error message."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()
                    client.embedding_provider = "google"

                    provider_error = Exception("Provider failed")

                    with patch.object(client, '_get_google_embedding_sync', side_effect=provider_error):
                        from testteller.core.utils.exceptions import EmbeddingGenerationError
                        with pytest.raises(EmbeddingGenerationError) as exc_info:
                            client.get_embedding_sync("test text")

                        error_msg = str(exc_info.value)
                        assert "Embedding provider 'google' failed" in error_msg
                        assert "Provider failed" in error_msg
                        assert "testteller configure" in error_msg

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_embedding_provider_fail_with_better_error_async(self, mock_claude_env_vars):
        """Test that embedding provider failing results in a helpful error message (async)."""
        with patch.dict(os.environ, mock_claude_env_vars):
            with patch('testteller.core.llm.claude_client.anthropic.Anthropic') as mock_anthropic:
                with patch('testteller.core.llm.claude_client.anthropic.AsyncAnthropic') as mock_async_anthropic:
                    mock_anthropic.return_value = Mock()
                    mock_async_anthropic.return_value = Mock()

                    client = ClaudeClient()
                    client.embedding_provider = "google"

                    provider_error = Exception("Provider failed")

                    with patch.object(client, '_get_google_embedding_async', side_effect=provider_error):
                        from testteller.core.utils.exceptions import EmbeddingGenerationError
                        with pytest.raises(EmbeddingGenerationError) as exc_info:
                            await client.get_embedding_async("test text")

                        error_msg = str(exc_info.value)
                        assert "Embedding provider 'google' failed" in error_msg
                        assert "Provider failed" in error_msg
                        assert "testteller configure" in error_msg
