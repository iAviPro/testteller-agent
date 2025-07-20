"""
Anthropic Claude API client implementation.
"""
import asyncio
import logging
import os
from typing import List

import anthropic

from .base_client import BaseLLMClient
from ..constants import (
    DEFAULT_CLAUDE_GENERATION_MODEL,
    DEFAULT_CLAUDE_EMBEDDING_PROVIDER,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_GEMINI_EMBEDDING_MODEL
)
from ..utils.retry_helpers import api_retry_async, api_retry_sync

logger = logging.getLogger(__name__)


class ClaudeClient(BaseLLMClient):
    """Client for interacting with Anthropic's Claude API with embedding support."""

    def __init__(self):
        """Initialize the Claude client with API key from settings or environment."""
        super().__init__("claude")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key)

        # Get embedding provider from settings - Claude doesn't do embeddings directly
        self.embedding_provider = self._get_embedding_provider()

        # Initialize embedding clients lazily
        self._openai_client = None
        self._openai_async_client = None
        self._gemini_client = None
        self._gemini_async_client = None

        logger.info("Initialized Claude client with generation model '%s' and embedding provider '%s'",
                    self.generation_model, self.embedding_provider)

    def _get_embedding_provider(self) -> str:
        """Get embedding provider from settings or environment."""
        try:
            from testteller.config import settings
            if settings and settings.llm:
                return getattr(settings.llm, 'claude_embedding_provider', DEFAULT_CLAUDE_EMBEDDING_PROVIDER)
        except Exception as e:
            logger.debug("Could not get embedding provider from settings: %s", e)
        
        return os.getenv("CLAUDE_EMBEDDING_PROVIDER", DEFAULT_CLAUDE_EMBEDDING_PROVIDER)

    def _get_env_key_name(self) -> str:
        """Get the environment variable name for Claude API key."""
        return "CLAUDE_API_KEY"
    
    def _get_default_models(self) -> tuple[str, str]:
        """Get default generation and embedding model names for Claude."""
        # Claude doesn't do embeddings directly, so we use the embedding provider's default
        embedding_model = DEFAULT_OPENAI_EMBEDDING_MODEL if self._get_embedding_provider() == "openai" else DEFAULT_GEMINI_EMBEDDING_MODEL
        return DEFAULT_CLAUDE_GENERATION_MODEL, embedding_model

    def _get_openai_client(self):
        """Lazy initialization of OpenAI client for embeddings."""
        if self._openai_client is None:
            import openai
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                from testteller.core.utils.exceptions import EmbeddingGenerationError
                raise EmbeddingGenerationError(
                    "OpenAI API key is required for embeddings when using Claude. "
                    "Please set OPENAI_API_KEY in your .env file.\n"
                    "💡 Run 'testteller configure' to set up your API keys properly."
                )
            self._openai_client = openai.OpenAI(api_key=openai_api_key)
        return self._openai_client

    def _get_openai_async_client(self):
        """Lazy initialization of OpenAI async client for embeddings."""
        if self._openai_async_client is None:
            import openai
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                from testteller.core.utils.exceptions import EmbeddingGenerationError
                raise EmbeddingGenerationError(
                    "OpenAI API key is required for embeddings when using Claude. "
                    "Please set OPENAI_API_KEY in your .env file.\n"
                    "💡 Run 'testteller configure' to set up your API keys properly."
                )
            self._openai_async_client = openai.AsyncOpenAI(
                api_key=openai_api_key)
        return self._openai_async_client

    def _get_gemini_client(self):
        """Lazy initialization of Gemini client for embeddings."""
        if self._gemini_client is None:
            import google.generativeai as genai
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                from testteller.core.utils.exceptions import EmbeddingGenerationError
                raise EmbeddingGenerationError(
                    "Google API key is required for Gemini embeddings when using Claude. "
                    "Please set GOOGLE_API_KEY in your .env file.\n"
                    "💡 Run 'testteller configure' to set up your API keys properly."
                )
            genai.configure(api_key=google_api_key)
            self._gemini_client = genai
        return self._gemini_client

    def _get_gemini_async_client(self):
        """Lazy initialization of Gemini async client for embeddings."""
        if self._gemini_async_client is None:
            import google.generativeai as genai
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                from testteller.core.utils.exceptions import EmbeddingGenerationError
                raise EmbeddingGenerationError(
                    "Google API key is required for Gemini embeddings when using Claude. "
                    "Please set GOOGLE_API_KEY in your .env file.\n"
                    "💡 Run 'testteller configure' to set up your API keys properly."
                )
            genai.configure(api_key=google_api_key)
            self._gemini_async_client = genai
        return self._gemini_async_client

    async def _get_google_embedding_async(self, text: str) -> List[float]:
        """Get embedding from Google Gemini API asynchronously."""
        try:
            gemini_client = self._get_gemini_async_client()
            response = await gemini_client.embed_content_async(
                model=DEFAULT_GEMINI_EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            return response['embedding']
        except Exception as e:
            logger.error("Error getting Google embedding: %s", e)
            raise

    def _get_google_embedding_sync(self, text: str) -> List[float]:
        """Get embedding from Google Gemini API synchronously."""
        try:
            gemini_client = self._get_gemini_client()
            response = gemini_client.embed_content(
                model=DEFAULT_GEMINI_EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            return response['embedding']
        except Exception as e:
            logger.error("Error getting Google embedding: %s", e)
            raise

    async def _get_openai_embedding_async(self, text: str) -> List[float]:
        """Get embedding from OpenAI API asynchronously."""
        try:
            openai_client = self._get_openai_async_client()
            response = await openai_client.embeddings.create(
                model=DEFAULT_OPENAI_EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error("Error getting OpenAI embedding: %s", e)
            raise

    def _get_openai_embedding_sync(self, text: str) -> List[float]:
        """Get embedding from OpenAI API synchronously."""
        try:
            openai_client = self._get_openai_client()
            response = openai_client.embeddings.create(
                model=DEFAULT_OPENAI_EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error("Error getting OpenAI embedding: %s", e)
            raise

    @api_retry_async
    async def get_embedding_async(self, text: str) -> List[float]:
        """
        Get embeddings for text asynchronously.

        Args:
            text: Text to get embeddings for

        Returns:
            List of embedding values
        """
        if not text or not text.strip():
            logger.warning(
                "Empty text provided for embedding, returning None.")
            return None

        try:
            if self.embedding_provider == "google":
                return await self._get_google_embedding_async(text)
            elif self.embedding_provider == "openai":
                return await self._get_openai_embedding_async(text)
            else:
                logger.error(
                    f"Unknown embedding provider: {self.embedding_provider}")
                from testteller.core.utils.exceptions import EmbeddingGenerationError
                raise EmbeddingGenerationError(
                    f"Unknown embedding provider: {self.embedding_provider}")
        except Exception as e:
            logger.error(
                f"Embedding provider '{self.embedding_provider}' failed: {e}")
            # Import here to avoid circular imports
            from testteller.core.utils.exceptions import EmbeddingGenerationError
            raise EmbeddingGenerationError(
                f"Embedding provider '{self.embedding_provider}' failed: {e}. "
                "Please check your API keys and network connection, or run 'testteller configure' to reconfigure."
            ) from e

    @api_retry_sync
    def get_embedding_sync(self, text: str) -> List[float]:
        """
        Get embeddings for text synchronously.

        Args:
            text: Text to get embeddings for

        Returns:
            List of embedding values
        """
        if not text or not text.strip():
            logger.warning(
                "Empty text provided for sync embedding, returning None.")
            return None

        try:
            if self.embedding_provider == "google":
                return self._get_google_embedding_sync(text)
            elif self.embedding_provider == "openai":
                return self._get_openai_embedding_sync(text)
            else:
                logger.error(
                    f"Unknown embedding provider: {self.embedding_provider}")
                from testteller.core.utils.exceptions import EmbeddingGenerationError
                raise EmbeddingGenerationError(
                    f"Unknown embedding provider: {self.embedding_provider}")
        except Exception as e:
            logger.error(
                f"Embedding provider '{self.embedding_provider}' failed: {e}")
            # Import here to avoid circular imports
            from testteller.core.utils.exceptions import EmbeddingGenerationError
            raise EmbeddingGenerationError(
                f"Embedding provider '{self.embedding_provider}' failed: {e}. "
                "Please check your API keys and network connection, or run 'testteller configure' to reconfigure."
            ) from e

    async def get_embeddings_async(self, texts: list[str]) -> list[list[float] | None]:
        """
        Get embeddings for multiple texts asynchronously.

        Args:
            texts: List of texts to get embeddings for

        Returns:
            List of embedding lists, with None for failed texts.
        """
        tasks = [self.get_embedding_async(text_chunk) for text_chunk in texts]
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)

        processed_embeddings = []
        for i, emb_or_exc in enumerate(embeddings):
            if isinstance(emb_or_exc, Exception):
                logger.error(
                    "Failed to get embedding for text chunk %d after retries: %s", i, emb_or_exc)
                processed_embeddings.append(None)
            else:
                processed_embeddings.append(emb_or_exc)
        return processed_embeddings

    @api_retry_sync
    def get_embeddings_sync(self, texts: list[str]) -> list[list[float] | None]:
        """
        Get embeddings for a list of texts synchronously.

        Args:
            texts: List of texts to get embeddings for

        Returns:
            List of embedding lists, with None for failed texts.
        """
        if not texts:
            return []

        # For batch processing, we'll process each text individually
        results = []
        for text in texts:
            embedding = self.get_embedding_sync(text)
            results.append(embedding)
        return results

    @api_retry_async
    async def generate_text_async(self, prompt: str) -> str:
        """
        Generate text using the Claude model asynchronously.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response
        """
        try:
            response = await self.async_client.messages.create(
                model=self.generation_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error("Error generating text with Claude async: %s", e)
            raise

    @api_retry_sync
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the Claude model.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response
        """
        try:
            response = self.client.messages.create(
                model=self.generation_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error("Error generating text with Claude: %s", e)
            raise
