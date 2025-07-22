"""
OpenAI API client implementation.
"""
import asyncio
import logging
from typing import List

import openai

from .base_client import BaseLLMClient
from ..constants import DEFAULT_OPENAI_GENERATION_MODEL, DEFAULT_OPENAI_EMBEDDING_MODEL
from ..utils.retry_helpers import api_retry_async, api_retry_sync

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """Client for interacting with OpenAI's API."""

    def __init__(self):
        """Initialize the OpenAI client with API key from settings or environment."""
        super().__init__("openai")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.async_client = openai.AsyncOpenAI(api_key=self.api_key)

    def _get_env_key_name(self) -> str:
        """Get the environment variable name for OpenAI API key."""
        return "OPENAI_API_KEY"
    
    def _get_default_models(self) -> tuple[str, str]:
        """Get default generation and embedding model names for OpenAI."""
        return DEFAULT_OPENAI_GENERATION_MODEL, DEFAULT_OPENAI_EMBEDDING_MODEL

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
            response = await self.async_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(
                "Error generating embedding for text: '%s...': %s", text[:50], e, exc_info=True)
            return None

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
                "Empty text provided for embedding, returning None.")
            return None
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(
                "Error generating embedding for text: '%s...': %s", text[:50], e, exc_info=True)
            return None

    @api_retry_sync
    def get_embeddings_sync(self, texts: list[str]) -> list[list[float] | None]:
        """
        Get embeddings for a list of texts synchronously in a single batch.

        Args:
            texts: List of texts to get embeddings for

        Returns:
            List of embedding lists, with None for failed texts.
        """
        if not texts:
            return []

        try:
            # Replace any empty strings with a single space to avoid API errors
            processed_texts = [text if text.strip() else " " for text in texts]

            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=processed_texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(
                "Error generating sync embeddings for a batch of %d texts: %s", len(texts), e, exc_info=True)
            # Return a list of Nones to indicate failure for all texts in the batch
            return [None] * len(texts)

    @api_retry_async
    async def generate_text_async(self, prompt: str) -> str:
        """
        Generate text using the OpenAI model asynchronously.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=self.generation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Error generating text with OpenAI async: %s", e)
            raise

    @api_retry_sync
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the OpenAI model.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.generation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Error generating text with OpenAI: %s", e)
            raise
