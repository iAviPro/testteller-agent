"""
Google Gemini API client implementation.
"""
import asyncio
import functools
import logging
from typing import List

import google.generativeai as genai

from .base_client import BaseLLMClient
from ..constants import DEFAULT_GEMINI_GENERATION_MODEL, DEFAULT_GEMINI_EMBEDDING_MODEL
from ..utils.retry_helpers import api_retry_async, api_retry_sync

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """Client for interacting with Google's Gemini API."""

    def __init__(self, api_key: str = None, generation_model: str = None, embedding_model: str = None):
        """Initialize the Gemini client with API key from settings or environment."""
        super().__init__("gemini")
        
        # Override with provided values if given
        if api_key:
            self.api_key = api_key
        if generation_model:
            self.generation_model = generation_model
        if embedding_model:
            self.embedding_model = embedding_model
            
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.generation_model)

    def _get_env_key_name(self) -> str:
        """Get the environment variable name for API key."""
        return "GOOGLE_API_KEY"
    
    def _get_default_models(self) -> tuple[str, str]:
        """Get default generation and embedding model names."""
        return DEFAULT_GEMINI_GENERATION_MODEL, DEFAULT_GEMINI_EMBEDDING_MODEL

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
            loop = asyncio.get_running_loop()
            func_to_run = functools.partial(
                genai.embed_content,
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            result = await loop.run_in_executor(None, func_to_run)
            return result['embedding']
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
                "Empty text provided for sync embedding, returning None.")
            return None
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(
                "Error generating sync embedding for text: '%s...': %s", text[:50], e, exc_info=True)
            return None

    async def get_embeddings_async(self, texts: list[str]) -> list[list[float] | None]:
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

            result = genai.embed_content(
                model=self.embedding_model,
                content=processed_texts,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(
                "Error generating sync embeddings for a batch of %d texts: %s", len(texts), e, exc_info=True)
            # Return a list of Nones to indicate failure for all texts in the batch
            return [None] * len(texts)

    @api_retry_async
    async def generate_text_async(self, prompt: str) -> str:
        """
        Generate text using the Gemini model asynchronously.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response
        """
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            logger.error("Error generating text with Gemini async: %s", e)
            raise

    @api_retry_sync
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the Gemini model.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error("Error generating text with Gemini: %s", e)
            raise
