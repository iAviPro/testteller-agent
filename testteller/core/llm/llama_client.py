"""
Llama client implementation using Ollama.
"""
import asyncio
import logging
import os
from typing import List

import httpx

from .base_client import BaseLLMClient
from ..constants import DEFAULT_LLAMA_GENERATION_MODEL, DEFAULT_LLAMA_EMBEDDING_MODEL, DEFAULT_OLLAMA_BASE_URL
from ..utils.retry_helpers import api_retry_async, api_retry_sync

logger = logging.getLogger(__name__)


class LlamaClient(BaseLLMClient):
    """Client for interacting with Llama models via Ollama."""

    def __init__(self):
        """Initialize the Llama client with Ollama configuration."""
        super().__init__("llama")
        self.base_url = self._get_ollama_base_url()

        logger.info("Initialized Llama client with generation model '%s', embedding model '%s', and Ollama URL '%s'",
                    self.generation_model, self.embedding_model, self.base_url)

    def _get_ollama_base_url(self) -> str:
        """Get Ollama base URL from settings or environment variables."""
        try:
            if hasattr(self, 'settings') and self.settings and self.settings.llm:
                base_url = self.settings.llm.__dict__.get('ollama_base_url')
                if base_url:
                    return base_url
        except Exception as e:
            logger.debug("Could not get Ollama base URL from settings: %s", e)

        base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        return base_url

    def _get_env_key_name(self) -> str:
        """Get the environment variable name for Llama API key (not used, but required by base class)."""
        return "OLLAMA_API_KEY"  # Not actually used since Ollama is local
    
    def _get_default_models(self) -> tuple[str, str]:
        """Get default generation and embedding model names for Llama."""
        return DEFAULT_LLAMA_GENERATION_MODEL, DEFAULT_LLAMA_EMBEDDING_MODEL
    
    def _get_api_key(self) -> str:
        """Override base class - Llama/Ollama doesn't require an API key."""
        return "not-required"  # Return dummy value since Ollama is local

    @api_retry_async
    async def get_embedding_async(self, text: str) -> List[float]:
        """
        Get embeddings for text asynchronously using Ollama.

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
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Use the standard /api/embeddings endpoint
                try:
                    response = await client.post(
                        f"{self.base_url}/api/embeddings",
                        json={
                            "model": self.embedding_model,
                            "prompt": text
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    # Handle both single and batch embedding response formats
                    if "embedding" in result:
                        # Single embedding response
                        return result["embedding"]
                    elif "embeddings" in result:
                        # Batch embedding response, get the first one
                        embeddings = result["embeddings"]
                        return embeddings[0] if embeddings else None
                    else:
                        logger.error("Unexpected response format from Ollama: %s", result)
                        return None
                except httpx.HTTPStatusError as e:
                    raise
        except Exception as e:
            logger.error(
                "Error generating embedding for text: '%s...': %s", text[:50], e, exc_info=True)
            return None

    @api_retry_sync
    def get_embedding_sync(self, text: str) -> List[float]:
        """
        Get embeddings for text synchronously using Ollama.

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
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                result = response.json()
                # Handle both single and batch embedding response formats
                if "embedding" in result:
                    # Single embedding response
                    return result["embedding"]
                elif "embeddings" in result:
                    # Batch embedding response, get the first one
                    embeddings = result["embeddings"]
                    return embeddings[0] if embeddings else None
                else:
                    logger.error("Unexpected response format from Ollama: %s", result)
                    return None
        except Exception as e:
            logger.error(
                "Error generating embedding for text: '%s...': %s", text[:50], e, exc_info=True)
            return None

    def get_embeddings_sync(self, texts: list[str]) -> list[list[float] | None]:
        """
        Get embeddings for a list of texts synchronously using Ollama.
        NOTE: Ollama API does not support batching, so this iterates through texts.

        Args:
            texts: List of texts to get embeddings for

        Returns:
            List of embedding lists, with None for failed texts.
        """
        if not texts:
            return []

        all_embeddings = []
        with httpx.Client(timeout=60.0) as client:
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    logger.warning(
                        "Empty text provided for sync embedding at index %d, skipping.", i)
                    all_embeddings.append(None)
                    continue

                try:
                    response = client.post(
                        f"{self.base_url}/api/embeddings",
                        json={
                            "model": self.embedding_model,
                            "prompt": text
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    # Handle both single and batch embedding response formats
                    if "embedding" in result:
                        # Single embedding response
                        all_embeddings.append(result["embedding"])
                    elif "embeddings" in result:
                        # Batch embedding response, get the first one
                        embeddings = result["embeddings"]
                        all_embeddings.append(embeddings[0] if embeddings else None)
                    else:
                        logger.error("Unexpected response format from Ollama: %s", result)
                        all_embeddings.append(None)
                except Exception as e:
                    logger.error(
                        "Error generating sync embedding for text at index %d ('%s...'): %s",
                        i, text[:50], e, exc_info=True
                    )
                    all_embeddings.append(None)

        return all_embeddings

    @api_retry_async
    async def generate_text_async(self, prompt: str) -> str:
        """
        Generate text using the Llama model asynchronously via Ollama.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response
        """
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.generation_model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["response"]
        except Exception as e:
            logger.error("Error generating text with Llama async: %s", e)
            raise

    @api_retry_sync
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the Llama model via Ollama.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response
        """
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.generation_model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["response"]
        except Exception as e:
            logger.error("Error generating text with Llama: %s", e)
            raise

    async def check_model_availability(self) -> bool:
        """
        Check if the configured models are available in Ollama.

        Returns:
            True if models are available, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json()

                available_models = [model["name"]
                                    for model in models.get("models", [])]

                generation_available = any(
                    self.generation_model in model for model in available_models)
                embedding_available = any(
                    self.embedding_model in model for model in available_models)

                if not generation_available:
                    logger.warning(
                        "Generation model '%s' not found in Ollama", self.generation_model)
                if not embedding_available:
                    logger.warning(
                        "Embedding model '%s' not found in Ollama", self.embedding_model)

                return generation_available and embedding_available
        except Exception as e:
            logger.error("Error checking Ollama model availability: %s", e)
            return False
