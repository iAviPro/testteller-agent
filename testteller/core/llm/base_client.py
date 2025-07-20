"""
Base LLM client with common functionality for all providers.
"""
import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import SecretStr

from testteller.config import settings
from ..utils.retry_helpers import api_retry_async, api_retry_sync

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Base class for all LLM clients with common functionality."""
    
    def __init__(self, provider_name: str):
        """Initialize base client with common setup."""
        self.provider_name = provider_name
        self.api_key = self._get_api_key()
        self.generation_model, self.embedding_model = self._get_model_names()
        
        logger.info(
            "Initialized %s client with generation model '%s' and embedding model '%s'",
            provider_name, self.generation_model, self.embedding_model
        )
    
    def _get_api_key(self) -> str:
        """Get API key from settings or environment variables."""
        try:
            if settings and settings.api_keys:
                api_key_attr = f"{self.provider_name.lower()}_api_key"
                api_key = getattr(settings.api_keys, api_key_attr, None)
                if isinstance(api_key, SecretStr):
                    return api_key.get_secret_value()
                elif isinstance(api_key, str) and api_key:
                    return api_key
        except Exception as e:
            logger.warning("Could not get API key from settings: %s", e)
        
        # Fallback to environment variable
        env_key = self._get_env_key_name()
        api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(f"No API key found in settings or environment variable {env_key}")
        
        return api_key
    
    def _get_model_names(self) -> tuple[str, str]:
        """Get generation and embedding model names from settings with fallbacks."""
        try:
            if settings and settings.llm:
                generation_key = f"{self.provider_name.lower()}_generation_model"
                embedding_key = f"{self.provider_name.lower()}_embedding_model"
                
                generation_model = getattr(settings.llm, generation_key, None)
                embedding_model = getattr(settings.llm, embedding_key, None)
                
                if generation_model and embedding_model:
                    return generation_model, embedding_model
        except Exception as e:
            logger.warning("Could not get model names from settings: %s. Using defaults.", e)
        
        # Use provider-specific defaults
        return self._get_default_models()
    
    @abstractmethod
    def _get_env_key_name(self) -> str:
        """Get the environment variable name for API key."""
        pass
    
    @abstractmethod
    def _get_default_models(self) -> tuple[str, str]:
        """Get default generation and embedding model names."""
        pass
    
    @abstractmethod
    async def get_embedding_async(self, text: str) -> List[float]:
        """Generate embedding for given text asynchronously."""
        pass
    
    @abstractmethod
    def get_embedding_sync(self, text: str) -> List[float]:
        """Generate embedding for given text synchronously."""
        pass
    
    @abstractmethod
    async def generate_text_async(self, prompt: str, **kwargs) -> str:
        """Generate text based on prompt asynchronously."""
        pass
    
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text based on prompt synchronously."""
        pass
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "name": self.provider_name,
            "generation_model": self.generation_model,
            "embedding_model": self.embedding_model
        }