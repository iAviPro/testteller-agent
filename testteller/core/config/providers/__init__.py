"""
LLM Provider Configuration Modules

This package contains provider-specific configuration implementations
for all supported LLM providers in TestTeller.
"""

from .base import BaseProviderConfig, ProviderConfigError
from .gemini import GeminiConfig
from .openai import OpenAIConfig  
from .claude import ClaudeConfig
from .llama import LlamaConfig

# Provider registry for dynamic instantiation
PROVIDER_CONFIGS = {
    'gemini': GeminiConfig,
    'openai': OpenAIConfig,
    'claude': ClaudeConfig,
    'llama': LlamaConfig
}

def get_provider_config(provider_name: str) -> BaseProviderConfig:
    """
    Get provider configuration instance by name.
    
    Args:
        provider_name: Name of the provider (gemini, openai, claude, llama)
        
    Returns:
        Provider configuration instance
        
    Raises:
        ProviderConfigError: If provider is not supported
    """
    if provider_name not in PROVIDER_CONFIGS:
        raise ProviderConfigError(f"Unsupported provider: {provider_name}")
    
    return PROVIDER_CONFIGS[provider_name]()

__all__ = [
    'BaseProviderConfig',
    'ProviderConfigError', 
    'GeminiConfig',
    'OpenAIConfig',
    'ClaudeConfig', 
    'LlamaConfig',
    'PROVIDER_CONFIGS',
    'get_provider_config'
]