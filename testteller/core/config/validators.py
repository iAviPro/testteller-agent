"""
Configuration validation utilities.

Provides validation functions for different types of configuration values
and overall configuration validation.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class ConfigurationValidator:
    """Validates configuration values and complete configurations."""
    
    def __init__(self):
        """Initialize validator."""
        self.errors = []
        self.warnings = []
    
    def validate_complete_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a complete configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Validate LLM provider configuration
        self._validate_llm_config(config)
        
        # Validate automation configuration
        self._validate_automation_config(config)
        
        # Validate paths and directories
        self._validate_paths_config(config)
        
        # Validate URLs and ports
        self._validate_network_config(config)
        
        # Check for conflicts
        self._check_config_conflicts(config)
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _validate_llm_config(self, config: Dict[str, Any]):
        """Validate LLM provider configuration."""
        provider = config.get('LLM_PROVIDER', '').lower()
        
        if not provider:
            self.errors.append("LLM_PROVIDER is required")
            return
        
        if provider not in ['gemini', 'openai', 'claude', 'llama']:
            self.errors.append(f"Unsupported LLM provider: {provider}")
            return
        
        # Provider-specific validation
        if provider == 'gemini':
            self._validate_gemini_config(config)
        elif provider == 'openai':
            self._validate_openai_config(config)
        elif provider == 'claude':
            self._validate_claude_config(config)
        elif provider == 'llama':
            self._validate_llama_config(config)
    
    def _validate_gemini_config(self, config: Dict[str, Any]):
        """Validate Gemini configuration."""
        api_key = config.get('GOOGLE_API_KEY', '')
        if not api_key:
            self.errors.append("GOOGLE_API_KEY is required for Gemini")
        elif not self.validate_api_key(api_key):
            self.errors.append("Invalid GOOGLE_API_KEY format")
        
        # Validate models
        gen_model = config.get('GEMINI_GENERATION_MODEL', '')
        if gen_model and not self.validate_model_name(gen_model):
            self.errors.append("Invalid GEMINI_GENERATION_MODEL format")
        
        emb_model = config.get('GEMINI_EMBEDDING_MODEL', '')
        if emb_model and not self.validate_model_name(emb_model):
            self.errors.append("Invalid GEMINI_EMBEDDING_MODEL format")
    
    def _validate_openai_config(self, config: Dict[str, Any]):
        """Validate OpenAI configuration."""
        api_key = config.get('OPENAI_API_KEY', '')
        if not api_key:
            self.errors.append("OPENAI_API_KEY is required for OpenAI")
        elif not self.validate_api_key(api_key):
            self.errors.append("Invalid OPENAI_API_KEY format")
        
        # Validate models
        gen_model = config.get('OPENAI_GENERATION_MODEL', '')
        if gen_model and not self.validate_model_name(gen_model):
            self.errors.append("Invalid OPENAI_GENERATION_MODEL format")
        
        emb_model = config.get('OPENAI_EMBEDDING_MODEL', '')
        if emb_model and not self.validate_model_name(emb_model):
            self.errors.append("Invalid OPENAI_EMBEDDING_MODEL format")
    
    def _validate_claude_config(self, config: Dict[str, Any]):
        """Validate Claude configuration."""
        api_key = config.get('CLAUDE_API_KEY', '')
        if not api_key:
            self.errors.append("CLAUDE_API_KEY is required for Claude")
        elif not self.validate_api_key(api_key):
            self.errors.append("Invalid CLAUDE_API_KEY format")
        
        # Validate embedding provider
        emb_provider = config.get('CLAUDE_EMBEDDING_PROVIDER', '').lower()
        if emb_provider and emb_provider not in ['google', 'openai']:
            self.errors.append("CLAUDE_EMBEDDING_PROVIDER must be 'google' or 'openai'")
        
        # Validate that required embedding provider API key is present
        if emb_provider == 'google' and not config.get('GOOGLE_API_KEY'):
            self.errors.append("GOOGLE_API_KEY is required when using Google embeddings with Claude")
        elif emb_provider == 'openai' and not config.get('OPENAI_API_KEY'):
            self.errors.append("OPENAI_API_KEY is required when using OpenAI embeddings with Claude")
        
        # Validate model
        gen_model = config.get('CLAUDE_GENERATION_MODEL', '')
        if gen_model and not self.validate_model_name(gen_model):
            self.errors.append("Invalid CLAUDE_GENERATION_MODEL format")
    
    def _validate_llama_config(self, config: Dict[str, Any]):
        """Validate Llama configuration."""
        base_url = config.get('OLLAMA_BASE_URL', '')
        if not base_url:
            self.errors.append("OLLAMA_BASE_URL is required for Llama")
        elif not self.validate_url(base_url):
            self.errors.append("Invalid OLLAMA_BASE_URL format")
        
        # Validate models
        gen_model = config.get('LLAMA_GENERATION_MODEL', '')
        if gen_model and not self.validate_model_name(gen_model):
            self.errors.append("Invalid LLAMA_GENERATION_MODEL format")
        
        emb_model = config.get('LLAMA_EMBEDDING_MODEL', '')
        if emb_model and not self.validate_model_name(emb_model):
            self.errors.append("Invalid LLAMA_EMBEDDING_MODEL format")
    
    def _validate_automation_config(self, config: Dict[str, Any]):
        """Validate automation configuration."""
        from ..constants import SUPPORTED_LANGUAGES, SUPPORTED_FRAMEWORKS
        
        language = config.get('AUTOMATION_LANGUAGE', '').lower()
        if language and language not in SUPPORTED_LANGUAGES:
            self.errors.append(f"Unsupported automation language: {language}")
        
        framework = config.get('AUTOMATION_FRAMEWORK', '').lower()
        if language and framework:
            supported_frameworks = SUPPORTED_FRAMEWORKS.get(language, [])
            if framework not in supported_frameworks:
                self.errors.append(f"Framework '{framework}' not supported for language '{language}'")
        
        # Validate base URL
        base_url = config.get('BASE_URL', '')
        if base_url and not self.validate_url(base_url):
            self.errors.append("Invalid BASE_URL format")
    
    def _validate_paths_config(self, config: Dict[str, Any]):
        """Validate file paths and directories."""
        # Validate output directory
        output_dir = config.get('AUTOMATION_OUTPUT_DIR', '')
        if output_dir and not self.validate_directory_path(output_dir):
            self.errors.append("Invalid AUTOMATION_OUTPUT_DIR path")
        
        # Validate chroma persist directory
        chroma_dir = config.get('CHROMA_DB_PERSIST_DIRECTORY', '')
        if chroma_dir and not self.validate_directory_path(chroma_dir):
            self.errors.append("Invalid CHROMA_DB_PERSIST_DIRECTORY path")
        
        # Validate temp clone directory
        temp_dir = config.get('TEMP_CLONE_DIR_BASE', '')
        if temp_dir and not self.validate_directory_path(temp_dir):
            self.errors.append("Invalid TEMP_CLONE_DIR_BASE path")
    
    def _validate_network_config(self, config: Dict[str, Any]):
        """Validate network-related configuration."""
        # Validate ChromaDB port
        chroma_port = config.get('CHROMA_DB_PORT', '')
        if chroma_port and not self.validate_port(str(chroma_port)):
            self.errors.append("Invalid CHROMA_DB_PORT")
        
        # Validate ChromaDB host
        chroma_host = config.get('CHROMA_DB_HOST', '')
        if chroma_host and not self.validate_hostname(chroma_host):
            self.warnings.append(f"ChromaDB host '{chroma_host}' may not be accessible")
    
    def _check_config_conflicts(self, config: Dict[str, Any]):
        """Check for configuration conflicts."""
        # Check for conflicting provider configurations
        provider = config.get('LLM_PROVIDER', '').lower()
        
        # Warn if multiple provider API keys are set but only one provider is selected
        providers_with_keys = []
        if config.get('GOOGLE_API_KEY'):
            providers_with_keys.append('gemini')
        if config.get('OPENAI_API_KEY'):
            providers_with_keys.append('openai')
        if config.get('CLAUDE_API_KEY'):
            providers_with_keys.append('claude')
        if config.get('OLLAMA_BASE_URL'):
            providers_with_keys.append('llama')
        
        if len(providers_with_keys) > 1 and provider in providers_with_keys:
            other_providers = [p for p in providers_with_keys if p != provider]
            self.warnings.append(f"Multiple provider credentials found. Using {provider}, ignoring: {', '.join(other_providers)}")
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format."""
        if not api_key or not isinstance(api_key, str):
            return False
        
        api_key = api_key.strip()
        
        # Basic validation - at least 10 characters, no obvious placeholder text
        if len(api_key) < 10:
            return False
        
        # Check for placeholder text
        placeholder_patterns = [
            'your_api_key_here',
            'replace_with_your_key',
            'api_key_placeholder',
            'enter_your_key'
        ]
        
        api_key_lower = api_key.lower()
        if any(pattern in api_key_lower for pattern in placeholder_patterns):
            return False
        
        # Check for reasonable characters (alphanumeric + common symbols)
        if not re.match(r'^[a-zA-Z0-9\-_\.]+$', api_key):
            return False
        
        return True
    
    @staticmethod
    def validate_model_name(model_name: str) -> bool:
        """Validate model name format."""
        if not model_name or not isinstance(model_name, str):
            return False
        
        model_name = model_name.strip()
        
        # Basic validation - reasonable length, allowed characters
        if not (3 <= len(model_name) <= 100):
            return False
        
        # Allow letters, numbers, hyphens, dots, underscores, colons
        if not re.match(r'^[a-zA-Z0-9\-\._:]+$', model_name):
            return False
        
        return True
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        if not url or not isinstance(url, str):
            return False
        
        url = url.strip()
        
        # Must start with http:// or https://
        if not (url.startswith('http://') or url.startswith('https://')):
            return False
        
        # Basic validation using regex
        url_pattern = r'^https?://[a-zA-Z0-9\-\._~:/?#[\]@!$&\'()*+,;=]+$'
        if not re.match(url_pattern, url):
            return False
        
        return True
    
    @staticmethod
    def validate_port(port_str: str) -> bool:
        """Validate port number."""
        try:
            port = int(port_str)
            return 1 <= port <= 65535
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_directory_path(path_str: str) -> bool:
        """Validate directory path."""
        if not path_str or not isinstance(path_str, str):
            return False
        
        try:
            path = Path(path_str)
            
            # Check for reasonable path length
            if len(str(path)) > 255:
                return False
            
            # Check for dangerous patterns
            dangerous_patterns = ['../', '../', '\\..\\', '/etc/', '/var/', '/usr/']
            path_str_lower = str(path).lower()
            if any(pattern in path_str_lower for pattern in dangerous_patterns):
                return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def validate_hostname(hostname: str) -> bool:
        """Validate hostname or IP address."""
        if not hostname or not isinstance(hostname, str):
            return False
        
        hostname = hostname.strip().lower()
        
        # Allow localhost
        if hostname == 'localhost':
            return True
        
        # Allow IP addresses (basic validation)
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(ip_pattern, hostname):
            # Validate IP ranges
            parts = hostname.split('.')
            try:
                return all(0 <= int(part) <= 255 for part in parts)
            except ValueError:
                return False
        
        # Allow hostnames
        hostname_pattern = r'^[a-zA-Z0-9\-\.]+$'
        if re.match(hostname_pattern, hostname) and len(hostname) <= 253:
            return True
        
        return False