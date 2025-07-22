"""
Base provider configuration interface.

Defines the common interface that all LLM provider configurations must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


class ProviderConfigError(Exception):
    """Exception raised for provider configuration errors."""
    pass


@dataclass
class ConfigurationStep:
    """Represents a single configuration step."""
    key: str
    prompt: str
    required: bool = True
    secret: bool = False
    default: Optional[str] = None
    validator: Optional[callable] = None
    error_message: str = "Invalid input"


@dataclass
class ProviderInfo:
    """Provider information and metadata."""
    name: str
    display_name: str
    description: str
    documentation_url: str
    setup_complexity: str  # "simple", "moderate", "complex"
    requirements: List[str]


class BaseProviderConfig(ABC):
    """Base class for all LLM provider configurations."""
    
    def __init__(self):
        """Initialize provider configuration."""
        self._config_data = {}
    
    @property
    @abstractmethod
    def provider_info(self) -> ProviderInfo:
        """Get provider information and metadata."""
        pass
    
    @property
    @abstractmethod
    def configuration_steps(self) -> List[ConfigurationStep]:
        """Get list of configuration steps for this provider."""
        pass
    
    @abstractmethod
    def validate_configuration(self, config: Dict[str, str]) -> tuple[bool, List[str]]:
        """
        Validate the configuration for this provider.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        pass
    
    @abstractmethod
    def test_connection(self, config: Dict[str, str]) -> tuple[bool, str]:
        """
        Test connection with the provider using the configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (success, message)
        """
        pass
    
    def configure(self, ui_helper) -> Dict[str, str]:
        """
        Run interactive configuration for this provider.
        
        Args:
            ui_helper: UI helper instance for user interaction
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        # Show provider information
        info = self.provider_info
        ui_helper.show_section_header(
            f"{info.display_name} Configuration",
            info.description,
            "🔵" if info.name == "gemini" else 
            "🟢" if info.name == "openai" else
            "🟣" if info.name == "claude" else "🟡"
        )
        
        # Show requirements if any
        if info.requirements:
            ui_helper.show_info("Requirements:", info.requirements)
        
        # Process each configuration step
        for step in self.configuration_steps:
            try:
                if step.secret:
                    value = ui_helper.get_api_key(step.prompt, info.display_name)
                else:
                    value = ui_helper.get_input(
                        step.prompt,
                        default=step.default,
                        validator=step.validator,
                        error_message=step.error_message
                    )
                
                config[step.key] = value
                
            except Exception as e:
                if step.required:
                    raise ProviderConfigError(f"Required configuration failed: {step.key}")
                else:
                    # Use default for optional steps
                    if step.default:
                        config[step.key] = step.default
        
        # Validate the complete configuration
        is_valid, errors = self.validate_configuration(config)
        if not is_valid:
            ui_helper.show_error("Configuration validation failed:", errors)
            raise ProviderConfigError("Configuration validation failed")
        
        # Store configuration
        self._config_data = config
        
        # Show success
        ui_helper.show_success(f"{info.display_name} configuration completed!")
        
        return config
    
    def get_required_env_vars(self) -> List[str]:
        """Get list of required environment variables for this provider."""
        return [step.key for step in self.configuration_steps if step.required]
    
    def get_optional_env_vars(self) -> List[str]:
        """Get list of optional environment variables for this provider."""
        return [step.key for step in self.configuration_steps if not step.required]
    
    def supports_embeddings(self) -> bool:
        """Check if this provider supports embeddings."""
        return any('embedding' in step.key.lower() for step in self.configuration_steps)
    
    def supports_generation(self) -> bool:
        """Check if this provider supports text generation."""
        return any('generation' in step.key.lower() for step in self.configuration_steps)
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration (sanitized for display)."""
        summary = {}
        for key, value in self._config_data.items():
            if any(secret_word in key.lower() for secret_word in ['key', 'token', 'secret']):
                summary[key] = "***" + value[-4:] if len(value) > 4 else "***"
            else:
                summary[key] = value
        return summary


def validate_api_key(api_key: str) -> bool:
    """Generic API key validation."""
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Basic validation - at least 10 characters, no spaces
    return len(api_key.strip()) >= 10 and ' ' not in api_key.strip()


def validate_model_name(model_name: str) -> bool:
    """Generic model name validation."""
    if not model_name or not isinstance(model_name, str):
        return False
    
    # Basic validation - reasonable length, allowed characters
    model_name = model_name.strip()
    return 3 <= len(model_name) <= 100 and model_name.replace('-', '').replace('.', '').replace('_', '').replace(':', '').isalnum()


def validate_url_with_port(url: str) -> bool:
    """Validate URL with optional port."""
    if not url or not isinstance(url, str):
        return False
    
    url = url.strip()
    
    # Check basic URL format
    if not (url.startswith('http://') or url.startswith('https://')):
        return False
    
    # Check for valid characters and structure
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return bool(parsed.netloc)
    except Exception:
        return False


def create_api_key_step(key: str, provider_name: str) -> ConfigurationStep:
    """Create a standard API key configuration step."""
    return ConfigurationStep(
        key=key,
        prompt=f"{provider_name} API key",
        required=True,
        secret=True,
        validator=validate_api_key,
        error_message="API key must be at least 10 characters long"
    )


def create_model_step(key: str, model_name: str, default_value: str) -> ConfigurationStep:
    """Create a standard model configuration step."""
    return ConfigurationStep(
        key=key,
        prompt=model_name,
        required=False,
        default=default_value,
        validator=validate_model_name,
        error_message="Invalid model name format"
    )