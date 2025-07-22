"""
OpenAI provider configuration.
"""

from typing import Dict, List
from .base import BaseProviderConfig, ProviderInfo, ConfigurationStep, validate_api_key, validate_model_name, create_api_key_step, create_model_step
from ...constants import DEFAULT_OPENAI_GENERATION_MODEL, DEFAULT_OPENAI_EMBEDDING_MODEL


class OpenAIConfig(BaseProviderConfig):
    """Configuration for OpenAI provider."""
    
    @property
    def provider_info(self) -> ProviderInfo:
        """Get OpenAI provider information."""
        return ProviderInfo(
            name="openai",
            display_name="OpenAI",
            description="Industry-leading GPT models with excellent reasoning capabilities",
            documentation_url="https://platform.openai.com/docs",
            setup_complexity="simple",
            requirements=[
                "OpenAI account",
                "Valid API key from https://platform.openai.com/api-keys",
                "Sufficient API credits"
            ]
        )
    
    @property
    def configuration_steps(self) -> List[ConfigurationStep]:
        """Get OpenAI configuration steps."""
        return [
            create_api_key_step("OPENAI_API_KEY", "OpenAI"),
            create_model_step("OPENAI_GENERATION_MODEL", "Generation model", DEFAULT_OPENAI_GENERATION_MODEL),
            create_model_step("OPENAI_EMBEDDING_MODEL", "Embedding model", DEFAULT_OPENAI_EMBEDDING_MODEL)
        ]
    
    def validate_configuration(self, config: Dict[str, str]) -> tuple[bool, List[str]]:
        """Validate OpenAI configuration."""
        errors = []
        
        # Check required API key
        api_key = config.get("OPENAI_API_KEY", "")
        if not api_key:
            errors.append("OpenAI API key is required")
        elif not validate_api_key(api_key):
            errors.append("Invalid OpenAI API key format")
        elif not api_key.startswith("sk-"):
            errors.append("OpenAI API key should start with 'sk-'")
        
        # Validate generation model
        gen_model = config.get("OPENAI_GENERATION_MODEL", "")
        if gen_model and not validate_model_name(gen_model):
            errors.append("Invalid OpenAI generation model name")
        
        # Validate embedding model
        emb_model = config.get("OPENAI_EMBEDDING_MODEL", "")
        if emb_model and not validate_model_name(emb_model):
            errors.append("Invalid OpenAI embedding model name")
        
        return len(errors) == 0, errors
    
    def test_connection(self, config: Dict[str, str]) -> tuple[bool, str]:
        """Test connection to OpenAI API."""
        try:
            # Import here to avoid circular dependencies
            from ...llm.openai_client import OpenAIClient
            
            api_key = config.get("OPENAI_API_KEY", "")
            if not api_key:
                return False, "API key not provided"
            
            # Create client with the provided configuration
            client = OpenAIClient(
                api_key=api_key,
                generation_model=config.get("OPENAI_GENERATION_MODEL", DEFAULT_OPENAI_GENERATION_MODEL),
                embedding_model=config.get("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL)
            )
            
            # Test with a simple generation request
            test_prompt = "Hello! Please respond with 'OK' to confirm the connection."
            response = client.generate_text(test_prompt)
            
            if response and len(response.strip()) > 0:
                return True, "Connection successful! OpenAI API is working."
            else:
                return False, "Connection failed: Empty response from OpenAI API"
                
        except Exception as e:
            error_msg = str(e)
            
            # Provide helpful error messages for common issues
            if "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                return False, "Authentication failed. Please check your API key."
            elif "quota" in error_msg.lower() or "credits" in error_msg.lower():
                return False, "API quota exceeded. Please check your usage and billing."
            elif "rate limit" in error_msg.lower():
                return False, "Rate limit exceeded. Please wait and try again."
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return False, "Network error. Please check your internet connection."
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                return False, "Model not found. Please check your model names."
            else:
                return False, f"Connection test failed: {error_msg}"