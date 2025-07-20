"""
Gemini (Google) provider configuration.
"""

from typing import Dict, List
from .base import BaseProviderConfig, ProviderInfo, ConfigurationStep, create_api_key_step, create_model_step, validate_api_key, validate_model_name
from ...constants import DEFAULT_GEMINI_GENERATION_MODEL, DEFAULT_GEMINI_EMBEDDING_MODEL


class GeminiConfig(BaseProviderConfig):
    """Configuration for Google Gemini provider."""
    
    @property
    def provider_info(self) -> ProviderInfo:
        """Get Gemini provider information."""
        return ProviderInfo(
            name="gemini",
            display_name="Google Gemini",
            description="Google's powerful multimodal AI with excellent code understanding",
            documentation_url="https://ai.google.dev/docs",
            setup_complexity="simple",
            requirements=[
                "Google AI Studio account",
                "Valid API key from https://makersuite.google.com/app/apikey"
            ]
        )
    
    @property
    def configuration_steps(self) -> List[ConfigurationStep]:
        """Get Gemini configuration steps."""
        return [
            create_api_key_step("GOOGLE_API_KEY", "Google"),
            create_model_step("GEMINI_GENERATION_MODEL", "Generation model", DEFAULT_GEMINI_GENERATION_MODEL),
            create_model_step("GEMINI_EMBEDDING_MODEL", "Embedding model", DEFAULT_GEMINI_EMBEDDING_MODEL)
        ]
    
    def validate_configuration(self, config: Dict[str, str]) -> tuple[bool, List[str]]:
        """Validate Gemini configuration."""
        errors = []
        
        # Check required API key
        api_key = config.get("GOOGLE_API_KEY", "")
        if not api_key:
            errors.append("Google API key is required")
        elif not validate_api_key(api_key):
            errors.append("Invalid Google API key format")
        
        # Validate generation model
        gen_model = config.get("GEMINI_GENERATION_MODEL", "")
        if gen_model and not validate_model_name(gen_model):
            errors.append("Invalid Gemini generation model name")
        
        # Validate embedding model
        emb_model = config.get("GEMINI_EMBEDDING_MODEL", "")
        if emb_model and not validate_model_name(emb_model):
            errors.append("Invalid Gemini embedding model name")
        
        return len(errors) == 0, errors
    
    def test_connection(self, config: Dict[str, str]) -> tuple[bool, str]:
        """Test connection to Gemini API."""
        try:
            # Import here to avoid circular dependencies
            from testteller.core.llm.gemini_client import GeminiClient
            
            api_key = config.get("GOOGLE_API_KEY", "")
            if not api_key:
                return False, "API key not provided"
            
            # Create client with the provided configuration
            client = GeminiClient(
                api_key=api_key,
                generation_model=config.get("GEMINI_GENERATION_MODEL", DEFAULT_GEMINI_GENERATION_MODEL),
                embedding_model=config.get("GEMINI_EMBEDDING_MODEL", DEFAULT_GEMINI_EMBEDDING_MODEL)
            )
            
            # Test with a simple generation request
            test_prompt = "Hello! Please respond with 'OK' to confirm the connection."
            response = client.generate_text(test_prompt)
            
            if response and len(response.strip()) > 0:
                return True, "Connection successful! Gemini API is working."
            else:
                return False, "Connection failed: Empty response from Gemini API"
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Connection test error: {error_msg}")
            
            # Provide helpful error messages for common issues
            if "API_KEY" in error_msg.upper() or "authentication" in error_msg.lower() or "401" in error_msg:
                return False, "Authentication failed. Please check your API key."
            elif "invalid" in error_msg.lower() and "api" in error_msg.lower():
                return False, "Invalid API key format. Please check your API key."
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                return False, "API quota exceeded. Please check your usage limits."
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return False, "Network error. Please check your internet connection."
            elif "404" in error_msg:
                return False, "API endpoint not found. Please check if the model exists."
            else:
                # Include more details for debugging
                return False, f"Connection test failed: {error_msg[:200]}"