"""
Claude (Anthropic) provider configuration.
"""

from typing import Dict, List
from .base import BaseProviderConfig, ProviderInfo, ConfigurationStep, validate_api_key, validate_model_name, create_api_key_step, create_model_step
from ...constants import DEFAULT_CLAUDE_GENERATION_MODEL, DEFAULT_CLAUDE_EMBEDDING_PROVIDER


class ClaudeConfig(BaseProviderConfig):
    """Configuration for Anthropic Claude provider."""
    
    @property
    def provider_info(self) -> ProviderInfo:
        """Get Claude provider information."""
        return ProviderInfo(
            name="claude",
            display_name="Anthropic Claude",
            description="Advanced reasoning AI with excellent context understanding",
            documentation_url="https://docs.anthropic.com/claude/docs",
            setup_complexity="moderate",
            requirements=[
                "Anthropic account",
                "Valid API key from https://console.anthropic.com/",
                "Additional embedding provider (Google or OpenAI)"
            ]
        )
    
    @property
    def configuration_steps(self) -> List[ConfigurationStep]:
        """Get Claude configuration steps."""
        return [
            create_api_key_step("CLAUDE_API_KEY", "Claude"),
            create_model_step("CLAUDE_GENERATION_MODEL", "Generation model", DEFAULT_CLAUDE_GENERATION_MODEL)
        ]
    
    def configure(self, ui_helper) -> Dict[str, str]:
        """
        Custom configuration flow for Claude (requires embedding provider setup).
        """
        config = {}
        
        # Show provider information
        info = self.provider_info
        ui_helper.show_section_header(
            f"🟣 {info.display_name} Configuration",
            info.description
        )
        
        # Show requirements
        ui_helper.show_info("Requirements:", info.requirements)
        
        # Step 1: Claude API Key
        claude_api_key = ui_helper.get_api_key("Claude API key", "Anthropic")
        config["CLAUDE_API_KEY"] = claude_api_key
        
        # Step 2: Embedding Provider Selection
        ui_helper.show_info(
            "Claude requires a separate embedding provider",
            ["Google (Gemini) embeddings", "OpenAI embeddings"]
        )
        
        embedding_providers = ["google", "openai"]
        descriptions = {
            "google": "Use Google Gemini for embeddings (recommended)",
            "openai": "Use OpenAI for embeddings"
        }
        
        embedding_provider = ui_helper.get_choice(
            "Select embedding provider",
            embedding_providers,
            default="google",
            descriptions=descriptions
        )
        
        config["CLAUDE_EMBEDDING_PROVIDER"] = embedding_provider
        
        # Step 3: Embedding Provider API Key
        if embedding_provider == "google":
            google_api_key = ui_helper.get_api_key(
                "Google API key (for embeddings)",
                "Google"
            )
            config["GOOGLE_API_KEY"] = google_api_key
        elif embedding_provider == "openai":
            openai_api_key = ui_helper.get_api_key(
                "OpenAI API key (for embeddings)",
                "OpenAI"
            )
            config["OPENAI_API_KEY"] = openai_api_key
        
        # Step 4: Generation Model
        generation_model = ui_helper.get_input(
            "Claude generation model",
            default=DEFAULT_CLAUDE_GENERATION_MODEL,
            validator=validate_model_name,
            error_message="Invalid model name format"
        )
        config["CLAUDE_GENERATION_MODEL"] = generation_model
        
        # Validate configuration
        is_valid, errors = self.validate_configuration(config)
        if not is_valid:
            ui_helper.show_error("Configuration validation failed:", errors)
            raise Exception("Configuration validation failed")
        
        # Store configuration
        self._config_data = config
        
        # Show success
        ui_helper.show_success(
            f"{info.display_name} configuration completed!",
            [
                f"Claude API key: configured",
                f"Embedding provider: {embedding_provider}",
                f"Generation model: {generation_model}"
            ]
        )
        
        return config
    
    def validate_configuration(self, config: Dict[str, str]) -> tuple[bool, List[str]]:
        """Validate Claude configuration."""
        errors = []
        
        # Check required Claude API key
        claude_api_key = config.get("CLAUDE_API_KEY", "")
        if not claude_api_key:
            errors.append("Claude API key is required")
        elif not validate_api_key(claude_api_key):
            errors.append("Invalid Claude API key format")
        
        # Check embedding provider
        embedding_provider = config.get("CLAUDE_EMBEDDING_PROVIDER", "")
        if not embedding_provider:
            errors.append("Embedding provider is required")
        elif embedding_provider not in ["google", "openai"]:
            errors.append("Embedding provider must be 'google' or 'openai'")
        
        # Check embedding provider API key
        if embedding_provider == "google":
            google_api_key = config.get("GOOGLE_API_KEY", "")
            if not google_api_key:
                errors.append("Google API key is required for Google embeddings")
            elif not validate_api_key(google_api_key):
                errors.append("Invalid Google API key format")
        elif embedding_provider == "openai":
            openai_api_key = config.get("OPENAI_API_KEY", "")
            if not openai_api_key:
                errors.append("OpenAI API key is required for OpenAI embeddings")
            elif not validate_api_key(openai_api_key):
                errors.append("Invalid OpenAI API key format")
            elif not openai_api_key.startswith("sk-"):
                errors.append("OpenAI API key should start with 'sk-'")
        
        # Validate generation model
        gen_model = config.get("CLAUDE_GENERATION_MODEL", "")
        if gen_model and not validate_model_name(gen_model):
            errors.append("Invalid Claude generation model name")
        
        return len(errors) == 0, errors
    
    def test_connection(self, config: Dict[str, str]) -> tuple[bool, str]:
        """Test connection to Claude API."""
        try:
            # Import here to avoid circular dependencies
            from ...llm.claude_client import ClaudeClient
            
            claude_api_key = config.get("CLAUDE_API_KEY", "")
            if not claude_api_key:
                return False, "Claude API key not provided"
            
            embedding_provider = config.get("CLAUDE_EMBEDDING_PROVIDER", "")
            if not embedding_provider:
                return False, "Embedding provider not specified"
            
            # Get embedding provider API key
            if embedding_provider == "google":
                embedding_api_key = config.get("GOOGLE_API_KEY", "")
                if not embedding_api_key:
                    return False, "Google API key not provided for embeddings"
            elif embedding_provider == "openai":
                embedding_api_key = config.get("OPENAI_API_KEY", "")
                if not embedding_api_key:
                    return False, "OpenAI API key not provided for embeddings"
            else:
                return False, f"Unsupported embedding provider: {embedding_provider}"
            
            # Create client with the provided configuration
            client = ClaudeClient(
                api_key=claude_api_key,
                generation_model=config.get("CLAUDE_GENERATION_MODEL", DEFAULT_CLAUDE_GENERATION_MODEL),
                embedding_provider=embedding_provider
            )
            
            # Test with a simple generation request
            test_prompt = "Hello! Please respond with 'OK' to confirm the connection."
            response = client.generate_text(test_prompt)
            
            if response and len(response.strip()) > 0:
                return True, "Connection successful! Claude API is working."
            else:
                return False, "Connection failed: Empty response from Claude API"
                
        except Exception as e:
            error_msg = str(e)
            
            # Provide helpful error messages for common issues
            if "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                return False, "Authentication failed. Please check your Claude API key."
            elif "credit" in error_msg.lower() or "usage" in error_msg.lower():
                return False, "API usage limit reached. Please check your Anthropic account."
            elif "rate limit" in error_msg.lower():
                return False, "Rate limit exceeded. Please wait and try again."
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return False, "Network error. Please check your internet connection."
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                return False, "Model not found. Please check your model name."
            elif "embedding" in error_msg.lower():
                return False, f"Embedding provider error. Please check your {embedding_provider} API key."
            else:
                return False, f"Connection test failed: {error_msg}"