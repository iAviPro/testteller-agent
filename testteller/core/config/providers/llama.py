"""
Llama (Ollama) provider configuration.
"""

from typing import Dict, List
from .base import BaseProviderConfig, ProviderInfo, ConfigurationStep, validate_url_with_port, validate_model_name, create_model_step
from ...constants import DEFAULT_LLAMA_GENERATION_MODEL, DEFAULT_LLAMA_EMBEDDING_MODEL, DEFAULT_OLLAMA_BASE_URL


class ConfigurationCancelledException(Exception):
    """Exception raised when user cancels configuration."""
    pass


def validate_ollama_url(url: str) -> bool:
    """Validate Ollama base URL format."""
    if not url:
        return False
    
    # Allow localhost variants
    localhost_patterns = [
        "http://localhost:",
        "http://127.0.0.1:",
        "http://0.0.0.0:"
    ]
    
    if any(url.startswith(pattern) for pattern in localhost_patterns):
        return validate_url_with_port(url)
    
    # Allow other valid URLs
    return validate_url_with_port(url)


class LlamaConfig(BaseProviderConfig):
    """Configuration for Llama (Ollama) provider."""
    
    @property
    def provider_info(self) -> ProviderInfo:
        """Get Llama provider information."""
        return ProviderInfo(
            name="llama",
            display_name="Llama (Ollama)",
            description="Privacy-focused local LLM deployment with Ollama",
            documentation_url="https://ollama.ai/",
            setup_complexity="moderate",
            requirements=[
                "Ollama installed locally",
                "Ollama service running",
                "Downloaded models (llama3.2, etc.)"
            ]
        )
    
    @property  
    def configuration_steps(self) -> List[ConfigurationStep]:
        """Get Llama configuration steps."""
        return [
            ConfigurationStep(
                key="OLLAMA_BASE_URL",
                prompt="Ollama base URL",
                required=True,
                default=DEFAULT_OLLAMA_BASE_URL,
                validator=validate_ollama_url,
                error_message="Invalid Ollama URL format (e.g., http://localhost:11434)"
            ),
            ConfigurationStep(
                key="LLAMA_GENERATION_MODEL",
                prompt="Generation model",
                required=False,
                default=DEFAULT_LLAMA_GENERATION_MODEL,
                validator=validate_model_name,
                error_message="Invalid model name format"
            ),
            ConfigurationStep(
                key="LLAMA_EMBEDDING_MODEL",
                prompt="Embedding model",
                required=False,
                default=DEFAULT_LLAMA_EMBEDDING_MODEL,
                validator=validate_model_name,
                error_message="Invalid model name format"
            )
        ]
    
    def configure(self, ui_helper) -> Dict[str, str]:
        """
        Custom configuration flow for Llama (includes service checking).
        """
        config = {}
        
        # Show provider information
        info = self.provider_info
        ui_helper.show_section_header(
            f"🟡 {info.display_name} Configuration",
            info.description
        )
        
        # Show requirements and installation info
        ui_helper.show_info("Requirements:", info.requirements)
        ui_helper.show_info(
            "Installation help:",
            [
                "1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh",
                "2. Start service: ollama serve",
                "3. Pull models: ollama pull llama3.2:3b"
            ]
        )
        
        # Step 1: Ollama Base URL
        base_url = ui_helper.get_input(
            "Ollama base URL",
            default=DEFAULT_OLLAMA_BASE_URL,
            validator=validate_ollama_url,
            error_message="Invalid Ollama URL format (e.g., http://localhost:11434)"
        )
        config["OLLAMA_BASE_URL"] = base_url
        
        # Step 2: Test connection before proceeding
        ui_helper.show_info("Testing Ollama connection...")
        connection_ok, connection_msg = self._test_ollama_connection(base_url)
        
        if not connection_ok:
            ui_helper.show_warning(
                "Ollama connection test failed",
                [
                    connection_msg,
                    "Please ensure Ollama is running and accessible",
                    "You can continue configuration and test later"
                ]
            )
            
            if not ui_helper.confirm("Continue with configuration anyway?", default=True):
                raise ConfigurationCancelledException("Configuration cancelled by user")
        else:
            ui_helper.show_success("Ollama connection successful!")
        
        # Step 3: Generation Model
        available_models = self._get_available_models(base_url) if connection_ok else []
        
        if available_models:
            ui_helper.show_info(f"Available models: {', '.join(available_models)}")
            
            # Check if default model is available
            if DEFAULT_LLAMA_GENERATION_MODEL in available_models:
                default_gen_model = DEFAULT_LLAMA_GENERATION_MODEL
            else:
                default_gen_model = available_models[0] if available_models else DEFAULT_LLAMA_GENERATION_MODEL
        else:
            default_gen_model = DEFAULT_LLAMA_GENERATION_MODEL
            ui_helper.show_warning(
                "Could not retrieve available models",
                ["Using default model name", "Ensure models are pulled after configuration"]
            )
        
        generation_model = ui_helper.get_input(
            "Generation model",
            default=default_gen_model,
            validator=validate_model_name,
            error_message="Invalid model name format"
        )
        config["LLAMA_GENERATION_MODEL"] = generation_model
        
        # Step 4: Embedding Model
        if available_models:
            # Check if default embedding model is available
            if DEFAULT_LLAMA_EMBEDDING_MODEL in available_models:
                default_emb_model = DEFAULT_LLAMA_EMBEDDING_MODEL
            else:
                # Look for embedding-capable models
                embedding_models = [m for m in available_models if any(keyword in m.lower() 
                                  for keyword in ['embed', 'embedding', 'nomic'])]
                default_emb_model = embedding_models[0] if embedding_models else default_gen_model
        else:
            default_emb_model = DEFAULT_LLAMA_EMBEDDING_MODEL
        
        embedding_model = ui_helper.get_input(
            "Embedding model",
            default=default_emb_model,
            validator=validate_model_name,
            error_message="Invalid model name format"
        )
        config["LLAMA_EMBEDDING_MODEL"] = embedding_model
        
        # Validate configuration
        is_valid, errors = self.validate_configuration(config)
        if not is_valid:
            ui_helper.show_error("Configuration validation failed:", errors)
            raise Exception("Configuration validation failed")
        
        # Store configuration
        self._config_data = config
        
        # Show success and next steps
        ui_helper.show_success(
            f"{info.display_name} configuration completed!",
            [
                f"Base URL: {base_url}",
                f"Generation model: {generation_model}",
                f"Embedding model: {embedding_model}"
            ]
        )
        
        if not connection_ok:
            ui_helper.show_info(
                "Next steps:",
                [
                    "1. Start Ollama service: ollama serve",
                    f"2. Pull generation model: ollama pull {generation_model}",
                    f"3. Pull embedding model: ollama pull {embedding_model}",
                    "4. Test configuration: testteller --help"
                ]
            )
        
        return config
    
    def validate_configuration(self, config: Dict[str, str]) -> tuple[bool, List[str]]:
        """Validate Llama configuration."""
        errors = []
        
        # Check required base URL
        base_url = config.get("OLLAMA_BASE_URL", "")
        if not base_url:
            errors.append("Ollama base URL is required")
        elif not validate_ollama_url(base_url):
            errors.append("Invalid Ollama base URL format")
        
        # Validate generation model
        gen_model = config.get("LLAMA_GENERATION_MODEL", "")
        if gen_model and not validate_model_name(gen_model):
            errors.append("Invalid Llama generation model name")
        
        # Validate embedding model
        emb_model = config.get("LLAMA_EMBEDDING_MODEL", "")
        if emb_model and not validate_model_name(emb_model):
            errors.append("Invalid Llama embedding model name")
        
        return len(errors) == 0, errors
    
    def test_connection(self, config: Dict[str, str]) -> tuple[bool, str]:
        """Test connection to Ollama API."""
        try:
            # Import here to avoid circular dependencies
            from ...llm.llama_client import LlamaClient
            
            base_url = config.get("OLLAMA_BASE_URL", "")
            if not base_url:
                return False, "Ollama base URL not provided"
            
            # Create client with the provided configuration
            client = LlamaClient(
                base_url=base_url,
                generation_model=config.get("LLAMA_GENERATION_MODEL", DEFAULT_LLAMA_GENERATION_MODEL),
                embedding_model=config.get("LLAMA_EMBEDDING_MODEL", DEFAULT_LLAMA_EMBEDDING_MODEL)
            )
            
            # Test with a simple generation request
            test_prompt = "Hello! Please respond with 'OK' to confirm the connection."
            response = client.generate_text(test_prompt)
            
            if response and len(response.strip()) > 0:
                return True, "Connection successful! Ollama API is working."
            else:
                return False, "Connection failed: Empty response from Ollama API"
                
        except Exception as e:
            error_msg = str(e)
            
            # Provide helpful error messages for common issues
            if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                return False, "Connection refused. Please ensure Ollama service is running."
            elif "timeout" in error_msg.lower():
                return False, "Connection timeout. Please check if Ollama is accessible."
            elif "not found" in error_msg.lower() and "model" in error_msg.lower():
                return False, "Model not found. Please pull the required models with 'ollama pull <model>'."
            elif "port" in error_msg.lower():
                return False, "Invalid port. Please check your Ollama base URL."
            else:
                return False, f"Connection test failed: {error_msg}"
    
    def _test_ollama_connection(self, base_url: str) -> tuple[bool, str]:
        """Test basic Ollama service availability."""
        try:
            import requests
            import json
            
            # Test the /api/tags endpoint to list models
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                return True, "Ollama service is running"
            else:
                return False, f"Ollama service returned status code: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "Could not connect to Ollama service"
        except requests.exceptions.Timeout:
            return False, "Connection to Ollama service timed out"
        except Exception as e:
            return False, f"Connection test error: {str(e)}"
    
    def _get_available_models(self, base_url: str) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            import requests
            
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                models = [model.get('name', '') for model in data.get('models', [])]
                return [model for model in models if model]  # Filter out empty names
            
            return []
            
        except Exception:
            return []