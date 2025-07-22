"""
Main configuration wizard orchestrator.

Coordinates the entire configuration process including LLM provider setup,
TestAutomator automation configuration, and additional settings.
"""

import logging
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .ui import UIHelper, UIMode
from .providers import get_provider_config, PROVIDER_CONFIGS
from .automation import TestAutomatorWizard
from .writers import ConfigurationWriter
from .validators import ConfigurationValidator
from ..constants import SUPPORTED_LLM_PROVIDERS, DEFAULT_LLM_PROVIDER

logger = logging.getLogger(__name__)


class ConfigurationCancelledException(Exception):
    """Exception raised when user cancels configuration."""
    pass


class ConfigurationWizard:
    """Main configuration wizard for TestTeller."""
    
    def __init__(self, ui_mode: UIMode = UIMode.CLI):
        """
        Initialize configuration wizard.
        
        Args:
            ui_mode: UI interaction mode (CLI, TUI, or AUTO)
        """
        self.ui = UIHelper(ui_mode)
        self.writer = ConfigurationWriter()
        self.validator = ConfigurationValidator()
        self.config_data = {}
        self.env_template = self._get_env_template()
    
    def run(self, env_path: Optional[Path] = None) -> bool:
        """
        Run the complete configuration wizard.
        
        Args:
            env_path: Path to .env file (defaults to current directory)
            
        Returns:
            True if configuration was successful
        """
        try:
            # Setup
            if not env_path:
                env_path = Path.cwd() / ".env"
            
            self.ui.set_progress(0, 3)  # 3 main steps
            
            # Step 0: Pre-flight checks
            if not self._check_prerequisites(env_path):
                return False
            
            # Step 1: LLM Provider Configuration
            self.ui.show_step_progress("LLM Provider Configuration")
            llm_config = self._configure_llm_provider()
            self.config_data.update(llm_config)
            
            # Step 2: Automator Agent Configuration
            self.ui.show_step_progress("TestAutomator (Automation) Setup")
            automation_config = self._configure_automation()
            self.config_data.update(automation_config)
            
            # Step 3: Validation and Writing
            self.ui.show_step_progress("Validation and Save")
            if not self._finalize_configuration(env_path):
                return False
            
            # Show completion
            self._show_completion(env_path)
            
            return True
            
        except KeyboardInterrupt:
            self.ui.show_info("Configuration cancelled by user.")
            return False
        except ConfigurationCancelledException:
            self.ui.show_info("💡 Configuration cancelled by user.")
            return False
        except Exception as e:
            logger.error(f"Configuration wizard failed: {e}")
            self.ui.show_error(f"Configuration failed: {e}")
            return False
    
    def run_quick_setup(self, provider: str, env_path: Optional[Path] = None) -> bool:
        """
        Run quick setup for a specific provider.
        
        Args:
            provider: LLM provider name
            env_path: Path to .env file
            
        Returns:
            True if successful
        """
        try:
            if not env_path:
                env_path = Path.cwd() / ".env"
            
            self.ui.show_header("TestTeller Quick Setup")
            
            # Configure specific provider
            provider_config = get_provider_config(provider)
            llm_config = provider_config.configure(self.ui)
            llm_config["LLM_PROVIDER"] = provider
            
            # Use defaults for automation
            automation_wizard = TestAutomatorWizard()
            automation_config = automation_wizard._get_default_config()
            
            # Combine configurations
            self.config_data.update(llm_config)
            self.config_data.update(automation_config)
            
            # Write configuration
            return self._finalize_configuration(env_path)
            
        except Exception as e:
            logger.error(f"Quick setup failed: {e}")
            self.ui.show_error(f"Quick setup failed: {e}")
            return False
    
    def _check_prerequisites(self, env_path: Path) -> bool:
        """Check prerequisites and handle existing configuration."""
        self.ui.show_header()
        
        # Check if .env file already exists
        if env_path.exists():
            self.ui.show_warning(
                f"Configuration file already exists: {env_path}",
                ["This will overwrite your existing configuration"]
            )
            
            if not self.ui.confirm("Do you want to reconfigure TestTeller?", default=False):
                self.ui.show_info("Configuration cancelled.")
                return False
        
        return True
    
    def _configure_llm_provider(self) -> Dict[str, str]:
        """Configure LLM provider."""
        config = {}
        
        # Show provider selection
        self.ui.show_section_header(
            "🤖 LLM Provider Selection",
            "Choose your AI provider for test case generation"
        )
        
        # Show provider information
        provider_descriptions = {
            "gemini": "Google's powerful multimodal AI (recommended)",
            "openai": "Industry-leading GPT models",
            "claude": "Advanced reasoning by Anthropic",
            "llama": "Privacy-focused local deployment"
        }
        
        provider = self.ui.get_choice(
            "Select your LLM provider",
            SUPPORTED_LLM_PROVIDERS,
            default=DEFAULT_LLM_PROVIDER,
            descriptions=provider_descriptions
        )
        
        config["LLM_PROVIDER"] = provider
        
        # Configure the selected provider
        try:
            provider_config = get_provider_config(provider)
            provider_settings = provider_config.configure(self.ui)
            config.update(provider_settings)
            
            # Test connection
            if self.ui.confirm("Test connection to verify configuration?", default=True):
                self.ui.show_info("Testing connection...")
                success, message = provider_config.test_connection(config)
                
                if success:
                    self.ui.show_success("Connection test successful!", [message])
                else:
                    self.ui.show_warning("Connection test failed", [message])
                    if not self.ui.confirm("Continue with configuration anyway?", default=True):
                        raise ConfigurationCancelledException("Configuration cancelled by user")
            
        except ConfigurationCancelledException:
            # Re-raise without logging - this is user cancellation, not an error
            raise
        except Exception as e:
            logger.error(f"Provider configuration failed: {e}")
            raise
        
        return config
    
    def _configure_automation(self) -> Dict[str, str]:
        """Configure Automator Agent automation with user choice."""
        try:
            # Ask user if they want to configure TestAutomator automation
            self.ui.show_section_header(
                "🔧 Automator Agent Configuration",
                "Configure automated test generation for multiple frameworks"
            )
            
            self.ui.show_info(
                "TestAutomator can generate automation code in multiple languages:",
                [
                    "• Python (pytest, unittest, playwright)",
                    "• JavaScript (jest, mocha, playwright)",
                    "• TypeScript (jest, playwright)",
                    "• Java (junit4, junit5, testng)"
                ]
            )
            
            configure_automation = self.ui.confirm(
                "Would you like to configure TestAutomator (Automation Agent)?", 
                default=True
            )
            
            if configure_automation:
                self.ui.show_info("Setting up TestAutomator automation configuration...")
                automation_wizard = TestAutomatorWizard()
                return automation_wizard.configure(self.ui)
            else:
                self.ui.show_info("Using default TestAutomator configuration from .env.example")
                automation_wizard = TestAutomatorWizard()
                default_config = automation_wizard._get_default_config()
                
                self.ui.show_info(
                    "Default automation settings applied:",
                    [f"• {key}: {value}" for key, value in default_config.items() if value]
                )
                
                self.ui.show_info(
                    "💡 You can configure TestAutomator later using:",
                    ["testteller configure --automator-agent"]
                )
                
                return default_config
            
        except Exception as e:
            logger.error(f"Automation configuration failed: {e}")
            # Return defaults if automation configuration fails
            self.ui.show_warning(
                "Automation configuration failed, using defaults",
                ["You can reconfigure later with 'testteller configure --automator-agent'"]
            )
            automation_wizard = TestAutomatorWizard()
            return automation_wizard._get_default_config()
    
    def _finalize_configuration(self, env_path: Path) -> bool:
        """Validate and write final configuration."""
        try:
            # Validate configuration
            self.ui.show_info("Validating configuration...")
            is_valid, errors, warnings = self.validator.validate_complete_configuration(self.config_data)
            
            if warnings:
                self.ui.show_warning("Configuration warnings:", warnings)
            
            if not is_valid:
                self.ui.show_error("Configuration validation failed:", errors)
                return False
            
            # Show configuration summary
            self.ui.show_configuration_summary(self.config_data)
            
            if not self.ui.confirm("Save this configuration?", default=True):
                self.ui.show_info("💡 Configuration not saved.")
                return False
            
            # Read additional configuration from .env.example
            env_example_path = env_path.parent / ".env.example"
            additional_config = {}
            
            if env_example_path.exists():
                self.ui.show_info("Loading additional default configurations from .env.example...")
                general_configs, provider_specific_configs = self.writer.read_env_example(env_example_path)
                
                # Merge general configs and relevant provider configs
                additional_config.update(general_configs)
                
                # Add relevant provider-specific configs
                current_provider = self.config_data.get("LLM_PROVIDER", "").lower()
                if current_provider:
                    provider_configs = {
                        k: v for k, v in provider_specific_configs.items()
                        if current_provider in k.lower() or k.upper().startswith(f"{current_provider.upper()}_")
                    }
                    additional_config.update(provider_configs)
                
                if additional_config:
                    self.ui.show_info(
                        f"Including {len(additional_config)} additional default settings:",
                        [f"• {key}" for key in list(additional_config.keys())[:5]]
                    )
                    if len(additional_config) > 5:
                        self.ui.show_info(f"  ... and {len(additional_config) - 5} more settings")

            # Write configuration file
            self.ui.show_info(f"Writing configuration to {env_path}...")
            
            success = self.writer.write_env_file(
                config=self.config_data,
                file_path=env_path,
                template_config=self.env_template,
                additional_config=additional_config
            )
            
            if not success:
                self.ui.show_error("Failed to write configuration file")
                return False
            
            success_message = f"Configuration saved to {env_path}"
            if additional_config:
                success_message += f" (including {len(additional_config)} default settings)"
            self.ui.show_success(success_message)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to finalize configuration: {e}")
            self.ui.show_error(f"Failed to save configuration: {e}")
            return False
    
    def _show_completion(self, env_path: Path):
        """Show completion message and next steps."""
        provider = self.config_data.get("LLM_PROVIDER", "")
        automation_enabled = self.config_data.get("AUTOMATION_LANGUAGE", "") != ""
        
        self.ui.show_completion(env_path, provider, automation_enabled)
    
    def _get_env_template(self) -> Dict[str, Dict[str, Any]]:
        """Get environment variable template for ordering."""
        return {
            "LLM_PROVIDER": {
                "description": "AI provider for test case generation",
                "required": True
            },
            "GOOGLE_API_KEY": {
                "description": "Google API key for Gemini",
                "required": False
            },
            "OPENAI_API_KEY": {
                "description": "OpenAI API key for GPT models",
                "required": False
            },
            "CLAUDE_API_KEY": {
                "description": "Anthropic API key for Claude",
                "required": False
            },
            "OLLAMA_BASE_URL": {
                "description": "Ollama base URL for local Llama models",
                "required": False
            },
            "AUTOMATION_LANGUAGE": {
                "description": "Primary language for test automation",
                "required": False
            },
            "AUTOMATION_FRAMEWORK": {
                "description": "Testing framework for automation",
                "required": False
            },
            "BASE_URL": {
                "description": "Base URL for testing",
                "required": False
            }
        }


# Convenience functions for different use cases
def run_full_wizard(ui_mode: UIMode = UIMode.CLI) -> bool:
    """Run the full configuration wizard."""
    wizard = ConfigurationWizard(ui_mode)
    return wizard.run()


def run_provider_only_setup(provider: str, ui_mode: UIMode = UIMode.CLI) -> bool:
    """Run setup for a specific provider only."""
    wizard = ConfigurationWizard(ui_mode)
    return wizard.run_quick_setup(provider)


def run_automation_only_setup(ui_mode: UIMode = UIMode.CLI) -> bool:
    """Run TestAutomator automation setup only."""
    try:
        ui_helper = UIHelper(ui_mode)
        ui_helper.show_header("Automator Agent Setup")
        
        automation_wizard = TestAutomatorWizard()
        config = automation_wizard.configure(ui_helper)
        
        # Write automation config to existing .env or create new one
        env_path = Path.cwd() / ".env"
        writer = ConfigurationWriter()
        
        # Read existing config if available
        existing_config = {}
        if env_path.exists():
            # Simple parsing of existing .env
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        existing_config[key.strip()] = value.strip().strip('"').strip("'")
        
        # Merge configurations
        existing_config.update(config)
        
        # Write updated configuration
        success = writer.write_env_file(existing_config, env_path)
        
        if success:
            ui_helper.show_success("Automator Agent configured successfully!")
            return True
        else:
            ui_helper.show_error("Failed to save automation configuration")
            return False
            
    except Exception as e:
        logger.error(f"Automation-only setup failed: {e}")
        return False