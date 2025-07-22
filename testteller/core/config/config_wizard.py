"""
Consolidated Configuration Wizard Module.

This module combines all configuration wizard functionality into a single, modular file
for easier maintenance and updates. It includes UI helpers, validation, provider configuration,
automation setup, and configuration file writing.
"""

import logging
import os
import re
import shutil
import typer
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigurationCancelledException(Exception):
    """Exception raised when user cancels configuration."""
    pass


# ==================== Constants ====================

# Import constants from the parent module
try:
    from ..constants import (
        SUPPORTED_LLM_PROVIDERS,
        DEFAULT_LLM_PROVIDER,
        SUPPORTED_LANGUAGES,
        DEFAULT_AUTOMATION_LANGUAGE,
        DEFAULT_AUTOMATION_FRAMEWORK,
        DEFAULT_AUTOMATION_OUTPUT_DIR,
        SUPPORTED_FRAMEWORKS,
        DEFAULT_GEMINI_GENERATION_MODEL,
        DEFAULT_GEMINI_EMBEDDING_MODEL,
        DEFAULT_LLAMA_GENERATION_MODEL,
        DEFAULT_LLAMA_EMBEDDING_MODEL,
        SUPPORTED_TEST_OUTPUT_FORMATS,
        DEFAULT_TEST_OUTPUT_FORMAT,
        DEFAULT_CHROMA_HOST,
        DEFAULT_CHROMA_PORT,
        DEFAULT_CHROMA_USE_REMOTE,
        DEFAULT_CHROMA_PERSIST_DIRECTORY,
        DEFAULT_BASE_URLS
    )
except ImportError:
    # Fallback definitions if constants can't be imported
    SUPPORTED_LLM_PROVIDERS = ["gemini", "openai", "claude", "llama"]
    DEFAULT_LLM_PROVIDER = "gemini"

    SUPPORTED_LANGUAGES = ["python", "javascript", "typescript", "java"]
    DEFAULT_AUTOMATION_LANGUAGE = "python"
    DEFAULT_AUTOMATION_FRAMEWORK = "pytest"
    DEFAULT_AUTOMATION_OUTPUT_DIR = "automation-tests"

    SUPPORTED_FRAMEWORKS = {
        "python": ["pytest", "unittest", "playwright", "cucumber"],
        "javascript": ["jest", "mocha", "playwright", "cypress", "cucumber"],
        "typescript": ["jest", "mocha", "playwright", "cypress", "cucumber"],
        "java": ["junit5", "junit4", "testng", "playwright", "karate", "cucumber"]
    }

    DEFAULT_GEMINI_GENERATION_MODEL = "gemini-2.0-flash"
    DEFAULT_GEMINI_EMBEDDING_MODEL = "text-embedding-004"
    
    DEFAULT_LLAMA_GENERATION_MODEL = "llama3.2:3b"
    DEFAULT_LLAMA_EMBEDDING_MODEL = "llama3.2:1b"

    # Test output format constants
    SUPPORTED_TEST_OUTPUT_FORMATS = ["md", "pdf", "docx"]
    DEFAULT_TEST_OUTPUT_FORMAT = "pdf"
    
    # ChromaDB constants
    DEFAULT_CHROMA_HOST = "localhost"
    DEFAULT_CHROMA_PORT = 8000
    DEFAULT_CHROMA_USE_REMOTE = False
    DEFAULT_CHROMA_PERSIST_DIRECTORY = "./chroma_data"
    
    # Base URL constants
    DEFAULT_BASE_URLS = {
        "python": "http://localhost:8000",
        "javascript": "http://localhost:3000", 
        "typescript": "http://localhost:3000",
        "java": "http://localhost:8080"
    }


# ==================== UI Helpers ====================

class UIMode(Enum):
    """UI interaction modes"""
    CLI = "cli"
    TUI = "tui"
    AUTO = "auto"


class UIHelper:
    """User interface helper for configuration wizard."""

    def __init__(self, mode: UIMode = UIMode.CLI):
        """Initialize UI helper."""
        self.mode = mode
        self._step_counter = 0
        self._total_steps = 0

    def set_progress(self, current_step: int, total_steps: int):
        """Set progress tracking for wizard steps."""
        self._step_counter = current_step
        self._total_steps = total_steps

    def show_header(self, title: str = "TestTeller Configuration Wizard"):
        """Show wizard header with branding."""
        print("\n" + "=" * 60)
        print(f"✨🤖 {title}")
        print("=" * 60)
        print("Setting up your AI-powered Testing Agent")
        if self._total_steps > 0:
            print(f"Progress: Step {self._step_counter}/{self._total_steps}")
        print()

    def show_section_header(self, title: str, description: str = "", icon: str = "⚙️"):
        """Show section header with description."""
        print(f"\n{icon} {title}")
        print("=" * len(f"{icon} {title}"))
        if description:
            print(f"{description}")
        print()

    def show_step_progress(self, step_name: str):
        """Show current step progress."""
        self._step_counter += 1
        if self._total_steps > 0:
            progress = (self._step_counter / self._total_steps) * 100
            print(
                f"\n📍 Step {self._step_counter}/{self._total_steps}: {step_name} ({progress:.0f}%)")
        else:
            print(f"\n📍 {step_name}")

    def get_api_key(self, prompt: str, provider: str = "") -> str:
        """Get API key with validation and security."""
        full_prompt = f"\n{prompt}"
        if provider:
            full_prompt += f" ({provider})"
        full_prompt += ":"

        while True:
            try:
                api_key = typer.prompt(
                    full_prompt,
                    hide_input=True,
                    confirmation_prompt=False
                )

                if api_key and len(api_key.strip()) > 10:
                    return api_key.strip()
                else:
                    print(
                        "❌ API key appears to be too short. Please check and try again.")

            except typer.Abort:
                print("\n❌ Configuration cancelled by user.")
                raise typer.Exit(1)
            except Exception as e:
                logger.error(f"Error getting API key: {e}")
                print("❌ Error reading input. Please try again.")

    def get_choice(self,
                   prompt: str,
                   choices: List[str],
                   default: Optional[str] = None,
                   descriptions: Optional[Dict[str, str]] = None) -> str:
        """Get user choice from a list of options."""
        print(f"\n{prompt}")

        for i, choice in enumerate(choices, 1):
            desc = ""
            if descriptions and choice in descriptions:
                desc = f" - {descriptions[choice]}"
            default_marker = " [default]" if choice == default else ""
            print(f"  {i}. {choice}{desc}{default_marker}")

        while True:
            try:
                if default:
                    user_input = typer.prompt(
                        f"\nEnter choice [1-{len(choices)}]",
                        default="",
                        show_default=False
                    ).strip()

                    if not user_input:
                        return default
                else:
                    user_input = typer.prompt(
                        f"\nEnter choice [1-{len(choices)}]")

                try:
                    choice_num = int(user_input)
                    if 1 <= choice_num <= len(choices):
                        return choices[choice_num - 1]
                    else:
                        print(
                            f"❌ Please enter a number between 1 and {len(choices)}")
                        continue
                except ValueError:
                    user_input_lower = user_input.lower()
                    matches = [choice for choice in choices if choice.lower(
                    ).startswith(user_input_lower)]

                    if len(matches) == 1:
                        return matches[0]
                    elif len(matches) > 1:
                        print(
                            f"❌ Ambiguous choice. Matches: {', '.join(matches)}")
                    else:
                        print(
                            f"❌ Invalid choice. Please select from: {', '.join(choices)}")

            except typer.Abort:
                print("\n❌ Configuration cancelled by user.")
                raise typer.Exit(1)

    def get_input(self,
                  prompt: str,
                  default: Optional[str] = None,
                  validator: Optional[Callable[[str], bool]] = None,
                  error_message: str = "Invalid input") -> str:
        """Get text input with optional validation."""
        while True:
            try:
                if default is not None:
                    value = typer.prompt(
                        f"\n{prompt} (default: {default if default else 'empty'})",
                        default=default,
                        show_default=False
                    )
                else:
                    value = typer.prompt(f"\n{prompt}")

                if validator and not validator(value):
                    print(f"❌ {error_message}")
                    continue

                return value.strip()

            except typer.Abort:
                print("\n❌ Configuration cancelled by user.")
                raise typer.Exit(1)

    def confirm(self, prompt: str, default: bool = True) -> bool:
        """Get yes/no confirmation from user."""
        try:
            return typer.confirm(f"\n{prompt}", default=default)
        except typer.Abort:
            print("\n❌ Configuration cancelled by user.")
            raise typer.Exit(1)

    def show_success(self, message: str, details: Optional[List[str]] = None):
        """Show success message with optional details."""
        print(f"\n✅ {message}")
        if details:
            for detail in details:
                print(f"   • {detail}")

    def show_warning(self, message: str, details: Optional[List[str]] = None):
        """Show warning message with optional details."""
        print(f"\n⚠️  {message}")
        if details:
            for detail in details:
                print(f"   • {detail}")

    def show_error(self, message: str, details: Optional[List[str]] = None):
        """Show error message with optional details."""
        print(f"\n❌ {message}")
        if details:
            for detail in details:
                print(f"   • {detail}")

    def show_info(self, message: str, details: Optional[List[str]] = None):
        """Show info message with optional details."""
        print(f"\n💡 {message}")
        if details:
            for detail in details:
                print(f"   • {detail}")

    def show_configuration_summary(self, config: Dict[str, Any]):
        """Show configuration summary before saving."""
        print("\n📋 Configuration Summary")
        print("=" * 30)

        llm_configs = {}
        automation_configs = {}
        other_configs = {}

        chromadb_configs = {}

        def mask_api_key(value):
            """Helper function to mask API keys showing first 6 characters followed by ..."""
            if len(value) > 6:
                return value[:6] + "..."
            else:
                return "***"

        for key, value in config.items():
            # Check if this is an API key regardless of provider
            if 'api_key' in key.lower() or 'token' in key.lower():
                # Always mask API keys and put them in LLM configs
                llm_configs[key] = mask_api_key(str(value))
            elif any(provider in key.lower() for provider in ['gemini', 'openai', 'claude', 'llama', 'ollama']):
                # Other LLM provider settings (models, URLs, etc.)
                llm_configs[key] = value
            elif 'automation' in key.lower() or key in ['BASE_URL', 'AUTOMATION_OUTPUT_DIR', 'TEST_OUTPUT_FORMAT']:
                automation_configs[key] = value
            elif 'chroma' in key.lower():
                chromadb_configs[key] = value
            else:
                other_configs[key] = value

        if llm_configs:
            print("\n🤖 LLM Provider:")
            for key, value in llm_configs.items():
                print(f"   • {key}: {value}")

        if automation_configs:
            print("\n🚀 Test Automator Agent:")
            for key, value in automation_configs.items():
                print(f"   • {key}: {value}")

        if chromadb_configs:
            print("\n🗄️ ChromaDB (Vector Database):")
            for key, value in chromadb_configs.items():
                print(f"   • {key}: {value}")

        if other_configs:
            print("\n📋 Additional Settings:")
            for key, value in other_configs.items():
                print(f"   • {key}: {value}")

    def show_completion(self, config_path: Path, provider: str, automation_enabled: bool):
        """Show completion message with next steps."""
        print("\n🎉 Configuration Complete!")
        print("=" * 30)
        print(f"✅ Configuration saved to: {config_path}")
        print(f"✅ LLM Provider: {provider}")
        print(
            f"✅ Test Automator Agent: {'Enabled' if automation_enabled else 'Disabled'}")

        print("\n🚀 Next Steps:")
        print("   1. Test your configuration:")
        print("      testteller --help")
        print("   2. Generate test cases:")
        print("      testteller generate 'your project description'")
        if automation_enabled:
            print("   3. Generate automation code:")
            print("      testteller automate test-cases.md")

        print("\n📚 Need help? Check the documentation or run:")
        print("   testteller --help")


# ==================== Validation ====================

def validate_port(port_str: str) -> bool:
    """Validate port number."""
    try:
        port = int(port_str)
        return 1 <= port <= 65535
    except ValueError:
        return False


def validate_url(url: str) -> bool:
    """Basic URL validation."""
    return url.startswith(('http://', 'https://')) and len(url) > 10


def validate_directory_path(path_str: str) -> bool:
    """Validate directory path (can be created)."""
    try:
        path = Path(path_str)
        return len(str(path)) > 0 and not str(path).startswith('/')
    except Exception:
        return False


def validate_api_key(api_key: str) -> bool:
    """Validate API key format."""
    if not api_key or not isinstance(api_key, str):
        return False

    api_key = api_key.strip()

    if len(api_key) < 10:
        return False

    placeholder_patterns = [
        'your_api_key_here',
        'replace_with_your_key',
        'api_key_placeholder',
        'enter_your_key'
    ]

    api_key_lower = api_key.lower()
    if any(pattern in api_key_lower for pattern in placeholder_patterns):
        return False

    if not re.match(r'^[a-zA-Z0-9\-_\.]+$', api_key):
        return False

    return True


# ==================== Provider Configuration ====================

class ProviderConfig:
    """Base class for provider configuration."""

    def configure(self, ui: UIHelper) -> Dict[str, str]:
        """Configure provider settings."""
        raise NotImplementedError

    def test_connection(self, config: Dict[str, str]) -> Tuple[bool, str]:
        """Test connection to provider."""
        return True, "Connection test not implemented"


class GeminiConfig(ProviderConfig):
    """Gemini provider configuration."""

    def configure(self, ui: UIHelper) -> Dict[str, str]:
        config = {}

        ui.show_info(
            "Configuring Google Gemini:",
            [
                "• Get your API key from: https://aistudio.google.com/app/apikey",
                "• Free tier includes 15 RPM, 1 million TPM",
                "• Supports both generation and embeddings"
            ]
        )

        api_key = ui.get_api_key("Enter your Google API key", "Gemini")
        config["GOOGLE_API_KEY"] = api_key

        if ui.confirm("Configure advanced model settings?", default=False):
            gen_model = ui.get_input(
                "Generation model",
                default=DEFAULT_GEMINI_GENERATION_MODEL,
                validator=lambda x: len(x) > 0,
                error_message="Model name cannot be empty"
            )
            config["GEMINI_GENERATION_MODEL"] = gen_model

            emb_model = ui.get_input(
                "Embedding model",
                default=DEFAULT_GEMINI_EMBEDDING_MODEL,
                validator=lambda x: len(x) > 0,
                error_message="Model name cannot be empty"
            )
            config["GEMINI_EMBEDDING_MODEL"] = emb_model

        return config

    def test_connection(self, config: Dict[str, str]) -> Tuple[bool, str]:
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.get("GOOGLE_API_KEY"))
            # Use the configured model or default to the constant
            model_name = config.get(
                "GEMINI_GENERATION_MODEL", DEFAULT_GEMINI_GENERATION_MODEL)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say hello")
            return True, "Successfully connected to Gemini"
        except Exception as e:
            return False, f"Failed to connect: {str(e)}"


class OpenAIConfig(ProviderConfig):
    """OpenAI provider configuration."""

    def configure(self, ui: UIHelper) -> Dict[str, str]:
        config = {}

        ui.show_info(
            "Configuring OpenAI:",
            [
                "• Get your API key from: https://platform.openai.com/api-keys",
                "• Requires paid account or credits",
                "• Supports GPT-4, GPT-3.5, and embeddings"
            ]
        )

        api_key = ui.get_api_key("Enter your OpenAI API key", "OpenAI")
        config["OPENAI_API_KEY"] = api_key

        if ui.confirm("Configure advanced model settings?", default=False):
            gen_model = ui.get_input(
                "Generation model",
                default="gpt-4",
                validator=lambda x: len(x) > 0,
                error_message="Model name cannot be empty"
            )
            config["OPENAI_GENERATION_MODEL"] = gen_model

            emb_model = ui.get_input(
                "Embedding model",
                default="text-embedding-ada-002",
                validator=lambda x: len(x) > 0,
                error_message="Model name cannot be empty"
            )
            config["OPENAI_EMBEDDING_MODEL"] = emb_model

        return config

    def test_connection(self, config: Dict[str, str]) -> Tuple[bool, str]:
        try:
            import openai
            openai.api_key = config.get("OPENAI_API_KEY")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True, "Successfully connected to OpenAI"
        except Exception as e:
            return False, f"Failed to connect: {str(e)}"


class ClaudeConfig(ProviderConfig):
    """Claude provider configuration."""

    def configure(self, ui: UIHelper) -> Dict[str, str]:
        config = {}

        ui.show_info(
            "Configuring Claude (Anthropic):",
            [
                "• Get your API key from: https://console.anthropic.com/",
                "• Supports Claude 3 models",
                "• Requires separate embedding provider"
            ]
        )

        api_key = ui.get_api_key("Enter your Claude API key", "Claude")
        config["CLAUDE_API_KEY"] = api_key

        ui.show_info(
            "Claude requires a separate embedding provider:",
            ["Choose between Google (Gemini) or OpenAI for embeddings"]
        )

        emb_provider = ui.get_choice(
            "Select embedding provider for Claude",
            ["google", "openai"],
            default="google",
            descriptions={
                "google": "Use Google's embedding models (requires Google API key)",
                "openai": "Use OpenAI's embedding models (requires OpenAI API key)"
            }
        )
        config["CLAUDE_EMBEDDING_PROVIDER"] = emb_provider

        if emb_provider == "google" and "GOOGLE_API_KEY" not in config:
            google_key = ui.get_api_key(
                "Enter Google API key for embeddings", "Google")
            config["GOOGLE_API_KEY"] = google_key
        elif emb_provider == "openai" and "OPENAI_API_KEY" not in config:
            openai_key = ui.get_api_key(
                "Enter OpenAI API key for embeddings", "OpenAI")
            config["OPENAI_API_KEY"] = openai_key

        if ui.confirm("Configure advanced model settings?", default=False):
            gen_model = ui.get_input(
                "Claude model",
                default="claude-3-opus-20240229",
                validator=lambda x: len(x) > 0,
                error_message="Model name cannot be empty"
            )
            config["CLAUDE_GENERATION_MODEL"] = gen_model

        return config

    def test_connection(self, config: Dict[str, str]) -> Tuple[bool, str]:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=config.get("CLAUDE_API_KEY"))
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return True, "Successfully connected to Claude"
        except Exception as e:
            return False, f"Failed to connect: {str(e)}"


class LlamaConfig(ProviderConfig):
    """Llama provider configuration."""

    def configure(self, ui: UIHelper) -> Dict[str, str]:
        config = {}

        ui.show_info(
            "Configuring Llama (via Ollama):",
            [
                "• Requires Ollama installed locally",
                "• Download from: https://ollama.ai",
                "• Privacy-focused, runs on your machine",
                "• Free and open source"
            ]
        )

        base_url = ui.get_input(
            "Ollama base URL",
            default="http://localhost:11434",
            validator=validate_url,
            error_message="Please enter a valid URL"
        )
        config["OLLAMA_BASE_URL"] = base_url

        ui.show_info(
            "Available models:",
            [
                f"• {DEFAULT_LLAMA_EMBEDDING_MODEL} - Compact Llama 3.2 model",
                f"• {DEFAULT_LLAMA_GENERATION_MODEL} - Standard Llama 3.2 model",
                "• codellama - Optimized for code",
                "• mistral - Fast and efficient",
                "• Run 'ollama list' to see installed models"
            ]
        )

        gen_model = ui.get_input(
            "Generation model",
            default=DEFAULT_LLAMA_GENERATION_MODEL,
            validator=lambda x: len(x) > 0,
            error_message="Model name cannot be empty"
        )
        config["LLAMA_GENERATION_MODEL"] = gen_model

        emb_model = ui.get_input(
            "Embedding model",
            default=DEFAULT_LLAMA_EMBEDDING_MODEL,
            validator=lambda x: len(x) > 0,
            error_message="Model name cannot be empty"
        )
        config["LLAMA_EMBEDDING_MODEL"] = emb_model

        return config

    def test_connection(self, config: Dict[str, str]) -> Tuple[bool, str]:
        try:
            import requests
            base_url = config.get("OLLAMA_BASE_URL", "").rstrip('/')
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return True, "Successfully connected to Ollama"
            else:
                return False, f"Ollama returned status code: {response.status_code}"
        except Exception as e:
            return False, f"Failed to connect to Ollama: {str(e)}"


PROVIDER_CONFIGS = {
    "gemini": GeminiConfig(),
    "openai": OpenAIConfig(),
    "claude": ClaudeConfig(),
    "llama": LlamaConfig()
}


def get_provider_config(provider: str) -> ProviderConfig:
    """Get provider configuration instance."""
    # Use internal provider configs directly
    # The providers module doesn't exist, so we skip the import attempt
    return PROVIDER_CONFIGS.get(provider.lower(), GeminiConfig())


# ==================== Automation Configuration ====================

class TestAutomatorWizard:
    """TestAutomator (Automation) configuration wizard."""

    def __init__(self):
        """Initialize TestAutomator wizard."""
        self.config_data = {}

    def configure(self, ui_helper: UIHelper, skip_prompt: bool = False) -> Dict[str, str]:
        """Run TestAutomator (Automation) configuration wizard."""
        try:
            config = {}

            if not skip_prompt:
                ui_helper.show_section_header(
                    "Test Automator Wizard",
                    "Configure automator agent which generates test automation code",
                    "🚀"
                )

                self._show_automation_overview(ui_helper)

                configure_automation = ui_helper.confirm(
                    "Would you like to configure test automation?",
                    default=True
                )

                if not configure_automation:
                    ui_helper.show_info(
                        "Test Automator Agent configuration skipped.")
                    return self._get_default_config()

            config.update(self._configure_language_and_framework(ui_helper))
            config.update(self._configure_testing_environment(ui_helper))
            config.update(self._configure_output_settings(ui_helper))
            config.update(self._configure_llm_enhancement(ui_helper))

            is_valid, errors = self._validate_configuration(config)
            if not is_valid:
                ui_helper.show_error(
                    "Configuration validation failed:", errors)
                raise Exception(
                    "Test Automator Agent configuration validation failed")

            self._show_configuration_summary(ui_helper, config)

            self.config_data = config

            return config

        except (KeyboardInterrupt, typer.Abort):
            print("\n\n🛑 Test Automator configuration interrupted by user")
            ui_helper.show_info(
                "Test Automator configuration cancelled. Using defaults.")
            return self._get_default_config()

    def _show_automation_overview(self, ui_helper: UIHelper):
        """Show automation capabilities overview."""
        ui_helper.show_info(
            "Test Automator can generate automated test code in multiple languages:",
            [
                f"Python: {', '.join(SUPPORTED_FRAMEWORKS['python'])}",
                f"JavaScript: {', '.join(SUPPORTED_FRAMEWORKS['javascript'])}",
                f"TypeScript: {', '.join(SUPPORTED_FRAMEWORKS['typescript'])}",
                f"Java: {', '.join(SUPPORTED_FRAMEWORKS['java'])}"
            ]
        )

        ui_helper.show_info(
            "Features include:",
            [
                "✅ Multi-framework test generation",
                "✅ AI-powered test enhancement (optional)",
                "✅ Realistic test data generation",
                "✅ Framework-specific best practices",
                "✅ Configuration files and dependencies"
            ]
        )

    def _configure_language_and_framework(self, ui_helper: UIHelper) -> Dict[str, str]:
        """Configure programming language and testing framework."""
        config = {}

        language_descriptions = {
            "python": "Popular for data science, web apps, and automation",
            "javascript": "Ubiquitous for web development and Node.js apps",
            "typescript": "Type-safe JavaScript for larger applications",
            "java": "Enterprise applications and Android development"
        }

        language = ui_helper.get_choice(
            "Select your primary programming language",
            SUPPORTED_LANGUAGES,
            default=DEFAULT_AUTOMATION_LANGUAGE,
            descriptions=language_descriptions
        )
        config["AUTOMATION_LANGUAGE"] = language

        available_frameworks = SUPPORTED_FRAMEWORKS[language]
        framework_descriptions = self._get_framework_descriptions(language)

        ui_helper.show_info(
            f"Available frameworks for {language}:", available_frameworks)

        framework = ui_helper.get_choice(
            f"Select testing framework for {language}",
            available_frameworks,
            default=available_frameworks[0],
            descriptions=framework_descriptions
        )
        config["AUTOMATION_FRAMEWORK"] = framework

        self._show_framework_info(ui_helper, language, framework)

        return config

    def _configure_testing_environment(self, ui_helper: UIHelper) -> Dict[str, str]:
        """Configure testing environment settings."""
        config = {}

        ui_helper.show_info(
            "Configure your testing environment:",
            ["Base URL for your application",
                "Timeout settings", "Environment variables"]
        )

        base_url = ui_helper.get_input(
            "Base URL for testing (where your app runs)",
            default="http://localhost:8000",
            validator=validate_url,
            error_message="Please enter a valid URL (http://... or https://...)"
        )
        config["BASE_URL"] = base_url

        return config

    def _configure_output_settings(self, ui_helper: UIHelper) -> Dict[str, str]:
        """Configure output and file settings."""
        config = {}

        ui_helper.show_info(
            "Configure output settings:",
            ["Where to save generated tests",
                "Test case output format", "File naming conventions"]
        )

        output_dir = ui_helper.get_input(
            "Output directory for generated tests",
            default=DEFAULT_AUTOMATION_OUTPUT_DIR,
            validator=validate_directory_path,
            error_message="Please enter a valid directory path"
        )
        config["AUTOMATION_OUTPUT_DIR"] = output_dir

        # Add test case output format selection
        format_descriptions = {
            "pdf": "PDF format - Professional, widely compatible",
            "md": "Markdown format - Developer-friendly, version control friendly",
            "docx": "Word document format - Business-friendly, easy to edit"
        }

        try:
            from ...constants import SUPPORTED_TEST_OUTPUT_FORMATS, DEFAULT_TEST_OUTPUT_FORMAT

            output_format = ui_helper.get_choice(
                "Select test case output format",
                SUPPORTED_TEST_OUTPUT_FORMATS,
                default=DEFAULT_TEST_OUTPUT_FORMAT,
                descriptions=format_descriptions
            )
            config["TEST_OUTPUT_FORMAT"] = output_format
        except ImportError:
            # Fallback if constants aren't available
            config["TEST_OUTPUT_FORMAT"] = "pdf"

        return config

    def _configure_llm_enhancement(self, ui_helper: UIHelper) -> Dict[str, str]:
        """Configure LLM enhancement options - set to default true."""
        config = {}

        # Always enable AI enhancement by default (simplified workflow)
        config["AUTOMATION_ENHANCE_BY_DEFAULT"] = "true"

        return config

    def _get_framework_descriptions(self, language: str) -> Dict[str, str]:
        """Get descriptions for frameworks in the given language."""
        descriptions = {
            "python": {
                "pytest": "Modern, feature-rich testing framework (recommended)",
                "unittest": "Built-in Python testing framework",
                "playwright": "Cross-browser E2E testing with Python",
                "cucumber": "Behavior-driven development (BDD) testing"
            },
            "javascript": {
                "jest": "Popular testing framework by Facebook",
                "mocha": "Flexible testing framework with rich ecosystem",
                "playwright": "Cross-browser E2E testing",
                "cypress": "Developer-friendly E2E testing",
                "cucumber": "Behavior-driven development (BDD) testing"
            },
            "typescript": {
                "jest": "Popular testing framework with TypeScript support",
                "mocha": "Flexible testing framework with TypeScript support",
                "playwright": "Cross-browser E2E testing with TypeScript",
                "cypress": "Developer-friendly E2E testing with TypeScript",
                "cucumber": "Behavior-driven development (BDD) testing"
            },
            "java": {
                "junit5": "Modern Java testing framework (recommended)",
                "junit4": "Legacy Java testing framework",
                "testng": "Feature-rich testing framework for Java",
                "playwright": "Cross-browser E2E testing for Java",
                "karate": "API testing framework for Java",
                "cucumber": "Behavior-driven development (BDD) testing"
            }
        }

        return descriptions.get(language, {})

    def _show_framework_info(self, ui_helper: UIHelper, language: str, framework: str):
        """Show information about the selected framework."""
        framework_info = {
            ("python", "pytest"): [
                "✅ Rich plugin ecosystem",
                "✅ Fixtures and parametrization",
                "✅ Parallel test execution",
                "✅ Excellent reporting"
            ],
            ("python", "playwright"): [
                "✅ Cross-browser testing",
                "✅ Auto-waiting for elements",
                "✅ Network interception",
                "✅ Visual testing support"
            ],
            ("javascript", "jest"): [
                "✅ Snapshot testing",
                "✅ Built-in mocking",
                "✅ Code coverage reports",
                "✅ Watch mode for development"
            ],
            ("javascript", "playwright"): [
                "✅ Multi-browser support",
                "✅ Mobile testing",
                "✅ Network mocking",
                "✅ Trace viewer for debugging"
            ],
            ("java", "junit5"): [
                "✅ Modern annotations",
                "✅ Parameterized tests",
                "✅ Dynamic tests",
                "✅ Extension model"
            ],
            ("java", "karate"): [
                "✅ API testing without code",
                "✅ Built-in JSON/XML handling",
                "✅ Parallel execution",
                "✅ Comprehensive reporting"
            ]
        }

        info = framework_info.get((language, framework))
        if info:
            ui_helper.show_info(f"{framework.title()} features:", info)

    def _validate_configuration(self, config: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate TestAutomator configuration."""
        errors = []

        language = config.get("AUTOMATION_LANGUAGE", "")
        if not language:
            errors.append("Automation language is required")
        elif language not in SUPPORTED_LANGUAGES:
            errors.append(f"Unsupported language: {language}")

        framework = config.get("AUTOMATION_FRAMEWORK", "")
        if not framework:
            errors.append("Automation framework is required")
        elif language and framework not in SUPPORTED_FRAMEWORKS.get(language, []):
            errors.append(
                f"Framework '{framework}' not supported for language '{language}'")

        base_url = config.get("BASE_URL", "")
        if base_url and not validate_url(base_url):
            errors.append("Invalid base URL format")

        output_dir = config.get("AUTOMATION_OUTPUT_DIR", "")
        if output_dir and not ConfigurationValidator.validate_directory_path(output_dir):
            errors.append("Invalid output directory path")

        return len(errors) == 0, errors

    def _show_configuration_summary(self, ui_helper: UIHelper, config: Dict[str, str]):
        """Show configuration summary."""
        language = config.get("AUTOMATION_LANGUAGE", "")
        framework = config.get("AUTOMATION_FRAMEWORK", "")
        base_url = config.get("BASE_URL", "")
        output_dir = config.get("AUTOMATION_OUTPUT_DIR", "")
        output_format = config.get("TEST_OUTPUT_FORMAT", "")

        summary_items = [
            f"Language: {language}",
            f"Framework: {framework}",
            f"Base URL: {base_url}",
            f"Output directory: {output_dir}"
        ]

        if output_format:
            summary_items.append(f"Test output format: {output_format}")

        ui_helper.show_success(
            "TestAutomator (Automation) configuration completed!",
            summary_items
        )

        ui_helper.show_info(
            "Usage examples:",
            [
                "testteller generate 'Create user registration tests'",
                "testteller automate test-cases.md",
                f"testteller automate test-cases.md --language {language} --framework {framework}"
            ]
        )

    def _get_default_config(self) -> Dict[str, str]:
        """Get default automation configuration."""
        default_config = {
            "AUTOMATION_LANGUAGE": DEFAULT_AUTOMATION_LANGUAGE,
            "AUTOMATION_FRAMEWORK": DEFAULT_AUTOMATION_FRAMEWORK,
            "BASE_URL": "http://localhost:8000",
            "AUTOMATION_OUTPUT_DIR": DEFAULT_AUTOMATION_OUTPUT_DIR,
            "TEST_TIMEOUT": "30000",
            "AUTOMATION_ENHANCE_BY_DEFAULT": "true"
        }

        # Add test output format if available
        try:
            from ...constants import DEFAULT_TEST_OUTPUT_FORMAT
            default_config["TEST_OUTPUT_FORMAT"] = DEFAULT_TEST_OUTPUT_FORMAT
        except ImportError:
            default_config["TEST_OUTPUT_FORMAT"] = "pdf"

        return default_config


# ==================== Configuration Writer ====================

class ConfigurationWriter:
    """Writes configuration to various file formats."""

    def __init__(self):
        """Initialize configuration writer."""
        self.backup_enabled = True

    def write_env_file(self,
                       config: Dict[str, Any],
                       file_path: Path,
                       template_config: Optional[Dict[str, Any]] = None,
                       additional_config: Optional[Dict[str, Any]] = None) -> bool:
        """Write configuration to .env file."""
        try:
            if self.backup_enabled and file_path.exists():
                self._create_backup(file_path)

            content_lines = []

            content_lines.extend(self._get_file_header())

            if template_config:
                content_lines.append("\n# Core Configuration")
                content_lines.append("# " + "=" * 50)

                for key in template_config.keys():
                    if key in config:
                        value = config[key]
                        comment = template_config[key].get('description', '')
                        content_lines.extend(
                            self._format_config_entry(key, value, comment))
            else:
                content_lines.append("\n# Configuration")
                content_lines.append("# " + "=" * 50)
                for key, value in config.items():
                    content_lines.extend(self._format_config_entry(key, value))

            if additional_config:
                content_lines.append("\n# Additional Configuration")
                content_lines.append("# " + "=" * 50)

                added_count = 0
                for key, value in additional_config.items():
                    if key not in config:
                        content_lines.extend(
                            self._format_config_entry(key, value))
                        added_count += 1
                
                logger.info(f"Added {added_count} additional configs out of {len(additional_config)} provided")

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content_lines))

            logger.info(f"Configuration written to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to write configuration to {file_path}: {e}")
            return False

    def _create_backup(self, file_path: Path):
        """Create backup of existing file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_suffix(f".backup_{timestamp}")

            shutil.copy2(file_path, backup_path)

            logger.info(f"Backup created: {backup_path}")

        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    def _get_file_header(self) -> List[str]:
        """Get file header with metadata."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return [
            f"# TestTeller Configuration",
            f"# Generated on: {timestamp}",
            f"# This file contains configuration for TestTeller AI agent",
            f"#",
            f"# DO NOT commit this file to version control if it contains API keys",
            f"# Use .env.example as a template for sharing configuration structure",
            f"#"
        ]

    def _format_config_entry(self, key: str, value: Any, comment: str = "") -> List[str]:
        """Format a single configuration entry."""
        lines = []

        if comment:
            lines.append(f"# {comment}")

        if isinstance(value, bool):
            value_str = "true" if value else "false"
        elif isinstance(value, (int, float)):
            value_str = str(value)
        else:
            value_str = str(value)

        lines.append(f"{key}={value_str}")

        return lines

    def read_env_example(self, env_example_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Read additional configuration from .env.example file."""
        general_configs = {}
        provider_specific_configs = {}

        if not env_example_path.exists():
            logger.warning(f".env.example not found: {env_example_path}")
            return general_configs, provider_specific_configs

        try:
            with open(env_example_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()

                    if not line or line.startswith('#'):
                        continue

                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")

                        if self._is_placeholder_value(value):
                            continue

                        if self._is_provider_specific(key):
                            provider_specific_configs[key] = value
                        else:
                            general_configs[key] = value

            logger.info(
                f"Read {len(general_configs)} general and {len(provider_specific_configs)} provider-specific configs")

        except Exception as e:
            logger.error(f"Failed to read .env.example: {e}")

        return general_configs, provider_specific_configs

    def _is_placeholder_value(self, value: str) -> bool:
        """Check if value is a placeholder."""
        placeholder_patterns = [
            'your_',
            '_here',
            'replace_with',
            'enter_your',
            'api_key_placeholder'
        ]

        value_lower = value.lower()
        return any(pattern in value_lower for pattern in placeholder_patterns)

    def _is_provider_specific(self, key: str) -> bool:
        """Check if configuration key is provider-specific."""
        provider_patterns = [
            'gemini', 'google',
            'openai', 'gpt',
            'claude', 'anthropic',
            'llama', 'ollama'
        ]

        key_lower = key.lower()
        return any(pattern in key_lower for pattern in provider_patterns)


# ==================== Configuration Validator ====================

class ConfigurationValidator:
    """Validates configuration values and complete configurations."""

    def __init__(self):
        """Initialize validator."""
        self.errors = []
        self.warnings = []

    def validate_complete_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Validate a complete configuration dictionary."""
        self.errors = []
        self.warnings = []

        self._validate_llm_config(config)
        self._validate_automation_config(config)
        self._validate_paths_config(config)
        self._validate_network_config(config)
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
        elif not ConfigurationValidator.validate_api_key(api_key):
            self.errors.append("Invalid GOOGLE_API_KEY format")

    def _validate_openai_config(self, config: Dict[str, Any]):
        """Validate OpenAI configuration."""
        api_key = config.get('OPENAI_API_KEY', '')
        if not api_key:
            self.errors.append("OPENAI_API_KEY is required for OpenAI")
        elif not ConfigurationValidator.validate_api_key(api_key):
            self.errors.append("Invalid OPENAI_API_KEY format")

    def _validate_claude_config(self, config: Dict[str, Any]):
        """Validate Claude configuration."""
        api_key = config.get('CLAUDE_API_KEY', '')
        if not api_key:
            self.errors.append("CLAUDE_API_KEY is required for Claude")
        elif not ConfigurationValidator.validate_api_key(api_key):
            self.errors.append("Invalid CLAUDE_API_KEY format")

        emb_provider = config.get('CLAUDE_EMBEDDING_PROVIDER', '').lower()
        if emb_provider and emb_provider not in ['google', 'openai']:
            self.errors.append(
                "CLAUDE_EMBEDDING_PROVIDER must be 'google' or 'openai'")

        if emb_provider == 'google' and not config.get('GOOGLE_API_KEY'):
            self.errors.append(
                "GOOGLE_API_KEY is required when using Google embeddings with Claude")
        elif emb_provider == 'openai' and not config.get('OPENAI_API_KEY'):
            self.errors.append(
                "OPENAI_API_KEY is required when using OpenAI embeddings with Claude")

    def _validate_llama_config(self, config: Dict[str, Any]):
        """Validate Llama configuration."""
        base_url = config.get('OLLAMA_BASE_URL', '')
        if not base_url:
            self.errors.append("OLLAMA_BASE_URL is required for Llama")
        elif not ConfigurationValidator.validate_url(base_url):
            self.errors.append("Invalid OLLAMA_BASE_URL format")

    def _validate_automation_config(self, config: Dict[str, Any]):
        """Validate automation configuration."""
        language = config.get('AUTOMATION_LANGUAGE', '').lower()
        if language and language not in SUPPORTED_LANGUAGES:
            self.errors.append(f"Unsupported automation language: {language}")

        framework = config.get('AUTOMATION_FRAMEWORK', '').lower()
        if language and framework:
            supported_frameworks = SUPPORTED_FRAMEWORKS.get(language, [])
            if framework not in supported_frameworks:
                self.errors.append(
                    f"Framework '{framework}' not supported for language '{language}'")

        base_url = config.get('BASE_URL', '')
        if base_url and not validate_url(base_url):
            self.errors.append("Invalid BASE_URL format")

    def _validate_paths_config(self, config: Dict[str, Any]):
        """Validate file paths and directories."""
        output_dir = config.get('AUTOMATION_OUTPUT_DIR', '')
        if output_dir and not ConfigurationValidator.validate_directory_path(output_dir):
            self.errors.append("Invalid AUTOMATION_OUTPUT_DIR path")

    def _validate_network_config(self, config: Dict[str, Any]):
        """Validate network-related configuration."""
        pass

    def _check_config_conflicts(self, config: Dict[str, Any]):
        """Check for configuration conflicts."""
        provider = config.get('LLM_PROVIDER', '').lower()

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
            self.warnings.append(
                f"Multiple provider credentials found. Using {provider}, ignoring: {', '.join(other_providers)}")

    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format."""
        if not api_key or not isinstance(api_key, str):
            return False

        api_key = api_key.strip()

        if len(api_key) < 10:
            return False

        placeholder_patterns = [
            'your_api_key_here',
            'replace_with_your_key',
            'api_key_placeholder',
            'enter_your_key'
        ]

        api_key_lower = api_key.lower()
        if any(pattern in api_key_lower for pattern in placeholder_patterns):
            return False

        if not re.match(r'^[a-zA-Z0-9\-_\.]+$', api_key):
            return False

        return True

    @staticmethod
    def validate_url(url: str) -> bool:
        """Basic URL validation."""
        return url.startswith(('http://', 'https://')) and len(url) > 10

    @staticmethod
    def validate_port(port_str: str) -> bool:
        """Validate port number."""
        try:
            port = int(port_str)
            return 1 <= port <= 65535
        except ValueError:
            return False

    @staticmethod
    def validate_directory_path(path_str: str) -> bool:
        """Validate directory path (can be created)."""
        try:
            path = Path(path_str)
            return len(str(path)) > 0 and not str(path).startswith('/')
        except Exception:
            return False


# ==================== Main Configuration Wizard ====================

class ConfigurationWizard:
    """Main configuration wizard for TestTeller."""

    def __init__(self, ui_mode: UIMode = UIMode.CLI):
        """Initialize configuration wizard."""
        self.ui = UIHelper(ui_mode)
        # Import the correct ConfigurationWriter from writers.py
        from .writers import ConfigurationWriter as ProperConfigurationWriter
        self.writer = ProperConfigurationWriter()
        self.validator = ConfigurationValidator()
        self.config_data = {}
        self.env_template = self._get_env_template()

    def run(self, env_path: Optional[Path] = None) -> bool:
        """Run the complete configuration wizard."""
        try:
            if not env_path:
                env_path = Path.cwd() / ".env"

            self.ui.set_progress(0, 3)

            if not self._check_prerequisites(env_path):
                return False

            self.ui.show_step_progress("LLM Provider Configuration")
            llm_config = self._configure_llm_provider()
            self.config_data.update(llm_config)

            self.ui.show_step_progress("TestAutomator (Automation) Setup")
            automation_config = self._configure_automation()
            self.config_data.update(automation_config)

            self.ui.show_step_progress("Validation and Save")
            if not self._finalize_configuration(env_path):
                return False

            self._show_completion(env_path)

            return True

        except KeyboardInterrupt:
            print("\n\n🛑 Configuration interrupted by user (Ctrl+C)")
            self.ui.show_info(
                "Configuration cancelled gracefully. No changes were made.")
            print(
                "💡 You can run the configuration wizard again anytime with: testteller configure")
            return False
        except ConfigurationCancelledException:
            self.ui.show_info("💡 Configuration cancelled by user.")
            return False
        except Exception as e:
            logger.error(f"Configuration wizard failed: {e}")
            self.ui.show_error(f"Configuration failed: {e}")
            return False

    def run_quick_setup(self, provider: str, env_path: Optional[Path] = None) -> bool:
        """Run quick setup for a specific provider."""
        try:
            if not env_path:
                env_path = Path.cwd() / ".env"

            self.ui.show_header("TestTeller Quick Setup")

            provider_config = get_provider_config(provider)
            llm_config = provider_config.configure(self.ui)
            llm_config["LLM_PROVIDER"] = provider

            automation_wizard = TestAutomatorWizard()
            automation_config = automation_wizard._get_default_config()

            self.config_data.update(llm_config)
            self.config_data.update(automation_config)

            return self._finalize_configuration(env_path)

        except KeyboardInterrupt:
            print("\n\n🛑 Quick setup interrupted by user (Ctrl+C)")
            self.ui.show_info(
                "Quick setup cancelled gracefully. No changes were made.")
            print("💡 You can run the quick setup again anytime.")
            return False
        except Exception as e:
            logger.error(f"Quick setup failed: {e}")
            self.ui.show_error(f"Quick setup failed: {e}")
            return False

    def _check_prerequisites(self, env_path: Path) -> bool:
        """Check prerequisites and handle existing configuration."""
        self.ui.show_header()

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

        self.ui.show_section_header(
            "🤖 LLM Provider Selection",
            "Choose your AI/LLM provider for test case generation"
        )

        provider_descriptions = {
            "Gemini": "Google's powerful multimodal AI (recommended)",
            "OpenAI": "Industry-leading GPT models",
            "Claude": "Advanced reasoning by Anthropic",
            "Llama": "Privacy-focused local deployment"
        }

        provider = self.ui.get_choice(
            "Select your LLM provider",
            SUPPORTED_LLM_PROVIDERS,
            default=DEFAULT_LLM_PROVIDER,
            descriptions=provider_descriptions
        )

        config["LLM_PROVIDER"] = provider

        try:
            provider_config = get_provider_config(provider)
            provider_settings = provider_config.configure(self.ui)
            config.update(provider_settings)

            if self.ui.confirm("Test connection to verify configuration?", default=True):
                self.ui.show_info("Testing connection...")
                success, message = provider_config.test_connection(config)

                if success:
                    self.ui.show_success(
                        "Connection test successful!", [message])
                else:
                    self.ui.show_warning("Connection test failed", [message])
                    if not self.ui.confirm("Continue with configuration anyway?", default=True):
                        raise ConfigurationCancelledException(
                            "Configuration cancelled by user")

        except ConfigurationCancelledException:
            # Re-raise without logging - this is user cancellation, not an error
            raise
        except Exception as e:
            logger.error(f"Provider configuration failed: {e}")
            raise

        return config

    def _configure_test_output_format(self) -> Dict[str, str]:
        """Configure preferred test case output format."""
        config = {}

        self.ui.show_section_header(
            "📄 Test Case Output Format",
            "Choose your preferred format for generated test cases"
        )

        self.ui.show_info(
            "Test cases can be generated in multiple formats:",
            [
                "• pdf - Professional documents with tables and formatting",
                "• md - Markdown, a developer-friendly format with table support",
                "• docx - Microsoft Word format for business collaboration"
            ]
        )

        # Make this optional by allowing user to press enter for default
        format_input = self.ui.get_input(
            f"Preferred test case output format",
            default=DEFAULT_TEST_OUTPUT_FORMAT,
            validator=lambda x: x == "" or x.lower(
            ) in [f.lower() for f in SUPPORTED_TEST_OUTPUT_FORMATS],
            error_message=f"Please enter one of: {', '.join(SUPPORTED_TEST_OUTPUT_FORMATS)} or press Enter for default"
        )

        if format_input.strip():
            # User provided a format
            output_format = format_input.lower()
        else:
            # Use default
            output_format = DEFAULT_TEST_OUTPUT_FORMAT

        config["TEST_OUTPUT_FORMAT"] = output_format

        if format_input.strip():
            self.ui.show_success(f"Test output format set to: {output_format}")
        else:
            self.ui.show_info(
                f"Using default test output format: {output_format}")

        return config

    def _configure_chromadb(self) -> Dict[str, str]:
        """Configure ChromaDB settings."""
        config = {}

        self.ui.show_section_header(
            "🗄️ ChromaDB Configuration",
            "Configure vector database for context storage and retrieval"
        )

        self.ui.show_info(
            "ChromaDB stores and retrieves context for better test generation:",
            [
                "• Local - Runs on your machine (recommended for privacy)",
                "• Remote - Connect to external ChromaDB server"
            ]
        )

        use_remote = self.ui.confirm(
            "Use remote ChromaDB server? (No = Local ChromaDB)",
            default=False
        )

        config["CHROMA_DB_USE_REMOTE"] = str(use_remote).lower()

        if use_remote:
            self.ui.show_info(
                "Configure remote ChromaDB connection:",
                [f"• Default host: {DEFAULT_CHROMA_HOST}", f"• Default port: {DEFAULT_CHROMA_PORT}"]
            )

            host = self.ui.get_input(
                "ChromaDB host",
                default=DEFAULT_CHROMA_HOST,
                validator=lambda x: len(x.strip()) > 0,
                error_message="Host cannot be empty"
            )
            config["CHROMA_DB_HOST"] = host

            port = self.ui.get_input(
                "ChromaDB port",
                default=str(DEFAULT_CHROMA_PORT),
                validator=validate_port,
                error_message="Please enter a valid port number (1-65535)"
            )
            config["CHROMA_DB_PORT"] = port

            self.ui.show_success(
                f"Remote ChromaDB configured: {host}:{port}"
            )
        else:
            # Set default values for local ChromaDB
            config["CHROMA_DB_HOST"] = DEFAULT_CHROMA_HOST
            config["CHROMA_DB_PORT"] = str(DEFAULT_CHROMA_PORT)

            persist_dir = self.ui.get_input(
                "ChromaDB data directory (local storage)",
                default=DEFAULT_CHROMA_PERSIST_DIRECTORY,
                validator=validate_directory_path,
                error_message="Please enter a valid directory path"
            )
            config["CHROMA_DB_PERSIST_DIRECTORY"] = persist_dir

            self.ui.show_success(
                f"Local ChromaDB configured: {persist_dir}"
            )

        return config

    def _configure_automation(self) -> Dict[str, str]:
        """Configure TestAutomator automation with user choice."""
        try:
            self.ui.show_section_header(
                "🚀 Test Automator Agent Configuration",
                "Configure automated test generation for multiple frameworks"
            )

            self.ui.show_info(
                "Test Automator Agent can generate automation code in multiple languages:",
                [
                    "• Python (pytest, unittest, playwright)",
                    "• JavaScript (jest, mocha, playwright)",
                    "• TypeScript (jest, playwright)",
                    "• Java (junit4, junit5, testng)"
                ]
            )

            configure_automation = self.ui.confirm(
                "Would you like to configure Test Automator Agent?",
                default=True
            )

            # Get automation configuration
            config = {}
            try:
                if configure_automation:
                    self.ui.show_info(
                        "Setting up Test Automator Agent configuration...")
                    automation_wizard = TestAutomatorWizard()
                    config = automation_wizard.configure(self.ui)
                else:
                    self.ui.show_info(
                        "Using default Test Automator Agent configuration from .env.example")
                    automation_wizard = TestAutomatorWizard()
                    config = automation_wizard._get_default_config()

                    self.ui.show_info(
                        "Default automation settings applied:",
                        [f"• {key}: {value}" for key,
                            value in config.items() if value]
                    )

                    self.ui.show_info(
                        "💡 You can configure Test Automator Agent later using:",
                        ["testteller configure --automator-agent"]
                    )
            except Exception as e:
                logger.warning(f"Automation configuration failed: {e}, using minimal defaults")
                config = {
                    "AUTOMATION_LANGUAGE": DEFAULT_AUTOMATION_LANGUAGE,
                    "AUTOMATION_FRAMEWORK": DEFAULT_AUTOMATION_FRAMEWORK,
                    "BASE_URL": DEFAULT_BASE_URLS.get(DEFAULT_AUTOMATION_LANGUAGE, "http://localhost:8000")
                }

            # Add test case output format configuration
            try:
                config.update(self._configure_test_output_format())
            except Exception as e:
                logger.warning(f"Test output format configuration failed: {e}, using defaults")
                config["TEST_OUTPUT_FORMAT"] = DEFAULT_TEST_OUTPUT_FORMAT

            # Add ChromaDB configuration
            try:
                config.update(self._configure_chromadb())
            except Exception as e:
                logger.warning(f"ChromaDB configuration failed: {e}, using defaults")
                config.update({
                    "CHROMA_DB_USE_REMOTE": str(DEFAULT_CHROMA_USE_REMOTE).lower(),
                    "CHROMA_DB_HOST": DEFAULT_CHROMA_HOST,
                    "CHROMA_DB_PORT": str(DEFAULT_CHROMA_PORT),
                    "CHROMA_DB_PERSIST_DIRECTORY": DEFAULT_CHROMA_PERSIST_DIRECTORY
                })

            return config

        except Exception as e:
            logger.error(f"Test Automator Agent configuration failed: {e}")
            self.ui.show_warning(
                "Automation configuration failed, using defaults",
                ["You can reconfigure later with 'testteller configure --automator-agent'"]
            )
            automation_wizard = TestAutomatorWizard()
            return automation_wizard._get_default_config()

    def _finalize_configuration(self, env_path: Path) -> bool:
        """Validate and write final configuration."""
        try:
            self.ui.show_info("Validating configuration...")
            is_valid, errors, warnings = self.validator.validate_complete_configuration(
                self.config_data)

            if warnings:
                self.ui.show_warning("Configuration warnings:", warnings)

            if not is_valid:
                self.ui.show_error("Configuration validation failed:", errors)
                return False

            self.ui.show_configuration_summary(self.config_data)

            if not self.ui.confirm("Save this configuration?", default=True):
                self.ui.show_info("💡 Configuration not saved.")
                return False

            # Read additional configuration from .env.example
            env_example_path = env_path.parent / ".env.example"
            additional_config = {}

            if env_example_path.exists():
                self.ui.show_info(
                    "Loading additional default configurations from .env.example...")
                general_configs, provider_specific_configs = self.writer.read_env_example(
                    env_example_path)
                
                

                # Merge general configs and relevant provider configs
                additional_config.update(general_configs)

                # Add relevant provider-specific configs
                current_provider = self.config_data.get("LLM_PROVIDER", "").lower()
                
                if current_provider:
                    # Provider-specific mapping for API keys and models
                    provider_key_mapping = {
                        'gemini': ['GOOGLE_API_KEY', 'GEMINI_EMBEDDING_MODEL', 'GEMINI_GENERATION_MODEL'],
                        'openai': ['OPENAI_API_KEY', 'OPENAI_EMBEDDING_MODEL', 'OPENAI_GENERATION_MODEL'],
                        'claude': ['CLAUDE_API_KEY', 'CLAUDE_GENERATION_MODEL', 'OPENAI_API_KEY'],  # Claude needs OpenAI for embeddings
                        'llama': ['LLAMA_EMBEDDING_MODEL', 'LLAMA_GENERATION_MODEL', 'OLLAMA_BASE_URL']
                    }
                    
                    # Add configs for current provider
                    if current_provider in provider_key_mapping:
                        for key in provider_key_mapping[current_provider]:
                            if key in provider_specific_configs:
                                additional_config[key] = provider_specific_configs[key]
                    
                    # Include all provider configs (model names and API key placeholders for all providers)
                    # Users should see all available options even if they're not using them now
                    for k, v in provider_specific_configs.items():
                        if k not in additional_config:
                            additional_config[k] = v

                # Ensure ALL general configs are included if not already set
                for config_key, config_value in general_configs.items():
                    if config_key not in self.config_data and config_key not in additional_config:
                        additional_config[config_key] = config_value
                        
                # Ensure critical configs are always present with defaults
                # Note: Most values should come from .env.example via the general_configs
                # These are only used if the value is missing from .env.example
                critical_defaults = {
                    'TEST_OUTPUT_FORMAT': DEFAULT_TEST_OUTPUT_FORMAT,
                    'CHROMA_DB_HOST': DEFAULT_CHROMA_HOST,
                    'CHROMA_DB_PORT': str(DEFAULT_CHROMA_PORT), 
                    'CHROMA_DB_USE_REMOTE': str(DEFAULT_CHROMA_USE_REMOTE).lower(),
                    'CHROMA_DB_PERSIST_DIRECTORY': DEFAULT_CHROMA_PERSIST_DIRECTORY,
                    'AUTOMATION_OUTPUT_DIR': DEFAULT_AUTOMATION_OUTPUT_DIR
                    # AUTOMATION_ENHANCE_BY_DEFAULT should come from .env.example
                }
                
                for config_key, default_value in critical_defaults.items():
                    if config_key not in self.config_data and config_key not in additional_config:
                        additional_config[config_key] = default_value

                
                if additional_config:
                    self.ui.show_info(
                        f"Including {len(additional_config)} additional default settings:",
                        [f"• {key}={value}" for key, value in list(
                            additional_config.items())[:5]]
                    )
                    if len(additional_config) > 5:
                        self.ui.show_info(
                            f"  ... and {len(additional_config) - 5} more settings")
                else:
                    self.ui.show_info("No additional configurations found to copy from .env.example")

            self.ui.show_info(f"Writing configuration to {env_path}...")

            # Ensure AUTOMATION_ENHANCE_BY_DEFAULT is always set to true
            if "AUTOMATION_ENHANCE_BY_DEFAULT" not in self.config_data:
                self.config_data["AUTOMATION_ENHANCE_BY_DEFAULT"] = "true"

            
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
        automation_enabled = self.config_data.get(
            "AUTOMATION_LANGUAGE", "") != ""

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
            },
            "TEST_OUTPUT_FORMAT": {
                "description": "Output format for test cases (md, pdf, docx)",
                "required": False
            },
            "CHROMA_DB_HOST": {
                "description": "ChromaDB host address",
                "required": False
            },
            "CHROMA_DB_PORT": {
                "description": "ChromaDB port number",
                "required": False
            },
            "CHROMA_DB_USE_REMOTE": {
                "description": "Use remote ChromaDB instance",
                "required": False
            },
            "CHROMA_DB_PERSIST_DIRECTORY": {
                "description": "ChromaDB persistence directory",
                "required": False
            },
            "AUTOMATION_OUTPUT_DIR": {
                "description": "Output directory for automation files",
                "required": False
            },
            "AUTOMATION_ENHANCE_BY_DEFAULT": {
                "description": "Enable automation enhancement by default",
                "required": False
            }
        }


# ==================== Convenience Functions ====================

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

        env_path = Path.cwd() / ".env"
        writer = ConfigurationWriter()

        existing_config = {}
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        existing_config[key.strip()] = value.strip().strip(
                            '"').strip("'")

        existing_config.update(config)

        success = writer.write_env_file(existing_config, env_path)

        if success:
            ui_helper.show_success(
                "Automator Agent configured successfully!")
            return True
        else:
            ui_helper.show_error("Failed to save automation configuration")
            return False

    except KeyboardInterrupt:
        print("\n\n🛑 Automation setup interrupted by user (Ctrl+C)")
        print("⚡ Automation setup cancelled gracefully. No changes were made.")
        print("💡 You can run the automation setup again anytime with: testteller configure --automator-agent")
        return False
    except Exception as e:
        logger.error(f"Automation-only setup failed: {e}")
        return False
