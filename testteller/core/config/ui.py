"""
User Interface utilities for configuration wizard.

Provides consistent UI patterns across CLI and TUI modes.
"""

import logging
import typer
from enum import Enum
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class UIMode(Enum):
    """UI interaction modes"""
    CLI = "cli"
    TUI = "tui"
    AUTO = "auto"


class UIHelper:
    """User interface helper for configuration wizard."""
    
    def __init__(self, mode: UIMode = UIMode.CLI):
        """
        Initialize UI helper.
        
        Args:
            mode: UI mode (CLI, TUI, or AUTO)
        """
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
        print(f"🔧 {title}")
        print("=" * 60)
        print("Setting up your AI-powered test automation platform")
        if self._total_steps > 0:
            print(f"Progress: Step {self._step_counter}/{self._total_steps}")
        print()
    
    def show_section_header(self, title: str, description: str = "", icon: str = "🔧"):
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
            print(f"\n📍 Step {self._step_counter}/{self._total_steps}: {step_name} ({progress:.0f}%)")
        else:
            print(f"\n📍 {step_name}")
    
    def get_api_key(self, prompt: str, provider: str = "") -> str:
        """
        Get API key with validation and security.
        
        Args:
            prompt: Input prompt text
            provider: Provider name for context
            
        Returns:
            API key string
        """
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
                
                if api_key and len(api_key.strip()) > 10:  # Basic validation
                    return api_key.strip()
                else:
                    print("❌ API key appears to be too short. Please check and try again.")
                    
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
        """
        Get user choice from a list of options.
        
        Args:
            prompt: Input prompt text
            choices: List of available choices
            default: Default choice if user presses enter
            descriptions: Optional descriptions for each choice
            
        Returns:
            Selected choice
        """
        print(f"\n{prompt}")
        
        # Show choices with descriptions
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
                    
                    if not user_input:  # User pressed enter
                        return default
                else:
                    user_input = typer.prompt(f"\nEnter choice [1-{len(choices)}]")
                
                # Try parsing as number
                try:
                    choice_num = int(user_input)
                    if 1 <= choice_num <= len(choices):
                        return choices[choice_num - 1]
                    else:
                        print(f"❌ Please enter a number between 1 and {len(choices)}")
                        continue
                except ValueError:
                    # Try matching as string
                    user_input_lower = user_input.lower()
                    matches = [choice for choice in choices if choice.lower().startswith(user_input_lower)]
                    
                    if len(matches) == 1:
                        return matches[0]
                    elif len(matches) > 1:
                        print(f"❌ Ambiguous choice. Matches: {', '.join(matches)}")
                    else:
                        print(f"❌ Invalid choice. Please select from: {', '.join(choices)}")
                        
            except typer.Abort:
                print("\n❌ Configuration cancelled by user.")
                raise typer.Exit(1)
    
    def get_input(self, 
                  prompt: str, 
                  default: Optional[str] = None,
                  validator: Optional[Callable[[str], bool]] = None,
                  error_message: str = "Invalid input") -> str:
        """
        Get text input with optional validation.
        
        Args:
            prompt: Input prompt text
            default: Default value
            validator: Optional validation function
            error_message: Error message for validation failure
            
        Returns:
            User input string
        """
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
        """
        Get yes/no confirmation from user.
        
        Args:
            prompt: Confirmation prompt
            default: Default value
            
        Returns:
            True if confirmed, False otherwise
        """
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
        
        # Group configurations by category
        llm_configs = {}
        automation_configs = {}
        other_configs = {}
        
        for key, value in config.items():
            if any(provider in key.lower() for provider in ['gemini', 'openai', 'claude', 'llama', 'ollama']):
                if 'api_key' in key.lower():
                    llm_configs[key] = "***" + value[-4:] if len(value) > 4 else "***"
                else:
                    llm_configs[key] = value
            elif 'automation' in key.lower() or key in ['BASE_URL', 'AUTOMATION_OUTPUT_DIR']:
                automation_configs[key] = value
            else:
                other_configs[key] = value
        
        # Show LLM configuration
        if llm_configs:
            print("\n🤖 LLM Provider:")
            for key, value in llm_configs.items():
                print(f"   • {key}: {value}")
        
        # Show TestAutomator configuration  
        if automation_configs:
            print("\n🧪 TestAutomator (Automation):")
            for key, value in automation_configs.items():
                print(f"   • {key}: {value}")
        
        # Show other configuration
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
        print(f"✅ TestAutomator Automation: {'Enabled' if automation_enabled else 'Disabled'}")
        
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
    
    def prompt_for_file_path(self, prompt: str, must_exist: bool = False) -> Path:
        """
        Prompt for file path with validation.
        
        Args:
            prompt: Input prompt
            must_exist: Whether file must exist
            
        Returns:
            Path object
        """
        while True:
            try:
                path_str = typer.prompt(f"\n{prompt}")
                path = Path(path_str).expanduser().resolve()
                
                if must_exist and not path.exists():
                    print(f"❌ File not found: {path}")
                    continue
                
                return path
                
            except typer.Abort:
                print("\n❌ Configuration cancelled by user.")
                raise typer.Exit(1)
            except Exception as e:
                print(f"❌ Invalid path: {e}")


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
        # Check if it's a reasonable path
        return len(str(path)) > 0 and not str(path).startswith('/')
    except Exception:
        return False