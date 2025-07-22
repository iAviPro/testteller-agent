"""
Configuration file writers.

Handles writing configuration to .env files and other formats.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


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
        """
        Write configuration to .env file.
        
        Args:
            config: Main configuration dictionary
            file_path: Path to .env file
            template_config: Template configuration for ordering
            additional_config: Additional configuration from .env.example
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup if file exists
            if self.backup_enabled and file_path.exists():
                self._create_backup(file_path)
            
            # Prepare file content
            content_lines = []
            
            # Add header comment
            content_lines.extend(self._get_file_header())
            
            # Write primary configuration (from template order)
            if template_config:
                content_lines.append("\n# Core Configuration")
                content_lines.append("# " + "=" * 50)
                
                for key in template_config.keys():
                    if key in config:
                        value = config[key]
                        comment = template_config[key].get('description', '')
                        content_lines.extend(self._format_config_entry(key, value, comment))
                
                # Write remaining main config items not in template
                for key, value in config.items():
                    if key not in template_config:
                        content_lines.extend(self._format_config_entry(key, value))
            else:
                # Write all main config if no template
                content_lines.append("\n# Configuration")
                content_lines.append("# " + "=" * 50)
                for key, value in config.items():
                    content_lines.extend(self._format_config_entry(key, value))
            
            # Write additional configuration
            if additional_config:
                content_lines.append("\n# Additional Configuration")
                content_lines.append("# " + "=" * 50)
                
                for key, value in additional_config.items():
                    if key not in config:  # Don't duplicate
                        content_lines.extend(self._format_config_entry(key, value))
            
            # Write to file
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
            
            # Copy existing file to backup
            import shutil
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
        
        # Add comment if provided
        if comment:
            lines.append(f"# {comment}")
        
        # Format value
        if isinstance(value, bool):
            value_str = "true" if value else "false"
        elif isinstance(value, (int, float)):
            value_str = str(value)
        else:
            value_str = str(value)
        
        # Add the configuration line
        lines.append(f"{key}={value_str}")
        
        return lines
    
    def validate_before_write(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate configuration before writing.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        from .validators import ConfigurationValidator
        
        validator = ConfigurationValidator()
        is_valid, errors, warnings = validator.validate_complete_configuration(config)
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
        
        return is_valid, errors
    
    def read_env_example(self, env_example_path: Path) -> tuple[Dict[str, str], Dict[str, str]]:
        """
        Read additional configuration from .env.example file.
        
        Args:
            env_example_path: Path to .env.example file
            
        Returns:
            Tuple of (general_configs, provider_specific_configs)
        """
        general_configs = {}
        provider_specific_configs = {}
        
        if not env_example_path.exists():
            logger.warning(f".env.example not found: {env_example_path}")
            return general_configs, provider_specific_configs
        
        try:
            with open(env_example_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key=value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        
                        # Include placeholder values for API keys and important configs
                        # Users need to see what to configure
                        if self._is_placeholder_value(value):
                            # Only skip truly meaningless placeholders, keep API key placeholders
                            if not self._is_important_placeholder(key, value):
                                continue
                        
                        # Categorize as provider-specific or general
                        if self._is_provider_specific(key):
                            provider_specific_configs[key] = value
                        else:
                            general_configs[key] = value
            
            logger.info(f"Read {len(general_configs)} general and {len(provider_specific_configs)} provider-specific configs")
            
        except Exception as e:
            logger.error(f"Failed to read .env.example: {e}")
        
        return general_configs, provider_specific_configs
    
    def _is_placeholder_value(self, value: str) -> bool:
        """Check if value is a placeholder."""
        # Only filter out obvious placeholder values for API keys and tokens
        # Keep structural configs like model names, URLs, etc.
        placeholder_patterns = [
            'your_',
            '_here',
            'replace_with',
            'enter_your',
            'api_key_placeholder'
        ]
        
        value_lower = value.lower()
        
        # Special handling for specific configs we want to keep
        if any(keep_pattern in value_lower for keep_pattern in [
            'text-embedding',  # Keep model names
            'gemini-',         # Keep Gemini model names
            'claude-',         # Keep Claude model names
            'llama',           # Keep Llama model names
            'gpt-',            # Keep GPT model names
            'localhost'        # Keep localhost URLs (they're not placeholders)
        ]):
            return False
            
        return any(pattern in value_lower for pattern in placeholder_patterns)
    
    def _is_important_placeholder(self, key: str, value: str) -> bool:
        """Check if a placeholder value represents an important config that should be included."""
        important_keys = [
            'api_key',
            'token',
            'secret',
            'password',
            'auth'
        ]
        
        key_lower = key.lower()
        return any(important_key in key_lower for important_key in important_keys)
    
    def _is_provider_specific(self, key: str) -> bool:
        """Check if configuration key is provider-specific."""
        # Keys that should always be treated as general configs
        always_general = [
            'github_token',
            'log_level',
            'log_format',
            'base_url'
        ]
        
        key_lower = key.lower()
        
        # Check if it's in the always general list
        if any(general_key in key_lower for general_key in always_general):
            return False
        
        provider_patterns = [
            'gemini', 'google',
            'openai', 'gpt',
            'claude', 'anthropic',
            'llama', 'ollama'
        ]
        
        return any(pattern in key_lower for pattern in provider_patterns)
    
    def export_configuration_summary(self, 
                                   config: Dict[str, Any], 
                                   output_path: Path,
                                   include_secrets: bool = False) -> bool:
        """
        Export configuration summary to a file.
        
        Args:
            config: Configuration dictionary
            output_path: Output file path
            include_secrets: Whether to include secret values
            
        Returns:
            True if successful
        """
        try:
            summary_lines = []
            
            # Header
            summary_lines.extend([
                "TestTeller Configuration Summary",
                "=" * 40,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ""
            ])
            
            # LLM Provider section
            provider = config.get('LLM_PROVIDER', 'Not configured')
            summary_lines.extend([
                "LLM Provider Configuration:",
                f"  Provider: {provider}",
                ""
            ])
            
            # Provider-specific details
            if provider.lower() == 'gemini':
                self._add_provider_summary(summary_lines, config, 'GOOGLE_API_KEY', 'GEMINI_', include_secrets)
            elif provider.lower() == 'openai':
                self._add_provider_summary(summary_lines, config, 'OPENAI_API_KEY', 'OPENAI_', include_secrets)
            elif provider.lower() == 'claude':
                self._add_provider_summary(summary_lines, config, 'CLAUDE_API_KEY', 'CLAUDE_', include_secrets)
            elif provider.lower() == 'llama':
                self._add_provider_summary(summary_lines, config, 'OLLAMA_BASE_URL', 'LLAMA_', include_secrets)
            
            # Automation section
            summary_lines.extend([
                "Automator Agent Configuration:",
                f"  Language: {config.get('AUTOMATION_LANGUAGE', 'Not configured')}",
                f"  Framework: {config.get('AUTOMATION_FRAMEWORK', 'Not configured')}",
                f"  Base URL: {config.get('BASE_URL', 'Not configured')}",
                f"  Output Directory: {config.get('AUTOMATION_OUTPUT_DIR', 'Not configured')}",
                ""
            ])
            
            # Other configuration
            other_configs = {k: v for k, v in config.items() 
                           if not any(pattern in k.upper() for pattern in 
                                    ['API_KEY', 'AUTOMATION_', 'BASE_URL', 'LLM_PROVIDER'])}
            
            if other_configs:
                summary_lines.extend(["Other Configuration:"])
                for key, value in other_configs.items():
                    summary_lines.append(f"  {key}: {value}")
            
            # Write summary
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))
            
            logger.info(f"Configuration summary exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration summary: {e}")
            return False
    
    def _add_provider_summary(self, 
                            lines: List[str], 
                            config: Dict[str, Any], 
                            api_key_name: str, 
                            prefix: str,
                            include_secrets: bool):
        """Add provider-specific summary to lines."""
        api_key = config.get(api_key_name, '')
        if api_key:
            if include_secrets:
                lines.append(f"  API Key: {api_key}")
            else:
                lines.append(f"  API Key: ***{api_key[-4:] if len(api_key) > 4 else '***'}")
        
        # Add model configurations
        for key, value in config.items():
            if key.startswith(prefix) and key != api_key_name:
                lines.append(f"  {key}: {value}")
        
        lines.append("")