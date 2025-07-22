"""
TestTeller Configuration System

A modular configuration system for setting up TestTeller with:
- LLM Provider configuration (Gemini, OpenAI, Claude, Llama)
- TestAutomator (Automation) Wizard for test generation setup
- Additional settings and validation

Usage:
    from testteller.config import ConfigurationWizard
    
    wizard = ConfigurationWizard()
    success = wizard.run()
"""

from .config_wizard import (
    ConfigurationWizard,
    UIHelper,
    UIMode,
    ConfigurationWriter,
    ConfigurationValidator,
    run_full_wizard,
    run_provider_only_setup,
    run_automation_only_setup
)

# Import TestAutomatorWizard from new location
try:
    from ...automator_agent.config import TestAutomatorWizard
except ImportError:
    TestAutomatorWizard = None

# Export validation functions for backward compatibility
validate_api_key = ConfigurationValidator.validate_api_key
validate_url = ConfigurationValidator.validate_url
validate_port = ConfigurationValidator.validate_port
validate_directory_path = ConfigurationValidator.validate_directory_path

# Import legacy config classes for backward compatibility using safe import
try:
    from testteller.config import (
        settings,
        AppSettings,
        ApiKeysSettings, 
        ChromaDBSettings,
        LLMSettings,
        ProcessingSettings,
        OutputSettings,
        LoggingSettings,
        ApiRetrySettings,
        CodeLoaderSettings
    )
except ImportError:
    # Fallback to None if import fails
    settings = None
    AppSettings = None
    ApiKeysSettings = None
    ChromaDBSettings = None
    LLMSettings = None
    ProcessingSettings = None
    OutputSettings = None
    LoggingSettings = None
    ApiRetrySettings = None
    CodeLoaderSettings = None

__all__ = [
    'ConfigurationWizard',
    'UIHelper', 
    'UIMode',
    'ConfigurationWriter',
    'ConfigurationValidator',
    'TestAutomatorWizard',
    'run_full_wizard',
    'run_provider_only_setup',
    'run_automation_only_setup',
    'validate_api_key',
    'validate_url',
    'validate_port',
    'validate_directory_path',
    'settings',
    'AppSettings',
    'ApiKeysSettings', 
    'ChromaDBSettings',
    'LLMSettings',
    'ProcessingSettings',
    'OutputSettings',
    'LoggingSettings',
    'ApiRetrySettings',
    'CodeLoaderSettings'
]