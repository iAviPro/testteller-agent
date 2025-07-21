"""
Test configuration module imports and basic functionality.
"""

import pytest


def test_config_imports():
    """Test that configuration modules can be imported without errors."""
    from testteller.core.config import (
        ConfigurationWizard,
        UIHelper,
        UIMode,
        ConfigurationWriter,
        ConfigurationValidator,
    )

    # Test that classes can be instantiated
    assert ConfigurationWizard is not None
    assert UIHelper is not None
    assert UIMode is not None
    assert ConfigurationWriter is not None
    assert ConfigurationValidator is not None


def test_automation_config_imports():
    """Test that automation configuration can be imported."""
    from testteller.automator_agent.config import TestAutomatorWizard

    assert TestAutomatorWizard is not None

    # Test instantiation
    wizard = TestAutomatorWizard()
    assert wizard is not None
    assert hasattr(wizard, 'configure')


def test_validation_functions():
    """Test that validation functions are accessible."""
    from testteller.core.config import (
        validate_api_key,
        validate_url,
        validate_port,
        validate_directory_path
    )

    # Test basic validation functions
    assert validate_api_key("sk-1234567890abcdef1234567890abcdef") is True
    assert validate_api_key("") is False

    assert validate_url("http://localhost:8000") is True
    assert validate_url("not-a-url") is False

    assert validate_port(8000) is True
    assert validate_port(0) is False
    assert validate_port(65536) is False

    assert validate_directory_path(".") is True
    assert validate_directory_path("/nonexistent/path/that/does/not/exist/12345") is False


def test_configuration_writer_basic():
    """Test ConfigurationWriter basic functionality."""
    from testteller.core.config import ConfigurationWriter

    writer = ConfigurationWriter()
    assert hasattr(writer, 'write_env_file')


def test_ui_mode_enum():
    """Test UIMode enum values."""
    from testteller.core.config import UIMode

    assert UIMode.CLI.value == "cli"
    assert UIMode.TUI.value == "tui"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
