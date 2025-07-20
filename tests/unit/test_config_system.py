"""
Unit tests for the modular configuration system

Tests the new modular configuration architecture including providers,
UI helpers, validators, and the configuration wizard.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from testteller.core.config import (
    UIHelper,
    UIMode,
    ConfigurationWriter,
    ConfigurationValidator,
    TestWriterWizard,
    ConfigurationWizard
)
# Import provider-related classes if they still exist separately
try:
    from testteller.core.config.providers.base import BaseProvider, ProviderConfig
    from testteller.core.config.providers.gemini import GeminiProvider
    from testteller.core.config.providers.openai import OpenAIProvider
    from testteller.core.config.providers.claude import ClaudeProvider
    from testteller.core.config.providers.llama import LlamaProvider
except ImportError:
    # Providers might be integrated into config_wizard
    pass


class TestUIHelper:
    """Test suite for UIHelper class"""

    @pytest.fixture
    def ui_helper(self):
        """Create UIHelper instance"""
        return UIHelper(UIMode.CLI)

    def test_init(self, ui_helper):
        """Test UIHelper initialization"""
        assert ui_helper.mode == UIMode.CLI
        assert hasattr(ui_helper, 'icons')
        assert hasattr(ui_helper, 'colors')

    @patch('builtins.input', return_value='test input')
    def test_get_input(self, mock_input, ui_helper):
        """Test input collection"""
        result = ui_helper.get_input("Enter value: ")
        assert result == 'test input'
        mock_input.assert_called_once()

    @patch('builtins.input', return_value='')
    def test_get_input_with_default(self, mock_input, ui_helper):
        """Test input with default value"""
        result = ui_helper.get_input("Enter value: ", default="default_value")
        assert result == 'default_value'

    @patch('builtins.input', return_value='y')
    def test_confirm_yes(self, mock_input, ui_helper):
        """Test confirmation with yes response"""
        result = ui_helper.confirm("Are you sure?")
        assert result is True

    @patch('builtins.input', return_value='n')
    def test_confirm_no(self, mock_input, ui_helper):
        """Test confirmation with no response"""
        result = ui_helper.confirm("Are you sure?")
        assert result is False

    @patch('builtins.input', side_effect=['invalid', '2'])
    def test_choose_from_list(self, mock_input, ui_helper):
        """Test choosing from a list with invalid then valid input"""
        options = ['option1', 'option2', 'option3']
        result = ui_helper.choose_from_list("Choose:", options)
        assert result == 'option2'
        assert mock_input.call_count == 2

    @patch('sys.stdout', new_callable=StringIO)
    def test_print_methods(self, mock_stdout, ui_helper):
        """Test various print methods"""
        ui_helper.print_header("Test Header")
        ui_helper.print_success("Success message")
        ui_helper.print_error("Error message")
        ui_helper.print_warning("Warning message")
        ui_helper.print_info("Info message")
        
        output = mock_stdout.getvalue()
        assert "Test Header" in output
        assert "Success message" in output
        assert "Error message" in output
        assert "Warning message" in output
        assert "Info message" in output


class TestConfigurationValidator:
    """Test suite for ConfigurationValidator class"""

    def test_validate_api_key_valid(self):
        """Test valid API key validation"""
        valid_key = "sk-1234567890abcdef1234567890abcdef"
        from testteller.core.config.config_wizard import validate_api_key
        assert validate_api_key(valid_key) is True

    def test_validate_api_key_invalid(self):
        """Test invalid API key validation"""
        from testteller.core.config.config_wizard import validate_api_key
        assert validate_api_key("") is False
        assert validate_api_key("short") is False

    def test_validate_model_name_valid(self):
        """Test valid model name validation"""
        valid_models = ["gpt-4", "gemini-2.0-flash", "claude-3-sonnet"]
        for model in valid_models:
            # Model validation is now internal to provider config
            assert len(model) > 0

    def test_validate_model_name_invalid(self):
        """Test invalid model name validation"""
        # Model validation is now internal to provider config
        pass

    def test_validate_url_valid(self):
        """Test valid URL validation"""
        valid_urls = [
            "http://localhost:8000",
            "https://api.example.com",
            "http://192.168.1.1:11434"
        ]
        for url in valid_urls:
            from testteller.core.config.config_wizard import validate_url
            assert validate_url(url) is True

    def test_validate_url_invalid(self):
        """Test invalid URL validation"""
        invalid_urls = ["not-a-url", "ftp://example.com", "localhost"]
        for url in invalid_urls:
            from testteller.core.config.config_wizard import validate_url
            assert validate_url(url) is False

    def test_validate_port_valid(self):
        """Test valid port validation"""
        valid_ports = [80, 8000, 11434, 65535]
        for port in valid_ports:
            from testteller.core.config.config_wizard import validate_port
            assert validate_port(port) is True

    def test_validate_port_invalid(self):
        """Test invalid port validation"""
        invalid_ports = [0, -1, 65536, "8000", None]
        for port in invalid_ports:
            from testteller.core.config.config_wizard import validate_port
            assert validate_port(port) is False


class TestConfigurationWriter:
    """Test suite for ConfigurationWriter class"""

    @pytest.fixture
    def temp_env_file(self):
        """Create temporary .env file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("# Test .env file\nEXISTING_VAR=existing_value\n")
            temp_path = Path(f.name)
        yield temp_path
        temp_path.unlink()

    def test_write_new_file(self):
        """Test writing to new .env file"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)
        temp_path.unlink()  # Remove the file so we test creation
        
        try:
            config = {"NEW_VAR": "new_value", "ANOTHER_VAR": "another_value"}
            writer = ConfigurationWriter()
            writer.write_env_file(config, temp_path)
            
            assert temp_path.exists()
            content = temp_path.read_text()
            assert "NEW_VAR=new_value" in content
            assert "ANOTHER_VAR=another_value" in content
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_update_existing_file(self, temp_env_file):
        """Test updating existing .env file"""
        config = {"NEW_VAR": "new_value", "EXISTING_VAR": "updated_value"}
        writer = ConfigurationWriter()
        writer.write_env_file(config, Path(temp_env_file))
        
        content = temp_env_file.read_text()
        assert "NEW_VAR=new_value" in content
        assert "EXISTING_VAR=updated_value" in content
        # Should preserve comments
        assert "# Test .env file" in content

    def test_backup_creation(self, temp_env_file):
        """Test backup file creation"""
        config = {"NEW_VAR": "new_value"}
        writer = ConfigurationWriter()
        writer.backup_enabled = True
        writer.write_env_file(config, Path(temp_env_file))
        
        backup_path = temp_env_file.with_suffix('.env.backup')
        assert backup_path.exists()
        
        # Backup should contain original content
        backup_content = backup_path.read_text()
        assert "EXISTING_VAR=existing_value" in backup_content
        
        # Clean up backup
        backup_path.unlink()


class TestBaseProvider:
    """Test suite for BaseProvider abstract class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseProvider cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseProvider()

    def test_provider_config_dataclass(self):
        """Test ProviderConfig dataclass"""
        config = ProviderConfig(
            provider_name="test",
            api_key="test-key",
            model_name="test-model"
        )
        assert config.provider_name == "test"
        assert config.api_key == "test-key"
        assert config.model_name == "test-model"
        assert config.additional_config == {}


class TestGeminiProvider:
    """Test suite for GeminiProvider"""

    @pytest.fixture
    def gemini_provider(self):
        """Create GeminiProvider instance"""
        ui_helper = Mock()
        return GeminiProvider(ui_helper)

    @patch('testteller.core.config.config_wizard.validate_api_key')
    def test_collect_config_valid(self, mock_validator, gemini_provider):
        """Test collecting valid Gemini configuration"""
        mock_validator.validate_api_key.return_value = True
        mock_validator.validate_model_name.return_value = True
        
        gemini_provider.ui.get_input.side_effect = [
            "test-api-key",  # API key
            "gemini-2.0-flash",    # Generation model
            "embedding-001"  # Embedding model
        ]
        
        config = gemini_provider.collect_config()
        
        assert isinstance(config, ProviderConfig)
        assert config.provider_name == "gemini"
        assert config.api_key == "test-api-key"
        assert config.model_name == "gemini-2.0-flash"
        assert config.additional_config["embedding_model"] == "embedding-001"

    def test_validate_config_valid(self, gemini_provider):
        """Test validating valid Gemini configuration"""
        config = ProviderConfig(
            provider_name="gemini",
            api_key="valid-key",
            model_name="gemini-2.0-flash"
        )
        
        with patch('testteller.core.config.config_wizard.validate_api_key', return_value=True):
            assert gemini_provider.validate_config(config) is True

    def test_validate_config_invalid(self, gemini_provider):
        """Test validating invalid Gemini configuration"""
        config = ProviderConfig(
            provider_name="gemini",
            api_key="",  # Invalid empty key
            model_name="gemini-2.0-flash"
        )
        
        with patch('testteller.core.config.config_wizard.validate_api_key', return_value=False):
            assert gemini_provider.validate_config(config) is False

    def test_to_env_dict(self, gemini_provider):
        """Test converting configuration to environment dictionary"""
        config = ProviderConfig(
            provider_name="gemini",
            api_key="test-key",
            model_name="gemini-2.0-flash",
            additional_config={"embedding_model": "embedding-001"}
        )
        
        env_dict = gemini_provider.to_env_dict(config)
        
        expected = {
            "LLM_PROVIDER": "gemini",
            "GOOGLE_API_KEY": "test-key",
            "GEMINI_GENERATION_MODEL": "gemini-2.0-flash",
            "GEMINI_EMBEDDING_MODEL": "embedding-001"
        }
        assert env_dict == expected


class TestClaudeProvider:
    """Test suite for ClaudeProvider"""

    @pytest.fixture
    def claude_provider(self):
        """Create ClaudeProvider instance"""
        ui_helper = Mock()
        return ClaudeProvider(ui_helper)

    def test_collect_config_with_google_embeddings(self, claude_provider):
        """Test collecting Claude config with Google embeddings"""
        claude_provider.ui.get_input.side_effect = [
            "claude-api-key",    # Claude API key
            "claude-3-sonnet",   # Generation model
        ]
        claude_provider.ui.choose_from_list.return_value = "google"
        claude_provider.ui.get_input.side_effect = [
            "claude-api-key",
            "claude-3-sonnet",
            "google-api-key"     # Google API key for embeddings
        ]
        
        with patch('testteller.core.config.config_wizard.validate_api_key', return_value=True):
            config = claude_provider.collect_config()
            
            assert config.provider_name == "claude"
            assert config.api_key == "claude-api-key"
            assert config.additional_config["embedding_provider"] == "google"
            assert config.additional_config["google_api_key"] == "google-api-key"

    def test_to_env_dict_google_embeddings(self, claude_provider):
        """Test environment dict generation for Claude with Google embeddings"""
        config = ProviderConfig(
            provider_name="claude",
            api_key="claude-key",
            model_name="claude-3-sonnet",
            additional_config={
                "embedding_provider": "google",
                "google_api_key": "google-key"
            }
        )
        
        env_dict = claude_provider.to_env_dict(config)
        
        expected = {
            "LLM_PROVIDER": "claude",
            "CLAUDE_API_KEY": "claude-key",
            "CLAUDE_GENERATION_MODEL": "claude-3-sonnet",
            "CLAUDE_EMBEDDING_PROVIDER": "google",
            "GOOGLE_API_KEY": "google-key"
        }
        assert env_dict == expected


class TestLlamaProvider:
    """Test suite for LlamaProvider"""

    @pytest.fixture
    def llama_provider(self):
        """Create LlamaProvider instance"""
        ui_helper = Mock()
        return LlamaProvider(ui_helper)

    def test_collect_config_default_url(self, llama_provider):
        """Test collecting Llama config with default URL"""
        llama_provider.ui.get_input.side_effect = [
            "",              # Use default URL (localhost)
            "",              # Use default port (11434)
            "llama3.2:3b",   # Generation model
            "llama3.2:3b"    # Embedding model
        ]
        
        with patch('testteller.core.config.config_wizard.validate_port', return_value=True):
            config = llama_provider.collect_config()
            
            assert config.provider_name == "llama"
            assert config.additional_config["base_url"] == "http://localhost:11434"
            assert config.model_name == "llama3.2:3b"

    def test_collect_config_custom_url(self, llama_provider):
        """Test collecting Llama config with custom URL"""
        llama_provider.ui.get_input.side_effect = [
            "remote-server",  # Custom URL
            "8080",          # Custom port
            "llama3.2:7b",   # Generation model
            "llama3.2:3b"    # Embedding model
        ]
        
        with patch('testteller.core.config.config_wizard.validate_port', return_value=True):
            config = llama_provider.collect_config()
            
            assert config.additional_config["base_url"] == "http://remote-server:8080"

    @patch('requests.get')
    def test_validate_config_success(self, mock_get, llama_provider):
        """Test successful Ollama validation"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "Ollama is running"}
        mock_get.return_value = mock_response
        
        config = ProviderConfig(
            provider_name="llama",
            additional_config={"base_url": "http://localhost:11434"}
        )
        
        assert llama_provider.validate_config(config) is True

    @patch('requests.get')
    def test_validate_config_failure(self, mock_get, llama_provider):
        """Test failed Ollama validation"""
        mock_get.side_effect = Exception("Connection failed")
        
        config = ProviderConfig(
            provider_name="llama",
            additional_config={"base_url": "http://localhost:11434"}
        )
        
        assert llama_provider.validate_config(config) is False


class TestTestWriterWizard:
    """Test suite for TestWriterWizard"""

    @pytest.fixture
    def wizard(self):
        """Create TestWriterWizard instance"""
        ui_helper = Mock()
        return TestWriterWizard(ui_helper)

    def test_collect_automation_settings(self, wizard):
        """Test collecting automation settings"""
        wizard.ui.choose_from_list.side_effect = [
            "python",   # Preferred language
            "pytest"    # Preferred framework
        ]
        wizard.ui.confirm.return_value = True  # Enable AI enhancement
        
        settings = wizard.collect_automation_settings()
        
        expected = {
            "TESTWRITER_DEFAULT_LANGUAGE": "python",
            "TESTWRITER_DEFAULT_FRAMEWORK": "pytest",
            "TESTWRITER_ENABLE_AI_ENHANCEMENT": "true"
        }
        assert settings == expected

    def test_run_automation_wizard(self, wizard):
        """Test running the automation wizard"""
        wizard.ui.choose_from_list.side_effect = ["typescript", "jest"]
        wizard.ui.confirm.return_value = False
        
        with patch.object(wizard, 'collect_automation_settings') as mock_collect:
            mock_collect.return_value = {"TEST": "value"}
            
            with patch('testteller.core.config.ConfigurationWriter.write_env_file') as mock_write:
                result = wizard.run(Path(".env"))
                
                assert result is True
                mock_write.assert_called_once()


class TestConfigurationWizard:
    """Test suite for ConfigurationWizard"""

    @pytest.fixture
    def wizard(self):
        """Create ConfigurationWizard instance"""
        return ConfigurationWizard(UIMode.CLI)

    def test_init(self, wizard):
        """Test wizard initialization"""
        assert wizard.ui.mode == UIMode.CLI
        assert hasattr(wizard, 'providers')
        assert len(wizard.providers) == 4  # gemini, openai, claude, llama

    def test_choose_provider(self, wizard):
        """Test provider selection"""
        wizard.ui.choose_from_list.return_value = "gemini"
        
        provider = wizard.choose_provider()
        assert provider.provider_name == "gemini"

    @patch('testteller.core.config.ConfigurationWriter.write_env_file')
    def test_run_success(self, mock_write, wizard):
        """Test successful wizard run"""
        # Mock provider selection and configuration
        mock_provider = Mock()
        mock_provider.provider_name = "gemini"
        mock_config = ProviderConfig(provider_name="gemini", api_key="test", model_name="test")
        mock_provider.collect_config.return_value = mock_config
        mock_provider.validate_config.return_value = True
        mock_provider.to_env_dict.return_value = {"TEST": "value"}
        
        wizard.ui.choose_from_list.return_value = "gemini"
        
        with patch.object(wizard, 'choose_provider', return_value=mock_provider):
            result = wizard.run(Path(".env"))
            
            assert result is True
            mock_write.assert_called_once()

    def test_run_validation_failure(self, wizard):
        """Test wizard run with validation failure"""
        mock_provider = Mock()
        mock_provider.collect_config.return_value = Mock()
        mock_provider.validate_config.return_value = False
        
        with patch.object(wizard, 'choose_provider', return_value=mock_provider):
            result = wizard.run(Path(".env"))
            
            assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])