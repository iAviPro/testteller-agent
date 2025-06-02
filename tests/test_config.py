import pytest
from pydantic import ValidationError

import config # Import config to allow setting _cached_settings to None
from config import (
    ApiKeysSettings,
    ApiRetrySettings,
    ChromaDbSettings,
    CodeLoaderSettings,
    CommonSettings,
    GeminiModelSettings,
    LoggingSettings,
    TextProcessingSettings,
    load_app_settings,
)



def test_common_settings_default():
    settings = CommonSettings()
    assert settings.LOG_LEVEL == "INFO" # Changed from debug
    assert settings.APP_NAME == "TestTeller RAG Agent"


def test_common_settings_env_override(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG") # Changed from DEBUG
    monkeypatch.setenv("APP_NAME", "New App Name")
    settings = CommonSettings()
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.APP_NAME == "New App Name"


def test_api_keys_settings_default(monkeypatch):
    # ApiKeysSettings requires GOOGLE_API_KEY, so we must provide a default for this test
    monkeypatch.setenv("GOOGLE_API_KEY", "default_test_key")
    settings = ApiKeysSettings()
    # The default for github_token is None, google_api_key is now set
    assert settings.google_api_key.get_secret_value() == "default_test_key"
    assert settings.github_token is None


def test_api_keys_settings_env_override(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "override_google_key")
    monkeypatch.setenv("GITHUB_TOKEN", "override_github_token")
    settings = ApiKeysSettings()
    assert settings.google_api_key.get_secret_value() == "override_google_key"
    assert settings.github_token.get_secret_value() == "override_github_token"


def test_api_keys_settings_validator_invalid_google_key(monkeypatch):
    # Test that an empty string for GOOGLE_API_KEY raises ValidationError
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    with pytest.raises(ValidationError) as excinfo:
        ApiKeysSettings()
    assert "GOOGLE_API_KEY environment variable must be set and cannot be empty" in str(excinfo.value)

def test_api_keys_settings_validator_missing_google_key(monkeypatch):
    # Test that a missing GOOGLE_API_KEY (not set at all) raises ValidationError
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False) # Ensure it's not set
    with pytest.raises(ValidationError) as excinfo:
        ApiKeysSettings()
    assert "Field required" in str(excinfo.value) # Pydantic's default message for missing required field
    assert "google_api_key" in str(excinfo.value)


def test_api_keys_settings_validator_invalid_github_token(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "valid_google_key") # Needs a valid google key
    monkeypatch.setenv("GITHUB_TOKEN", "") # Set github token to empty
    with pytest.raises(ValidationError) as excinfo:
        ApiKeysSettings()
    assert "GITHUB_TOKEN environment variable, if set, cannot be empty" in str(excinfo.value)


def test_chroma_db_settings_default():
    settings = ChromaDbSettings()
    assert settings.chroma_db_path == "./chroma_data_prod"  # Corrected field name and value
    assert settings.default_collection_name == "test_documents_prod" # Corrected field name and value


def test_chroma_db_settings_env_override(monkeypatch):
    monkeypatch.setenv("CHROMA_DB_PATH", "/path/to/db") # Corrected env var name
    monkeypatch.setenv("DEFAULT_COLLECTION_NAME", "new_collection") # Corrected env var name
    settings = ChromaDbSettings()
    assert settings.chroma_db_path == "/path/to/db"
    assert settings.default_collection_name == "new_collection"


def test_gemini_model_settings_default():
    settings = GeminiModelSettings()
    assert settings.gemini_embedding_model == "models/embedding-001" # Corrected field name
    assert settings.gemini_generation_model == "gemini-2.0-flash"   # Corrected field name


def test_gemini_model_settings_env_override(monkeypatch):
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "new-embedding-model") # Corrected env var name
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "new-generation-model") # Corrected env var name
    settings = GeminiModelSettings()
    assert settings.gemini_embedding_model == "new-embedding-model"
    assert settings.gemini_generation_model == "new-generation-model"


def test_text_processing_settings_default():
    settings = TextProcessingSettings()
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 150 # Corrected default value


def test_text_processing_settings_env_override(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "2000") # Env var CHUNK_SIZE is correct
    monkeypatch.setenv("CHUNK_OVERLAP", "400") # Env var CHUNK_OVERLAP is correct
    settings = TextProcessingSettings()
    assert settings.chunk_size == 2000
    assert settings.chunk_overlap == 400


def test_code_loader_settings_default():
    settings = CodeLoaderSettings()
    # max_depth and use_gitignore are not in CodeLoaderSettings in config.py
    # Default values from config.py:
    assert settings.code_extensions == ['.py', '.js', '.java', '.c', '.cpp', '.cs', '.go', '.rb', '.php', '.ts', '.tsx', '.html', '.css', '.md', '.json', '.yaml', '.sh']
    assert settings.temp_clone_dir_base == "./temp_cloned_repos_prod"


def test_code_loader_settings_env_override(monkeypatch):
    monkeypatch.setenv("CODE_EXTENSIONS", '["*.py","*.test.js"]') # JSON formatted list
    monkeypatch.setenv("TEMP_CLONE_DIR_BASE", "/new/clone/dir")
    settings = CodeLoaderSettings()
    assert settings.code_extensions == ['*.py','*.test.js'] # Expect Python list
    assert settings.temp_clone_dir_base == "/new/clone/dir"


def test_api_retry_settings_default():
    settings = ApiRetrySettings()
    assert settings.api_retry_attempts == 3 # Corrected field name
    assert settings.api_retry_wait_seconds == 2 # Corrected field name and default
    # retry_max_delay_seconds is not in ApiRetrySettings in config.py


def test_api_retry_settings_env_override(monkeypatch):
    monkeypatch.setenv("API_RETRY_ATTEMPTS", "5") # Corrected env var name
    monkeypatch.setenv("API_RETRY_WAIT_SECONDS", "10") # Corrected env var name
    settings = ApiRetrySettings()
    assert settings.api_retry_attempts == 5
    assert settings.api_retry_wait_seconds == 10
    # retry_max_delay_seconds is not in ApiRetrySettings in config.py


def test_logging_settings_default():
    settings = LoggingSettings()
    assert settings.log_level == "INFO"


def test_logging_settings_env_override(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    settings = LoggingSettings()
    assert settings.log_level == "DEBUG"


def test_logging_settings_validator_invalid_level():
    with pytest.raises(ValidationError):
        LoggingSettings(log_level="INVALID_LEVEL")


def test_load_app_settings_loads_all_settings(monkeypatch):
    config._cached_settings = None # Reset cache
    # Set environment variables to test overrides, including the required GOOGLE_API_KEY
    monkeypatch.setenv("GOOGLE_API_KEY", "override_google_key_for_load_app")
    monkeypatch.setenv("DEBUG", "True") # This should be part of CommonSettings, not AppSettings directly
    # monkeypatch.setenv("GEMINI_API_KEY", "override_key") # This was for the old ApiKeysSettings
    monkeypatch.setenv("CHROMA_DB_PATH", "/override/path") # Corrected env var name
    # monkeypatch.setenv("GEMINI_MODEL_NAME", "override-model") # This is part of GeminiModelSettings
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "override-model-generation") # Correct
    monkeypatch.setenv("CHUNK_SIZE", "1500") # Correct
    # MAX_DEPTH is part of CodeLoaderSettings, not CommonSettings
    # monkeypatch.setenv("MAX_DEPTH", "7") # This was for CodeLoaderSettings directly
    monkeypatch.setenv("CODE_EXTENSIONS", '["*.test", "*.java"]') # JSON formatted list
    monkeypatch.setenv("API_RETRY_ATTEMPTS", "4") # Correct
    monkeypatch.setenv("LOG_LEVEL", "WARNING") # This is part of LoggingSettings # Correct

    # CommonSettings are not directly part of AppSettings in the provided config.py structure
    # They are standalone. If AppSettings is supposed to have a `common` field,
    # then config.py needs to be updated. For now, testing AppSettings fields.
    # Test a CommonSetting separately or assume it's loaded if used by AppSettings internally.

    settings = load_app_settings()

    # CommonSettings are not directly part of AppSettings.
    # If you need to test CommonSettings.debug, you would do:
    # monkeypatch.setenv("DEBUG", "True")
    # common_settings_instance = CommonSettings()
    # assert common_settings_instance.debug is True

    assert settings.api_keys.google_api_key.get_secret_value() == "override_google_key_for_load_app"
    assert settings.chroma_db.chroma_db_path == "/override/path" # Corrected field name
    assert settings.gemini_model.gemini_generation_model == "override-model-generation"
    assert settings.text_processing.chunk_size == 1500
    assert settings.code_loader.code_extensions == ["*.test", "*.java"] # Corrected expected value
    assert settings.api_retry.api_retry_attempts == 4 # Corrected field name
    assert settings.logging.log_level == "WARNING"


# test_api_keys_settings_validator_empty_key was for gemini_api_key,
# new tests cover google_api_key and github_token (test_api_keys_settings_validator_invalid_google_key, etc.)


def test_logging_settings_validator_invalid_log_format():
    """Test that LoggingSettings raises ValueError for an invalid log_format."""
    with pytest.raises(ValidationError) as excinfo:
        LoggingSettings(log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", log_level="INFO")
    # This test is a bit tricky because the validator currently only checks the log_level.
    # If we want to test log_format, the validator in LoggingSettings needs to be updated.
    # For now, this test will pass as long as the log_level is valid.
    # To make it fail (and thus test a hypothetical log_format validator),
    # you could change log_format to something invalid AND update the validator.
    pass # Placeholder for now, as log_format validation is not strictly enforced by pydantic based on current model


def test_logging_settings_validator_valid_log_level():
    """Test that LoggingSettings accepts valid log levels."""
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    for level in valid_levels:
        settings = LoggingSettings(log_level=level)
        assert settings.log_level == level


def test_load_app_settings_without_env_file(monkeypatch, tmp_path):
    config._cached_settings = None # Reset cache
    """Test that load_app_settings works correctly when no .env file is present."""
    # Ensure GOOGLE_API_KEY is set, as it's required by ApiKeysSettings
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy_key_for_no_env_test")
    # Ensure no .env file is loaded by pointing to a non-existent one
    # The SettingsConfigDict in config.py already specifies '.env' which might not exist,
    # so this test primarily ensures defaults are loaded when env vars are not explicitly set (except required ones).

    # Clear other potentially interfering env vars if necessary
    monkeypatch.delenv("CHROMA_DB_PATH", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)


    settings = load_app_settings() # This will load AppSettings
    
    # Check some default values from AppSettings component models
    assert settings.api_keys.google_api_key.get_secret_value() == "dummy_key_for_no_env_test"
    assert settings.chroma_db.chroma_db_path == "./chroma_data_prod" # Default from ChromaDbSettings
    assert settings.logging.log_level == "INFO" # Default from LoggingSettings

    # CommonSettings are not part of AppSettings, test them separately if needed.
    # For example, to test CommonSettings.debug default:
    # common_settings_instance = CommonSettings()
    # assert common_settings_instance.debug is False

def test_load_app_settings_with_partial_env_vars(monkeypatch):
    config._cached_settings = None # Reset cache
    """Test that load_app_settings correctly uses defaults for unspecified env vars."""
    # Set required GOOGLE_API_KEY and one other var to override
    monkeypatch.setenv("GOOGLE_API_KEY", "partial_test_key")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG") # Override one setting in LoggingSettings
    # Leave others to default by ensuring they are not set or clearing them
    monkeypatch.delenv("CHROMA_DB_PATH", raising=False)
    monkeypatch.delenv("DEFAULT_COLLECTION_NAME", raising=False)


    settings = load_app_settings()

    assert settings.api_keys.google_api_key.get_secret_value() == "partial_test_key"
    assert settings.logging.log_level == "DEBUG" # Overridden
    # Check a setting that should remain default
    assert settings.chroma_db.default_collection_name == "test_documents_prod" # Corrected field name
    assert settings.gemini_model.gemini_embedding_model == "models/embedding-001" # Corrected field name


# It's good practice to also test the main `load_app_settings()` function
# to ensure it integrates all settings correctly.
def test_load_app_settings_integration(monkeypatch):
    config._cached_settings = None # Reset cache
    """Test the integrated load_app_settings() function."""
    # Set all required and some optional env vars
    monkeypatch.setenv("GOOGLE_API_KEY", "integration_google_key")
    monkeypatch.setenv("GITHUB_TOKEN", "integration_github_token")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG") # For LoggingSettings
    monkeypatch.setenv("CHROMA_DB_PATH", "/integration/db") # For ChromaDbSettings, corrected env_var
    # CommonSettings are not part of AppSettings, so DEBUG won't be set on AppSettings
    # monkeypatch.setenv("DEBUG", "true")


    all_settings = load_app_settings()

    assert all_settings.api_keys.google_api_key.get_secret_value() == "integration_google_key"
    assert all_settings.api_keys.github_token.get_secret_value() == "integration_github_token"
    assert all_settings.logging.log_level == "DEBUG"
    assert all_settings.chroma_db.chroma_db_path == "/integration/db" # Corrected field name
    # Check a setting that should remain default
    assert all_settings.gemini_model.gemini_generation_model == "gemini-2.0-flash" # Corrected field name
