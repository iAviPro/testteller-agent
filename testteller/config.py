# config.py
"""
Configuration module for TestTeller RAG agent.
This module uses Pydantic to manage application settings, including API keys,
ChromaDB settings, and other parameters.
"""
import logging
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Load .env file from the current working directory or parent directories


def load_env():
    """Load environment variables from .env file."""
    current_dir = Path.cwd()
    env_path = None

    # Search for .env in current and parent directories
    while current_dir.parent != current_dir:
        test_path = current_dir / '.env'
        if test_path.is_file():
            env_path = test_path
            break
        current_dir = current_dir.parent

    if env_path:
        load_dotenv(env_path)
        logger.info("Loaded .env from: %s", env_path)
        return True
    return False


# Load .env file
load_env()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global variable to track if settings are loaded
settings = None


class CommonSettings(BaseSettings):
    """Common application settings."""
    APP_NAME: str = "TestTeller RAG Agent"
    APP_VERSION: str = "0.1.1"

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.getcwd(), '.env'),
        env_file_encoding='utf-8',
        extra='ignore',  # Ignore extra fields from .env
        case_sensitive=False
    )


class ApiKeysSettings(BaseSettings):
    """API Key configurations."""
    model_config = SettingsConfigDict(extra='ignore', case_sensitive=False)

    google_api_key: SecretStr = Field(..., env="GOOGLE_API_KEY",
                                      description="Google Gemini API Key")
    github_token: Optional[SecretStr] = Field(
        None, env="GITHUB_TOKEN", description="GitHub Personal Access Token for private repos")

    @field_validator("google_api_key")
    @classmethod
    def validate_google_api_key(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value():
            raise ValueError(
                "GOOGLE_API_KEY environment variable must be set and cannot be empty.")
        return v

    @field_validator("github_token")
    @classmethod
    def validate_github_token(cls, v: Optional[SecretStr]) -> Optional[SecretStr]:
        if v is not None and not v.get_secret_value():
            raise ValueError(
                "GITHUB_TOKEN environment variable, if set, cannot be empty.")
        return v


class ChromaDbSettings(BaseSettings):
    """ChromaDB specific configurations."""
    model_config = SettingsConfigDict(extra='ignore', case_sensitive=False)

    chroma_db_path: str = Field(
        default="./chroma_data_non_prod", env="CHROMA_DB_PATH", description="Path to ChromaDB persistent storage")
    default_collection_name: str = Field(
        default="test_documents_non_prod", env="DEFAULT_COLLECTION_NAME", description="Default ChromaDB collection name")
    chroma_db_host: Optional[str] = Field(
        default=None, env="CHROMA_DB_HOST", description="ChromaDB server host (for HttpClient)")
    chroma_db_port: Optional[int] = Field(
        default=None, env="CHROMA_DB_PORT", description="ChromaDB server port (for HttpClient)")


class GeminiModelSettings(BaseSettings):
    """Gemini model name configurations."""
    model_config = SettingsConfigDict(extra='ignore', case_sensitive=False)

    gemini_embedding_model: str = Field(
        default="text-embedding-004", env="GEMINI_EMBEDDING_MODEL", description="Gemini model for embeddings")
    gemini_generation_model: str = Field(
        default="gemini-2.0-flash", env="GEMINI_GENERATION_MODEL", description="Gemini model for generation")


class TextProcessingSettings(BaseSettings):
    """Text processing configurations for ingestion."""
    model_config = SettingsConfigDict(extra='ignore', case_sensitive=False)

    chunk_size: int = Field(
        default=1000, env="CHUNK_SIZE", description="Size of text chunks for ingestion")
    chunk_overlap: int = Field(
        default=150, env="CHUNK_OVERLAP", description="Overlap between text chunks")


class CodeLoaderSettings(BaseSettings):
    """Code loader specific configurations."""
    model_config = SettingsConfigDict(extra='ignore', case_sensitive=False)

    code_extensions: List[str] = Field(
        default=['.py', '.js', '.java', '.c', '.cpp', '.cs', '.go', '.rb',
                 '.php', '.ts', '.tsx', '.html', '.css', '.md', '.json', '.yml', '.sh', '.yaml', '.ini', '.md'],
        # Pydantic handles comma-separated string from env for List[str]
        env="CODE_EXTENSIONS",
        description="Supported code file extensions"
    )
    temp_clone_dir_base: str = Field(
        default="./temp_cloned_repos", env="TEMP_CLONE_DIR_BASE", description="Base directory for cloning GitHub repos")


class ApiRetrySettings(BaseSettings):
    """API call retry configurations."""
    model_config = SettingsConfigDict(extra='ignore', case_sensitive=False)

    api_retry_attempts: int = Field(
        default=3, env="API_RETRY_ATTEMPTS", description="Number of retry attempts for API calls")
    api_retry_wait_seconds: int = Field(
        default=2, env="API_RETRY_WAIT_SECONDS", description="Initial wait time in seconds for API retries")


class LoggingSettings(BaseSettings):
    """Logging configurations."""
    model_config = SettingsConfigDict(extra='ignore', case_sensitive=False)

    log_level: str = Field(
        default="INFO", env="LOG_LEVEL", description="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)")
    log_format: str = Field(
        default="JSON", env="LOG_FORMAT", description="Log format ('json' or 'text')")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(
                f"Invalid log level '{v}'. Must be one of {allowed_levels}.")
        return v.upper()

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        allowed_formats = ["json", "text"]
        if v.lower() not in allowed_formats:
            raise ValueError(
                f"Invalid log format '{v}'. Must be 'json' or 'text'")
        return v.lower()


class OutputSettings(BaseSettings):
    """Output configurations."""
    model_config = SettingsConfigDict(extra='ignore', case_sensitive=False)

    output_file_path: Optional[str] = Field(
        default="testteller_output.md",
        env="OUTPUT_FILE_PATH",
        description="Path to save the generated output (default: testteller_output.md in current directory)"
    )

    @field_validator("output_file_path")
    @classmethod
    def validate_output_file_path(cls, v: Optional[str]) -> Optional[str]:
        if v and not v.endswith('.md'):
            raise ValueError("Output file path must end with .md extension")
        return v


# --- Main Application Settings ---


class AppSettings(BaseSettings):
    """
    Main application settings, composing settings from different modules.
    All settings can be configured via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.getcwd(), '.env'),
        env_file_encoding='utf-8',
        extra='ignore',  # Ignore extra fields from .env
        case_sensitive=False
    )

    api_keys: ApiKeysSettings = Field(default_factory=ApiKeysSettings)
    chroma_db: ChromaDbSettings = Field(default_factory=ChromaDbSettings)
    gemini_model: GeminiModelSettings = Field(
        default_factory=GeminiModelSettings)
    text_processing: TextProcessingSettings = Field(
        default_factory=TextProcessingSettings)
    code_loader: CodeLoaderSettings = Field(default_factory=CodeLoaderSettings)
    api_retry: ApiRetrySettings = Field(default_factory=ApiRetrySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)

    @classmethod
    def load_settings(cls) -> Optional['AppSettings']:
        """
        Load settings with graceful error handling.
        Returns None if required settings are missing.
        """
        try:
            return cls()
        except ValidationError as e:
            if any('google_api_key' in err['loc'] for err in e.errors()):
                return None
            raise


# Initialize settings with graceful error handling
try:
    settings = AppSettings.load_settings()
except Exception as e:
    print(f"Error loading settings: {e}")
    settings = None
